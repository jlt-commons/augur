# augur: plan

augur is the statistics core of bayesian-news-model (bnm), taken out as its
own library: Bayesian posteriors that learn online with forgetting,
calibration of probabilities against outcomes, event-probability (hazard)
models, combining forecasters, change-point detection, and forecast scoring.

There are two consumers:

- bnm, which keeps its event model and its news domain and takes the
  statistics from augur;
- samizdat, which uses augur to make decisions from evidence and to learn a
  project over time: which steers work, which memories help, when a branch
  has stalled, how far to trust a judge, what a person allows.

Steps 1, 3 and 4 of the order of work are done in v0.1.0 (everything that
moves, `beta`, `gamma`, `layer`, `ledger`); step 2 is bnm's switch.

## Constraints

1. **No dependencies.** Plain Jolt (`clojure.core`, `Math`); runs on the
   same Jolt floor as its consumers (samizdat is at 0.8.22).
2. **Pure functions over values.** No IO, no clock reads, no logging, no
   atoms. Time is an argument (a run number, a turn, a bin), never read
   from the system. The random generator is a value (`augur.math/rng`), so
   the same seed gives the same result.
3. **No constants that decide behaviour.** Every prior, forgetting rate,
   shrinkage and bin edge is an argument, with a `defaults` map per
   namespace to read from. samizdat keeps its numbers in `gates.edn` and
   passes them in; bnm passes the values it uses today. A `def` holding a
   tuning number is a bug in this library.
4. **Persisted state is EDN.** Anything a consumer stores — posteriors,
   ledgers, calibration tables, fitted coefficients — is maps, vectors and
   numbers, readable with `clojure.edn/read-string`. Primitive arrays are
   allowed inside a computation and in hot fitting loops, never in returned
   state. Where bnm returns `double[]` today (`hazard/fit`), augur returns a
   vector, or offers `->state` / `<-state` beside the array version.
5. **Bounded state.** Anything that grows with use (a ledger's resolved rows,
   a key space) has a cap given by the caller, and what falls off the end is
   folded into decayed counts instead of being lost.
6. **Safe on a fiber.** Nothing parks, so samizdat's fiber rules
   (no parking inside lazy bodies) are met trivially. Keep it that way: no
   callbacks into consumer code from inside `map`/`for`.
7. **Cheap enough to call every turn.** Updating a posterior or a ledger is
   O(1) in the history. Fitting (`logit/fit`, `trees/fit`) is for run end or
   offline, and documented as such.
8. **Licence: Apache-2.0**, as bnm is. samizdat (GPL-3.0) can depend on it.
   `samizdat.stats` moves in under the same licence, which its author
   can grant; it ports thinkingbox's `eval_utils.py`, whose licence has to be
   checked before the move.

## What moves from bnm

| bnm namespace | augur namespace | Notes |
|---|---|---|
| `bnm.math` | `augur.math` | log-gamma, logit/expit, clamp, the seedable rng, normal/gamma/beta/poisson/negative-binomial samplers, quantile, `solve-spd`, normal quantile and cdf, Nelder-Mead. Moves whole. |
| `bnm.fstats` special functions | `augur.special` | `gamma-p`, `chi2-sf`, `phi`, `phi-inv`, incomplete beta, `t-cdf`, merged with `samizdat.stats`'s `lgamma`, `betainc`, `betaincinv`. One implementation of each, held to scipy values in the tests (samizdat's `stats_test` already pins them). |
| `bnm.calibrate`: `isotonic`, `calibrate`, `learn`, `isotonic-map`, `logistic-map` | `augur.calibrate` | Probability recalibration. `edges`, `shrink` and `forget` become arguments. The key is any value, not a "channel". |
| `bnm.calibrate`: spread, coverage, shock, `widen*` | stays in bnm | These recalibrate bnm's simulated count samples by `[level channel stratum]` rows. They can move later, if a second use for widening a predictive sample appears. |
| `bnm.hazard`: `fit`, `fit-shrunk`, `predict` | `augur.logit` | Weighted Bayesian logistic regression: ridge, or the horseshoe-like reweighting. Rows are `{:x :y :w}`. |
| `bnm.hazard`: `pool-weights`, `pool-weights-ml`, `hedge-weights`, `median-pool`, `pool`, `fit-stack`, `predict-stack` | `augur.pool` | Combining forecasters. |
| `bnm.hazard`: `auc`, `auc-within`, `avg-precision`, `recall-parts`, `alarm-shares`, `recall-at-budget`, `brier`, `usefulness`, `area-skill`, `matched-concordance`, the paired and bootstrap gains, `cusum`, `recalibration-gain`, `stack-out-of-sample`, `pick-model` | `augur.score` | Scoring and comparing probability forecasts. |
| `bnm.fstats` forecast tests | `augur.ftest` | Diebold-Mariano, Clark-West, out-of-sample R², Pesaran-Timmermann, QLIKE, CRPS (normal, t), log scores, Kupiec, Christoffersen, Berkowitz, Newey-West, Hansen's SPA. |
| `bnm.fstats` Sharpe family | stays in bnm | Trading-specific (`sharpe-stats`, probabilistic and deflated Sharpe). |
| `bnm.changepoint` | `augur.changepoint` | Adams & MacKay online change-point detection. `max-run` becomes an argument. |
| `bnm.trees` | `augur.trees` | Shallow gradient-boosted trees for probabilities. |
| `bnm.analog` | `augur.analog` | Nearest-past-cases forecasts, shrunk toward the base rate. |
| `bnm.eval`: `pit-bounds`, `pit-histogram`, the CRPS of a sample | `augur.score` | Only these generic parts; the rest of `eval` reads bnm's node history. |

Stays in bnm: `model` (the self-exciting event model), `track`, the rest of
`eval`, `store`, and the whole domain (`geo`, `ontology`, `events`,
`extract`, `series`, `structure`, `slow`, `stocks`, `polecon`, `trade`,
`vol`, `mix`, `ties`, `sync`, `calm`, `invariants`, `onset`, `time`, `csv`).

`bnm.model` could be generalised later (number of channels and node kinds
as configuration, not taken from `ontology` and `geo`). No samizdat
requirement calls for it now, so it is out of scope.

## What is new

bnm has the ideas but not as a reusable API. samizdat's requirements below
need these.

### `augur.beta` — keyed Beta posteriors with forgetting

The workhorse: how likely something is to happen, learned from outcomes,
with old evidence fading. It generalises `samizdat.stats` (which only reads a
posterior from k of n) and the per-bin counts inside `bnm.calibrate/learn`.

```clojure
(beta/prior a b)                        ; => {:a a :b b :a0 a :b0 b}
(beta/jeffreys)                         ; Beta(1/2, 1/2)
(beta/from-counts k n prior)            ; posterior after k successes in n
(beta/observe post outcome opts)        ; one outcome (true/false, or a weight in [0 1])
                                        ; opts {:forget f :weight w}: decay, then add
(beta/decay post f steps)               ; evidence fades toward the prior:
                                        ; a' = a0 + f^steps * (a - a0), same for b
(beta/mean post)
(beta/evidence post)                    ; a + b - a0 - b0: effective observations
(beta/interval post level)              ; central credible interval [lo hi]
(beta/prob-above post x)                ; P(p > x)
(beta/prob-between post lo hi)          ; samizdat's prob-in-zone
(beta/prob-greater post-a post-b)       ; P(p_A > p_B), samizdat's prob-a-gt-b
(beta/sample rng post)                  ; [x rng'], for Thompson sampling
(beta/pass-at-k n c k) (beta/pass-power-k n c k)   ; moved from samizdat.stats
```

`decay` takes a step count because samizdat's clock is irregular: a memory
idle for five runs decays by `f^5` at once, read lazily from the stored
`idle_runs`, without a write each run.

### `augur.layer` — shrinking toward a parent

Empirical Bayes across samizdat's layers (project, machine, shipped
default), and across bnm's hierarchy. A child's prior is its parent's
posterior mean at a fixed strength, so a new project starts from the
machine's experience and earns its own estimate as evidence comes in.

```clojure
(layer/prior-from parent-post strength)    ; Beta prior with mean = parent mean,
                                           ; a0 + b0 = strength
(layer/posterior [child parent grandparent] {:strength s})
    ; folds from the top: each layer's prior is the one above's posterior,
    ; at strength s (one number, or one per layer)
(layer/explain [child parent ...] opts)
    ; {:mean :interval :evidence :by-layer [{:mean :evidence} ...]}
    ; what a supervisor or a person reads: how much is this project's own
```

The same for rates (Gamma) is wanted, see `augur.gamma`.

### `augur.gamma` — keyed Gamma-Poisson rates with forgetting

How often something happens per unit of exposure (tool errors per turn,
compactions per 100 turns), the same operations as `augur.beta`:
`prior`, `observe` with `{:count :exposure :forget}`, `decay`, `mean`,
`interval`, `prob-above`, `sample`, and `layer` support. bnm's centred,
scale-free priors are the model for the defaults.

### `augur.ledger` — forecasts registered, resolved, scored, recalibrated

Generalises the cycle `bnm.track` runs for bnm's nodes. A forecast is made
now, its outcome is known later, and the scores of past forecasts recalibrate
new ones. This is what lets samizdat measure its own decisions continuously
instead of by hand after a campaign.

```clojure
(ledger/empty opts)
    ; opts {:max-open n :max-rows n :forget f :calibration {...}}

(ledger/register L {:id id :key k :p p :at t
                    :bench {:prior 0.4 :always 0.0}   ; optional benchmark forecasts
                    :x features})                     ; optional, kept for refitting
    ; => L. An id already open is an error, not an overwrite.

(ledger/resolve L id outcome {:at t})
    ; => L. Scores the forecast and its benchmarks (Brier, log), adds to the
    ;    key's calibration table and decayed score sums, keeps the row
    ;    (bounded by :max-rows, oldest folded into the sums).
    ;    Resolving an unknown id is a no-op that `ledger/stats` counts.

(ledger/expire L pred {:at t})
    ; => L. Open forecasts that will never resolve (their run ended) are
    ;    dropped and counted per key, never scored as negatives.

(ledger/calibrated L k p)        ; p recalibrated from k's history, or p itself
(ledger/record L k)
    ; {:n :brier :log :skill {bench-name skill} :reliability [[p-bin rate n] ...]
    ;  :open n :expired n}
(ledger/rows L k)                ; resolved rows {:p :y :x :at}, for refitting
(ledger/keys L)
```

The key is any EDN value (`[:gate :no-edits "glm"]`). One ledger can hold
many decisions. A consumer that wants layers keeps one ledger per layer and
combines the per-key posteriors with `augur.layer`.

## samizdat's requirements, as API

samizdat's architecture fixes where each piece goes:

- augur calls and the ledger are mechanism, in `src/`;
- the decision drawn from a posterior is made in a cell;
- the priors, forgetting rates and strengths live in `gates.edn`.

Every learned table also has to be discoverable: the supervisor can list it
and read it with its evidence, under `introspect` or a tool of its own.
samizdat stores augur state as EDN in its sqlite database (per project) and
in `~/.config/samizdat` (per machine).

The uses, most valuable first.

### R1. Whether a gate works

A gate fires and steers the branch. Whether the branch did what it asked is
known a turn or a few later. Today compliance is counted by hand from the
events table (39 of 44 obeyed on one run, 0 of 6 on another).

- On firing: `ledger/register` with key `[:gate gate-id model-family]`,
  `p` from the layered posterior, `:at` the turn, `:bench` the shipped
  default rate.
- On the turn that settles it: `ledger/resolve` with whether the branch
  obeyed. What "obeyed" means per gate is samizdat's, not augur's.
- At run end: `ledger/expire` what is still open.
- A cell reads `layer/explain` over `[project machine shipped]` and decides
  whether to keep the gate as a suggestion, escalate it to a withhold, or
  silence it. Thresholds for those come from `gates.edn`.

Needs: `augur.beta`, `augur.layer`, `augur.ledger`.

### R2. What a memory is worth

`samizdat.memory/effective-salience` adds hand-tuned bonuses (recency,
effectiveness, log-damped corroboration, confidence). The rows already keep
`success_count`, `failure_count`, `corroborations` and `idle_runs`.

- `beta/from-counts` on success and failure under a prior per memory kind
  (from `gates.edn`), then `beta/decay` by `idle_runs` at the configured
  per-run forgetting.
- Rank by `beta/mean`, or by `beta/sample` when the recall step should
  explore memories with little evidence. The rng is seeded per run so a
  replay recalls the same memories.
- Corroboration by other runs enters as evidence weight, not as a bonus.

Needs: `augur.beta` with `decay` by step count and a seedable `sample`.

### R3. When a branch has stalled

Beam supersession and the stall gates use fixed turn counts. A run that
spent 238 turns re-reading correct code and a healthy run that looked
stalled both show that turn counts alone misjudge it.

- Offline or at run end, from past tapes: rows `{:x features :y shipped?
  :w weight}` with features such as turns since the last write, the trend
  in failing tests, repeated reads. Fit with `logit/fit-shrunk`, score with
  `score/auc`, `score/avg-precision`, `score/brier` and
  `score/paired-gain` against the turn-count rule as the benchmark.
  Coefficients are stored as an EDN vector with the feature names beside
  them.
- Per turn: `logit/predict` for the chance the branch ships, and
  `changepoint/run` (or an incremental form, see below) over the branch's
  progress series for the chance it broke from its trend recently.
- The beam's cell decides from those, with thresholds in `gates.edn`.

Needs: `augur.logit` with vector state, `augur.score`, `augur.changepoint`
with an incremental API:

```clojure
(changepoint/init {:hazard h :prior-var v :max-run n})
(changepoint/step state x)       ; => [state' {:recent p :run expected-length}]
```

bnm's `run` over a whole series stays, written on top of `step`.

### R4. How far to trust the judge and critic

The judge and critic give verdicts; whether verification later passed is the
outcome.

- `ledger/register` per verdict, keyed `[:judge verdict-kind model-family]`,
  `p` the stated confidence or the verdict's prior rate.
- `ledger/calibrated` turns a verdict into a calibrated probability;
  `calibrate/isotonic-map` or `calibrate/logistic-map` over
  `ledger/rows` when refitted in bulk.

Needs: `augur.ledger`, `augur.calibrate` with keys of any value.

### R5. What a person allows

In attended mode a person's answer to an ask is a label. In auto mode the
classifier decides.

- `augur.beta` per `[command-head path-class reason]`, layered project over
  machine, forgetting per run.
- The posterior only adds evidence to the classifier's prompt, or lets a
  class a person has allowed many times running skip the model call. It
  never overrides the deterministic reach check, and never learns to allow
  a write outside the project. Those are samizdat's rules; augur provides
  the posterior and its interval, and samizdat requires the interval's
  lower end to clear the threshold, not just the mean.

Needs: `augur.beta`, `augur.layer`.

### R6. The arena's reliability statistics

`dev/samizdat/dev/arena.clj` uses `samizdat.stats` (`cred-int`,
`prob-in-zone`, `pass-at-k`, `pass-power-k`, `prob-a-gt-b`). These move to
`augur.beta` and `augur.special`. The values must match what `stats_test`
pins today. `samizdat.stats` is then deleted, and `base_test` and the test
runner updated.

### Requirements that apply to all of them

- **Every number is passed in.** samizdat reads them from `gates.edn`, with
  provenance, as for every threshold.
- **State round-trips through EDN**: `(= s (edn/read-string
  (pr-str s)))` for every state type, tested.
- **Explanations.** Every posterior and ledger key can produce a plain map
  (mean, interval, evidence, per-layer share) that samizdat can render into
  a prompt or a TUI panel without knowing augur's internals.
- **Deterministic.** Given the same state, inputs and seed, the same
  output, so samizdat's replay can check a run.
- **Missing state is empty, not an error.** `nil` for a posterior means the
  prior; `nil` for a ledger means `ledger/empty`.

## Order of work

1. **Create augur with what moves unchanged**: `math`, `special` (merged with
   `samizdat.stats`), `calibrate` (probability part, constants as
   arguments), `logit`, `pool`, `score`, `ftest`, `changepoint`, `trees`,
   `analog`. Port bnm's tests for each, and samizdat's `stats_test`.
2. **Switch bnm to augur** as a git dependency pinned to a full sha. bnm
   passes today's constants explicitly. The check: a pinned backtest
   (one saved origin through `bnm.refit`) gives the same report as before
   the switch, number for number.
3. **Add `beta`, `gamma`, `layer`**, with tests that the forgetting and
   layering behave as stated (a fresh child equals its parent; evidence
   moves it; decay returns it to the prior).
4. **Add `ledger`**. Test it against `bnm.track`'s behaviour on a recorded
   sequence, then optionally rebuild `bnm.track` on it.
5. **samizdat R6**: replace `samizdat.stats`. The smallest change, and it
   proves the dependency.
6. **samizdat R1** (gate effectiveness): the ledger, what "obeyed" means
   for each gate, the cell, `gates.edn` entries, the supervisor's view.
   Measure it on a campaign against the hand counts.
7. **samizdat R2, R3, R4, R5**, each measured before it changes a decision:
   it first runs alongside the current rule and is recorded, and replaces it
   only when it does better.

## Open questions

- thinkingbox's licence, for the code `samizdat.stats` ported.
- Whether `augur.ledger` should hold one layer or know about layers itself.
  The plan above keeps it to one and lets `augur.layer` combine them.
- Where samizdat keeps machine-wide state: a file under
  `~/.config/samizdat` or a machine database. This is samizdat's choice;
  augur only needs it to be EDN.
- Whether R3's features are computed in a cell (behaviour, editable at
  runtime) or in `src/` (mechanism). The rule says the choice of features is
  behaviour, so a cell.
