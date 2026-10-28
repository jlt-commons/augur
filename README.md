# augur

Statistics for forecasting and for learning from outcomes, in plain
[Jolt](https://github.com/jolt-lang/jolt) Clojure with no dependencies:
Bayesian posteriors that learn online with forgetting, shrinkage toward a
parent, a ledger that scores forecasts as they resolve and recalibrates new
ones, probability calibration, logistic hazards, combining forecasters,
change-point detection, and forecast scoring and tests.

It is the statistics core of
[bayesian-news-model](https://github.com/yogthos/bayesian-news-model), taken
out as its own library.

## Using it

```clojure
;; deps.edn
{:deps {io.github.jlt-commons/augur {:git/url "https://github.com/jlt-commons/augur"
                                     :git/tag "v0.11.0"
                                     :git/sha "e0869b05f494d1a868bc2f04040b468d7653a11b"}}}
```

```clojure
(require '[augur.beta :as beta] '[augur.layer :as layer] '[augur.ledger :as ledger])

;; how often a gate is obeyed, learned per run with forgetting
(def project (-> (beta/jeffreys)
                 (beta/observe true {:forget 0.95})
                 (beta/observe false {:forget 0.95})))
(def machine (beta/from-counts 39 44 (beta/jeffreys)))

(layer/explain [project machine] {:strength 20.0 :level 0.9})
;; => {:mean 0.843 :interval [0.702 0.948] :evidence 1.95 :own 0.089 :by-layer [...]}

;; forecasts made now, resolved later, scored against a benchmark
(-> (ledger/empty {:max-open 1000 :max-rows 500 :max-keys 200 :forget 0.99
                   :calibration augur.calibrate/defaults})
    (ledger/register {:id 1 :key [:gate :no-edits] :p 0.8 :at 12 :bench {:shipped 0.6}})
    (ledger/resolve 1 true {:at 14})
    (ledger/record [:gate :no-edits]))
;; => {:n 1.0 :brier 0.04 :log -0.223 :skill {:shipped 0.75} :reliability [[0.75 1.0 1.0]] :open 0 :expired 0}
```

## Namespaces

| namespace | what it does |
|---|---|
| `augur.beta` | Beta posteriors on a probability: observe outcomes, forget by step count, mean, credible interval, P(p > x), P(p_A > p_B), Thompson samples; pass@k and pass^k |
| `augur.gamma` | Gamma posteriors on a rate per unit of exposure, with the same operations |
| `augur.layer` | A child's prior from its parent's posterior at a fixed strength, folded over any number of layers, with an explanation of how much is the child's own |
| `augur.ledger` | Forecasts registered and resolved later: Brier and log scores against benchmarks, decayed per key, recalibration from the key's record, rows for refitting; bounded |
| `augur.calibrate` | Recalibrating probabilities: an online binned table with forgetting, the isotonic (CORP) map, the logistic map |
| `augur.logit` | Weighted Bayesian logistic regression: ridge or horseshoe-like shrinkage, with per-row offsets |
| `augur.hier` | Effects over a tree of units: nested random intercepts with variances by empirical Bayes, and dynamic AR(1) states per node for counts (bottom-up evidence, top-down scaling), each level's persistence and size chosen by one-step-ahead predictive likelihood; window multipliers and the chance of any event in a group under shared states |
| `augur.dynamic` | Dynamic generalized linear models (logistic, Poisson) with discounting and linear-Bayes updates, the dynamic linear model with its noise learned (Student t), their log predictive scores and predictive moments, dynamic model averaging over them, and coupling of models' coefficients by certainty |
| `augur.enkf` | The ensemble Kalman filter's stochastic update (centred perturbations), derivative-free: the analysis of an ensemble of states and parameters, repeated with inflated noise for the ensemble smoother with multiple data assimilation; linear identities kept by projection onto them |
| `augur.pool` | Combining forecasters: EM linear pools, Hedge weights, the log-odds median, a log-linear stack |
| `augur.score` | AUC, average precision, recall at a budget, Brier, usefulness, area skill, matched concordance, paired bootstrap gains (one- and two-way), a risk-adjusted CUSUM, out-of-sample stacks and drift refits, the PIT and CRPS of a sample |
| `augur.ftest` | Tests of forecasts: out-of-sample R², Clark-West, Diebold-Mariano, Pesaran-Timmermann, QLIKE, CRPS and log scores, Kupiec, Christoffersen, Berkowitz, Newey-West, Hansen's SPA |
| `augur.changepoint` | Bayesian online change-point detection, incremental (`init`, `step`) or over a series (`run`) |
| `augur.trees` | Shallow gradient-boosted trees for probabilities |
| `augur.analog` | Nearest-past-cases forecasts shrunk toward the base rate |
| `augur.special` | Incomplete gamma and beta functions and their inverses, the normal and t distributions, Gauss-Legendre nodes |
| `augur.math` | log-gamma, logit/expit, a seedable generator and samplers, quantiles, a Cholesky solve, Nelder-Mead |

## Conventions

- **Pure functions over values.** No IO, clock, logging or atoms. Time is an
  argument. The generator is a value: `augur.math/rng-state` gives a vector
  of six doubles and `with-rng` returns a draw with the state after it, so
  the same state gives the same draw.
- **Numbers are arguments.** Priors, forgetting rates, shrinkage, strengths,
  bin edges and thresholds are passed in, and a function that needs one
  throws if it is missing. A namespace's `defaults` map records the values
  bayesian-news-model uses; it is never applied silently. Only iteration
  limits and tolerances have built-in values.
- **State is EDN.** Posteriors, ledgers, calibration tables, change-point
  detectors, tree models and stored coefficients (`augur.logit/->state`)
  round-trip through `pr-str` and `clojure.edn/read-string`. Primitive
  arrays appear only inside computations and as the return of fits meant
  for immediate use.
- **Bounded state.** What grows with use has a cap from the caller, and what
  falls off is folded into decayed counts.
- **Missing state is empty.** A nil posterior is the prior given as
  `:prior`; a nil ledger reads as empty.
- Updating a posterior or a ledger is O(1) in its history. Fitting
  (`logit/fit`, `trees/fit`, `pool/pool-weights-ml`) is for the end of a run
  or offline.

## Testing

```
jolt -A:test -M
```

Special functions and tests are held to values computed with scipy.

## Licence

Apache License 2.0, see [LICENSE](LICENSE).
