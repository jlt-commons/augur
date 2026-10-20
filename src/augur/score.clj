(ns augur.score
  "Scoring and comparing probability forecasts of binary outcomes, and
  predictive samples of counts: ROC AUC (pooled and within periods),
  average precision, recall at an alert budget, Brier score, Sarlin's
  usefulness, the area skill of an error diagram, matched concordance
  against a benchmark, paired bootstrap gains over a benchmark, a
  risk-adjusted CUSUM for calibration drift, the out-of-sample gain of a
  drift refit, out-of-sample stacks, choosing a model on scored years, and
  the non-randomized PIT and CRPS of a sample.

  Rows that carry a period use integer periods; where periods are pooled
  into blocks (resampled together, or refit together), `:per-block` says
  how many periods a block holds and is required: 4 makes years of
  quarters. Resampling takes :reps and a :seed, so the same call gives the
  same interval."
  (:require [augur.math :as m]
            [augur.pool :as pool]))

(set! *unchecked-math* true)

(defn- need
  "Throws naming the options in `opts` that are nil."
  [f opts]
  (when-let [missing (seq (keep (fn [[k v]] (when (nil? v) k)) opts))]
    (throw (ex-info (str "augur.score/" (name f) " needs " (vec missing)) {:missing (vec missing)}))))

(defn auc
  "Area under the ROC curve (Mann-Whitney, ties averaged), or nil without
  both outcomes."
  [scores labels]
  (let [pairs (vec (sort-by first (map vector scores labels)))
        n (count pairs)
        n-pos (count (filter second pairs))
        n-neg (- n n-pos)]
    (when (and (pos? n-pos) (pos? n-neg))
      (let [rank-sum (loop [i 0 acc 0.0]
                       (if (>= i n)
                         acc
                         (let [s (first (pairs i))
                               j (loop [j i] (if (and (< j n) (= s (first (pairs j)))) (recur (inc j)) j))
                               r (/ (+ (inc i) j) 2.0)
                               pos (count (filter second (subvec pairs i j)))]
                           (recur j (+ acc (* r pos))))))]
        (/ (- rank-sum (/ (* n-pos (inc n-pos)) 2.0)) (* n-pos n-neg))))))

(defn auc-within
  "AUC within each period, averaged with weights by positives, so it scores
  ranking across units only, not how rates move between periods: `rows` are
  [period score label]. nil when no period has both outcomes."
  [rows]
  (let [per (for [[_ g] (group-by first rows)
                  :let [a (auc (map second g) (map #(nth % 2) g))]
                  :when a]
              [a (count (filter #(nth % 2) g))])
        w (reduce + (map second per))]
    (when (pos? w)
      (/ (reduce + (map (fn [[a n]] (* a n)) per)) (double w)))))

(defn avg-precision
  "Area under the precision-recall curve, or nil without positives."
  [scores labels]
  (let [n-pos (count (filter true? labels))]
    (when (pos? n-pos)
      (loop [xs (sort-by (comp - first) (map vector scores labels)) i 1 hits 0 acc 0.0]
        (if-let [[[_ l] & more] (seq xs)]
          (if l
            (recur more (inc i) (inc hits) (+ acc (/ (inc hits) (double i))))
            (recur more (inc i) hits acc))
          (/ acc n-pos))))))

(defn recall-parts
  "{period [hits positives]} at the top `budget` fraction of each period's
  rows ([period score label]) by score; rows tied at the cutoff share the
  slots left, so the order they come in does not matter."
  [rows budget]
  (into {} (for [[q g] (group-by first rows)
                 :let [k (long (Math/ceil (* budget (count g))))]]
             [q [(loop [[[_ tied] & more] (sort-by #(- (double (key %))) (group-by #(double (second %)) g))
                        left k acc 0.0]
                   (if (or (nil? tied) (<= left 0))
                     acc
                     (let [n (count tied) hits (count (filter #(nth % 2) tied))
                           t (min n left)]
                       (recur more (- left t) (+ acc (* hits (/ t (double n))))))))
                 (count (filter #(nth % 2) g))]])))

(defn alarm-shares
  "Each row's share of an alarm at the top `budget` of its period by score
  (rows [period score ...], in their order): 1 above the cutoff, 0 below,
  and for rows tied at it the slots left over their number, as in
  `recall-parts`."
  [rows budget]
  (let [rows (vec rows)
        out (double-array (count rows))]
    (doseq [[_ idx] (group-by #(first (rows %)) (range (count rows)))
            :let [k (long (Math/ceil (* budget (count idx))))]]
      (loop [[[_ tied] & more] (sort-by #(- (double (key %))) (group-by #(double (second (rows %))) idx))
             left k]
        (when (and tied (pos? left))
          (let [n (count tied) t (min n left)]
            (doseq [i tied] (aset out (int i) (/ t (double n))))
            (recur more (- left t))))))
    out))

(defn paired-recall-gain-2way
  "`paired-recall-gain` on rows [period p-model p-bench outcome unit], the
  interval resampling units and blocks of periods both (the pigeonhole
  bootstrap, Owen 2007): each row weighted by how often its unit and its
  block were drawn, so a unit's overlapping windows and a block's shared
  shocks stay together. Alarms are fixed by the full sample. Options as
  `paired-recall-gain`."
  [rows budget {:keys [reps seed alpha per-block]}]
  (need :paired-recall-gain-2way {:reps reps :seed seed :alpha alpha :per-block per-block})
  (let [per-block (long per-block)]
   (let [rows (vec rows)
         n (count rows)
         ha (alarm-shares (map (fn [[q a]] [q a]) rows) budget)
         hb (alarm-shares (map (fn [[q _ b]] [q b]) rows) budget)
         ys (boolean-array (map #(boolean (nth % 3)) rows))
         units (vec (distinct (map #(nth % 4) rows)))
         years (vec (distinct (map #(quot (first %) per-block) rows)))
         ui (let [ix (zipmap units (range))] (int-array (map #(ix (nth % 4)) rows)))
         yi (let [ix (zipmap years (range))] (int-array (map #(ix (quot (first %) per-block)) rows)))
         ;; only onsets enter the gain: loop over them alone
         pos-idx (int-array (filter #(aget ys %) (range n)))
         npos (alength pos-idx)
         gain (fn [^doubles wu ^doubles wy]
                (loop [k 0 num 0.0 pos 0.0]
                  (if (< k npos)
                    (let [i (aget pos-idx k)
                          w (* (aget wu (aget ui i)) (aget wy (aget yi i)))]
                      (recur (inc k) (+ num (* w (- (aget ha i) (aget hb i)))) (+ pos w)))
                    (if (pos? pos) (/ num pos) 0.0))))
         draw (fn [g k] (let [c (double-array k)]
                          (dotimes [_ k] (let [j (min (dec k) (long (* k (m/next-double! g))))] (aset c j (inc (aget c j)))))
                          c))
         g (m/rng seed)
         diffs (vec (sort (for [_ (range reps)] (gain (draw g (count units)) (draw g (count years))))))
         at (fn [q] (nth diffs (min (dec (count diffs)) (long (* q (count diffs))))))]
     {:diff (gain (double-array (count units) 1.0) (double-array (count years) 1.0))
      :lo (at (/ alpha 2)) :hi (at (- 1.0 (/ alpha 2)))})))

(defn recall-at-budget
  "Of the positives, the share among the top `budget` fraction of rows by
  score in their period: `rows` are [period score label]. Rows tied at the
  cutoff share the slots left (`recall-parts`)."
  [rows budget]
  (let [parts (vals (recall-parts rows budget))
        pos (reduce + (map second parts))]
    (when (pos? pos)
      (/ (reduce + (map first parts)) (double pos)))))

(defn paired-recall-gain
  "`paired-gain` for recall at `budget` on rows [period p-model p-bench
  outcome], from per-period hits, so each resample sums blocks instead of
  rescoring rows. Options: :reps resamples, :seed, :alpha (the interval is
  1 - alpha; alpha over the number of tests for a Bonferroni bound) and
  :per-block periods per resampled block (4 for quarters in years)."
  [rows budget {:keys [reps seed alpha per-block]}]
  (need :paired-recall-gain {:reps reps :seed seed :alpha alpha :per-block per-block})
  (let [pa (recall-parts (map (fn [[q a _ y]] [q a y]) rows) budget)
        pb (recall-parts (map (fn [[q _ b y]] [q b y]) rows) budget)
        ;; per year: [model hits, bench hits, positives]
        blocks (vec (for [[_ qs] (group-by #(quot % (long per-block)) (keys pa))]
                      (reduce (fn [[x y z] q] [(+ x (first (pa q))) (+ y (first (pb q))) (+ z (second (pa q)))])
                              [0.0 0.0 0.0] qs)))
        score (fn [bs] (let [[x y z] (reduce #(mapv + %1 %2) [0.0 0.0 0.0] bs)]
                         (if (pos? z) (/ (- x y) z) 0.0)))
        k (count blocks)
        g (m/rng seed)
        diffs (vec (sort (for [_ (range reps)]
                           (score (map (fn [_] (blocks (min (dec k) (long (* k (m/next-double! g)))))) (range k))))))
        at (fn [q] (nth diffs (min (dec (count diffs)) (long (* q (count diffs))))))]
    {:diff (score blocks) :lo (at (/ alpha 2)) :hi (at (- 1.0 (/ alpha 2)))}))

(defn brier
  "The mean squared error of probabilities ps for boolean outcomes ys."
  [ps ys]
  (/ (reduce + (map (fn [p y] (let [d (- (double p) (if y 1.0 0.0))] (* d d))) ps ys)) (double (count ps))))

(defn usefulness
  "Sarlin's (2013) relative usefulness of probabilities ps for outcomes ys,
  for a policymaker who weighs missed events by mu and false alarms by
  1 - mu: the share of the loss of ignoring the forecast (always or never
  acting, whichever is cheaper) that the best threshold saves. 1 is a
  perfect forecast, 0 one worth nothing. Every threshold is tried in one
  sweep down the sorted forecasts."
  [ps ys mu]
  (let [n (count ys)
        pos (count (filter true? ys))
        neg (- n pos)
        p1 (/ pos (double n)) p2 (- 1.0 p1)
        loss (fn [tp fp]
               (+ (* mu p1 (if (pos? pos) (/ (- pos tp) (double pos)) 0.0))
                  (* (- 1.0 mu) p2 (if (pos? neg) (/ fp (double neg)) 0.0))))
        ;; alarms at threshold p are all forecasts >= p: walk the groups of
        ;; equal forecasts from the highest down
        groups (partition-by first (sort-by #(- (double (first %))) (map vector ps ys)))
        best (loop [[g & more] groups tp 0 fp 0 best (loss 0 0)]
               (if-not g
                 best
                 (let [t (count (filter #(true? (second %)) g))
                       tp (+ tp t) fp (+ fp (- (count g) t))]
                   (recur more tp fp (min best (loss tp fp))))))
        ignore (min (* mu p1) (* (- 1.0 mu) p2))]
    (if (pos? ignore) (/ (- ignore best) ignore) 0.0)))

(defn block-bootstrap-auc
  "A 90% interval for the pooled AUC of rows [period p y], resampling whole
  periods with replacement `reps` times (seeded), so that the correlation
  within a period is kept."
  [rows reps seed]
  (let [blocks (vec (vals (group-by first rows)))
        k (count blocks)
        g (m/rng seed)
        aucs (sort (for [_ (range reps)
                         :let [rs (mapcat (fn [_] (blocks (min (dec k) (long (* k (m/next-double! g)))))) (range k))
                               ys (map #(nth % 2) rs)]
                         :when (and (some true? ys) (some false? ys))]
                     (auc (map second rs) ys)))
        at (fn [q] (nth aucs (min (dec (count aucs)) (long (* q (count aucs))))))]
    [(at 0.05) (at 0.95)]))

(defn cusum
  "A risk-adjusted Bernoulli CUSUM (Steiner, Cook, Farewell & Treasure 2000)
  watching forecasts ps against outcomes ys, in order, for a drift of the
  odds by `ratio` (above 1: events more frequent than forecast; below 1:
  less): S <- max(0, S + log-likelihood ratio of the outcome). Sequential, so
  it watches calibration continuously without the false alarms of repeated
  tests. {:max the largest S, :last S now, :signal the index where S first
  crossed :h, or nil}. When outcomes overlap (each event inside
  `overlap` successive forecast windows), each increment counts 1/overlap,
  so the evidence is counted about once."
  [ps ys ratio & {:keys [h overlap] :or {overlap 1}}]
  (need :cusum {:h h})
  (let [lr (Math/log ratio)
        k (/ 1.0 (double overlap))]
    (loop [[p & ps] ps [y & ys] ys i 0 s 0.0 mx 0.0 sig nil]
      (if (nil? p)
        {:max mx :last s :signal sig}
        (let [p (min (max (double p) 1e-6) (- 1.0 1e-6))
              ;; log of P(y | odds x ratio) / P(y | p)
              w (* k (- (if (true? y) lr 0.0) (Math/log (+ 1.0 (* p (- ratio 1.0))))))
              s (max 0.0 (+ s w))]
          (recur ps ys (inc i) s (max mx s) (or sig (when (> s h) i))))))))

(defn recalibration-gain
  "Whether a drift is worth refitting for, out of sample: the mean log score,
  over rows [quarter p outcome] with quarter >= `from`, of shifting each
  block's forecasts (:per-block periods: 4 quarters to a year) by an
  intercept on the logit refit on the outcomes known by the block's start
  (a row is known `:overlap` periods on, 1 by default), discounted by
  `:discount` per block (0.8, a half-life of about three, follows a lasting
  shift), minus that of the forecasts as given. Positive when the shift has
  lasted; negative when outcomes come in waves a refit only chases. nil
  without outcomes to score."
  [rows from {:keys [overlap discount per-block] :or {overlap 1}}]
  (need :recalibration-gain {:discount discount :per-block per-block})
  (let [per-block (long per-block)
        clip #(min (max (double %) 1e-6) (- 1.0 1e-6))
        logit #(let [p (clip %)] (Math/log (/ p (- 1.0 p))))
        sig #(/ 1.0 (+ 1.0 (Math/exp (- %))))
        ll (fn [p y] (Math/log (if y p (- 1.0 p))))
        intercept (fn [known q0]
                    ;; Newton on the weighted log likelihood, with a unit prior at 0
                    (loop [a 0.0 it 0]
                      (let [[g h] (reduce (fn [[g h] [q p y]]
                                            (let [w (Math/pow discount (/ (- q0 q) (double per-block)))
                                                  r (sig (+ a (logit p)))]
                                              [(+ g (* w (- (if y 1.0 0.0) r))) (+ h (* w r (- 1.0 r)))]))
                                          [(- a) 1.0] known)
                            d (/ g h)]
                        (if (or (> it 30) (< (Math/abs d) 1e-9)) (+ a d) (recur (+ a d) (inc it))))))
        scored (filter #(>= (first %) from) rows)]
    (when (seq scored)
      (let [diffs (for [[yr this] (group-by #(quot (first %) per-block) scored)
                        :let [q0 (* per-block yr)
                              a (intercept (filter #(<= (+ (first %) overlap) q0) rows) q0)]
                        [_ p y] this]
                    (- (ll (clip (sig (+ a (logit p)))) y) (ll (clip p) y)))]
        (/ (reduce + diffs) (double (count diffs)))))))

(defn stack-out-of-sample
  "Rows [quarter ps outcome], ps the forecasters' probabilities, forecast
  out of sample by a stack (`augur.pool/fit-stack`): each block of
  :per-block periods from period `from` on by one fit on the rows known by
  its start (a row is known :overlap periods on, 1 by default), discounted
  per block by :discount (1, none, by default), under prior precision
  :prior. Returns [period p outcome] for those blocks, with whatever else
  each row carried after its outcome. With :by, a function of a row, each
  group it makes gets its own stack."
  [rows from {:keys [overlap discount prior by per-block] :or {overlap 1 discount 1.0} :as opts}]
  (need :stack-out-of-sample {:prior prior :per-block per-block})
  (if by
    (vec (mapcat #(stack-out-of-sample % from (dissoc opts :by))
                 (vals (group-by by rows))))
    (vec (for [[yr this] (sort-by key (group-by #(quot (first %) (long per-block)) (filter #(>= (first %) from) rows)))
               :let [q0 (* (long per-block) yr)
                     known (filter #(<= (+ (first %) overlap) q0) rows)
                     beta (pool/fit-stack (map second known) (map #(nth % 2) known)
                                          (map #(Math/pow discount (/ (- q0 (first %)) (double per-block))) known)
                                          {:prior prior})]
               [q ps y & more] this]
           (into [q (pool/predict-stack beta ps) y] more)))))

(defn pick-model
  "The model whose forecasts, {model [[period p outcome unit] ...]}, have
  the best mean log score over the blocks lo-hi (a period's block is it
  over `per-block`: 4 for years of quarters), on the rows every model
  with rows there shares (a combination that starts later is not scored on
  other years than its members): chosen before the years held out, as the
  members' priors are. nil when none has rows there."
  [preds lo hi per-block]
  (let [in (into {} (for [[m rows] preds
                          :let [rs (into {} (for [[q p y unit] rows :when (<= lo (quot q (long per-block)) hi)] [[q unit] [p y]]))]
                          :when (seq rs)]
                      [m rs]))
        common (when (seq in) (reduce (fn [ks rs] (filter #(contains? rs %) ks)) (keys (val (first in))) (vals in)))
        score (fn [rs]
                (when (seq common)
                  (/ (reduce + (for [k common
                                     :let [[p y] (rs k) p (min (max (double p) 1e-6) (- 1.0 1e-6))]]
                                 (Math/log (if y p (- 1.0 p)))))
                     (double (count common)))))
        scored (keep (fn [[m rs]] (when-let [s (score rs)] [m s])) in)]
    (when (seq scored) (key (apply max-key val (into {} scored))))))

(defn paired-gain
  "How much better a model scores than a benchmark on the same rows [period
  p-model p-bench outcome] by `metric` (a function of rows [period p
  outcome], higher better): {:diff :lo :hi}, the 90% interval resampling
  whole blocks of :per-block periods :reps times (from :seed), which keeps
  the correlation of units within a block and of a block's periods."
  [rows metric {:keys [reps seed per-block]}]
  (need :paired-gain {:reps reps :seed seed :per-block per-block})
  (let [score (fn [rs] (- (or (metric (map (fn [[q a _ y]] [q a y]) rs)) 0.0)
                          (or (metric (map (fn [[q _ b y]] [q b y]) rs)) 0.0)))
        blocks (vec (vals (group-by #(quot (first %) (long per-block)) rows)))
        k (count blocks)
        g (m/rng seed)
        diffs (vec (sort (for [_ (range reps)]
                           (score (mapcat (fn [_] (blocks (min (dec k) (long (* k (m/next-double! g)))))) (range k))))))
        at (fn [q] (nth diffs (min (dec (count diffs)) (long (* q (count diffs))))))]
    {:diff (score rows) :lo (at 0.05) :hi (at 0.95)}))

(defn paired-usefulness-gain
  "`paired-gain` for `usefulness` at `mu`, with the same draws: a resample
  is integer weights on the rows (how often each one's block was drawn),
  swept down one sort per model instead of rescoring copied rows. Options
  as `paired-gain`."
  [rows mu {:keys [reps seed per-block]}]
  (need :paired-usefulness-gain {:reps reps :seed seed :per-block per-block})
  (let [blocks (vec (vals (group-by #(quot (first %) (long per-block)) rows)))
        rows (vec (apply concat blocks))
        n (count rows)
        bix (long-array (mapcat (fn [i b] (repeat (count b) i)) (range) blocks))
        ys (boolean-array (map #(boolean (nth % 3)) rows))
        sweep (fn [col]
                (let [order (int-array (sort-by #(- (double (nth (rows %) col))) (range n)))
                      ps (double-array (map #(double (nth (rows %) col)) order))]
                  (fn [^doubles w]
                    (let [[pos tot] (loop [i 0 pos 0.0 tot 0.0]
                                      (if (< i n)
                                        (recur (inc i) (if (aget ys i) (+ pos (aget w i)) pos) (+ tot (aget w i)))
                                        [pos tot]))
                          neg (- tot pos)
                          p1 (/ pos tot) p2 (- 1.0 p1)
                          loss (fn [tp fp]
                                 (+ (* mu p1 (if (pos? pos) (/ (- pos tp) pos) 0.0))
                                    (* (- 1.0 mu) p2 (if (pos? neg) (/ fp neg) 0.0))))
                          best (loop [i 0 tp 0.0 fp 0.0 best (loss 0.0 0.0)]
                                 (if (< i n)
                                   (let [r (aget order i) wi (aget w r)
                                         tp (if (aget ys r) (+ tp wi) tp)
                                         fp (if (aget ys r) fp (+ fp wi))
                                         ;; a threshold only between distinct forecasts
                                         end? (or (= i (dec n)) (not= (aget ps i) (aget ps (inc i))))]
                                     (recur (inc i) tp fp (if end? (min best (loss tp fp)) best)))
                                   best))
                          ignore (min (* mu p1) (* (- 1.0 mu) p2))]
                      (if (pos? ignore) (/ (- ignore best) ignore) 0.0)))))
        ua (sweep 1) ub (sweep 2)
        score (fn [w] (- (double (ua w)) (double (ub w))))
        k (count blocks)
        g (m/rng seed)
        diffs (vec (sort (for [_ (range reps)]
                           (let [c (long-array k)]
                             (dotimes [_ k]
                               (let [b (min (dec k) (long (* k (m/next-double! g))))]
                                 (aset c b (inc (aget c b)))))
                             (score (double-array (map #(aget c (int %)) bix)))))))
        at (fn [q] (nth diffs (min (dec (count diffs)) (long (* q (count diffs))))))]
    {:diff (score (double-array n 1.0)) :lo (at 0.05) :hi (at 0.95)}))

(defn area-skill
  "The area skill score of an error diagram (Molchan 1990; Zechar & Jordan
  2008) for rows [period p outcome weight]: alarms are raised by rank
  within each period (the alert budgets' order), the alarm volume is the
  share of the weights covered, and the score is one minus the area under
  the miss rate. 0.5 is a random order; with weights the benchmark's
  probabilities, 0.5 is no better than the benchmark (Kossobokov's
  reference-weighted volume). nil without onsets."
  [rows]
  (let [pos (count (filter #(nth % 2) rows))
        total (reduce + (map #(double (nth % 3)) rows))]
    (when (and (pos? pos) (pos? total))
      (let [;; each row's place in its period, 0 the highest, ties sharing the mean
            ranked (mapcat (fn [[_ g]]
                             (let [n (count g)]
                               (loop [[t & more] (partition-by second (sort-by #(- (double (second %))) g)) at 0 out []]
                                 (if-not t
                                   out
                                   (let [k (count t) frac (/ (+ at (* 0.5 k)) n)]
                                     (recur more (+ at k) (into out (map #(vector frac (nth % 2) (nth % 3)) t))))))))
                           (group-by first rows))
            area (loop [[t & more] (partition-by first (sort-by first ranked)) miss 1.0 area 0.0]
                   (if-not t
                     area
                     (let [dt (/ (reduce + (map #(double (nth % 2)) t)) total)
                           dm (/ (count (filter second t)) (double pos))]
                       (recur more (- miss dm) (+ area (* dt (- miss (* 0.5 dm))))))))]
        (- 1.0 area)))))

(defn matched-concordance
  "Signal beyond a benchmark: each event (rows {:period :p-model :p-bench
  :y}) is set against the `k` non-events of its period whose benchmark
  probabilities are closest to its own, and the share of those pairs the
  model ranks the event above is returned (ties count a half). A model that
  knows only what the benchmark knows scores about 0.5. nil without events."
  [rows k]
  (let [pairs (for [[_ g] (group-by :period rows)
                    :let [neg (filter (complement :y) g)]
                    e (filter :y g)
                    c (take k (sort-by #(Math/abs (- (double (:p-bench %)) (double (:p-bench e)))) neg))]
                (let [a (double (:p-model e)) b (double (:p-model c))]
                  (cond (> a b) 1.0 (= a b) 0.5 :else 0.0)))]
    (when (seq pairs) (/ (reduce + pairs) (double (count pairs))))))

;; --- predictive samples ----------------------------------------------------------------------

(defn crps-sample
  "CRPS of the sorted sample `xs` (a double array) for the outcome `a`."
  ^double [^doubles xs ^double a]
  (let [n (alength xs)
        s1 (areduce xs i s 0.0 (+ s (Math/abs (- (aget xs i) a))))
        s2 (areduce xs i s 0.0 (+ s (* (aget xs i) (- (* 2 i) n -1))))]
    (- (/ s1 n) (/ s2 (* n (double n))))))

(defn pit-bounds
  "[F(a-) F(a)]: the predictive cdf just below and at the outcome a, from
  the sorted sample xs. For counts the PIT is an interval, not a point; its
  non-randomized histogram spreads each outcome uniformly over it (Czado,
  Gneiting & Held 2009)."
  [^doubles xs a]
  (let [n (alength xs)
        below (areduce xs i c 0 (if (< (aget xs i) a) (inc c) c))
        at (areduce xs i c 0 (if (<= (aget xs i) a) (inc c) c))]
    [(/ below (double n)) (/ at (double n))]))

(defn pit-histogram
  "The non-randomized PIT histogram over `bins` equal bins: the mean over
  outcomes of the mass each [lo hi] puts in each bin. Flat when the
  forecasts are calibrated; a U when they are too narrow; a hump when too
  wide."
  [bounds bins]
  (let [cdf (fn [[lo hi] u] (cond (>= u hi) 1.0 (<= u lo) 0.0 :else (/ (- u lo) (- hi lo))))
        n (count bounds)]
    (for [j (range 1 (inc bins))]
      (/ (reduce + (map #(- (cdf % (/ j (double bins))) (cdf % (/ (dec j) (double bins)))) bounds)) n))))
