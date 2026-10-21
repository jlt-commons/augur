(ns augur.score-test
  (:require [augur.math :as m]
            [augur.score :as score]
            [clojure.test :refer [deftest is testing]]))

(deftest scores-rare-event-forecasts
  (let [ys [true false false true false false false false]
        good [0.9 0.1 0.2 0.8 0.1 0.3 0.2 0.1]
        flat (repeat 8 0.25)]
    (is (= 1.0 (score/auc good ys)))
    (is (= 0.5 (score/auc flat ys)))
    (is (= 1.0 (score/avg-precision good ys)))
    (testing "AUC within periods ignores how rates move between them"
      ;; period 1 has the higher scores and most positives, but within each
      ;; period the ranking is reversed
      (let [rows [[0 0.1 true] [0 0.2 false] [1 0.8 true] [1 0.9 false] [1 0.85 true]]]
        (is (> (score/auc (map second rows) (map #(nth % 2) rows)) 0.3))
        (is (= 0.0 (score/auc-within rows))))
      (is (nil? (score/auc-within [[0 0.3 true] [1 0.2 false]])) "no period has both outcomes"))
    (testing "recall when flagging the top quarter of each period"
      (is (= 1.0 (score/recall-at-budget (map vector (repeat 0) good ys) 0.25)))
      (is (= 0.0 (score/recall-at-budget (map vector (repeat 0) (reverse good) ys) 0.25))))))

(deftest policy-usefulness
  ;; Sarlin (2013): the share of the loss of ignoring the forecast that it saves
  (let [ys [true true false false false false false false false false]]
    (testing "a perfect forecast saves all of it"
      (is (< (Math/abs (- 1.0 (score/usefulness [0.9 0.8 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1] ys 0.5))) 1e-12)))
    (testing "a constant forecast saves none"
      (is (< (Math/abs (score/usefulness (repeat 10 0.3) ys 0.5)) 1e-12)))))

(deftest bootstrap-intervals
  (let [rows (for [yr (range 20) i (range 10)] [yr (if (< i 3) (+ 0.5 (* 0.01 i)) (* 0.01 i)) (< i 3)])
        [lo hi] (score/block-bootstrap-auc rows 200 7)]
    (testing "resamples whole periods, and the interval holds the estimate"
      (is (<= lo (score/auc (map second rows) (map #(nth % 2) rows)) hi))
      (is (<= 0.0 lo hi 1.0)))))

(deftest calibration-cusum
  ;; risk-adjusted Bernoulli CUSUM (Steiner et al. 2000): accumulates the log
  ;; likelihood ratio of the outcomes under odds scaled by `ratio` against the
  ;; forecasts as given, reset at zero
  (let [ps (repeat 400 0.1)]
    (testing "calibrated forecasts stay quiet"
      (let [ys (map #(zero? (mod % 10)) (range 400))]
        (is (< (:max (score/cusum ps ys 2.0 :h 4.0)) 4.0))
        (is (< (:max (score/cusum ps ys 0.5 :h 4.0)) 4.0))))
    (testing "events twice as frequent as forecast raise the upward chart"
      (let [ys (map #(zero? (mod % 5)) (range 400))
            up (score/cusum ps ys 2.0 :h 4.0)]
        (is (> (:max up) 4.0))
        (is (some? (:signal up)) "and it says when")))
    (testing "events half as frequent raise the downward chart"
      (let [ys (map #(zero? (mod % 20)) (range 400))]
        (is (> (:max (score/cusum ps ys 0.5 :h 4.0)) 4.0))))
    (testing "overlapping windows count each increment once over the overlap"
      (let [ys (map #(zero? (mod % 5)) (range 400))]
        (is (< (Math/abs (- (:max (score/cusum ps ys 2.0 :overlap 4 :h 4.0)) (/ (:max (score/cusum ps ys 2.0 :h 4.0)) 4.0))) 1e-9))))))

(deftest recalibration-gain-says-whether-refitting-helps
  ;; [quarter p outcome] rows; the gain is the mean log score, over quarters
  ;; from `from`, of refitting an intercept on outcomes known each year minus
  ;; that of the forecasts as given
  (let [rows (fn [rate-of] (for [q (range 160) u (range 40)]
                             [q 0.05 (< (mod (hash [q u]) 1000) (* 1000 (rate-of q)))]))]
    (testing "a lasting shift is worth refitting for"
      (is (pos? (score/recalibration-gain (rows (constantly 0.15)) 120 {:overlap 4 :discount 0.8 :per-block 4}))))
    (testing "calibrated forecasts gain nothing"
      (is (< (score/recalibration-gain (rows (constantly 0.05)) 120 {:overlap 4 :discount 0.8 :per-block 4}) 0.002)))
    (testing "waves that come and go: refitting chases them and loses"
      (is (neg? (score/recalibration-gain (rows #(if (< (mod % 24) 4) 0.25 0.012)) 120 {:overlap 4 :discount 0.8 :per-block 4}))))
    (testing "nil without outcomes to score"
      (is (nil? (score/recalibration-gain (rows (constantly 0.05)) 500 {:overlap 4 :discount 0.8 :per-block 4}))))))

(deftest the-model-of-record-is-chosen-on-the-tuning-years
  ;; {model [[quarter p outcome] ...]}: the best mean log score over the
  ;; tuning years wins, whatever happens after them
  (let [ys (fn [q] (zero? (mod q 10)))
        good (for [q (range 0 160)] [q (if (ys q) 0.5 0.05) (ys q)])
        bad (for [q (range 0 160)] [q (if (< q 40) 0.1 (if (ys q) 0.9 0.01)) (ys q)])]
    (is (= :good (score/pick-model {:good good :bad bad} 0 9 4)))
    (is (= :bad (score/pick-model {:good good :bad bad} 10 39 4)) "later years favour the other")
    (is (nil? (score/pick-model {:good good} 50 60 4)) "nil without rows in the years")))

(deftest paired-gain-over-a-benchmark
  ;; rows [period p-model p-bench outcome]: the metric's difference, with a
  ;; 90% interval resampling whole years (4 periods each)
  (let [g (m/rng 3)
        rows (vec (for [q (range 120) _ (range 30)
                        :let [y (< (m/next-double! g) 0.05)
                              noise (m/next-double! g)]]
                    [q (if y (+ 0.3 (* 0.1 noise)) (* 0.3 noise)) (* 0.4 (m/next-double! g)) y]))
        auc-of (fn [rs] (score/auc (map second rs) (map #(nth % 2) rs)))
        {:keys [diff lo hi]} (score/paired-gain rows auc-of {:reps 200 :seed 1 :per-block 4})]
    (is (> diff 0.3) "the model separates, the benchmark is noise")
    (is (< 0.0 lo diff hi))
    (testing "a model against itself gains nothing"
      (let [same (mapv (fn [[q p _ y]] [q p p y]) rows)
            {:keys [diff lo hi]} (score/paired-gain same auc-of {:reps 50 :seed 1 :per-block 4})]
        (is (= 0.0 diff lo hi))))))

(deftest matched-controls-measure-signal-beyond-the-benchmark
  ;; each event against the k non-events of its period the benchmark rates
  ;; closest: the share of pairs the model orders right (ties a half)
  (let [rows (for [q (range 40) u (range 50)
                   :let [hist (/ (mod u 10) 10.0)
                         y (and (= 0 (mod (+ q u) 7)) (> hist 0.4))]]
               {:period q :p-bench hist :p-model (if y 0.95 hist) :y y})]
    (is (> (score/matched-concordance rows 3) 0.95) "it knows which of the alike will tip")
    (is (< (Math/abs (- 0.5 (score/matched-concordance (map #(assoc % :p-model (:p-bench %)) rows) 3))) 0.1)
        "the benchmark against its own matches is a coin")))

(deftest usefulness-matches-every-threshold-tried
  ;; the sweep gives what trying each threshold against every row gives
  (let [g (m/rng 11)
        ps (vec (repeatedly 500 #(Math/floor (* 50 (m/next-double! g)))))
        ys (mapv #(< (m/next-double! g) (/ % 100.0)) ps)
        brute (fn [ps ys mu]
                (let [n (count ys) pos (count (filter true? ys)) neg (- n pos)
                      p1 (/ pos (double n)) p2 (- 1.0 p1)
                      loss (fn [thr] (let [alarm (filter #(>= (first %) thr) (map vector ps ys))
                                           tp (count (filter #(true? (second %)) alarm))
                                           fp (- (count alarm) tp)]
                                       (+ (* mu p1 (/ (- pos tp) (double pos))) (* (- 1.0 mu) p2 (/ fp (double neg))))))
                      best (apply min (loss Double/POSITIVE_INFINITY) (map loss (distinct ps)))
                      ignore (min (* mu p1) (* (- 1.0 mu) p2))]
                  (/ (- ignore best) ignore)))]
    (doseq [mu [0.5 0.9 0.95]]
      (is (< (Math/abs (- (brute ps ys mu) (score/usefulness ps ys mu))) 1e-12)))))

(deftest a-stack-refit-each-year-out-of-sample
  ;; rows [quarter [p1 p2] outcome]: each year from `from` on is forecast by
  ;; a stack fit only on the rows known by its start, so a second forecaster
  ;; that knows something the first does not raises the score held out
  (let [g (m/rng 11)
        rows (vec (for [q (range 40 120) _ (range 60)]
                    (let [a (m/normal! g) b (m/normal! g)
                          p (m/expit (+ -2.5 a b))]
                      [q [(m/expit (+ -2.5 a)) (m/expit (+ -2.5 b))] (< (m/next-double! g) p)])))
        ll (fn [rs] (/ (reduce + (for [[_ p y] rs] (Math/log (if y p (- 1.0 p))))) (double (count rs))))
        both (score/stack-out-of-sample rows 80 {:per-block 4 :prior 1.0})
        first-only (score/stack-out-of-sample (mapv (fn [[q ps y]] [q [(first ps)] y]) rows) 80 {:per-block 4 :prior 1.0})]
    (is (= (count (filter #(>= (first %) 80) rows)) (count both)) "every row from `from` on, once")
    (is (every? #(>= (first %) 80) both))
    (is (> (ll both) (+ 0.02 (ll first-only))) "the second forecaster's information is used")
    (testing "a year's stack sees nothing from that year"
      (let [poisoned (mapv (fn [[q ps y]] (if (>= q 100) [q ps (not y)] [q ps y])) rows)
            a (filter #(< (first %) 100) (score/stack-out-of-sample rows 80 {:per-block 4 :prior 1.0}))
            b (filter #(< (first %) 100) (score/stack-out-of-sample poisoned 80 {:per-block 4 :prior 1.0}))]
        (is (= (map second a) (map second b)))))))

(deftest a-stack-per-group-carries-the-rows-along
  ;; the second forecaster helps only in group :b; a stack per group finds it
  (let [g (m/rng 12)
        rows (vec (for [q (range 40 120) i (range 60)]
                    (let [grp (if (even? i) :a :b) a (m/normal! g) b (m/normal! g)
                          p (m/expit (+ -2.5 a (if (= grp :b) (* 2.0 b) 0.0)))]
                      [q [(m/expit (+ -2.5 a)) (m/expit (+ -2.5 b))] (< (m/next-double! g) p) grp [q i]])))
        out (score/stack-out-of-sample rows 80 {:by #(nth % 3) :per-block 4 :prior 1.0})]
    (is (= (set (map #(nth % 4) (filter #(>= (first %) 80) rows))) (set (map #(nth % 4) out)))
        "every held-out row comes back once, with what it carried")
    (is (= (count out) (count (set (map #(nth % 4) out)))))
    (let [ll (fn [rs] (/ (reduce + (for [[_ p y] rs] (Math/log (if y p (- 1.0 p))))) (double (count rs))))
          pooled (score/stack-out-of-sample rows 80 {:per-block 4 :prior 1.0})]
      (is (> (ll out) (ll pooled)) "a stack per group beats one for all"))))

(deftest recall-at-a-budget-shares-ties-at-the-cutoff
  ;; ten rows in one period, a budget of 2: one clear top row, then four tied,
  ;; one of them a positive; the second slot holds a quarter of the tie
  (let [rows (concat [[1 0.9 true]]
                     (for [y [true false false false]] [1 0.5 y])
                     (for [_ (range 5)] [1 0.1 false]))]
    (is (< (Math/abs (- (/ 1.25 2.0) (score/recall-at-budget rows 0.2))) 1e-12))
    (is (= (score/recall-at-budget rows 0.2) (score/recall-at-budget (reverse rows) 0.2))
        "the order of the rows does not matter")))

(deftest recall-gain-from-per-period-parts
  ;; recall at a budget is hits over positives summed by period, so the
  ;; bootstrap resamples per-year sums instead of rescoring every row
  (let [g (m/rng 21)
        rows (vec (for [q (range 40 80) i (range 50)]
                    (let [a (m/normal! g) y (< (m/next-double! g) (m/expit (+ -3.0 (* 1.5 a))))]
                      [q (m/expit a) (m/expit (+ a (m/normal! g))) y])))
        parts (score/recall-parts (map (fn [[q a _ y]] [q a y]) rows) 0.1)
        r (score/recall-at-budget (map (fn [[q a _ y]] [q a y]) rows) 0.1)
        fast (score/paired-recall-gain rows 0.1 {:reps 200 :seed 1 :alpha 0.1 :per-block 4})
        slow (score/paired-gain rows #(score/recall-at-budget % 0.1) {:reps 200 :seed 1 :per-block 4})]
    (is (< (Math/abs (- r (/ (reduce + (map first (vals parts))) (reduce + (map second (vals parts)))))) 1e-12))
    (is (< (Math/abs (- (:diff fast) (:diff slow))) 1e-12) "the same gain")
    (is (<= (:lo fast) (:diff fast) (:hi fast)))
    (is (< (Math/abs (- (:lo fast) (:lo slow))) 0.05) "about the same interval")
    (is (= {:diff 0.0 :lo 0.0 :hi 0.0} (score/paired-recall-gain (mapv (fn [[q a _ y]] [q a a y]) rows) 0.1 {:reps 50 :seed 1 :alpha 0.1 :per-block 4})))))

(deftest paired-usefulness-gain-matches-the-general-bootstrap
  ;; resampled years become integer row weights over one sort, so the same
  ;; draws give the same intervals as rescoring the resampled rows
  (let [g (m/rng 22)
        rows (vec (for [q (range 40 80) i (range 30)]
                    (let [a (m/normal! g) y (< (m/next-double! g) (m/expit (+ -3.0 (* 1.5 a))))]
                      [q (/ (Math/round (* 20 (m/expit a))) 20.0) (m/expit (+ a (m/normal! g))) y])))
        u (fn [rs] (score/usefulness (map second rs) (map #(nth % 2) rs) 0.9))
        fast (score/paired-usefulness-gain rows 0.9 {:reps 100 :seed 1 :per-block 4})
        slow (score/paired-gain rows u {:reps 100 :seed 1 :per-block 4})]
    (doseq [k [:diff :lo :hi]]
      (is (< (Math/abs (- (k fast) (k slow))) 1e-9) (str k)))))

(deftest models-are-picked-on-the-rows-they-share
  ;; b starts late, on easy rows, and would win on its own rows; on the
  ;; rows both have, a is better
  (let [a [[8000 0.1 false "A"] [8001 0.9 true "A"] [8010 0.6 true "A"]]
        b [[8010 0.7 true "A"]]
        c [[8000 0.5 false "A"] [8001 0.5 true "A"] [8010 0.65 true "A"]]]
    (is (= "b" (score/pick-model {"a" a "b" b} 2000 2003 4)) "on 8010 alone b is ahead")
    (is (= "a" (score/pick-model {"a" a "c" c} 2000 2003 4)))
    (is (= "a" (score/pick-model {"a" a "late" [[9000 0.99 true "A"]]} 2000 2003 4)) "no rows in the years: not a candidate")))

(deftest area-skill-of-error-diagrams
  ;; rows [period p y w]: alarms by rank within the period, alarm volume as
  ;; the share of w; 1 - area under the miss curve (Molchan; Zechar & Jordan)
  (let [g (m/rng 41)
        ref (vec (for [q (range 40) _ (range 100)] [q (m/expit (- (* 1.5 (m/normal! g)) 4.0))]))
        ys (mapv (fn [[_ r]] (< (m/next-double! g) r)) ref)
        rows (fn [score w] (mapv (fn [[q r] y] [q (score r) y (w r)]) ref ys))]
    (testing "a random order scores about a half"
      (is (< (Math/abs (- 0.5 (score/area-skill (rows (fn [_] (m/next-double! g)) (constantly 1.0))))) 0.06)))
    (testing "the reference itself is skilful by row count, but a half by its own mass"
      (is (> (score/area-skill (rows identity (constantly 1.0))) 0.75))
      (is (< (Math/abs (- 0.5 (score/area-skill (rows identity identity)))) 0.06)))
    (testing "knowing the outcomes is near one"
      (is (> (score/area-skill (mapv (fn [[q r] y] [q (if y 1.0 0.0) y 1.0]) ref ys)) 0.95)))))

(deftest recall-gain-intervals-widen-for-many-tests
  (let [g (m/rng 23)
        rows (vec (for [q (range 40 80) _ (range 50)]
                    (let [a (m/normal! g) y (< (m/next-double! g) (m/expit (+ -3.0 (* 1.5 a))))]
                      [q (m/expit a) (m/expit (+ a (m/normal! g))) y])))
        one (score/paired-recall-gain rows 0.1 {:reps 2000 :seed 1 :alpha 0.1 :per-block 4})
        many (score/paired-recall-gain rows 0.1 {:reps 2000 :seed 1 :alpha (/ 0.1 12) :per-block 4})]
    (is (= (:diff one) (:diff many)))
    (is (< (:lo many) (:lo one)))
    (is (> (:hi many) (:hi one)))))

(deftest two-way-bootstrap
  ;; units whose gain persists over time: resampling years alone treats
  ;; their years as independent; resampling units and years both does not
  (let [g (m/rng 43)
        rows (vec (for [u (range 60) q (range 40 88)
                        :let [good (< u 30) a (m/normal! g)
                              y (< (m/next-double! g) (m/expit (+ -3.0 (* 1.5 a))))]]
                    [q (m/expit (if good a (* 0.2 a))) (m/expit (* 0.6 a)) y (str "u" u)]))
        year (score/paired-recall-gain (mapv #(subvec % 0 4) rows) 0.1 {:reps 400 :seed 1 :alpha 0.1 :per-block 4})
        both (score/paired-recall-gain-2way rows 0.1 {:reps 400 :seed 1 :alpha 0.1 :per-block 4})]
    (is (< (Math/abs (- (:diff year) (:diff both))) 1e-9) "the same gain")
    (is (> (- (:hi both) (:lo both)) (- (:hi year) (:lo year))) "a wider interval")
    (testing "a model equal to its bench gains nothing at any draw"
      (is (= {:diff 0.0 :lo 0.0 :hi 0.0}
             (score/paired-recall-gain-2way (mapv (fn [[q a _ y u]] [q a a y u]) rows) 0.1 {:reps 50 :seed 1 :alpha 0.1 :per-block 4}))))))

(deftest non-randomized-pit
  (testing "the predictive cdf just below and at the outcome, from sorted samples"
    (is (= [0.25 0.75] (score/pit-bounds (double-array [0.0 1.0 1.0 2.0]) 1.0)))
    (is (= [1.0 1.0] (score/pit-bounds (double-array [0.0 1.0]) 5.0)))
    (is (= [0.0 0.0] (score/pit-bounds (double-array [3.0 4.0]) 1.0))))
  (testing "the histogram spreads each outcome's mass uniformly over its interval (Czado et al. 2009)"
    (let [h (score/pit-histogram [[0.0 0.5] [0.5 1.0]] 2)]
      (is (< (Math/abs (- 0.5 (first h))) 1e-12))
      (is (< (Math/abs (- 0.5 (second h))) 1e-12)))
    (is (every? #(< (Math/abs (- 0.25 %)) 1e-12) (score/pit-histogram [[0.0 1.0]] 4)))
    (testing "a point mass at a bin edge falls in the bin below"
      (is (= [1.0 0.0] (mapv double (score/pit-histogram [[0.5 0.5]] 2)))))))

(deftest crps-of-a-sample
  (testing "a point forecast's CRPS is its absolute error"
    (is (< (Math/abs (- 2.0 (score/crps-sample (double-array [3.0]) 1.0))) 1e-12)))
  (testing "E|X - a| - E|X - X'| / 2 over the sample"
    (let [xs (double-array [0.0 1.0 2.0 4.0])
          a 1.5
          n 4.0
          e1 (/ (reduce + (map #(Math/abs (- % a)) xs)) n)
          e2 (/ (reduce + (for [x xs y xs] (Math/abs (- x y)))) (* n n))]
      (is (< (Math/abs (- (- e1 (* 0.5 e2)) (score/crps-sample xs a))) 1e-12)))))

(deftest options-are-required
  (is (thrown? Exception (score/paired-gain [[0 0.1 0.2 true]] (constantly 0.0) {:reps 10 :seed 1})))
  (is (thrown? Exception (score/cusum [0.1] [true] 2.0))))


(deftest non-finite-scores-throw
  ;; NaN equals nothing, so a tie scan on it never advanced: auc hung
  (let [ps [0.1 ##NaN 0.3 0.2] ys [true false false true]
        rows (map vector [1 1 2 2] ps ys)
        calls {:auc #(score/auc ps ys)
               :avg-precision #(score/avg-precision ps ys)
               :recall-parts #(score/recall-parts rows 0.5)
               :alarm-shares #(score/alarm-shares rows 0.5)
               :usefulness #(score/usefulness ps ys 0.9)}]
    (doseq [[k f] calls]
      (let [r (deref (future (try (f) :no-error (catch Exception e (ex-message e)))) 10000 :hung)]
        (is (and (string? r) (re-find #"non-finite" r)) (str k " " r))))))

(def ^:private brier (fn [p y] (let [d (- p (if y 1.0 0.0))] (* d d))))

(defn- sim-rows
  "Rows [period p-model p-bench outcome unit group]: outcomes drawn from the
  model's probabilities, so the model is right and the bench, which lacks
  the signal z, is not; `beta` scales the signal."
  [seed beta groups]
  (let [g (m/rng seed)]
    (vec (for [grp groups unit (range 60) q (range 40)
               :let [z (m/normal! g)
                     lb -3.0
                     pm (m/expit (+ lb (* beta z)))
                     pb (m/expit lb)]]
           [q pm pb (< (m/next-double! g) pm) (str "u" unit) grp]))))

(deftest paired-loss-finds-a-real-gain-and-no-false-one
  (let [opts {:reps 300 :seed 2 :alpha 0.1 :per-block 4}
        real (score/paired-loss-2way (sim-rows 3 0.8 ["a" "b"]) brier opts)
        none (score/paired-loss-2way (sim-rows 3 0.0 ["a" "b"]) brier opts)]
    (is (pos? (:diff real)) "positive: the model's loss is lower")
    (is (pos? (:lo real)) real)
    (is (<= (:lo none) 0.0 (:hi none)) none)
    (testing "groups weigh equally: the mean of each group's mean"
      (let [rows (concat (map #(assoc % 5 "big") (sim-rows 4 0.8 ["x"]))
                         (take 200 (map #(assoc % 5 "small") (sim-rows 5 0.0 ["y"]))))
            r (score/paired-loss-2way rows brier opts)
            by (fn [grp] (:diff (score/paired-loss-2way (filter #(= grp (nth % 5)) rows) brier opts)))]
        (is (< (Math/abs (- (:diff r) (/ (+ (by "big") (by "small")) 2.0))) 1e-12))))
    (is (= (count (:draws real)) 300))))

(deftest romano-wolf-controls-the-family-and-finds-real-cells
  (let [rows (concat (sim-rows 6 0.9 ["real"]) (sim-rows 7 0.0 ["null1" "null2" "null3"]))
        cells (score/romano-wolf rows brier {:reps 300 :seed 3 :alpha 0.05 :per-block 4 :cell #(nth % 5)})]
    (is (= #{"real" "null1" "null2" "null3"} (set (keys cells))))
    (is (:reject (cells "real")) (cells "real"))
    (is (not-any? #(:reject (cells %)) ["null1" "null2" "null3"]) cells)
    (testing "adjusted p-values never below the unadjusted"
      (is (every? #(>= (:p-adj %) (:p %)) (vals cells))))))

(deftest romano-wolf-skips-a-cell-a-draw-missed
  ;; a cell with one unit is missed by about a third of the draws; counted
  ;; as a mean of zero there, its centred t is large when its difference is
  ;; negative, and the max-T would mask a real cell
  (let [real (sim-rows 6 0.9 ["real"])
        tiny (for [q (range 40)] [q 0.9 0.1 false "lonely" "tiny"])
        cells (score/romano-wolf (concat real tiny) brier {:reps 300 :seed 3 :alpha 0.05 :per-block 4 :cell #(nth % 5)})]
    (is (:reject (cells "real")) (cells "real"))
    (is (not (:reject (cells "tiny"))))
    (is (< (:se (cells "tiny")) 1e-6) "every row the same difference: no spread when drawn, and no draw counted when missed")))
