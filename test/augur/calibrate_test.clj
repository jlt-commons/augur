(ns augur.calibrate-test
  (:require [augur.calibrate :as cal]
            [clojure.edn]
            [clojure.test :refer [deftest is testing]]))

(deftest overconfidence-is-learned-away
  ;; forecasts of 0.9 that come true half the time
  (let [outcomes (for [i (range 200)] {:key :violence :p 0.9 :outcome (even? i)})
        c (cal/learn {} outcomes cal/defaults)]
    (is (< 0.45 (cal/calibrate c :violence 0.9 cal/defaults) 0.55))
    (is (= 0.3 (cal/calibrate c :protest 0.3 cal/defaults)) "an unseen channel passes through")
    (is (= 0.03 (cal/calibrate c :violence 0.03 cal/defaults)) "an empty bin keeps the raw value")))

(deftest pool-adjacent-violators
  (testing "the weighted least-squares nondecreasing fit"
    (is (= [1.0 2.5 2.5 4.0] (cal/isotonic [1.0 3.0 2.0 4.0] [1.0 1.0 1.0 1.0])))
    (is (= [1.0 2.25 2.25] (cal/isotonic [1.0 3.0 2.0] [1.0 1.0 3.0])))
    (is (= [2.0 2.0 2.0] (cal/isotonic [3.0 2.0 1.0] [1.0 1.0 1.0])))
    (is (= [] (cal/isotonic [] [])))))

(deftest calibration-is-monotone
  ;; forecasts of 0.45 come true 60% of the time, forecasts of 0.65 only 30%:
  ;; the calibrated map may not rank the higher forecast lower
  (let [outcomes (concat (for [i (range 100)] {:key :violence :p 0.45 :outcome (< (mod i 10) 6)})
                         (for [i (range 100)] {:key :violence :p 0.65 :outcome (< (mod i 10) 3)}))
        c (cal/learn {} outcomes cal/defaults)]
    (is (<= (cal/calibrate c :violence 0.45 cal/defaults) (cal/calibrate c :violence 0.65 cal/defaults)))
    (is (< 0.35 (cal/calibrate c :violence 0.45 cal/defaults) 0.5) "the two pooled")))

(deftest isotonic-recalibration-map
  ;; forecasts of 0.1 come true 20% of the time, of 0.5 30%, of 0.9 30%
  (let [pairs (concat (for [i (range 10)] [0.1 (< i 2)]) (for [i (range 10)] [0.5 (< i 3)])
                      (for [i (range 10)] [0.9 (< i 3)]))
        f (cal/isotonic-map (map first pairs) (map second pairs))]
    (is (< (Math/abs (- 0.2 (f 0.1))) 1e-12))
    (is (< (Math/abs (- 0.3 (f 0.9))) 1e-12))
    (is (< (Math/abs (- 0.2 (f 0.3))) 1e-12) "between forecasts, the step below")
    (is (< (Math/abs (- 0.2 (f 0.01))) 1e-12) "below the lowest, the lowest step")))

(deftest logistic-recalibration
  ;; overconfident forecasts: the truth is half as extreme on the logit scale
  (let [logit #(Math/log (/ % (- 1.0 %)))
        sigm #(/ 1.0 (+ 1.0 (Math/exp (- %))))
        ps (for [i (range 2000)] (sigm (- (* 6.0 (/ (mod i 100) 100.0)) 3.0)))
        ys (for [[i p] (map-indexed vector ps)] (< (mod (* i 7919) 1000) (* 1000 (sigm (* 0.5 (logit p))))))
        f (cal/logistic-map ps ys)]
    (testing "pulls extreme forecasts in, and keeps their order"
      (is (< (f 0.95) 0.9))
      (is (> (f 0.05) 0.1))
      (is (< (f 0.3) (f 0.31))))))

(deftest the-table-is-edn-and-forgets
  (let [o (fn [p y] {:key [:gate :x] :p p :outcome y})
        c (cal/learn nil [(o 0.3 true) (o 0.3 false)] cal/defaults)
        c2 (cal/learn c [] cal/defaults)]
    (is (= c (clojure.edn/read-string (pr-str c))))
    (is (= 2.0 (reduce + (get-in c [[:gate :x] :n]))))
    (is (< (Math/abs (- 1.8 (reduce + (get-in c2 [[:gate :x] :n])))) 1e-12) "a batch discounts the old counts by :forget")))
