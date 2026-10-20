(ns augur.trees-test
  (:require [augur.logit :as logit]
            [augur.math :as m]
            [augur.score :as score]
            [augur.trees :as trees]
            [clojure.test :refer [deftest is testing]]))

(defn- rows [n seed f]
  (let [g (m/rng seed)]
    (vec (for [_ (range n)
               :let [a (- (m/next-double! g) 0.5) b (- (m/next-double! g) 0.5) c (m/next-double! g)]]
           {:x (double-array [1.0 a b c]) :y (f a b c g) :w 1.0}))))

(deftest learns-an-interaction-a-logit-cannot
  ;; onset when exactly one of two inputs is high: no linear term carries it
  (let [xor (fn [a b _ g] (if (not= (pos? a) (pos? b)) (< (m/next-double! g) 0.9) (< (m/next-double! g) 0.1)))
        train (rows 2000 1 xor) test (rows 1000 2 xor)
        model (trees/fit train (assoc trees/defaults :rounds 50))
        logit (logit/fit train 4 {:prior 1.0})
        ys (map :y test)]
    (is (> (score/auc (map #(trees/predict model (:x %)) test) ys) 0.85))
    (is (< (score/auc (map #(logit/predict logit (:x %)) test) ys) 0.6))))

(deftest predictions-are-probabilities-near-the-rate
  (let [train (rows 500 3 (fn [_ _ _ g] (< (m/next-double! g) 0.2)))
        model (trees/fit train (assoc trees/defaults :rounds 20))
        ps (map #(trees/predict model (:x %)) train)]
    (is (every? #(< 0.0 % 1.0) ps))
    (is (< (Math/abs (- 0.2 (/ (reduce + ps) (count ps)))) 0.05))))

(deftest weights-count
  (let [train (mapv (fn [r] (assoc r :w (if (:y r) 1.0 0.25))) (rows 1000 4 (fn [_ _ _ g] (< (m/next-double! g) 0.2))))
        model (trees/fit train (assoc trees/defaults :rounds 0))]
    (testing "with no rounds, the weighted base rate: 0.2 / (0.2 + 0.8 x 0.25) = 0.5"
      (is (< (Math/abs (- 0.5 (trees/predict model (double-array [1.0 0.0 0.0 0.0])))) 0.05)))))

(deftest deterministic
  (let [train (rows 300 5 (fn [a _ _ g] (< (m/next-double! g) (if (pos? a) 0.6 0.2))))]
    (is (= (trees/fit train (assoc trees/defaults :rounds 10)) (trees/fit train (assoc trees/defaults :rounds 10))))))

(deftest rare-flags-and-few-valued-inputs-can-be-split
  ;; a flag on in 5% of rows that carries the onsets
  (let [g (m/rng 9)
        train (vec (for [i (range 1000)
                         :let [flag (if (< (mod i 100) 5) 1.0 0.0) level (double (mod i 3))]]
                     {:x (double-array [1.0 flag level]) :y (< (m/next-double! g) (if (pos? flag) 0.6 0.02)) :w 1.0}))
        model (trees/fit train (assoc trees/defaults :rounds 20))]
    (is (> (trees/predict model (double-array [1.0 1.0 0.0])) (* 5 (trees/predict model (double-array [1.0 0.0 0.0])))))
    (is (= [0.5 1.5] (trees/cuts (map :x train) 2 16)) "every gap of a three-valued input")))

(deftest missing-values-go-the-same-way-in-fitting-and-predicting
  (let [train (vec (for [i (range 200)] {:x (double-array [1.0 (if (< i 100) Double/NaN (double i))]) :y (< i 100) :w 1.0}))
        model (trees/fit train (assoc trees/defaults :rounds 20))]
    (is (> (trees/predict model (double-array [1.0 Double/NaN])) 0.8))))
