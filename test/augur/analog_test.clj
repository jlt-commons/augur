(ns augur.analog-test
  (:require [augur.analog :as an]
            [augur.math :as m]
            [clojure.test :refer [deftest is testing]]))

(defn- rows [n seed]
  ;; onsets only in one corner of the inputs: a shape no line separates
  (let [g (m/rng seed)]
    (vec (for [_ (range n)
               :let [a (m/normal! g) b (m/normal! g)
                     hot (and (> a 1.0) (> b 1.0))]]
           {:x (double-array [1.0 a b]) :y (< (m/next-double! g) (if hot 0.5 0.01)) :w 1.0}))))

(deftest neighbours-carry-the-onsets-near-them
  (let [model (an/fit (rows 4000 1) {:k 100 :prior 10.0})
        hot (an/predict model (double-array [1.0 1.8 1.8]))
        cold (an/predict model (double-array [1.0 -1.0 0.0]))]
    (is (> hot 0.25))
    (is (< cold 0.05))
    (is (< 0.0 cold hot 1.0))))

(deftest weights-and-shrinkage
  (testing "with no neighbours' onsets the forecast is the shrunk base rate, not zero"
    (let [train (vec (for [i (range 200)] {:x (double-array [1.0 (double i)]) :y (< i 4) :w 1.0}))
          model (an/fit train {:k 10 :prior 10.0})
          p (an/predict model (double-array [1.0 150.0]))]
      (is (< (Math/abs (- p (/ (* 10.0 0.02) 20.0))) 1e-9) "prior x base over k + prior")))
  (testing "a row's weight counts it that many times"
    (let [train [{:x (double-array [1.0 0.0]) :y true :w 3.0} {:x (double-array [1.0 0.1]) :y false :w 1.0}]
          model (an/fit train {:k 2 :prior 0.0})]
      (is (< (Math/abs (- 0.75 (an/predict model (double-array [1.0 0.0])))) 1e-9)))))

(deftest constant-columns-are-ignored
  (let [train (vec (for [i (range 50)] {:x (double-array [1.0 7.0 (double i)]) :y (< i 5) :w 1.0}))
        model (an/fit train {:k 5 :prior 0.0})]
    (is (> (an/predict model (double-array [1.0 7.0 1.0])) 0.5))))
