(ns augur.enkf-test
  (:require [augur.enkf :as e]
            [augur.math :as m]
            [clojure.test :refer [deftest is testing]]))

(defn- mean [xs] (/ (reduce + xs) (double (count xs))))
(defn- sd [xs] (let [mu (mean xs)] (Math/sqrt (/ (reduce + (map #(* (- % mu) (- % mu)) xs)) (dec (count xs))))))

(deftest a-linear-gaussian-update-is-the-kalman-filter
  ;; one state, observed directly: the ensemble's mean and spread after
  ;; the update approach the Kalman posterior N((m/p + y/r)/(1/p + 1/r), ...)
  (let [g (m/rng 1)
        ens (vec (repeatedly 4000 #(vector (+ 1.0 (* 2.0 (m/normal! g))))))
        post (e/update ens (fn [x] [(first x)]) [3.0] [1.0] g)
        xs (map first post)
        pv (/ 1.0 (+ (/ 1.0 4.0) 1.0))]
    (is (< (Math/abs (- (mean xs) (* pv (+ (/ 1.0 4.0) 3.0)))) 0.05))
    (is (< (Math/abs (- (sd xs) (Math/sqrt pv))) 0.05))))

(deftest an-unobserved-part-moves-by-its-covariance
  ;; a parameter never observed, correlated with the observed state through
  ;; the ensemble, is learned from the observation (the augmented state)
  (let [g (m/rng 2)
        ens (vec (for [_ (range 2000)] (let [k (m/normal! g)] [k (+ (* 2.0 k) (* 0.1 (m/normal! g)))])))
        post (e/update ens (fn [x] [(second x)]) [2.0] [0.01] g)]
    (is (< (Math/abs (- 1.0 (mean (map first post)))) 0.05) "y = 2k observed at 2: k near 1")
    (is (< (sd (map first post)) 0.1))))

(deftest several-observations
  (let [g (m/rng 3)
        ens (vec (repeatedly 3000 #(vector (m/normal! g) (m/normal! g))))
        post (e/update ens identity [1.0 -1.0] [0.25 0.25] g)]
    (is (< (Math/abs (- 0.8 (mean (map first post)))) 0.05))
    (is (< (Math/abs (+ 0.8 (mean (map second post)))) 0.05))))

(deftest bad-inputs-are-refused
  (let [g (m/rng 4) ens [[1.0] [2.0] [3.0]]]
    (is (thrown? clojure.lang.ExceptionInfo (e/update ens identity [1.0] [0.0] g)) "zero noise")
    (is (thrown? clojure.lang.ExceptionInfo (e/update ens identity [1.0] [Double/NaN] g)))
    (is (thrown? clojure.lang.ExceptionInfo (e/update [[1.0]] identity [1.0] [1.0] g)) "one member")
    (is (thrown? clojure.lang.ExceptionInfo (e/update ens identity [1.0 2.0] [1.0] g)) "a variance per observation")
    (is (thrown? clojure.lang.ExceptionInfo (e/update [[1.0] [Double/NaN] [3.0]] identity [1.0] [1.0] g)) "a member not finite")))

(deftest many-observations-at-once
  ;; a state observed thirty times (an ensemble smoother's case): the
  ;; posterior of a mean from thirty noisy draws
  (let [g (m/rng 5)
        ens (vec (repeatedly 2000 #(vector (* 3.0 (m/normal! g)))))
        ys (vec (repeatedly 30 #(+ 1.0 (m/normal! g))))
        post (e/update ens (fn [x] (vec (repeat 30 (first x)))) ys (vec (repeat 30 1.0)) g)
        pv (/ 1.0 (+ (/ 1.0 9.0) 30.0))]
    (is (< (Math/abs (- (mean (map first post)) (* pv (reduce + ys)))) 0.03))
    (is (< (Math/abs (- (sd (map first post)) (Math/sqrt pv))) 0.02))))
