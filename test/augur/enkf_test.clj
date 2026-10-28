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

;; -- linear equality constraints (accounting identities) --------------------------------

(defn- dot [a b] (reduce + (map * a b)))

(deftest a-projection-lands-on-the-constraints-nearest
  ;; x1 + x2 + x3 = 6 and x1 - x3 = 0: the nearest point to (1 2 3) is
  ;; (2 2 2) (the residual along the rows' span)
  (let [a [[1.0 1.0 1.0] [1.0 0.0 -1.0]]
        p (e/project [1.0 2.0 3.0] a [6.0 0.0])]
    (is (every? #(< (Math/abs (- 2.0 %)) 1e-12) p))
    (testing "a point already on the constraints does not move"
      (is (= [2.0 2.0 2.0] (mapv #(/ (Math/round (* 1e12 %)) 1e12) (e/project [2.0 2.0 2.0] a [6.0 0.0])))))
    (testing "a redundant row (a combination of the others) is allowed"
      (let [p2 (e/project [1.0 2.0 3.0] (conj a [2.0 1.0 0.0]) [6.0 0.0 6.0])]
        (is (every? #(< (Math/abs (- 2.0 %)) 1e-12) p2))))
    (testing "contradictory rows are refused"
      (is (thrown? clojure.lang.ExceptionInfo (e/project [1.0 2.0 3.0] (conj a [2.0 1.0 0.0]) [6.0 0.0 7.0]))))))

(deftest an-update-keeps-identities-every-member-holds
  ;; members on x1 + x2 - x3 = 0 (an identity), the first observed: the
  ;; update moves members along the ensemble's own deviations, so the
  ;; identity holds after it; update-constrained also projects, so it holds
  ;; when the members did not quite satisfy it
  (let [g (m/rng 7)
        ens (vec (for [_ (range 50)] (let [x1 (* 1e6 (+ 1.0 (* 0.1 (m/normal! g)))) x2 (* 1e6 (m/normal! g))] [x1 x2 (+ x1 x2)])))
        a [[1.0 1.0 -1.0]]
        post (e/update ens (fn [x] [(first x)]) [1.2e6] [1e8] g)
        cpost (e/update-constrained (mapv #(update % 2 + 1e-3) ens) (fn [x] [(first x)]) [1.2e6] [1e8] g {:a a :b [0.0]})]
    (is (every? #(< (Math/abs (dot (first a) %)) 1e-6) post))
    (is (every? #(< (Math/abs (dot (first a) %)) 1e-8) cpost))
    (is (< (Math/abs (- 1.2e6 (mean (map first cpost)))) 3e4))
    (testing "a member's own right-hand side"
      (let [bs (vec (for [i (range 50)] [(double i)]))
            q (e/update-constrained ens (fn [x] [(first x)]) [1.2e6] [1e8] g {:a a :b-of (fn [i] (bs i))})]
        (is (every? (fn [[i x]] (< (Math/abs (- (dot (first a) x) i)) 1e-8)) (map-indexed vector q)))))))
