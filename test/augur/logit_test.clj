(ns augur.logit-test
  (:require [augur.math :as m]
            [augur.logit :as logit]
            [augur.score :as score]
            [clojure.edn]
            [clojure.test :refer [deftest is testing]]))

(defn- synthetic [n seed]
  ;; onsets more likely where x1 is high, unrelated to x2
  (let [g (m/rng seed)]
    (vec (for [_ (range n)]
           (let [x1 (m/normal! g) x2 (m/normal! g)
                 p (m/expit (+ -3.0 (* 1.5 x1)))]
             {:x (double-array [1.0 x1 x2]) :y (< (m/next-double! g) p) :w 1.0})))))

(deftest fits-a-logistic-hazard
  (let [rows (synthetic 4000 1)
        beta (logit/fit rows 3 {:prior 1.0})]
    (is (< 1.2 (aget beta 1) 1.8) (vec beta))
    (is (< (Math/abs (aget beta 2)) 0.2) (vec beta))
    (testing "a warm start converges to the same place"
      (let [b2 (logit/fit rows 3 {:prior 1.0 :start beta :iterations 2})]
        (is (< (Math/abs (- (aget b2 1) (aget beta 1))) 1e-3))))))

(deftest stays-finite-on-rare-separable-events
  ;; few positives, many inputs on large scales, one that separates the
  ;; positives: the fit must not run away, even from a bad start
  (let [g (m/rng 4)
        rows (vec (for [i (range 3000)]
                    (let [y (< i 30)
                          xs (concat [1.0 (if y 3.0 0.0)]
                                     (for [_ (range 60)] (* 50.0 (m/normal! g))))]
                      {:x (double-array xs) :y y :w (if y 1.0 6.7)})))
        cold (logit/fit rows 62 {:prior 2.0})
        ;; warm-started far from the answer, as a yearly refit can be
        warm (logit/fit rows 62 {:prior 2.0 :start (double-array 62 3.0)})
        ps (map #(logit/predict warm (:x %)) rows)]
    (is (every? #(not (Double/isNaN %)) warm))
    (is (every? #(<= 0.0 % 1.0) ps))
    (is (> (score/auc ps (map :y rows)) 0.9))
    (is (< (Math/abs (- (logit/predict cold (:x (rows 0))) (logit/predict warm (:x (rows 0))))) 1e-3)
        "a bad start ends where a cold one does")))

(deftest horseshoe-like-shrinkage
  ;; one strong input among many weak ones: the adaptive penalty keeps the
  ;; strong coefficient nearer its value and shrinks the rest at least as hard
  (let [g (m/rng 37)
        k 21
        rows (vec (for [_ (range 3000)]
                    (let [x (double-array (cons 1.0 (repeatedly (dec k) #(m/normal! g))))]
                      {:x x :y (< (m/next-double! g) (m/expit (+ -3.0 (* 1.2 (aget x 1))))) :w 1.0})))
        ridge (logit/fit rows k {:prior 100.0})
        hs (logit/fit-shrunk rows k {:prior 100.0})
        noise (fn [^doubles b] (reduce + (map #(Math/abs (aget b %)) (range 2 k))))]
    (is (> (aget hs 1) (aget ridge 1)) "the signal is freed")
    (is (< (Math/abs (- 1.2 (aget hs 1))) (Math/abs (- 1.2 (aget ridge 1)))))
    (is (<= (noise hs) (* 1.05 (noise ridge))) "the noise no less shrunk")))

(deftest coefficients-store-as-edn
  (let [rows (synthetic 400 3)
        beta (logit/fit rows 3 {:prior 1.0})
        state (logit/->state beta)]
    (is (= state (clojure.edn/read-string (pr-str state))))
    (is (= (vec beta) (vec (logit/<-state state))))
    (is (= (logit/predict beta (:x (first rows))) (logit/predict state (vec (:x (first rows)))))
        "predict takes either form")
    (is (thrown? Exception (logit/fit rows 3 {})) ":prior is required")))

(deftest offsets-are-fixed-parts-of-the-linear-predictor
  ;; y depends on x1 and a known per-row offset; fitting with the offset
  ;; recovers the same slope as fitting it as an input whose coefficient is 1
  (let [g (m/rng 7)
        rows (vec (for [_ (range 4000)]
                    (let [x1 (m/normal! g) o (* 2.0 (m/normal! g))
                          p (m/expit (+ -2.0 (* 1.0 x1) o))]
                      {:x (double-array [1.0 x1]) :o o :y (< (m/next-double! g) p) :w 1.0})))
        with (logit/fit rows 2 {:prior 0.01})
        without (logit/fit (map #(dissoc % :o) rows) 2 {:prior 0.01})]
    (is (< 0.85 (aget with 1) 1.15) (vec with))
    (is (< -2.2 (aget with 0) -1.8) (vec with))
    (testing "ignoring the offset attenuates the slope"
      (is (< (aget without 1) (aget with 1))))
    (testing "predict adds the offset"
      (is (< (Math/abs (- (logit/predict with [1.0 0.5] 1.5)
                          (m/expit (+ (aget with 0) (* 0.5 (aget with 1)) 1.5))))
             1e-12)))))

(deftest the-prior-precision-can-be-estimated
  ;; coefficients drawn with precision 4 on standardized inputs: the
  ;; evidence (Laplace EM) step finds it from either side
  (let [g (m/rng 11)
        k 9
        truth (double-array (cons -2.0 (repeatedly (dec k) #(* 0.5 (m/normal! g)))))
        rows (vec (for [_ (range 6000)]
                    (let [x (double-array (cons 1.0 (repeatedly (dec k) #(m/normal! g))))]
                      {:x x :y (< (m/next-double! g) (logit/predict truth x)) :w 1.0})))
        from-weak (logit/fit-evidence rows k {:prior 0.01 :rounds 30})
        from-strong (logit/fit-evidence rows k {:prior 1000.0 :rounds 30})]
    (is (< 1.0 (:prior from-weak) 16.0) (:prior from-weak))
    (is (< (Math/abs (- (Math/log (:prior from-weak)) (Math/log (:prior from-strong)))) 0.05)
        [(:prior from-weak) (:prior from-strong)])
    (testing "the coefficients are the fit at the estimated prior"
      (is (< (reduce max (map #(Math/abs (- %1 %2)) (:beta from-weak)
                              (logit/fit rows k {:prior (:prior from-weak)})))
             1e-3))))
  (is (thrown? Exception (logit/fit-evidence [] 2 {:prior 1.0})) "needs :rounds"))

(deftest one-evidence-step
  (let [g (m/rng 12)
        rows (vec (for [_ (range 2000)]
                    (let [x (double-array [1.0 (m/normal! g) (m/normal! g) (m/normal! g)])]
                      {:x x :y (< (m/next-double! g) (logit/predict (double-array [-2.0 0.5 -0.3 0.0]) x)) :w 1.0 :o 0.1})))
        beta (logit/fit rows 4 {:prior 10.0})
        step (logit/evidence-step rows 4 beta 10.0)]
    (is (= step (:prior (logit/fit-evidence rows 4 {:prior 10.0 :rounds 1})))
        "fit-evidence's first round is a fit then this step")
    (is (< 0.0 step 1e6))))
