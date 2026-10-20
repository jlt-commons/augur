(ns augur.pool-test
  (:require [augur.math :as m]
            [augur.pool :as pool]
            [clojure.test :refer [deftest is testing]]))

(deftest combines-forecasters
  ;; one forecaster knows which unit (a fixed risk), another knows when (a
  ;; common shock); onsets need both
  (let [g (m/rng 7)
        rows (vec (for [t (range 200) u (range 20)]
                    (let [risky (< u 4) shock (zero? (rem t 5))
                          p (m/expit (+ -5.0 (if risky 2.5 0.0) (if shock 2.5 0.0)))]
                      {:who (m/expit (+ -4.0 (if risky 2.0 0.0)))
                       :when (m/expit (+ -4.0 (if shock 2.0 0.0)))
                       :noise (m/next-double! g)
                       :y (< (m/next-double! g) p)})))
        ys (map :y rows)
        ws (repeat (count rows) 1.0)]
    (testing "a linear pool puts its weight on forecasters that help"
      (let [w (pool/pool-weights (map (juxt :who :noise) rows) ys ws)]
        (is (> (first w) 0.8) (vec w))
        (is (< (Math/abs (- 1.0 (reduce + w))) 1e-9))))
    (testing "a log-linear stack beats each forecaster when they know different things"
      (let [st (pool/fit-stack (map (juxt :who :when) rows) ys ws {:prior 1.0})
            ps (map #(pool/predict-stack st [(:who %) (:when %)]) rows)
            ll (fn [ps] (reduce + (map (fn [p y] (Math/log (if y p (- 1.0 p)))) ps ys)))]
        (is (> (ll ps) (ll (map :who rows))))
        (is (> (ll ps) (ll (map :when rows))))))))

(deftest hedge-weights-follow-each-members-record
  (let [ps [[0.9 0.5] [0.9 0.5] [0.1 0.5] [0.1 0.5]]
        ys [true true false false]
        w (pool/hedge-weights ps ys [1.0 1.0 1.0 1.0] 1.0)]
    (is (< (Math/abs (- 1.0 (+ (aget w 0) (aget w 1)))) 1e-9))
    (is (> (aget w 0) 0.7) "the sharper member, right every time")
    (is (= [0.5 0.5] (vec (pool/hedge-weights ps ys [1.0 1.0 1.0 1.0] 0.0))) "no learning: equal")
    (testing "per onset: twice the rows with the same record give the same weights"
      (is (< (Math/abs (- (aget w 0) (aget (pool/hedge-weights (into ps ps) (into ys ys) (repeat 8 1.0) 1.0) 0))) 1e-9)))))

(deftest converged-pool-weights
  ;; two good forecasters and many noisy copies: plain EM from equal weights
  ;; crawls; the accelerated fit reaches the maximum
  (let [g (m/rng 31)
        n 3000
        truth (vec (for [_ (range n)] (- (m/normal! g) 3.0)))
        ys (mapv #(< (m/next-double! g) (m/expit %)) truth)
        ps (mapv (fn [z] (vec (concat [(m/expit z) (m/expit (+ z (* 0.3 (m/normal! g))))]
                                      (for [_ (range 10)] (m/expit (+ -3.0 (* 0.2 (m/normal! g)))))))) truth)
        ws (repeat n 1.0)
        ll (fn [w] (reduce + (map (fn [row y] (let [p (pool/pool w row)] (Math/log (if y p (- 1.0 p))))) ps ys)))
        slow (pool/pool-weights ps ys ws :iterations 20000)
        fast (pool/pool-weights-ml ps ys ws)]
    (is (< (Math/abs (- (ll fast) (ll slow))) 1e-3) "the same maximum")
    (is (> (ll fast) (ll (pool/pool-weights ps ys ws)) ) "beyond 200 plain steps")
    (is (< (Math/abs (- 1.0 (reduce + fast))) 1e-9))
    (is (every? #(>= % 0.0) fast))))

(deftest median-of-members
  ;; the untrained combination that won the epidemic forecast hubs (Cramer
  ;; et al. 2022; Sherratt et al. 2023): the median of the members' log odds
  (is (< (Math/abs (- 0.2 (pool/median-pool [0.1 0.2 0.9]))) 1e-12))
  (is (< (Math/abs (- (m/expit (* 0.5 (+ (m/logit 0.1) (m/logit 0.3)))) (pool/median-pool [0.1 0.3]))) 1e-12)
      "an even count takes the mean of the middle two log odds"))
