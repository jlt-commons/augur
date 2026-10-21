(ns augur.dynamic-test
  (:require [augur.dynamic :as d]
            [augur.math :as m]
            [clojure.test :refer [deftest is testing]]))

(defn- rows
  "n rows [x y] with x = [1 z1 z2], outcomes from coefficients `beta`."
  [g n beta]
  (vec (for [_ (range n)]
         (let [x [1.0 (m/normal! g) (m/normal! g)]
               eta (reduce + (map * beta x))]
           {:x x :y (< (m/next-double! g) (m/expit eta))}))))

(defn- run
  "The state after stepping through `periods` (each a vector of rows)."
  [state periods]
  (reduce d/step state periods))

(deftest learns-static-coefficients
  (let [g (m/rng 1)
        truth [-1.0 1.2 -0.8]
        s (run (d/logistic {:mean [0 0 0] :var [10 10 10] :discount 1.0})
               (repeatedly 60 #(rows g 50 truth)))]
    (is (every? true? (map #(< (Math/abs (- %1 %2)) 0.25) (d/means s) truth)) (d/means s))))

(deftest a-discount-follows-a-change
  ;; the second coefficient flips sign halfway: a discount below one
  ;; forgets the old sign, one does not
  (let [g (m/rng 2)
        before (repeatedly 40 #(rows g 40 [-1.0 1.5 0.0]))
        after (repeatedly 40 #(rows g 40 [-1.0 -1.5 0.0]))
        fixed (run (d/logistic {:mean [0 0 0] :var [10 10 10] :discount 1.0}) (concat before after))
        moving (run (d/logistic {:mean [0 0 0] :var [10 10 10] :discount 0.9}) (concat before after))]
    (is (< (second (d/means moving)) -1.0) (d/means moving))
    (is (> (second (d/means fixed)) (second (d/means moving))))))

(deftest informative-priors-hold-until-the-data-say-otherwise
  (let [g (m/rng 3)
        prior {:mean [-1.0 1.0 0.0] :var [0.01 0.01 1.0] :discount 1.0}
        few (run (d/logistic prior) [(rows g 20 [-1.0 -1.0 0.0])])
        many (run (d/logistic (assoc prior :var [0.01 1.0 1.0])) (repeatedly 50 #(rows g 50 [-1.0 -1.0 0.0])))]
    (is (> (second (d/means few)) 0.8) "a tight prior and little data: the prior")
    (is (< (second (d/means many)) -0.6) "a looser prior and much data: the data")))

(deftest predictions-widen-with-uncertainty
  (let [s (d/logistic {:mean [2.0 0 0] :var [0.0001 0 0] :discount 1.0})
        unsure (d/logistic {:mean [2.0 0 0] :var [25.0 0 0] :discount 1.0})
        x [1.0 0.0 0.0]]
    (is (< (Math/abs (- (d/predict s x) (m/expit 2.0))) 1e-3))
    (is (< (d/predict unsure x) (- (d/predict s x) 0.1)) "an uncertain log odds pulls toward a half")))

(deftest poisson-counts
  (let [g (m/rng 4)
        draw (fn [lambda] (loop [k 0 p (Math/exp (- lambda)) s p u (m/next-double! g)]
                            (if (> u s) (let [k (inc k) p (/ (* p lambda) k)] (recur k p (+ s p) u)) k)))
        periods (repeatedly 60 (fn [] (vec (for [_ (range 30)]
                                             (let [x [1.0 (m/normal! g)]]
                                               {:x x :y (draw (Math/exp (+ 0.5 (* 0.7 (x 1)))))})))))
        s (run (d/poisson {:mean [0 0] :var [10 10] :discount 1.0}) periods)]
    (is (< (Math/abs (- (first (d/means s)) 0.5)) 0.15) (d/means s))
    (is (< (Math/abs (- (second (d/means s)) 0.7)) 0.15) (d/means s))))

(deftest model-averaging-finds-what-matters
  ;; three models on different inputs; the outcome depends on the second
  ;; input until halfway, then on the third: the weights follow
  (let [g (m/rng 5)
        period (fn [beta] (vec (for [_ (range 40)]
                                 (let [x [1.0 (m/normal! g) (m/normal! g)]]
                                   {:x x :y (< (m/next-double! g) (m/expit (reduce + (map * beta x))))}))))
        models {:first (d/logistic {:mean [0 0] :var [4 4] :discount 0.98 :inputs [0 1]})
                :second (d/logistic {:mean [0 0] :var [4 4] :discount 0.98 :inputs [0 2]})
                :none (d/logistic {:mean [0] :var [4] :discount 0.98 :inputs [0]})}
        avg (d/averaged models {:forget 0.95})
        mid (reduce d/step avg (repeatedly 40 #(period [-1.0 2.0 0.0])))
        end (reduce d/step mid (repeatedly 40 #(period [-1.0 0.0 2.0])))]
    (is (> (get (d/weights mid) :first) 0.8) (d/weights mid))
    (is (> (get (d/weights end) :second) 0.8) (d/weights end))
    (testing "the averaged prediction is the weighted one"
      (let [x [1.0 0.5 -0.5]
            w (d/weights end)]
        (is (< (Math/abs (- (d/predict end x)
                            (reduce + (for [[k mdl] (:models end)] (* (w k) (d/predict mdl x))))))
               1e-12))))))

(deftest options-are-required
  (is (thrown? Exception (d/logistic {:mean [0] :var [1]})) "needs :discount")
  (is (thrown? Exception (d/logistic {:mean [0 0] :var [1] :discount 1.0})) "one variance per mean"))
