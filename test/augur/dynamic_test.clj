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

(deftest coupling-pulls-means-by-certainty
  (let [a (d/logistic {:mean [0.0 0.0] :var [1.0 1.0] :discount 1.0})
        sure (d/logistic {:mean [5.0 2.0] :var [1.0 0.01] :discount 1.0})
        unsure (d/logistic {:mean [5.0 2.0] :var [1.0 100.0] :discount 1.0})
        [a1 _] (d/couple a sure {:indices [1] :strength 0.5})
        [a2 _] (d/couple a unsure {:indices [1] :strength 0.5})]
    (is (= 0.0 (first (d/means a1))) "an index not listed stays")
    (is (> (second (d/means a1)) 1.5) "toward a confident partner, far")
    (is (< (second (d/means a2)) 0.1) "toward an unsure one, little")
    (is (= (vec (:c a)) (vec (:c a1))) "variances stay: no evidence counted twice"))
  (testing "both ways at once"
    (let [a (d/logistic {:mean [0.0 0.0] :var [1.0 1.0] :discount 1.0})
          b (d/logistic {:mean [0.0 2.0] :var [1.0 1.0] :discount 1.0})
          [a' b'] (d/couple a b {:indices [1] :strength 1.0})]
      (is (< (Math/abs (- (second (d/means a')) 1.0)) 1e-12))
      (is (< (Math/abs (- (second (d/means b')) 1.0)) 1e-12)))))

(deftest no-weight-dies
  ;; a weight that underflows to zero could never come back
  (let [avg (d/averaged {:a (d/poisson {:mean [0.0] :var [400.0] :discount 1.0})
                         :b (d/poisson {:mean [0.0] :var [1.0] :discount 1.0})}
                        {:forget 0.99})
        after (d/step avg [{:x [1.0] :y 50}])]
    (is (every? pos? (vals (d/weights after))) (d/weights after))))

(deftest numeric-outcomes
  (let [s (reduce d/step (d/logistic {:mean [0.0] :var [4.0] :discount 1.0}) (repeat 20 [{:x [1.0] :y 0}]))]
    (is (< (d/predict s [1.0]) 0.2) "0 is no event, as false is")))

(deftest poisson-evidence-integrates-uncertainty
  ;; an unsure model's predictive spreads its mass: integrated over its
  ;; uncertainty, a zero count is likelier than a Poisson at its inflated
  ;; mean would say, and likelier than the sure model says
  (let [unsure (d/poisson {:mean [1.0] :var [2.0] :discount 1.0})
        sure (d/poisson {:mean [1.0] :var [0.0001] :discount 1.0})]
    (is (> (d/log-score unsure [{:x [1.0] :y 0}]) (d/log-score sure [{:x [1.0] :y 0}])))))

(deftest the-linear-predictor-s-moments
  (let [s (d/logistic {:mean [0.5 1.0] :var [0.2 0.3] :discount 1.0})
        {:keys [f q]} (d/predictive s [1.0 2.0] 0.25)]
    (is (< (Math/abs (- f 2.75)) 1e-12) "x'm plus the offset")
    (is (< (Math/abs (- q (+ 0.2 (* 4 0.3)))) 1e-12) "x'Cx")
    (testing "what predict integrates"
      (is (< (Math/abs (- (d/predict s [1.0 2.0] 0.25) (m/expit (/ f (Math/sqrt (+ 1.0 (/ (* Math/PI q) 8.0))))))) 1e-12))))
  (testing "a model average: the mixture's mean and variance"
    (let [a (d/logistic {:mean [1.0] :var [0.1] :discount 1.0})
          b (d/logistic {:mean [-1.0] :var [0.3] :discount 1.0})
          avg (assoc (d/averaged {:a a :b b} {:forget 1.0}) :w {:a 0.25 :b 0.75})
          {:keys [f q]} (d/predictive avg [1.0])]
      (is (< (Math/abs (- f -0.5)) 1e-12))
      ;; within, 0.25 0.1 + 0.75 0.3 = 0.25, plus between, 1 - 0.5^2 = 0.75
      (is (< (Math/abs (- q 1.0)) 1e-9)))))

;; --- the dynamic linear model (West & Harrison 1997, ch. 4: the
;; observation variance unknown, learned with the coefficients) ----------

(defn- linear-rows [g n beta sd]
  (vec (for [_ (range n)]
         (let [x [1.0 (m/normal! g) (m/normal! g)]]
           {:x x :y (+ (reduce + (map * beta x)) (* sd (m/normal! g)))}))))

(deftest linear-learns-coefficients-and-noise
  (let [g (m/rng 11)
        s (run (d/linear {:mean [0 0 0] :var [100 100 100] :discount 1.0})
               (repeatedly 40 #(linear-rows g 25 [2.0 3.0 -1.0] 0.5)))]
    (is (every? true? (map #(< (Math/abs (- %1 %2)) 0.05) (d/means s) [2.0 3.0 -1.0])) (d/means s))
    (is (< 0.2 (:s s) 0.3) "the observation variance, 0.25")
    (is (< (Math/abs (- 4.6 (d/predict s [1.0 1.0 0.4]))) 0.1) "the mean")
    (let [{:keys [f q s]} (d/predictive s [1.0 1.0 0.4])]
      (is (< 4.5 f 4.7))
      (is (< q 0.01) "the coefficients well known")
      (is (< 0.2 s 0.3) "and the noise beside them"))))

(deftest linear-follows-a-change
  (let [g (m/rng 12)
        periods (concat (repeatedly 30 #(linear-rows g 20 [0.0 1.0 0.0] 0.3))
                        (repeatedly 30 #(linear-rows g 20 [0.0 -1.0 0.0] 0.3)))
        moving (run (d/linear {:mean [0 0 0] :var [10 10 10] :discount 0.9}) periods)
        fixed (run (d/linear {:mean [0 0 0] :var [10 10 10] :discount 1.0}) periods)]
    (is (< (second (d/means moving)) -0.9))
    (is (> (second (d/means fixed)) -0.2) "without a discount, the average of both")))

(deftest linear-scores-and-averages
  ;; the log predictive density (Student t) is higher under the right
  ;; model, and the average finds it
  (let [g (m/rng 13)
        periods (repeatedly 40 #(linear-rows g 20 [0.0 0.0 2.0] 0.5))
        a (d/linear {:mean [0 0] :var [10 10] :discount 1.0 :inputs [0 1]})
        b (d/linear {:mean [0 0] :var [10 10] :discount 1.0 :inputs [0 2]})
        sa (run a (take 10 periods)) sb (run b (take 10 periods))
        test (nth periods 20)]
    (is (> (d/log-score sb test) (+ 20.0 (d/log-score sa test))))
    (let [avg (run (d/averaged {:z1 a :z2 b} {:forget 1.0}) periods)]
      (is (> (get (d/weights avg) :z2) 0.99))
      (is (< (Math/abs (- 2.0 (d/predict avg [1.0 0.0 1.0]))) 0.1)))))

(deftest linear-weights-and-offsets
  ;; a row's :w scales its precision, its :o is a fixed part of the mean
  (let [g (m/rng 14)
        rows (fn [] (vec (for [_ (range 20)] (let [x [1.0 (m/normal! g)]] {:x x :y (+ 5.0 (* 2.0 (second x)) (* 0.1 (m/normal! g))) :o 5.0}))))
        s (run (d/linear {:mean [0 0] :var [10 10] :discount 1.0}) (repeatedly 20 rows))]
    (is (< (Math/abs (first (d/means s))) 0.05) "the offset carries the 5")
    (is (< (Math/abs (- 2.0 (second (d/means s)))) 0.05))
    (is (thrown? clojure.lang.ExceptionInfo (d/linear {:mean [0] :var [1]})) "a discount is required")))

(deftest linear-mixture-density
  ;; an average's log score: the row's weight a power on the mixture
  (let [a (d/linear {:mean [0.0] :var [1.0] :discount 1.0})
        b (d/linear {:mean [3.0] :var [1.0] :discount 1.0})
        avg (d/averaged {:a a :b b} {:forget 1.0})
        row {:x [1.0] :y 1.0}
        la (d/log-score a [row]) lb (d/log-score b [row])
        mix (Math/log (+ (* 0.5 (Math/exp la)) (* 0.5 (Math/exp lb))))]
    (is (< (Math/abs (- (d/log-score avg [row]) mix)) 1e-9))
    (is (< (Math/abs (- (d/log-score avg [(assoc row :w 2.0)]) (* 2.0 mix))) 1e-9))))

(defn- close? [a b tol] (every? true? (map #(< (Math/abs (- (double %1) (double %2))) tol) a b)))

(deftest linear-matches-the-batch-posterior
  ;; with no discount, the sequential update is the conjugate batch
  ;; posterior (normal-inverse-gamma): prior C0* = var/s0, d0 = n0 s0, P =
  ;; C0*^-1 + X'X, b = P^-1 (C0*^-1 m0 + X'y), S = (d0 + y'y + m0'C0*^-1 m0
  ;; - b'Pb)/(n0 + N), C = S P^-1; here in one input, closed form
  (let [m0 2.0 v 4.0 s0 1.5 n0 3.0
        xs [0.5 -1.0 2.0 1.5] ys [1.0 -2.5 4.2 2.9]
        st (reduce #(d/step %1 [%2]) (d/linear {:mean [m0] :var [v] :discount 1.0 :s0 s0 :n0 n0})
                   (map (fn [x y] {:x [x] :y y}) xs ys))
        c0 (/ v s0) p (+ (/ 1.0 c0) (reduce + (map #(* % %) xs)))
        b (/ (+ (/ m0 c0) (reduce + (map * xs ys))) p)
        sv (/ (+ (* n0 s0) (reduce + (map #(* % %) ys)) (/ (* m0 m0) c0) (- (* b b p))) (+ n0 (count xs)))]
    (is (close? (d/means st) [b] 1e-12))
    (is (< (Math/abs (- (:s st) sv)) 1e-12))
    (is (< (Math/abs (- (aget ^doubles (:c st) 0) (/ sv p))) 1e-12))
    (is (= (+ n0 4.0) (:n st)))))

(deftest a-weight-is-the-row-repeated
  ;; :w 2 is the same row twice in the period
  (let [mk #(d/linear {:mean [0.0 0.0] :var [4.0 4.0] :discount 1.0 :s0 1.0 :n0 3.0})
        row {:x [1.0 0.5] :y 3.0}
        once (d/step (mk) [(assoc row :w 2.0)])
        twice (d/step (mk) [row row])]
    (is (close? (d/means once) (d/means twice) 1e-12))
    (is (< (Math/abs (- (:s once) (:s twice))) 1e-12))
    (is (close? (vec (:c once)) (vec (:c twice)) 1e-12))
    (is (= (:n once) (:n twice)))))

(deftest linear-densities-and-discounts
  (testing "one degree of freedom: the Cauchy"
    (let [st (d/linear {:mean [1.0] :var [0.5] :discount 1.0 :s0 2.0 :n0 1.0})
          big-q 2.5 e 1.0]
      (is (< (Math/abs (- (d/log-score st [{:x [1.0] :y 2.0}])
                          (- (Math/log (* Math/PI (Math/sqrt big-q) (+ 1.0 (/ (* e e) big-q)))))))
             1e-12))))
  (testing "the predictive under a discount: x'Cx/delta, and the noise beside it"
    (let [st (d/linear {:mean [0.0 0.0] :var [1.0 2.0] :discount 0.8 :s0 0.3 :n0 4.0})
          {:keys [q s n]} (d/predictive st [1.0 1.0])]
      (is (< (Math/abs (- q (/ 3.0 0.8))) 1e-12))
      (is (= 0.3 s)) (is (= 4.0 n))))
  (testing "a period without rows: only the discount"
    (let [st (d/step (d/linear {:mean [1.0] :var [1.0] :discount 0.5 :s0 0.3 :n0 4.0}) [])]
      (is (= [1.0] (d/means st))) (is (= 0.3 (:s st))) (is (= 4.0 (:n st)))
      (is (= 2.0 (aget ^doubles (:c st) 0)))))
  (testing "the averaged predictive carries the noise"
    (let [avg (d/averaged {:a (d/linear {:mean [0.0] :var [1.0] :discount 1.0 :s0 0.2})
                           :b (d/linear {:mean [0.0] :var [1.0] :discount 1.0 :s0 0.4})} {:forget 1.0})]
      (is (< (Math/abs (- 0.3 (:s (d/predictive avg [1.0])))) 1e-12))))
  (testing "a row without inputs throws, as for the other models"
    (is (thrown? Exception (d/step (d/linear {:mean [0.0] :var [1.0] :discount 1.0}) [{:y 1.0} {:x [1.0] :y 2.0}]))))
  (testing "the noise's prior must be positive"
    (is (thrown? clojure.lang.ExceptionInfo (d/linear {:mean [0.0] :var [1.0] :discount 1.0 :s0 0.0})))
    (is (thrown? clojure.lang.ExceptionInfo (d/linear {:mean [0.0] :var [1.0] :discount 1.0 :n0 0.0})))))

(deftest averaging-linear-models-on-sequential-evidence
  ;; a period's evidence for a linear model is the product of each row's
  ;; predictive given the rows before it: the marginal likelihood, so many
  ;; rows of an uncertain model are not counted as if independent
  (let [mk #(d/linear {:mean [0.0] :var [10.0] :discount 1.0 :s0 1.0 :n0 2.0})
        rows (vec (for [y [0.5 1.5 1.0 0.8]] {:x [1.0] :y y}))
        seq-ll (first (reduce (fn [[ll st] r] [(+ ll (d/log-score st [r])) (d/step st [r])]) [0.0 (mk)] rows))
        other (d/linear {:mean [5.0] :var [0.01] :discount 1.0 :s0 1.0 :n0 2.0})
        other-ll (first (reduce (fn [[ll st] r] [(+ ll (d/log-score st [r])) (d/step st [r])]) [0.0 other] rows))
        avg (d/step (d/averaged {:a (mk) :b other} {:forget 1.0}) rows)
        expect (/ 1.0 (+ 1.0 (Math/exp (- other-ll seq-ll))))]
    (is (< (Math/abs (- (get (d/weights avg) :a) (+ (* (- 1.0 1e-6) expect) 0.5e-6))) 1e-9))))
