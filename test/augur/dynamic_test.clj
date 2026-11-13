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

;; --- slopes pooled across groups of places (partial pooling) -------------

(defn- grouped-rows
  "n rows per group, x = [1 z1 z2 g], the outcome from intercept a and
  group g's slopes (slopes g)."
  [g n a slopes]
  (vec (for [grp (range (count slopes)) _ (range n)]
         (let [z1 (m/normal! g) z2 (m/normal! g)
               [b1 b2] (slopes grp)
               eta (+ a (* b1 z1) (* b2 z2))]
           {:x [1.0 z1 z2 (double grp)] :y (< (m/next-double! g) (m/expit eta))}))))

(defn- pooled-model [tau]
  (d/logistic {:mean [0 0 0] :var [4 1 1] :discount 1.0 :inputs [0 1 2]
               :by {:index 3 :groups 2 :slopes [1 2] :tau tau}}))

(deftest opposite-slopes-in-two-groups-are-recovered
  ;; a slope of +1 in one region and -1 in the other: one pooled slope
  ;; averages them to zero, slopes by group recover both
  (let [g (m/rng-mixed 21)
        periods (repeatedly 40 #(grouped-rows g 25 -1.0 [[1.0 0.0] [-1.0 0.0]]))
        one (run (d/logistic {:mean [0 0 0] :var [4 1 1] :discount 1.0 :inputs [0 1 2]}) periods)
        by (run (pooled-model 1.0) periods)
        [s0 s1] (d/slopes by)]
    (is (< (Math/abs (second (d/means one))) 0.2) "pooled: the slopes cancel")
    (is (< (Math/abs (- (first s0) 1.0)) 0.25) s0)
    (is (< (Math/abs (- (first s1) -1.0)) 0.25) s1)
    (is (< (Math/abs (second s0)) 0.2) "the irrelevant input stays near zero")
    (testing "the forecast reads its own group's slope"
      (is (> (d/predict by [1.0 2.0 0.0 0.0]) (+ 0.2 (d/predict by [1.0 2.0 0.0 1.0])))))))

(deftest a-common-slope-is-pooled
  ;; one slope everywhere; a group with few rows borrows from the one with
  ;; many, so its slope is better known than on its own
  (let [g (m/rng-mixed 22)
        periods (vec (for [_ (range 30)]
                       (let [many (grouped-rows g 40 -1.0 [[1.0 0.0] [1.0 0.0]])]
                         ;; group 1 keeps 2 rows a period
                         (into (filterv #(zero? (nth (:x %) 3)) many)
                               (take 2 (filter #(= 1.0 (nth (:x %) 3)) many))))))
        alone (run (d/logistic {:mean [0 0 0] :var [4 1 1] :discount 1.0 :inputs [0 1 2]})
                   (map (fn [p] (filterv #(= 1.0 (nth (:x %) 3)) p)) periods))
        by (run (pooled-model 0.1) periods)
        [_ s1] (d/slopes by)]
    (is (< (Math/abs (- (first s1) 1.0)) 0.15) s1)
    (is (< (Math/abs (- (first s1) 1.0)) (Math/abs (- (second (d/means alone)) 1.0)))
        "closer than the group on its own")))

(deftest averaging-over-tau-learns-how-much-to-pool
  ;; models at several tau, averaged: a small tau wins where the slopes are
  ;; common, a large one where they differ
  (let [avg #(d/averaged {:t0 (pooled-model 0.02) :t1 (pooled-model 1.0)} {:forget 1.0})
        common (let [g (m/rng-mixed 23)] (run (avg) (repeatedly 40 #(grouped-rows g 25 -1.0 [[1.0 0.0] [1.0 0.0]]))))
        apart (let [g (m/rng-mixed 24)] (run (avg) (repeatedly 40 #(grouped-rows g 25 -1.0 [[1.0 0.0] [-1.0 0.0]]))))]
    (is (> (:t0 (d/weights common)) 0.5) (d/weights common))
    (is (> (:t1 (d/weights apart)) 0.99) (d/weights apart))))

(deftest an-irrelevant-covariate-shrinks-to-zero
  ;; an input with no effect in any group: its common slope and every
  ;; group's stay near zero, nearer than separate unpooled models' and
  ;; held there by the deviations' prior
  (let [g (m/rng-mixed 25)
        periods (repeatedly 30 #(grouped-rows g 8 -1.0 [[0.0 0.0] [0.0 0.0]]))
        by (run (pooled-model 0.05) periods)
        sep (fn [grp] (second (d/means (run (d/logistic {:mean [0 0 0] :var [4 1 1] :discount 1.0 :inputs [0 1 2]})
                                            (map (fn [p] (filterv #(= (double grp) (nth (:x %) 3)) p)) periods)))))]
    (doseq [s (d/slopes by)]
      (is (< (Math/abs (first s)) 0.12) s))
    (is (< (reduce + (map #(Math/abs (first %)) (d/slopes by))) (+ (Math/abs (sep 0)) (Math/abs (sep 1))))))
  (testing "the deviations are held, not discounted: pooling does not dissolve"
    (let [st (d/step (d/logistic {:mean [0 0] :var [4 1] :discount 0.5 :inputs [0 1]
                                  :by {:index 2 :groups 2 :slopes [1] :tau 0.1}})
                     [])
          ^doubles c (:c st)
          k (:k st)]
      (is (= 4 k) "intercept, one common slope, a deviation per group")
      (is (< (Math/abs (- (aget c 0) 8.0)) 1e-12) "the common part discounted")
      (is (< (Math/abs (- (aget c (+ (* 2 k) 2)) 0.01)) 1e-12) "a deviation held at tau^2"))))

(deftest pooled-slopes-need-a-group-in-range
  (is (thrown? clojure.lang.ExceptionInfo
               (d/predict (pooled-model 0.1) [1.0 0.0 0.0 2.0])))
  (is (thrown? clojure.lang.ExceptionInfo
               (d/logistic {:mean [0 0] :var [1 1] :discount 1.0 :by {:index 2 :groups 2 :slopes [5] :tau 0.1}}))))

;; --- stacking: weights that blend, not select -----------------------------

(defn- noisy-copy-periods
  "Periods of rows x = [1 z n1 n2 n3], the outcome from z alone; the n's are noise."
  [g periods n]
  (vec (for [_ (range periods)]
         (vec (for [_ (range n)]
                (let [z (m/normal! g)
                      x [1.0 z (m/normal! g) (m/normal! g) (m/normal! g)]]
                  {:x x :y (< (m/next-double! g) (m/expit (+ -2.0 z)))}))))))

(defn- good-and-noisy []
  {:good (d/logistic {:mean [0 0] :var [4 1] :discount 0.99 :inputs [0 1]})
   :noisy (d/logistic {:mean [0 0 0 0 0] :var [4 1 1 1 1] :discount 0.99 :inputs [0 1 2 3 4]})})

(deftest stacking-blends-where-posterior-weights-select
  (let [periods (noisy-copy-periods (m/rng-mixed 31) 120 30)
        st (run (d/averaged (good-and-noisy) {:forget 0.95 :weighting {:rule :stack :rate 0.02 :floor 0.05}}) periods)
        hist (fn [avg] (map d/weights (reductions d/step avg periods)))
        dma-path (hist (d/averaged (good-and-noisy) {:forget 0.95}))
        st-path (hist (d/averaged (good-and-noisy) {:forget 0.95 :weighting {:rule :stack :rate 0.02 :floor 0.05}}))]
    (is (> (apply max (map #(apply max (vals %)) dma-path)) 0.95) "posterior weights collapse onto one model")
    (is (every? #(>= (apply min (vals %)) (- 0.025 1e-12)) st-path) "stacking keeps every weight above floor / n")
    (is (< (apply max (map #(apply max (vals %)) st-path)) 0.95) "stacking does not")
    (testing "blending is not ignoring: the noisy copy keeps about half the weight"
      (is (< 0.4 (:good (d/weights st)) 0.6) (d/weights st)))))

(deftest stacking-still-finds-what-matters
  (let [g (m/rng-mixed 32)
        period (fn [beta] (vec (for [_ (range 40)]
                                 (let [x [1.0 (m/normal! g) (m/normal! g)]]
                                   {:x x :y (< (m/next-double! g) (m/expit (reduce + (map * beta x))))}))))
        models {:first (d/logistic {:mean [0 0] :var [4 4] :discount 0.98 :inputs [0 1]})
                :second (d/logistic {:mean [0 0] :var [4 4] :discount 0.98 :inputs [0 2]})}
        end (run (d/averaged models {:forget 1.0 :weighting {:rule :stack :rate 0.05 :floor 0.02}})
                 (repeatedly 40 #(period [-1.0 0.0 2.0])))]
    (is (> (:second (d/weights end)) 0.8) (d/weights end))))

(deftest weighting-needs-a-known-rule
  (is (thrown? clojure.lang.ExceptionInfo
               (d/averaged (good-and-noisy) {:forget 1.0 :weighting {:rule :nope}})))
  (is (thrown? clojure.lang.ExceptionInfo
               (d/averaged (good-and-noisy) {:forget 1.0 :weighting {:rule :stack :rate 0.1}}))
      "stacking needs a floor"))

(deftest tempered-posterior-weights
  (let [periods (noisy-copy-periods (m/rng-mixed 36) 120 30)
        path (fn [opts] (map d/weights (rest (reductions d/step (d/averaged (good-and-noisy) opts) periods))))
        plain (path {:forget 0.95})
        one (path {:forget 0.95 :weighting {:rule :temper :power 1.0}})
        tempered (path {:forget 0.95 :weighting {:rule :temper :power 0.1}})
        extreme (fn [ws] (/ (reduce + (map #(apply max (vals %)) ws)) (count ws)))]
    (is (= plain one) "a power of one is the posterior")
    (is (< (extreme tempered) (extreme plain)) "a smaller power selects less sharply")
    (is (> (/ (reduce + (map :good tempered)) (count tempered)) 0.5) "and still favours the model without noise"))
  (is (thrown? clojure.lang.ExceptionInfo (d/averaged (good-and-noisy) {:forget 1.0 :weighting {:rule :temper}}))))

(deftest held-coefficients-are-not-discounted
  (let [st (d/step (d/logistic {:mean [0 0 0] :var [4 1 1] :discount 0.5 :hold [2]}) [])
        ^doubles c (:c st)]
    (is (< (Math/abs (- (aget c 0) 8.0)) 1e-12) "the intercept discounted")
    (is (< (Math/abs (- (aget c 4) 2.0)) 1e-12))
    (is (< (Math/abs (- (aget c 8) 1.0)) 1e-12) "a held slope keeps its variance"))
  (testing "an empty input held settles nearer zero than one that drifts"
    (let [g (m/rng-mixed 41)
          periods (vec (repeatedly 200 #(vec (for [_ (range 20)]
                                               (let [x [1.0 (m/normal! g)]]
                                                 {:x x :y (< (m/next-double! g) 0.1)})))))
          path (fn [opts] (map #(second (d/means %)) (rest (reductions d/step (d/logistic opts) periods))))
          drift (path {:mean [0 0] :var [4 0.1] :discount 0.98})
          held (path {:mean [0 0] :var [4 0.1] :discount 0.98 :hold [1]})
          rms (fn [xs] (Math/sqrt (/ (reduce + (map #(* % %) (drop 100 xs))) 100.0)))]
      (is (< (rms held) (* 0.5 (rms drift))) [(rms held) (rms drift)])))
  (is (thrown? clojure.lang.ExceptionInfo (d/logistic {:mean [0] :var [1] :discount 1.0 :hold [3]}))))

;; --- a prior on the models' weights ------------------------------------------

(deftest weights-start-from-and-fade-toward-a-prior
  (let [same {:a (d/logistic {:mean [0 0] :var [4 1] :discount 0.99 :inputs [0 1]})
              :b (d/logistic {:mean [0 0] :var [4 1] :discount 0.99 :inputs [0 1]})
              :c (d/logistic {:mean [0 0] :var [4 1] :discount 0.99 :inputs [0 1]})}
        g (m/rng-mixed 51)
        periods (vec (repeatedly 60 #(vec (for [_ (range 20)] (let [x [1.0 (m/normal! g)]] {:x x :y (< (m/next-double! g) 0.2)})))))
        avg (d/averaged same {:forget 0.95 :prior {:a 0.6}})]
    (is (< (Math/abs (- 0.6 (:a (d/weights avg)))) 1e-12) "the named share")
    (is (< (Math/abs (- 0.2 (:b (d/weights avg)))) 1e-12) "the rest evenly")
    (testing "models that fit alike stay at the prior, not at equal weights"
      (let [end (run avg periods)]
        (is (< (Math/abs (- 0.6 (:a (d/weights end)))) 1e-4) (d/weights end))))
    (testing "without one, equal weights as before"
      (is (= (d/weights (run (d/averaged same {:forget 0.95}) periods))
             (d/weights (run (d/averaged same {:forget 0.95 :prior {}}) periods))))))
  (testing "evidence still moves the weights off the prior"
    (let [g (m/rng-mixed 52)
          period (fn [] (vec (for [_ (range 40)]
                               (let [x [1.0 (m/normal! g) (m/normal! g)]]
                                 {:x x :y (< (m/next-double! g) (m/expit (+ -1.0 (* 2.0 (nth x 2)))))}))))
          models {:first (d/logistic {:mean [0 0] :var [4 4] :discount 0.98 :inputs [0 1]})
                  :second (d/logistic {:mean [0 0] :var [4 4] :discount 0.98 :inputs [0 2]})}
          end (run (d/averaged models {:forget 0.99 :prior {:first 0.9}}) (repeatedly 40 period))]
      (is (> (:second (d/weights end)) 0.8) (d/weights end))))
  (is (thrown? clojure.lang.ExceptionInfo (d/averaged {:a (d/logistic {:mean [0] :var [1] :discount 1.0})} {:forget 1.0 :prior {:z 0.5}}))
      "a prior on a model that is not there")
  (is (thrown? clojure.lang.ExceptionInfo (d/averaged {:a (d/logistic {:mean [0] :var [1] :discount 1.0})
                                                       :b (d/logistic {:mean [0] :var [1] :discount 1.0})} {:forget 1.0 :prior {:a 1.5}}))
      "shares above one"))

(deftest a-held-or-pooled-model-s-mean-is-x-m
  ;; the discount widens the coefficients' covariance, never their means:
  ;; the predictive mean is x'm whether some coefficients are held or not
  (let [x [1.0 1.0]
        plain (d/logistic {:mean [1.0 1.0] :var [1 1] :discount 0.5})
        held (d/logistic {:mean [1.0 1.0] :var [1 1] :discount 0.5 :hold [1]})
        pooled (d/logistic {:mean [1.0 1.0 0.0] :var [1 1 1] :discount 0.5 :by {:index 2 :groups 2 :slopes [1] :tau 0.1}})]
    (is (< (Math/abs (- 2.0 (:f (d/predictive plain x)))) 1e-12))
    (is (< (Math/abs (- 2.0 (:f (d/predictive held x)))) 1e-12) (:f (d/predictive held x)))
    (is (< (Math/abs (- 2.0 (:f (d/predictive pooled [1.0 1.0 0])))) 1e-12) (:f (d/predictive pooled [1.0 1.0 0])))
    (testing "the variance: the discounted part widened, the held not"
      (is (< (Math/abs (- 3.0 (:q (d/predictive held x)))) 1e-12) (:q (d/predictive held x))))))

;; --- ranking signals by their proven increment (bnm-56c.2) -------------------

(defn- strong-and-weak-periods
  "Periods of rows x = [1 z1 z2 n1 n2 n3]: the outcome from a strong z1 and a
  weak z2; the n's are noise."
  [g periods n]
  (vec (for [_ (range periods)]
         (vec (for [_ (range n)]
                (let [z1 (m/normal! g) z2 (m/normal! g)
                      x [1.0 z1 z2 (m/normal! g) (m/normal! g) (m/normal! g)]]
                  {:x x :y (< (m/next-double! g) (m/expit (+ -2.0 (* 1.5 z1) (* 0.5 z2))))}))))))

(defn- ranked-models []
  {:base (d/logistic {:mean [0] :var [4] :discount 0.99 :inputs [0]})
   :strong (d/logistic {:mean [0 0] :var [4 1] :discount 0.99 :inputs [0 1]})
   :weak (d/logistic {:mean [0 0] :var [4 1] :discount 0.99 :inputs [0 2]})
   ;; the strong model with the noise beside it: nested in :strong
   :noisy (d/logistic {:mean [0 0 0 0 0] :var [4 1 1 1 1] :discount 0.99 :inputs [0 1 3 4 5]})})

(def ^:private evidence-rule {:rule :evidence :base :base :scale 0.005 :parents {:noisy :strong}})

(deftest evidence-orders-signals-and-keeps-the-weak-one-secondary
  (let [periods (strong-and-weak-periods (m/rng-mixed 61) 150 40)
        path (map d/weights (rest (reductions d/step (d/averaged (ranked-models) {:forget 0.98 :weighting evidence-rule}) periods)))
        end (last path)]
    (is (> (:strong end) (:weak end) 0.05) end)
    (is (< (:noisy end) 0.05) "the noise nested in the strong model adds nothing over it")
    (is (< (apply max (vals end)) 0.95) "no selection")
    (is (< (Math/abs (- 1.0 (reduce + (vals end)))) 1e-9))
    (testing "the order is stable once the evidence is in"
      (is (every? #(> (:strong %) (:weak %)) (drop 50 path))))))

(deftest evidence-starts-on-the-base
  (let [avg (d/averaged (ranked-models) {:forget 0.98 :weighting evidence-rule})]
    (is (= 1.0 (:base (d/weights avg))))
    (is (< (Math/abs (- (d/predict avg [1.0 2.0 0 0 0 0]) (d/predict (:base (:models avg)) [1.0 2.0 0 0 0 0]))) 1e-12)))
  (is (thrown? clojure.lang.ExceptionInfo (d/averaged (ranked-models) {:forget 1.0 :weighting {:rule :evidence :base :nope :scale 0.01}})))
  (is (thrown? clojure.lang.ExceptionInfo (d/averaged (ranked-models) {:forget 1.0 :weighting {:rule :evidence :base :base}}))))

(deftest a-weaker-increment-is-capped-by-a-stronger-one
  ;; base 0, strong +1, weak +3 (log odds); weights 0.5 and 0.3: uncapped the
  ;; weak model would move the forecast 0.9, more than the strong one's 0.5
  (let [mk (fn [b] (d/logistic {:mean [b] :var [1e-12] :discount 1.0 :inputs [0]}))
        avg (-> (d/averaged {:base (mk 0.0) :strong (mk 1.0) :weak (mk 3.0)} {:forget 1.0 :weighting {:rule :evidence :base :base :scale 0.01}})
                (assoc :w {:base 0.2 :strong 0.5 :weak 0.3} :gains {:strong 0.02 :weak 0.01}))
        uncapped (assoc-in avg [:weighting :cap] false)]
    (is (< (Math/abs (- 1.0 (:f (d/predictive avg [1.0])))) 1e-9) (d/predictive avg [1.0]))
    (is (< (Math/abs (- 1.4 (:f (d/predictive uncapped [1.0])))) 1e-9))))

(deftest group-shrinkage-shrinks-noise-and-lets-signal-escape
  (let [periods (strong-and-weak-periods (m/rng-mixed 62) 200 40)
        opts {:mean [0 0 0 0 0 0] :var [4 1 1 1 1 1] :discount 0.98}
        plain (run (d/logistic opts) periods)
        shrunk (run (d/logistic (assoc opts :group-shrink {:groups [[1] [2] [3 4 5]] :tau 0.5})) periods)
        rms (fn [s ix] (Math/sqrt (/ (reduce + (map #(let [v (nth (d/means s) %)] (* v v)) ix)) (count ix))))]
    (is (< (rms shrunk [3 4 5]) (* 0.6 (rms plain [3 4 5]))) [(rms shrunk [3 4 5]) (rms plain [3 4 5])])
    (is (< (Math/abs (- (nth (d/means shrunk) 1) (nth (d/means plain) 1))) (* 0.15 (nth (d/means plain) 1)))
        "the strong slope escapes")
    (is (> (nth (d/means shrunk) 2) 0.25) "the weak one is kept")
    (let [[s w n] (d/group-scales shrunk)]
      (is (> s w n) [s w n])))
  (testing "no groups, no change"
    (let [periods (strong-and-weak-periods (m/rng-mixed 63) 20 20)
          opts {:mean [0 0 0 0 0 0] :var [4 1 1 1 1 1] :discount 0.98}]
      (is (= (d/means (run (d/logistic opts) periods))
             (d/means (run (d/logistic (assoc opts :group-shrink {:groups [] :tau 0.5})) periods)))))))

(deftest synthesis-recalibrates-its-agents
  ;; one agent twice too sure (log odds -2 + 2z, the truth -1 + z), held
  ;; fixed: the synthesis learns a slope near one half on it
  (let [g (m/rng-mixed 64)
        periods (vec (for [_ (range 120)]
                       (vec (for [_ (range 40)]
                              (let [x [1.0 (m/normal! g)]]
                                {:x x :y (< (m/next-double! g) (m/expit (+ -1.0 (nth x 1))))})))))
        sure (d/logistic {:mean [-2.0 2.0] :var [1e-12 1e-12] :discount 1.0})
        base (d/logistic {:mean [-1.5] :var [1e-12] :discount 1.0 :inputs [0]})
        end (run (d/averaged {:sure sure :base base} {:forget 1.0 :weighting {:rule :synth :discount 0.99 :var 1.0}}) periods)
        ;; the synthesis' inputs in sorted order: :base, :sure
        [_ _ theta-sure] (d/means (:synth end))]
    (is (< 0.35 theta-sure 0.65) (d/means (:synth end)))
    (is (> (:sure (d/weights end)) 0.5))
    (is (< (Math/abs (- (d/predict end [1.0 1.0]) (m/expit 0.0))) 0.06) (d/predict end [1.0 1.0]))))

;; Issue-time evidence (bnm-56c loop audit, issue 2). Outcomes become known
;; `lag` periods after their forecast; B's input matters with a sign that
;; flips every era of `lag` periods. Scored when a row is learned, B has
;; meanwhile learned the earlier rows of the same era (hindsight) and looks
;; skilful; scored by the predictive it issued, it read the previous era's
;; sign and has no skill.
(defn- delayed-stream
  "Runs `avg` over a delayed stream: each period issues `n` rows, records
  each model's issue-time predictive (d/issued), and learns the rows
  issued `lag` periods before. The final average."
  [avg seed]
  (let [g (m/rng seed)
        lag 5 n 30 periods 120
        sign (vec (for [_ (range (inc (quot periods lag)))] (if (< (m/next-double! g) 0.5) -1.0 1.0)))]
    (loop [t 0 avg avg pending []]
      (if (= t periods)
        avg
        (let [{ready true waiting false} (group-by #(<= (:known-at %) t) pending)
              avg (d/step avg (mapv #(dissoc % :known-at) ready))
              made (vec (for [_ (range n)]
                          (let [z (if (< (m/next-double! g) 0.5) -1.0 1.0)
                                x [1.0 z]]
                            {:x x :y (< (m/next-double! g) (m/expit (* 3.0 (sign (quot t lag)) z)))
                             :issued (d/issued avg x 0.0) :known-at (+ t lag)})))]
          (recur (inc t) avg (into (vec waiting) made)))))))

(defn- two-models []
  {:base (d/logistic {:mean [0.0] :var [1.0] :discount 0.99 :inputs [0]})
   :b (d/logistic {:mean [0.0 0.0] :var [1.0 1.0] :discount 0.8})})

(deftest evidence-at-issue-does-not-reward-hindsight
  (let [learn (delayed-stream (d/averaged (two-models) {:forget 0.995}) 3)
        issue (delayed-stream (d/averaged (two-models) {:forget 0.995 :evidence-at :issue}) 3)]
    (is (> (:b (d/weights learn)) 0.9) "scored at learn time, hindsight selects B")
    (is (> (:base (d/weights issue)) 0.99) "scored as issued, the base keeps the weight")
    (testing "the other rules read the same evidence"
      (let [ev (delayed-stream (d/averaged (two-models) {:forget 0.995 :evidence-at :issue
                                                         :weighting {:rule :evidence :base :base :scale 0.005}}) 3)]
        (is (> (:base (d/weights ev)) 0.99) (d/weights ev)))
      (let [stk (delayed-stream (d/averaged (two-models) {:forget 0.995 :evidence-at :issue
                                                          :weighting {:rule :stack :rate 0.05 :floor 0.0}}) 3)
            stk-learn (delayed-stream (d/averaged (two-models) {:forget 0.995
                                                                :weighting {:rule :stack :rate 0.05 :floor 0.0}}) 3)]
        (is (< (:b (d/weights stk)) (:b (d/weights stk-learn))))))))

(deftest issued-evidence-is-the-issued-forecast-s-log-score
  (let [g (m/rng 4)
        avg (d/averaged (two-models) {:forget 0.99 :evidence-at :issue})
        avg (d/step avg (mapv #(assoc % :issued (d/issued avg (:x %) 0.0)) (rows g 40 [0.5 1.0 0.0])))
        x [1.0 0.7]
        issued (d/issued avg x 0.3)
        p (d/predict (get-in avg [:models :b]) x 0.3)]
    (is (= (set (keys issued)) #{:base :b}))
    (is (< (Math/abs (- (Math/log p) (:b (d/evidence avg [{:x x :y true :o 0.3 :issued issued}])))) 1e-12))
    (testing "issued now, the issue-time evidence is the learn-time one"
      (let [now (d/evidence avg [{:x x :y false :o 0.3 :issued issued}])
            learned (d/evidence (dissoc avg :evidence-at) [{:x x :y false :o 0.3}])]
        (is (every? #(< (Math/abs (- (now %) (learned %))) 1e-12) [:base :b]) [now learned])))
    (is (thrown? clojure.lang.ExceptionInfo (d/step avg [{:x x :y true}])) "a row without its issued predictive")
    (is (thrown? clojure.lang.ExceptionInfo (d/averaged (two-models) {:forget 0.99 :evidence-at :later})))))

;; --- shared effects: one period's rows sharing a common shock ---------------

(defn- shock-periods
  "T periods of n rows each: an intercept and a time-common input w_t (the
  same at every row of a period) that carries nothing, the outcomes moved
  by a period shock N(0, s^2) common to the period's rows. With `shared`,
  each row carries the period's shared effect [t 0 s^2]."
  [seed T n s shared]
  (let [g (m/rng seed)]
    (vec (for [t (range T)]
           (let [w (m/normal! g)
                 z (* s (m/normal! g))]
             (vec (for [_ (range n)]
                    (cond-> {:x [1.0 w] :y (< (m/next-double! g) (m/expit (+ -1.0 z)))}
                      shared (assoc :shared [t 0.0 (* s s)])))))))))

(defn- coef-var [state i] (aget ^doubles (:c state) (+ (* i (:k state)) i)))

(deftest a-shared-effect-counts-a-period-once-for-time-common-inputs
  (let [fresh (d/logistic {:mean [0.0 0.0] :var [4.0 1.0] :discount 1.0})
        after (fn [n shared] (run fresh (shock-periods 3 40 n 1.0 shared)))]
    (testing "rows read as independent: 500 rows a period teach a time-common slope ten times what 50 do"
      (is (> (/ (coef-var (after 50 false) 1) (coef-var (after 500 false) 1)) 5.0)))
    (testing "with the period's shock shared, a period teaches it at most one observation of the shock's variance, however many rows"
      (let [r (/ (coef-var (after 50 true) 1) (coef-var (after 500 true) 1))]
        (is (< r 1.5) r))
      (is (> (coef-var (after 500 true) 1) (/ 1.0 (+ 1.0 40.0 (* 3.0 40.0)))) "about 1/(1 + 40 w^2) after 40 periods"))
    (testing "and the spurious slope stays near zero"
      (is (< (Math/abs (nth (d/means (after 50 true)) 1)) 0.5) (d/means (after 50 true))))))

(deftest a-shared-effect-of-zero-variance-changes-nothing
  (let [fresh (d/logistic {:mean [0.0 0.0] :var [4.0 1.0] :discount 0.99})
        plain (shock-periods 4 10 20 1.0 false)
        zero (mapv (fn [rs] (mapv #(assoc % :shared [:p 0.0 0.0]) rs)) plain)
        a (run fresh plain) b (run fresh zero)]
    (is (every? #(< (Math/abs (double %)) 1e-12) (map - (d/means a) (d/means b))))
    (is (every? #(< (Math/abs (double %)) 1e-12) (map - (vec (:c a)) (vec (:c b)))))
    (is (< (Math/abs (- (d/log-score fresh (first plain)) (d/log-score fresh (first zero)))) 1e-9))))

(deftest a-shared-shock-is-one-observation-of-evidence
  ;; a period where every row has the event against a model at p = 0.27:
  ;; scored independently its evidence grows with the rows; with the shock
  ;; shared, the shock is paid for once, so the gap grows with n
  (let [mdl (d/logistic {:mean [-1.0] :var [1e-6] :discount 1.0})
        period (fn [n shared] (vec (for [_ (range n)] (cond-> {:x [1.0] :y true} shared (assoc :shared [:t 0.0 1.0])))))
        gap (fn [n] (- (d/log-score mdl (period n true)) (d/log-score mdl (period n false))))]
    (is (pos? (gap 50)))
    (is (> (gap 100) (* 1.8 (gap 50))) [(gap 50) (gap 100)])
    (testing "the shared score per row approaches the shock's, not the model's"
      (is (> (/ (d/log-score mdl (period 200 true)) 200) -0.1)))))

(deftest averaging-weighs-a-shared-shock-once
  ;; two models that differ only in a time-common slope; one shocked period
  ;; that the slope happens to fit. Read as 100 independent rows it locks
  ;; the weights; shared, it moves them about as one row of evidence would
  (let [a (d/logistic {:mean [-1.0 0.0] :var [1e-6 1e-6] :discount 1.0})
        b (d/logistic {:mean [-1.0 1.0] :var [1e-6 1e-6] :discount 1.0})
        rows (fn [shared] (vec (for [i (range 100)] (cond-> {:x [1.0 1.5] :y (< i 60)} shared (assoc :shared [:t 0.0 1.0])))))
        w (fn [shared] (:b (d/weights (d/step (d/averaged {:a a :b b} {:forget 1.0}) (rows shared)))))]
    (is (> (w false) 0.999))
    (is (< (w true) 0.9) (w true))
    (testing "and under :evidence-at :issue the same, from the issued predictives"
      (let [avg (d/averaged {:a a :b b} {:forget 1.0 :evidence-at :issue})
            rs (mapv #(assoc % :issued (d/issued avg (:x %) 0.0)) (rows true))]
        (is (< (:b (d/weights (d/step avg rs))) 0.9))))))

(deftest a-shared-effect-is-refused-for-linear-models
  (is (thrown? clojure.lang.ExceptionInfo
               (d/step (d/linear {:mean [0.0] :var [1.0] :discount 1.0}) [{:x [1.0] :y 1.0 :shared [:t 0.0 1.0]}])))
  (is (thrown? clojure.lang.ExceptionInfo
               (d/step (d/logistic {:mean [0.0] :var [1.0] :discount 1.0}) [{:x [1.0] :y true :shared [:t 0.0 -1.0]}]))))

;; --- shared effects of several dimensions: a common shock and a slope on each row's exposure ---

(defn- slope-periods
  "T periods of n rows: row i has a fixed exposure e_i; an intercept and a
  period input w_t times e_i that carries nothing; the outcomes moved by a
  period shock z_t + u_t e_i (sds s and su). `shared`: nil, :scalar (the
  rows share [t 0 s^2], the slope left out) or :slope (they share
  [t [0 0] [[s^2 0] [0 su^2]]], each loading [1 e_i])."
  [seed T n s su shared]
  (let [g (m/rng seed)
        es (vec (for [i (range n)] (- (* 2.0 (/ (double i) (max 1 (dec n)))) 1.0)))]
    (vec (for [t (range T)]
           (let [w (m/normal! g) z (* s (m/normal! g)) u (* su (m/normal! g))]
             (vec (for [e es]
                    (cond-> {:x [1.0 (* w e)] :y (< (m/next-double! g) (m/expit (+ -1.0 z (* u e))))}
                      (= :scalar shared) (assoc :shared [t 0.0 (* s s)])
                      (= :slope shared) (assoc :shared [t [0.0 0.0] [[(* s s) 0.0] [0.0 (* su su)]]] :load [1.0 e])))))))))

(deftest a-shared-slope-counts-a-period-once-for-an-exposure-input
  (let [fresh (d/logistic {:mean [0.0 0.0] :var [4.0 1.0] :discount 1.0})
        after (fn [n shared] (run fresh (slope-periods 5 40 n 1.0 1.0 shared)))]
    (testing "a scalar shock leaves an exposure-weighted input taught once per row"
      (is (> (/ (coef-var (after 200 :scalar) 1) (coef-var (after 2000 :scalar) 1)) 5.0)))
    (testing "with the slope shared too, at most one observation of its variance a period"
      ;; (exposures spread over [-1, 1]: under about 200 rows a period do not
      ;; yet reach the bound, about 1/(1 + 40 w^2))
      (let [r (/ (coef-var (after 200 :slope) 1) (coef-var (after 2000 :slope) 1))]
        (is (< r 1.5) r))
      (is (> (coef-var (after 2000 :slope) 1) (/ 1.0 (+ 1.0 40.0 (* 3.0 40.0))))))))

(deftest a-shared-state-of-one-dimension-is-the-scalar-one
  (let [fresh (d/logistic {:mean [0.0 0.0] :var [4.0 1.0] :discount 0.99})
        scalar (slope-periods 6 10 20 1.0 0.0 :scalar)
        vec1 (mapv (fn [rs] (mapv #(let [[id m v] (:shared %)] (assoc % :shared [id [m] [[v]]] :load [1.0])) rs)) scalar)
        a (run fresh scalar) b (run fresh vec1)]
    (is (every? #(< (Math/abs (double %)) 1e-9) (map - (d/means a) (d/means b))))
    (is (every? #(< (Math/abs (double %)) 1e-9) (map - (vec (:c a)) (vec (:c b)))))
    (is (< (Math/abs (- (d/log-score fresh (first scalar)) (d/log-score fresh (first vec1)))) 1e-9))
    (testing "and the averaging's evidence, as issued"
      (let [avg (d/averaged {:a fresh :b (d/logistic {:mean [-1.0 0.5] :var [1.0 1.0] :discount 0.99})} {:forget 1.0 :evidence-at :issue})
            iss (fn [rs] (mapv #(assoc % :issued (d/issued avg (:x %) 0.0)) rs))
            ea (d/evidence avg (iss (first scalar))) eb (d/evidence avg (iss (first vec1)))]
        (is (every? #(< (Math/abs (- (ea %) (eb %))) 1e-9) [:a :b]))))))

(deftest averaging-weighs-a-shared-slope-once
  ;; two models that differ in an exposure slope; one period whose shock
  ;; loads on the exposure in the slope's favour: read as 100 rows, or with
  ;; only the scalar shock shared, it locks the weights; with the slope
  ;; shared it moves them about as one row would
  (let [a (d/logistic {:mean [-1.0 0.0] :var [1e-6 1e-6] :discount 1.0})
        b (d/logistic {:mean [-1.0 1.0] :var [1e-6 1e-6] :discount 1.0})
        es (vec (for [i (range 100)] (- (* 0.02 i) 1.0)))
        rows (fn [shared] (vec (for [[i e] (map-indexed vector es)]
                                 (cond-> {:x [1.0 e] :y (< (mod (* 7 i) 100) (+ 27 (* 30 e)))}
                                   (= :scalar shared) (assoc :shared [:t 0.0 1.0])
                                   (= :slope shared) (assoc :shared [:t [0.0 0.0] [[1.0 0.0] [0.0 1.0]]] :load [1.0 e])))))
        w (fn [shared] (:b (d/weights (d/step (d/averaged {:a a :b b} {:forget 1.0}) (rows shared)))))]
    (is (> (w nil) 0.99))
    (is (> (w :scalar) 0.99))
    (is (< (w :slope) 0.9) (w :slope))))

(deftest a-shared-state-of-several-dimensions-is-refused-when-malformed
  (let [mdl (d/logistic {:mean [0.0] :var [1.0] :discount 1.0})]
    (doseq [bad [{:x [1.0] :y true :shared [:t [0.0 0.0] [[1.0 0.0] [0.0 1.0]]]}
                 {:x [1.0] :y true :shared [:t [0.0 0.0] [[1.0 0.0] [0.0 1.0]]] :load [1.0]}
                 {:x [1.0] :y true :shared [:t [0.0 0.0] [[1.0 0.0]]] :load [1.0 1.0]}
                 {:x [1.0] :y true :shared [:t [0.0 0.0] [[1.0 0.0] [0.0 -1.0]]] :load [1.0 1.0]}]]
      (is (thrown? clojure.lang.ExceptionInfo (d/step mdl [bad])) (pr-str bad)))))

;; --- :evidence-at :sequential: a period's rows each given the ones before it ---

(deftest sequential-evidence-of-one-row-is-the-log-score
  (let [mdl (d/logistic {:mean [-1.0 0.3] :var [0.5 2.0] :discount 0.99})
        r [{:x [1.0 1.5] :y true}]
        avg (d/averaged {:a mdl} {:forget 1.0 :evidence-at :sequential})]
    (is (< (Math/abs (- ((d/evidence avg r) :a) (d/log-score mdl r))) 1e-12))))

(deftest sequential-evidence-counts-a-coefficient-s-uncertainty-once
  ;; two models alike but for an input whose coefficient is 0 in both, one
  ;; of them unsure of it (variance 4). Rows scored independently each
  ;; integrate the coefficient on their own, so the unsure model's hedge
  ;; is paid or rewarded once per row; given the rows before it, the
  ;; coefficient is one quantity for the period
  (let [sure (d/logistic {:mean [-1.0 0.0] :var [1e-6 1e-6] :discount 1.0})
        unsure (d/logistic {:mean [-1.0 0.0] :var [1e-6 4.0] :discount 1.0})
        period (fn [n] (vec (for [i (range n)] {:x [1.0 (- (* 4.0 (/ (double i) (dec n))) 2.0)] :y (< (mod (* 7 i) 10) 5)})))
        gap (fn [ev n] (let [e (ev (period n))] (- (e :unsure) (e :sure))))
        indep (d/averaged {:sure sure :unsure unsure} {:forget 1.0})
        seqn (d/averaged {:sure sure :unsure unsure} {:forget 1.0 :evidence-at :sequential})
        ev-i #(d/evidence indep %) ev-s #(d/evidence seqn %)]
    (is (> (Math/abs (gap ev-i 200)) (* 3.0 (Math/abs (gap ev-i 50)))) "independent: the gap grows with the rows")
    (is (< (Math/abs (gap ev-s 200)) (+ 1.0 (Math/abs (gap ev-s 50)))) [(gap ev-s 50) (gap ev-s 200)])
    (testing "the weights it gives: one period of 200 rows locks them independently, not sequentially"
      (let [w (fn [avg] (:unsure (d/weights (d/step avg (period 200)))))]
        (is (or (> (w indep) 0.99) (< (w indep) 0.01)) (w indep))
        (is (< 0.05 (w seqn) 0.95) (w seqn))))))

(deftest sequential-evidence-with-a-shared-slope
  ;; rows sharing a shock and a slope on their exposure: given the rows
  ;; before it, each row's predictive carries what the period's rows have
  ;; taught the shared state and the coefficients jointly. One shared
  ;; dimension at variance 0 and mean 0 with a load of 1 is no effect
  (let [mdl (d/logistic {:mean [-1.0 0.0] :var [0.5 1.0] :discount 0.99})
        rows (first (slope-periods 7 1 60 1.0 1.0 :slope))
        avg (d/averaged {:a mdl} {:forget 1.0 :evidence-at :sequential})
        plain (mapv #(dissoc % :shared :load) rows)
        zero (mapv #(assoc % :shared [:t [0.0] [[0.0]]] :load [1.0]) plain)]
    (is (< (Math/abs (- ((d/evidence avg plain) :a) ((d/evidence avg zero) :a))) 1e-9))
    (is (not= ((d/evidence avg plain) :a) ((d/evidence avg rows) :a)))
    (testing "the averaging's step reads it"
      (let [b (d/logistic {:mean [-1.0 1.0] :var [0.5 1.0] :discount 0.99})
            two (d/averaged {:a mdl :b b} {:forget 1.0 :evidence-at :sequential})
            e (d/evidence two rows)
            w (d/weights (d/step two rows))]
        (is (< (Math/abs (- (/ (w :b) (w :a)) (Math/exp (- (e :b) (e :a))))) 1e-6))))))

(deftest evidence-at-takes-learn-issue-sequential-or-issue-joint
  (is (thrown? clojure.lang.ExceptionInfo (d/averaged {:a (d/logistic {:mean [0.0] :var [1.0] :discount 1.0})} {:forget 1.0 :evidence-at :joint}))))

;; --- :evidence-at :issue-joint: the joint predictive each model issued for an origin ---

(defn- chol-lower
  "The lower Cholesky factor of a small positive semi-definite matrix."
  [a]
  (let [p (count a)]
    (reduce (fn [l j]
              (let [dd (- (get-in a [j j]) (reduce + 0.0 (for [s (range j)] (let [x (get-in l [j s])] (* x x)))))
                    ljj (Math/sqrt (max dd 0.0))
                    l (assoc-in l [j j] ljj)]
                (reduce (fn [l i] (assoc-in l [i j] (if (pos? ljj)
                                                      (/ (- (get-in a [i j]) (reduce + 0.0 (for [s (range j)] (* (get-in l [i s]) (get-in l [j s]))))) ljj)
                                                      0.0)))
                        l (range (inc j) p))))
            (vec (repeat p (vec (repeat p 0.0)))) (range p))))

(defn- quadrature
  "The log marginal of outcomes ys at inputs xs under theta ~ N(mu, S), by
  brute force: theta = mu + L v on a grid of 41 points over [-6, 6] in
  each of v's dimensions (the trapezoid rule, spectrally accurate for a
  smooth integrand against a Gaussian)."
  [type mu S xs ys]
  (let [p (count mu) L (chol-lower S)
        npts 41 lim 6.0
        h (/ (* 2.0 lim) (dec npts))
        grid (vec (for [i (range npts)] (+ (- lim) (* i h))))
        wts (mapv #(* h (/ (Math/exp (* -0.5 % %)) (Math/sqrt (* 2 Math/PI)))) grid)
        idx (fn idx [dm] (if (zero? dm) [[]] (for [i (range npts) r (idx (dec dm))] (cons i r))))
        ll (fn [eta y] (case type
                         :logistic (if y (- eta (Math/log1p (Math/exp eta))) (- (Math/log1p (Math/exp eta))))
                         :poisson (- (* y eta) (Math/exp eta) (m/log-gamma (+ 1.0 y)))))]
    (Math/log
     (reduce + 0.0 (for [ii (idx p)]
                     (let [v (mapv grid ii)
                           th (mapv (fn [a] (+ (mu a) (reduce + 0.0 (for [b (range p)] (* (get-in L [a b]) (v b)))))) (range p))]
                       (* (reduce * (map wts ii))
                          (Math/exp (reduce + 0.0 (map (fn [x y] (ll (reduce + 0.0 (map * x th)) y)) xs ys))))))))))

(defn- with-cov
  "A model of `type` with means `mean` and covariance `c` (no discount)."
  [type mean c]
  (assoc ((if (= :poisson type) d/poisson d/logistic) {:mean mean :var (map #(get-in c [% %]) (range (count mean))) :discount 1.0})
         :c (double-array (apply concat c))))

(defn- scaled [s c] (mapv (fn [r] (mapv #(* s (double %)) r)) c))

(deftest the-joint-score-is-the-joint-marginal
  ;; two correlated coefficients and a period's shared shock, integrated
  ;; jointly over the rows: Laplace against brute-force quadrature over all
  ;; three. Laplace's error is a few hundredths of a nat for the whole
  ;; origin, whatever its rows, and grows slowly with the prior's width
  (let [c [[0.8 0.3] [0.3 1.2]]
        rows (fn [s n] (vec (for [i (range n)] {:x [1.0 (- (* 0.3 i) 1.5)] :y (odd? (* i 7 (inc i))) :shared [:t 0.2 (* s 0.5)]})))
        quad (fn [s rs] (quadrature :logistic [-0.5 0.4 0.2] [[(* s 0.8) (* s 0.3) 0.0] [(* s 0.3) (* s 1.2) 0.0] [0.0 0.0 (* s 0.5)]]
                                    (map #(conj (:x %) 1.0) rs) (map :y rs)))]
    (doseq [[s n tol] [[1.0 1 0.03] [1.0 4 0.03] [1.0 12 0.03] [15.0 12 0.1]]]
      (let [rs (rows s n)
            lap (d/joint-score (with-cov :logistic [-0.5 0.4] (scaled s c)) rs)
            q (quad s rs)]
        (is (< (Math/abs (- lap q)) tol) [s n lap q])))
    (testing "a Poisson model"
      (let [rs [{:x [1.0 0.5] :y 2 :shared [:t 0.0 0.2]} {:x [1.0 -1.0] :y 0 :shared [:t 0.0 0.2]} {:x [1.0 1.0] :y 3 :shared [:t 0.0 0.2]}]
            lap (d/joint-score (with-cov :poisson [0.2 0.3] [[0.3 0.1] [0.1 0.4]]) rs)
            q (quadrature :poisson [0.2 0.3 0.0] [[0.3 0.1 0.0] [0.1 0.4 0.0] [0.0 0.0 0.2]] (map #(conj (:x %) 1.0) rs) (map :y rs))]
        (is (< (Math/abs (- lap q)) 0.01) [lap q])))
    (testing "a shared state of two dimensions, each row loading [1 e]"
      (let [v [[0.4 0.1] [0.1 0.3]]
            rs (vec (for [[e y] [[-1.0 false] [-0.3 true] [0.4 false] [1.0 true] [0.8 true]]]
                      {:x [1.0] :y y :shared [:t [0.1 0.0] v] :load [1.0 e]}))
            lap (d/joint-score (with-cov :logistic [-0.3] [[0.6]]) rs)
            q (quadrature :logistic [-0.3 0.1 0.0] [[0.6 0.0 0.0] [0.0 0.4 0.1] [0.0 0.1 0.3]] (map #(into [1.0] (:load %)) rs) (map :y rs))]
        (is (< (Math/abs (- lap q)) 0.03) [lap q])))
    (testing "the discount applies as step's: the state's covariance over the discount"
      (let [mdl (assoc (with-cov :logistic [-0.5 0.4] c) :discount 0.5)
            rs (rows 1.0 4)]
        (is (< (Math/abs (- (d/joint-score mdl rs) (d/joint-score (with-cov :logistic [-0.5 0.4] (scaled 2.0 c)) rs))) 1e-9))))))

(deftest the-joint-score-with-nothing-uncertain-is-the-rows-likelihood
  (let [mdl (with-cov :logistic [-0.5 0.4] [[0.0 0.0] [0.0 0.0]])
        rs [{:x [1.0 0.5] :y true :o 0.1} {:x [1.0 -1.0] :y false} {:x [1.0 2.0] :y true :shared [:t 0.3 0.0]}]
        ll (+ (Math/log (m/expit (+ -0.5 0.2 0.1))) (Math/log (- 1.0 (m/expit -0.9))) (Math/log (m/expit (+ -0.5 0.8 0.3))))]
    (is (< (Math/abs (- (d/joint-score mdl rs) ll)) 1e-12))
    (testing "a row's issued moments set its mean"
      (is (< (Math/abs (- (d/joint-score mdl [{:x [1.0 0.5] :y true :f 1.0 :q 0.0}]) (Math/log (m/expit 1.0)))) 1e-12)))))

(deftest a-row-is-scored-at-the-probability-it-was-issued
  ;; a row issued over a mixture of offsets carries its mixture's log odds
  ;; with no spread (:q 0): its mean is moved so that, under the
  ;; coefficients' variance, it scores what was issued
  (let [mdl (with-cov :logistic [-0.5 0.4] [[0.8 0.3] [0.3 1.2]])
        f (m/logit 0.3)]
    (is (< (Math/abs (- (d/joint-score mdl [{:x [1.0 0.5] :y true :f f :q 0.0}]) (Math/log 0.3))) 0.03))
    (is (< (Math/abs (- (d/joint-score mdl [{:x [1.0 0.5] :y false :f f :q 0.0}]) (Math/log 0.7))) 0.03))))

(defn- issued-rows
  "`rows` as `avg` issues them now: each with every model's predictive and
  the models' state."
  [avg rows]
  (let [st (d/issue-state avg)]
    (mapv #(assoc % :issued (d/issued avg (:x %) (or (:o %) 0.0)) :issue-state st) rows)))

(deftest joint-evidence-counts-a-coefficient-s-uncertainty-once-per-origin
  ;; the models of sequential-evidence-counts-a-coefficient-s-uncertainty-once,
  ;; scored as issued: per row (:issue) the unsure model's hedge is paid
  ;; once per row and the gap grows with the rows; jointly, once per origin
  (let [sure (d/logistic {:mean [-1.0 0.0] :var [1e-6 1e-6] :discount 1.0})
        unsure (d/logistic {:mean [-1.0 0.0] :var [1e-6 4.0] :discount 1.0})
        period (fn [n] (vec (for [i (range n)] {:x [1.0 (- (* 4.0 (/ (double i) (dec n))) 2.0)] :y (< (mod (* 7 i) 10) 5)})))
        per-row (d/averaged {:sure sure :unsure unsure} {:forget 1.0 :evidence-at :issue})
        joint (d/averaged {:sure sure :unsure unsure} {:forget 1.0 :evidence-at :issue-joint})
        gap (fn [avg n] (let [e (d/evidence avg (issued-rows avg (period n)))] (- (e :unsure) (e :sure))))]
    (is (> (Math/abs (gap per-row 200)) (* 3.0 (Math/abs (gap per-row 50)))) "per row: the gap grows with the rows")
    (is (< (Math/abs (gap joint 200)) (+ 1.0 (Math/abs (gap joint 50)))) [(gap joint 50) (gap joint 200)])
    (is (< (Math/abs (gap joint 1000)) 5.0) (gap joint 1000))
    (testing "and a shock common to the rows, as issued, counted once: the evidence per origin stays bounded however many rows"
      (let [a (d/logistic {:mean [-1.0] :var [1e-6] :discount 1.0})
            b (d/logistic {:mean [0.5] :var [1e-6] :discount 1.0})
            avg (d/averaged {:a a :b b} {:forget 1.0 :evidence-at :issue-joint})
            per (d/averaged {:a a :b b} {:forget 1.0 :evidence-at :issue})
            rows (fn [n] (vec (for [i (range n)] {:x [1.0] :y (< (mod (* 7 i) 10) 6) :shared [:t 0.0 1.0]})))
            g (fn [avg n] (let [e (d/evidence avg (issued-rows avg (rows n)))] (- (e :b) (e :a))))]
        (is (< (Math/abs (- (g avg 2000) (g avg 200))) 0.5) [(g avg 200) (g avg 2000)])
        (testing "(with the coefficients known, the per-row rule integrates the shock jointly too: the two agree)"
          (is (< (Math/abs (- (g avg 2000) (g per 2000))) 0.01) [(g avg 2000) (g per 2000)]))))))

(deftest joint-evidence-is-per-origin
  (let [a (d/logistic {:mean [-1.0 0.0] :var [0.5 2.0] :discount 0.95})
        b (d/logistic {:mean [0.0 0.3] :var [1.0 1.0] :discount 0.95})
        avg (d/averaged {:a a :b b} {:forget 1.0 :evidence-at :issue-joint})
        g (m/rng 9)
        o1 (issued-rows avg (rows g 20 [-0.5 1.0 0.0]))
        avg2 (d/step avg o1)
        o2 (issued-rows avg2 (rows g 20 [-0.5 1.0 0.0]))
        e (fn [rs] (d/evidence avg2 rs))]
    (testing "two origins' rows learned in one period: each origin scored jointly under its own issue state, the scores summed"
      (doseq [k [:a :b]]
        (is (< (Math/abs (- ((e (into o1 o2)) k) (+ ((e o1) k) ((e o2) k)))) 1e-9))
        (is (< (Math/abs (- ((e o1) k) (d/joint-score (k (d/issue-state avg)) (map #(assoc % :f (get-in % [:issued k :f]) :q (get-in % [:issued k :q])) o1)))) 1e-12))))
    (testing "the weights follow the joint evidence"
      (let [ev (d/evidence avg o1) w (d/weights (d/step avg o1))]
        (is (< (Math/abs (- (Math/log (/ (w :b) (w :a))) (- (ev :b) (ev :a)))) 1e-4))))
    (is (thrown? clojure.lang.ExceptionInfo (d/evidence avg (mapv #(dissoc % :issue-state) o1))) "a row without its issue state")))

(defn- delayed-stream-joint
  "delayed-stream, each row also carrying the models' issue state."
  [avg seed]
  (let [g (m/rng seed)
        lag 5 n 30 periods 120
        sign (vec (for [_ (range (inc (quot periods lag)))] (if (< (m/next-double! g) 0.5) -1.0 1.0)))]
    (loop [t 0 avg avg pending []]
      (if (= t periods)
        avg
        (let [{ready true waiting false} (group-by #(<= (:known-at %) t) pending)
              avg (d/step avg (mapv #(dissoc % :known-at) ready))
              st (d/issue-state avg)
              made (vec (for [_ (range n)]
                          (let [z (if (< (m/next-double! g) 0.5) -1.0 1.0)
                                x [1.0 z]]
                            {:x x :y (< (m/next-double! g) (m/expit (* 3.0 (sign (quot t lag)) z)))
                             :issued (d/issued avg x 0.0) :issue-state st :known-at (+ t lag)})))]
          (recur (inc t) avg (into (vec waiting) made)))))))

(deftest joint-evidence-does-not-reward-hindsight
  ;; B fits only in hindsight (evidence-at-issue-does-not-reward-hindsight):
  ;; scored jointly as issued it gets no weight, and the sequential rule at
  ;; learn time, which also counts an origin once, still rewards it
  (let [joint (delayed-stream-joint (d/averaged (two-models) {:forget 0.995 :evidence-at :issue-joint}) 3)
        sq (delayed-stream-joint (d/averaged (two-models) {:forget 0.995 :evidence-at :sequential}) 3)]
    (is (> (:base (d/weights joint)) 0.99) (d/weights joint))
    (is (> (:b (d/weights sq)) 0.9) (d/weights sq))))
