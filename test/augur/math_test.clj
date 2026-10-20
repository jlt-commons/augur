(ns augur.math-test
  (:require [augur.math :as m]
            [clojure.edn]
            [clojure.test :refer [deftest is]]))

(defn- close? [a b tol] (< (Math/abs (- a b)) tol))

(deftest log-gamma-matches-factorials
  (doseq [n (range 1 20)]
    (is (close? (m/log-gamma (double n))
                (Math/log (double (reduce * 1N (range 1 n)))) 1e-9)))
  (is (close? (m/log-gamma 0.5) (Math/log (Math/sqrt Math/PI)) 1e-12)))

(deftest samplers-have-the-right-moments
  (let [g (m/rng 123) n 20000
        mean (fn [xs] (/ (reduce + xs) (double n)))]
    (is (close? (mean (repeatedly n #(m/gamma! g 2.5 2.0))) 5.0 0.15))
    (is (close? (mean (repeatedly n #(m/gamma! g 0.4 1.0))) 0.4 0.03))
    (is (close? (mean (repeatedly n #(double (m/poisson! g 4.0)))) 4.0 0.1))
    (is (close? (mean (repeatedly n #(m/beta! g 2.0 6.0))) 0.25 0.01))))

(deftest generator-is-reproducible
  (is (= (repeatedly 5 #(m/next-double! (m/rng 9)))
         (repeatedly 5 #(m/next-double! (m/rng 9))))))

(deftest normal-quantiles
  (is (< (Math/abs (m/normal-quantile 0.5)) 1e-9))
  (is (< (Math/abs (- 1.2815515655446004 (m/normal-quantile 0.9))) 1e-6))
  (is (< (Math/abs (+ 2.3263478740408408 (m/normal-quantile 0.01))) 1e-6)))

(deftest normal-cdf-inverts-the-quantile
  (doseq [p [0.001 0.05 0.3 0.5 0.8 0.99]]
    (is (< (Math/abs (- p (m/normal-cdf (m/normal-quantile p)))) 1e-6) p)))

(deftest nelder-mead-finds-the-rosenbrock-valley-floor
  (let [{:keys [x f]} (m/nelder-mead (fn [[a b]] (+ (Math/pow (- 1.0 a) 2) (* 100.0 (Math/pow (- b (* a a)) 2))))
                                     [-1.2 1.0] {:tol 1e-14 :max-iter 5000})]
    (is (< f 1e-10))
    (is (< (Math/abs (- 1.0 (x 0))) 1e-4))
    (is (< (Math/abs (- 1.0 (x 1))) 1e-4))))

(deftest a-stored-generator-replays
  (let [st (m/rng-state 9)
        [x st'] (m/with-rng st m/next-double!)
        [y _] (m/with-rng st' m/next-double!)]
    (is (= st (clojure.edn/read-string (pr-str st))) "the state is EDN")
    (is (= x (m/next-double! (m/rng 9))) "the first draw is the array generator's")
    (is (= [x y] (let [g (m/rng 9)] [(m/next-double! g) (m/next-double! g)])) "and so is the next")
    (is (= x (first (m/with-rng st m/next-double!))) "the same state, the same draw")))
