(ns augur.ftest-test
  "Expected values from numpy and scipy on the same inputs."
  (:require [augur.ftest :as fs]
            [augur.math :as m]
            [clojure.test :refer [deftest is testing]]))

(defn- close? [a b] (< (Math/abs (- (double a) (double b))) (* 1e-9 (max 1.0 (Math/abs (double b))))))

(def y [0.005 0.0169 -0.006 -0.0306 -0.0132 -0.0347 0.0074 0.0586 -0.0147 -0.0198 0.0246 0.0193 0.0092 -0.0322 0.0038 0.0328 -0.0488 -0.0133 -0.071 -0.0466 -0.0687 -0.0044 -0.0457 0.0159 0.0113 -0.0025 -0.0957 -0.0165 0.0031 0.0095 -0.0562 -0.0141 -0.0341 -0.0274 0.0474 -0.0273 0.0037 0.0404 -0.0183 0.0005 0.0094 0.0076 -0.044 0.008 0.0594 -0.0569 0.0394 0.0098 -0.0207 0.085 0.0355 -0.043 0.008 0.0281 -0.0026 0.0323 0.0023 0.0317 0.0625 -0.022])
(def f [0.0101 -0.0063 0.0045 -0.0423 -0.0188 -0.0138 0.0317 0.0544 -0.0416 -0.0273 0.0293 -0.0515 -0.0086 -0.0101 0.0414 0.033 -0.022 -0.0125 -0.0263 0.0342 -0.031 -0.0079 -0.0006 0.0036 -0.0 -0.0317 -0.0266 -0.0158 0.0384 0.0249 -0.0151 0.0183 -0.0179 0.0258 0.0166 0.0118 -0.0351 0.025 -0.0536 -0.0584 -0.0038 -0.0222 -0.0058 0.0722 -0.0046 -0.0333 0.0205 0.0202 -0.009 0.0218 0.0342 0.0052 -0.0261 0.0086 0.0028 -0.0194 0.011 -0.0137 0.0504 0.0017])
(def b (vec (repeat 60 0.003)))

(deftest newey-west
  (is (= 3 (fs/nw-lags 60)))
  (is (close? 0.001394088062962963 (fs/long-run-variance y 3))))

(deftest return-forecasts
  (is (close? 0.09068412398372161 (fs/oos-r2 y f b)))
  (let [{:keys [stat p]} (fs/clark-west y f b 3)]
    (is (close? 4.834833064450931 stat))
    (is (close? 6.662870761786621e-07 p)))
  (let [{:keys [stat p]} (fs/diebold-mariano (map #(let [e (- %1 %2)] (* e e)) y f)
                                             (map #(let [e (- %1 %2)] (* e e)) y b) 3)]
    (is (close? -0.6007158396505118 stat))
    (is (close? 0.548029267211511 p)))
  (let [{:keys [hit stat p]} (fs/pesaran-timmermann y f)]
    (is (close? (/ 2.0 3.0) hit))
    (is (close? 2.6445341229533574 stat))
    (is (close? 0.0040901729104615425 p)))
  (testing "forecasts of one sign carry no timing information"
    (let [{:keys [hit p]} (fs/pesaran-timmermann [1.0 -1.0 1.0] [0.1 0.2 0.3])]
      (is (close? (/ 2.0 3.0) hit))
      (is (= 1.0 p))))
  (testing "identical forecasts give no difference to test"
    (let [{:keys [stat p]} (fs/clark-west y b b 3)]
      (is (= 0.0 stat))
      (is (= 0.5 p)))
    (is (= 1.0 (:p (fs/diebold-mariano [1.0 2.0 3.0] [1.0 2.0 3.0] 2))))))

(deftest distributions
  (is (close? 0.024097654305104653 (fs/crps-normal 0.01 0.04 -0.03)))
  (is (< (Math/abs (- 0.024153222509929205 (fs/crps-t 0.01 0.04 5.0 -0.03))) 1e-8))
  (is (close? 1.7032915654316123 (fs/log-t 0.01 0.04 5.0 -0.03)))
  (is (close? 0.0 (fs/qlike 2.0 2.0)))
  (is (> (fs/qlike 1.0 2.0) 0.0)))

(deftest coverage
  (let [{:keys [stat p]} (fs/kupiec 9 60 0.1)]
    (is (close? 1.4682137342721902 stat))
    (is (close? 0.2256277377816313 p)))
  (let [{:keys [stat p]} (fs/christoffersen [0 0 1 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0])]
    (is (close? 0.5310044432812475 stat))
    (is (close? 0.4661848765790597 p)))
  (testing "Berkowitz: uniform PITs pass, a biased and too-narrow forecast fails"
    (let [g (m/rng 5)
          u (repeatedly 400 #(m/next-double! g))
          z (repeatedly 400 #(+ 0.4 (* 1.5 (m/normal! g))))]
      (is (> (:p (fs/berkowitz u)) 0.05))
      (is (< (:p (fs/berkowitz (map m/normal-cdf z))) 0.001)))))

(deftest superior-predictive-ability
  (let [g (m/rng 11)
        noise (fn [] (vec (repeatedly 500 #(m/normal! g))))
        ;; columns: each model's per-period gain over the benchmark
        null [(noise) (noise) (noise)]
        edge [(noise) (mapv #(+ 0.25 %) (noise)) (noise)]]
    (is (> (:p (fs/spa null {:block 10 :reps 999 :seed 1})) 0.1))
    (is (< (:p (fs/spa edge {:block 10 :reps 999 :seed 1})) 0.01))))

(deftest series-that-never-move
  (testing "a model no different from the benchmark adds nothing to SPA"
    (let [g (m/rng 3)
          null (vec (repeatedly 200 #(m/normal! g)))]
      (is (= 1.0 (:p (fs/spa [(repeat 200 0.0)] {:block 10 :reps 99 :seed 1}))))
      (is (> (:p (fs/spa [(repeat 200 0.0) null] {:block 10 :reps 999 :seed 1})) 0.1)))))
