(ns augur.beta-test
  "Reference values are scipy's (betainc, betaincinv, 200-point
  Gauss-Legendre for P(A > B)), as samizdat's stats tests pin them."
  (:require [augur.beta :as beta]
            [clojure.edn]
            [clojure.test :refer [deftest is testing]]))

(defn- close? [expected actual tol]
  (and (number? actual) (< (Math/abs (- (double expected) (double actual))) tol)))

(defn- post [k n] (beta/from-counts k n (beta/jeffreys)))

(deftest a-credible-interval-is-honest-at-small-n
  (let [[lo hi] (beta/interval (post 2 3) 0.95)]
    (is (close? 0.1767 lo 1e-3))
    (is (close? 0.9613 hi 1e-3)))
  (let [[lo hi] (beta/interval (post 14 20) 0.95)]
    (is (close? 0.4828 lo 1e-3))
    (is (close? 0.8639 hi 1e-3)))
  (testing "zero successes is not zero: Jeffreys' prior keeps the upper end open"
    (let [[lo hi] (beta/interval (post 0 3) 0.95)]
      (is (< lo 0.001))
      (is (close? 0.5356 hi 1e-3)))))

(deftest the-chance-a-rate-lies-in-a-zone
  (is (close? 0.1059 (beta/prob-between (post 20 20) 0.0625 0.9375) 1e-3))
  (is (close? 0.1059 (beta/prob-between (post 0 20) 0.0625 0.9375) 1e-3))
  (is (close? 0.9996 (beta/prob-between (post 14 20) 0.0625 0.9375) 1e-3))
  (is (close? 0.9480 (beta/prob-between (post 2 3) 0.0625 0.9375) 1e-3))
  (is (close? (- 1.0 (beta/cdf (post 2 3) 0.4)) (beta/prob-above (post 2 3) 0.4) 1e-15)))

(deftest pass-at-k-and-pass-power-k
  (is (close? 0.7 (beta/pass-at-k 20 14 1) 1e-12))
  (is (close? (- 1.0 (/ 6.0 15504.0)) (beta/pass-at-k 20 14 5) 1e-12) "1 - C(6,5)/C(20,5)")
  (is (= 1.0 (beta/pass-at-k 20 19 5)) "fewer failures than k: certain")
  (is (= 0.0 (beta/pass-at-k 20 0 5)))
  (is (close? (Math/pow 0.7 5) (beta/pass-power-k 20 14 5) 1e-12))
  (is (= 0.0 (beta/pass-power-k 0 0 3)) "no runs is no reliability, not an error"))

(deftest the-probability-one-arm-beats-another
  (is (close? 0.9724 (beta/prob-greater (post 14 20) (post 8 20)) 2e-3))
  (is (close? 0.5 (beta/prob-greater (post 10 20) (post 10 20)) 1e-6) "the same record is a coin flip")
  (is (< (beta/prob-greater (post 8 20) (post 14 20)) 0.03)))

(deftest evidence-fades-to-the-prior
  (let [p (beta/prior 2.0 3.0)
        q (beta/from-counts 30 40 p)]
    (is (= 40.0 (beta/evidence q)))
    (is (close? 2.0 (:a (beta/decay q 0.5 200)) 1e-12) "decay returns it to the prior")
    (is (close? 0.4 (beta/mean (beta/decay q 0.5 200)) 1e-12))
    (is (close? (:a (beta/decay q 0.9 5)) (:a (reduce (fn [x _] (beta/decay x 0.9 1)) q (range 5))) 1e-12)
        "five steps at once are five single steps")
    (is (close? 20.0 (beta/evidence (beta/decay q 0.5 1)) 1e-12))))

(deftest observing-outcomes
  (let [p (beta/jeffreys)]
    (is (= (beta/from-counts 1 1 p) (beta/observe p true {})))
    (is (= (beta/observe p true {}) (beta/observe nil true {:prior p})) "nil is the prior")
    (is (close? 2.5 (:a (beta/observe p 1.0 {:weight 2.0})) 1e-12))
    (is (close? 0.75 (:a (beta/observe p 0.25 {})) 1e-12) "a partial success")
    (is (= (beta/observe (beta/decay (post 3 4) 0.9 1) false {})
           (beta/observe (post 3 4) false {:forget 0.9}))
        "forgetting comes before the new outcome")
    (is (thrown? Exception (beta/observe p 1.5 {})))
    (is (thrown? Exception (beta/observe nil true {})))))

(deftest thompson-samples-are-replayable
  (let [st [12345.0 12345.0 12345.0 12345.0 12345.0 12345.0]
        p (post 14 20)
        [x st'] (beta/sample st p)
        draws (loop [st st n 0 acc []] (if (< n 4000) (let [[x st] (beta/sample st p)] (recur st (inc n) (conj acc x))) acc))]
    (is (= x (first (beta/sample st p))) "the same state, the same draw")
    (is (not= st st'))
    (is (close? (beta/mean p) (/ (reduce + draws) (count draws)) 0.01))))

(deftest state-is-edn
  (let [p (beta/observe (post 3 7) true {:forget 0.95})]
    (is (= p (clojure.edn/read-string (pr-str p))))
    (is (= {:mean (beta/mean p) :interval (beta/interval p 0.9) :evidence (beta/evidence p)}
           (beta/explain p 0.9)))))
