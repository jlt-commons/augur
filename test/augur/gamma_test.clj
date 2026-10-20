(ns augur.gamma-test
  (:require [augur.gamma :as gamma]
            [clojure.edn]
            [clojure.test :refer [deftest is testing]]))

(defn- close? [expected actual tol]
  (< (Math/abs (- (double expected) (double actual))) tol))

(deftest a-rate-from-counts-and-exposure
  (let [p (gamma/centred 0.1 10.0)
        q (gamma/observe p {:count 30 :exposure 100.0})]
    (is (close? 0.1 (gamma/mean p) 1e-15) "centred on the rate")
    (is (close? (/ 31.0 110.0) (gamma/mean q) 1e-15))
    (is (= 100.0 (gamma/evidence q)))
    (let [[lo hi] (gamma/interval q 0.9)]
      (is (< lo (gamma/mean q) hi))
      (is (close? 0.05 (gamma/cdf q lo) 1e-9))
      (is (close? 0.95 (gamma/cdf q hi) 1e-9)))
    (testing "scipy: gamma.ppf(0.5, 31, scale=1/110)"
      (is (close? 0.27879373988779926 (gamma/quantile q 0.5) 1e-9)))
    (is (close? (- 1.0 (gamma/cdf q 0.3)) (gamma/prob-above q 0.3) 1e-15))))

(deftest evidence-fades-to-the-prior
  (let [p (gamma/prior 2.0 4.0)
        q (gamma/observe p {:count 50 :exposure 20.0})]
    (is (close? 0.5 (gamma/mean (gamma/decay q 0.5 200)) 1e-12))
    (is (= (gamma/observe (gamma/decay q 0.8 1) {:count 1})
           (gamma/observe q {:count 1 :forget 0.8})))
    (is (= (gamma/observe p {:count 2}) (gamma/observe nil {:count 2 :prior p})))
    (is (thrown? Exception (gamma/observe p {:exposure 1.0})))))

(deftest samples-and-state
  (let [q (gamma/observe (gamma/centred 0.2 5.0) {:count 40 :exposure 200.0})
        st [1.0 2.0 3.0 4.0 5.0 6.0]
        draws (loop [st st n 0 acc []] (if (< n 4000) (let [[x st] (gamma/sample st q)] (recur st (inc n) (conj acc x))) acc))]
    (is (close? (gamma/mean q) (/ (reduce + draws) (count draws)) 0.005))
    (is (= q (clojure.edn/read-string (pr-str q))))))
