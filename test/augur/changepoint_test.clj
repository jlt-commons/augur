(ns augur.changepoint-test
  (:require [augur.changepoint :as cp]
            [augur.math :as m]
            [clojure.edn]
            [clojure.test :refer [deftest is]]))

(defn- series [g n f] (vec (for [t (range n)] (+ (f t) (* 0.3 (m/normal! g))))))

(deftest a-shift-in-level-is-a-recent-break
  (let [g (m/rng 3)
        xs (series g 60 #(if (< % 40) 0.0 2.0))
        out (cp/run xs {:hazard (/ 1.0 20) :recent 2 :prior-var 1.0 :max-run 400})]
    (is (= 60 (count out)))
    (is (> (:recent (out 41)) 0.5) "just after the shift")
    (is (< (:recent (out 38)) 0.2) "in the long calm before it")
    (is (< (:run (out 41)) (:run (out 38))) "the expected run length restarts")))

(deftest a-steady-series-rarely-breaks
  (let [g (m/rng 4)
        out (cp/run (series g 80 (constantly 1.0)) {:hazard (/ 1.0 20) :recent 2 :prior-var 1.0 :max-run 400})]
    (is (< (/ (reduce + (map :recent (drop 10 out))) 70.0) 0.2))
    (is (> (:run (last out)) 20.0))))

(deftest stepping-matches-the-whole-series
  (let [g (m/rng 5)
        xs (series g 50 #(if (< % 30) 0.0 1.5))
        opts {:hazard 0.05 :recent 2 :prior-var 1.0 :max-run 20}
        stepped (loop [st (cp/init opts) [x & more] xs out []]
                  (if x (let [[st r] (cp/step st x)] (recur st more (conj out r))) out))
        state (reduce #(first (cp/step %1 %2)) (cp/init opts) xs)]
    (is (= (cp/run xs opts) stepped))
    (is (<= (count (:ps state)) 20) "bounded by :max-run")
    (is (= state (clojure.edn/read-string (pr-str state))) "the state is EDN")))

(deftest options-are-required
  (is (thrown? Exception (cp/init {:hazard 0.05 :recent 2}))))
