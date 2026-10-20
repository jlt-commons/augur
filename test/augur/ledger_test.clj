(ns augur.ledger-test
  (:require [augur.calibrate :as cal]
            [augur.ledger :as ledger]
            [clojure.edn]
            [clojure.test :refer [deftest is testing]]))

(defn- close? [expected actual tol]
  (< (Math/abs (- (double expected) (double actual))) tol))

(def opts {:max-open 100 :max-rows 50 :max-keys 10 :forget 1.0 :calibration (assoc cal/defaults :forget 1.0)})

(def k [:gate :no-edits "glm"])

(defn- run
  "A ledger after forecasts [p outcome] under key k, each registered then resolved."
  [L pairs]
  (reduce (fn [L [i [p y]]]
            (-> L
                (ledger/register {:id [:f i] :key k :p p :at i :bench {:prior 0.5}})
                (ledger/resolve [:f i] y {:at (inc i)})))
          L (map-indexed vector pairs)))

(deftest forecasts-are-scored-against-their-benchmarks
  (let [L (run (ledger/empty opts) [[0.9 true] [0.8 true] [0.2 false] [0.3 true]])
        r (ledger/record L k)]
    (is (= 4.0 (:n r)))
    (is (close? (/ (+ 0.01 0.04 0.04 0.49) 4.0) (:brier r) 1e-12))
    (is (close? (/ (+ (Math/log 0.9) (Math/log 0.8) (Math/log 0.8) (Math/log 0.3)) 4.0) (:log r) 1e-12))
    (is (close? (- 1.0 (/ 0.58 1.0)) (get-in r [:skill :prior]) 1e-12) "Brier skill over always 0.5")
    (is (= 0 (:open r)))
    (is (= 4 (count (ledger/rows L k))))
    (is (= [:f 0] (:id (first (ledger/rows L k)))))))

(deftest recalibration-learns-the-keys-record
  (let [L (run (ledger/empty opts) (for [i (range 200)] [0.9 (even? i)]))]
    (is (< 0.45 (ledger/calibrated L k 0.9) 0.55) "overconfident forecasts pulled in")
    (is (= 0.3 (ledger/calibrated L [:other-key] 0.3)) "a key with no record passes through")
    (is (some (fn [[_ rate n]] (and (close? 0.5 rate 1e-9) (= 200.0 n))) (:reliability (ledger/record L k))))))

(deftest ids-and-unknowns
  (let [L (ledger/register (ledger/empty opts) {:id 1 :key k :p 0.5 :at 0})]
    (is (thrown? Exception (ledger/register L {:id 1 :key k :p 0.4 :at 1})) "an open id is not overwritten")
    (is (= 1 (:unknown (ledger/stats (ledger/resolve L 99 true {:at 1})))) "an unknown id is counted")
    (is (= 1 (:open (ledger/stats (ledger/resolve L 99 true {:at 1})))))
    (is (thrown? Exception (ledger/register nil {:id 2 :key k :p 0.5 :at 0})))))

(deftest expired-forecasts-are-never-scored
  (let [L (-> (ledger/empty opts)
              (ledger/register {:id 1 :key k :p 0.9 :at 0 :run "a"})
              (ledger/register {:id 2 :key k :p 0.9 :at 0 :run "b"})
              (ledger/expire #(= 1 (:id %)) {:at 5}))
        r (ledger/record L k)]
    (is (= 1 (:expired r)))
    (is (= 1 (:open r)))
    (is (= 0.0 (:n r)))
    (is (nil? (:brier r)))))

(deftest state-is-bounded
  (testing "open forecasts beyond :max-open expire oldest first"
    (let [L (reduce #(ledger/register %1 {:id %2 :key k :p 0.5 :at %2}) (ledger/empty (assoc opts :max-open 3)) (range 5))]
      (is (= #{2 3 4} (set (keys (:open L)))))
      (is (= 2 (:expired (ledger/record L k))))))
  (testing "rows beyond :max-rows are dropped, their scores kept"
    (let [L (run (ledger/empty (assoc opts :max-rows 3)) (repeat 10 [0.5 true]))]
      (is (= 3 (count (ledger/rows L k))))
      (is (= 10.0 (:n (ledger/record L k))))))
  (testing "keys beyond :max-keys fold into ::other"
    (let [L (reduce (fn [L i] (-> L (ledger/register {:id i :key [:k i] :p 0.5 :at i}) (ledger/resolve i true {:at i})))
                    (ledger/empty (assoc opts :max-keys 3)) (range 5))]
      (is (= 4 (count (ledger/keys L))) "three keys and ::other")
      (is (= 2.0 (:n (ledger/record L ::ledger/other))))
      (is (= #{[:k 2] [:k 3] [:k 4]} (set (remove #{::ledger/other} (ledger/keys L)))) "the least recent go"))))

(deftest forgetting-weighs-recent-forecasts-more
  (let [L (run (ledger/empty (assoc opts :forget 0.5)) [[0.9 false] [0.9 true]])]
    (is (close? 1.5 (:n (ledger/record L k)) 1e-12))
    (is (close? (/ (+ (* 0.5 0.81) 0.01) 1.5) (:brier (ledger/record L k)) 1e-12))))

(deftest state-is-edn
  (let [L (-> (run (ledger/empty opts) [[0.9 true] [0.2 false]])
              (ledger/register {:id "open" :key [:judge :pass] :p 0.7 :at 3 :x {:turns 12}}))]
    (is (= L (clojure.edn/read-string (pr-str L))))
    (is (= {:n 0.0 :brier nil :log nil :skill {} :reliability [] :open 0 :expired 0} (ledger/record nil k))
        "nil reads as an empty ledger")))
