(ns augur.test-runner
  (:require [clojure.test :as t]
            [augur.math-test]
            [augur.special-test]
            [augur.ftest-test]
            [augur.calibrate-test]
            [augur.logit-test]
            [augur.pool-test]
            [augur.score-test]
            [augur.changepoint-test]
            [augur.trees-test]
            [augur.analog-test]
            [augur.beta-test]
            [augur.gamma-test]
            [augur.layer-test]
            [augur.ledger-test]
            [augur.hier-test]))

(def namespaces
  '[augur.math-test augur.special-test augur.ftest-test augur.calibrate-test
    augur.logit-test augur.pool-test augur.score-test augur.changepoint-test
    augur.trees-test augur.analog-test augur.beta-test augur.gamma-test
    augur.layer-test augur.ledger-test augur.hier-test])

(defn -main [& _]
  (let [{:keys [fail error]} (apply t/run-tests namespaces)]
    (System/exit (if (zero? (+ fail error)) 0 1))))
