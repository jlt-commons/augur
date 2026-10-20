(ns augur.layer-test
  (:require [augur.beta :as beta]
            [augur.gamma :as gamma]
            [augur.layer :as layer]
            [clojure.test :refer [deftest is testing]]))

(defn- close? [expected actual tol]
  (< (Math/abs (- (double expected) (double actual))) tol))

(def machine (beta/from-counts 80 100 (beta/prior 1.0 1.0)))
(def shipped (beta/from-counts 5 10 (beta/prior 1.0 1.0)))

(deftest a-fresh-child-is-its-parent
  (let [fresh (beta/prior 1.0 1.0)
        post (layer/posterior [fresh machine] {:strength 20.0})]
    (is (close? (beta/mean machine) (beta/mean post) 1e-12))
    (is (close? (beta/mean machine) (beta/mean (layer/posterior [nil machine] {:strength 20.0})) 1e-12)
        "nil is a layer with no evidence")
    (is (= 0.0 (:own (layer/explain [fresh machine] {:strength 20.0 :level 0.9}))))))

(deftest evidence-moves-it
  (let [child (beta/from-counts 2 20 (beta/prior 1.0 1.0))
        post (layer/posterior [child machine] {:strength 20.0})]
    (is (< (beta/mean post) (beta/mean machine)))
    (is (> (beta/mean post) (beta/mean child)))
    (is (close? (/ (+ (* 20.0 (beta/mean machine)) 2.0) 40.0) (beta/mean post) 1e-12)))
  (testing "a stronger link holds it nearer the parent"
    (let [child (beta/from-counts 2 20 (beta/prior 1.0 1.0))]
      (is (> (beta/mean (layer/posterior [child machine] {:strength 200.0}))
             (beta/mean (layer/posterior [child machine] {:strength 5.0})))))))

(deftest decay-returns-it-to-the-parent
  (let [child (beta/from-counts 2 20 (beta/prior 1.0 1.0))]
    (is (close? (beta/mean machine)
                (beta/mean (layer/posterior [(beta/decay child 0.5 100) machine] {:strength 20.0})) 1e-9))))

(deftest three-layers-fold-from-the-top
  (let [child (beta/from-counts 9 10 (beta/prior 1.0 1.0))
        two (layer/posterior [machine shipped] {:strength 10.0})
        three (layer/posterior [child machine shipped] {:strength [30.0 10.0]})]
    (is (close? (/ (+ (* 30.0 (beta/mean two)) 9.0) 40.0) (beta/mean three) 1e-12))
    (let [e (layer/explain [child machine shipped] {:strength [30.0 10.0] :level 0.9})]
      (is (= 10.0 (:evidence e)))
      (is (close? 0.25 (:own e) 1e-12))
      (is (= 3 (count (:by-layer e))))
      (is (close? (beta/mean child) (:mean (first (:by-layer e))) 1e-12)))))

(deftest rates-layer-too
  (let [parent (gamma/observe (gamma/centred 0.1 1.0) {:count 50 :exposure 1000.0})
        fresh (gamma/centred 1.0 1.0)
        child (gamma/observe fresh {:count 10 :exposure 20.0})]
    (is (close? (gamma/mean parent) (gamma/mean (layer/posterior [fresh parent] {:strength 50.0})) 1e-12))
    (is (> (gamma/mean (layer/posterior [child parent] {:strength 50.0})) (gamma/mean parent)))))
