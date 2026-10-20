(ns augur.special-test
  "Expected values from scipy (betainc, betaincinv, chi2.sf, t.cdf, norm)."
  (:require [augur.special :as sp]
            [clojure.test :refer [deftest is testing]]))

(defn- close? [a b tol] (< (Math/abs (- (double a) (double b))) tol))

(deftest distributions
  (is (close? 0.07363827012030295 (sp/chi2-sf 3.2 1) 1e-12))
  (is (close? 0.2018965179946554 (sp/chi2-sf 3.2 2) 1e-12))
  (is (close? 0.057558451972636406 (sp/chi2-sf 7.5 3) 1e-12))
  (is (close? 0.1251503170853385 (sp/t-cdf -1.3 5.0) 1e-12))
  (is (close? 0.9432370873896009 (sp/t-cdf 2.1 3.5) 1e-12)))

(deftest the-normal-to-double-precision
  (is (close? 0.5 (sp/phi 0.0) 1e-16))
  (is (close? 0.9750021048517795 (sp/phi 1.96) 1e-15))
  (is (close? 2.8665157187919344e-07 (sp/phi-sf 5.0) 1e-18) "no cancellation in the upper tail")
  (doseq [p [1e-10 0.001 0.3 0.5 0.975 0.999999]]
    (is (close? p (sp/phi (sp/phi-inv p)) (* 1e-13 (max p 1e-3))) p)))

(deftest the-regularized-incomplete-beta
  (is (close? 0.5248 (sp/beta-inc 2 3 0.4) 1e-12) "a closed form: I_0.4(2,3)")
  (is (close? 0.5 (sp/beta-inc 0.5 0.5 0.5) 1e-12) "symmetric at the middle")
  (is (close? 0.4 (sp/beta-inc 1 1 0.4) 1e-12) "Beta(1,1) is uniform")
  (is (close? 0.10591207961185312 (sp/beta-inc 20.5 0.5 0.9375) 1e-9)
      "a half-integer with mass piled at 1")
  (is (= 0.0 (sp/beta-inc 2 3 0.0)))
  (is (= 1.0 (sp/beta-inc 2 3 1.0))))

(deftest the-inverse-recovers-the-quantile
  (is (close? 0.17673609713125732 (sp/beta-inc-inv 2.5 1.5 0.025) 1e-9))
  (is (close? 0.9612523822148348 (sp/beta-inc-inv 2.5 1.5 0.975) 1e-9))
  (is (close? 0.4 (sp/beta-inc-inv 2 3 0.5248) 1e-9)))

(deftest gauss-legendre-integrates-polynomials-exactly
  (let [[xs ws] (sp/gauss-legendre 5)
        int (fn [f] (reduce + (map #(* %2 (f %1)) xs ws)))]
    (is (close? 2.0 (int (constantly 1.0)) 1e-14))
    (is (close? (/ 2.0 3.0) (int #(* % %)) 1e-14))
    (is (close? (/ 2.0 9.0) (int #(Math/pow % 8)) 1e-14) "degree 2n - 1 = 9 and below")))
