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

;; --- seeds: rng fills its state linearly, seed-of mixes -------------------------------

(defn- corr [xs ys]
  (let [n (double (count xs))
        mx (/ (reduce + xs) n) my (/ (reduce + ys) n)
        sxy (reduce + (map #(* (- %1 mx) (- %2 my)) xs ys))
        sxx (reduce + (map #(* (- % mx) (- % mx)) xs))
        syy (reduce + (map #(* (- % my) (- % my)) ys))]
    (/ sxy (Math/sqrt (* sxx syy)))))

(defn- draws
  "The draws at indices `ds` of a generator: one vector per stream."
  [g ds]
  (let [top (inc (apply max ds))
        xs (vec (repeatedly top #(m/next-double! g)))]
    (mapv xs ds)))

(defn- neighbour-corrs
  "For streams (make s) over seeds s = 1, 1+step, ..., n of them: the
  correlation at each draw index in `ds` between each stream and the next."
  [make step n ds]
  (let [rows (mapv #(draws (make (inc (* % step))) ds) (range (inc n)))]
    (mapv (fn [j] (corr (mapv #(nth % j) (pop rows)) (mapv #(nth % j) (rest rows))))
          (range (count ds)))))

(def ^:private ds [0 1 2 3 44 1000])

(deftest rng-output-is-unchanged
  (is (= [2.0 12345.0 2.0 7933.0 12345.0 7933.0] (vec (m/rng 1))))
  (doseq [[s xs] {1 [0.5909521507374121 0.7664460862103818 0.23130270864604122]
                  42 [0.308585652426308 0.38708519994135054 0.21204070143971265]
                  7919 [0.8643826751018867 0.45536875834611756 0.17058563336772187]
                  123456789 [0.3131982866081511 0.5076614864155625 0.31800836560915696]}]
    (let [g (m/rng s)] (is (= xs (vec (repeatedly 3 #(m/next-double! g)))) s))))

(deftest rng-streams-from-evenly-spaced-seeds-are-correlated
  ;; the defect seed-of exists for: kept as a record of why
  (is (some #(> (Math/abs %) 0.3) (neighbour-corrs m/rng 1 2000 ds)))
  (is (some #(> (Math/abs %) 0.3) (neighbour-corrs m/rng 7919 2000 ds))))

(deftest seed-of-is-splitmix64
  ;; the same values bnm.sfc-analysis/seed-of gave, so its runs do not move
  (is (= -2152535657050944081 (m/seed-of 0)) "SplitMix64's first output from state 0")
  (is (= [-7995527694508729151 -4838594755968170389 -5182195735961227806 -8278551758858648694 2173552090174352397]
         [(m/seed-of 1) (m/seed-of 1 2) (m/seed-of 1 2 1) (m/seed-of 7919 0) (m/seed-of -5 3)]))
  (is (not= (m/seed-of 1 2) (m/seed-of 2 1)) "the keys are ordered"))

(deftest rng-mixed-is-rng-of-seed-of
  (is (= (vec (m/rng (m/seed-of 3 4))) (vec (m/rng-mixed 3 4))))
  (is (= (vec (m/rng (m/seed-of 3))) (vec (m/rng-mixed 3)))))

(deftest mixed-streams-from-evenly-spaced-seeds-are-uncorrelated
  (doseq [step [1 2 7919 1000003]
          r (neighbour-corrs m/rng-mixed step 4000 ds)]
    (is (< (Math/abs r) 0.05) [step r])))

(deftest mixed-streams-by-index-are-uncorrelated
  ;; one seed, stream index i = 0, 1, 2, ...: neighbouring streams, and the
  ;; same stream index under neighbouring seeds
  (doseq [r (neighbour-corrs #(m/rng-mixed 17 %) 1 4000 ds)]
    (is (< (Math/abs r) 0.05) [:index r]))
  (doseq [r (neighbour-corrs #(m/rng-mixed % 5) 1 4000 ds)]
    (is (< (Math/abs r) 0.05) [:seed r])))
