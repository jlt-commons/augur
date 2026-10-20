(ns augur.ftest
  "Tests of forecasts: out-of-sample R² (Campbell & Thompson 2008) and
  Clark & West (2007) for nested point forecasts, Diebold & Mariano for
  losses, Pesaran & Timmermann (1992) for signs, QLIKE (Patton 2011) for
  variance, CRPS and log scores, interval coverage (Kupiec 1995,
  Christoffersen 1998), PIT calibration (Berkowitz 2001) and Hansen's
  (2005) test of superior predictive ability. Long-run variances are
  Newey-West with Bartlett weights. Degenerate inputs (forecasts of one
  sign, losses that never differ) give an answer, not a division by zero."
  (:require [augur.math :as m]
            [augur.special :refer [chi2-sf phi phi-sf phi-inv t-cdf]]))

;; --- long-run variance ----------------------------------------------------------------

(defn nw-lags
  "The usual Newey-West truncation: floor(4 (n/100)^(2/9))."
  ^long [n]
  (long (Math/floor (* 4.0 (Math/pow (/ n 100.0) (/ 2.0 9.0))))))

(defn long-run-variance
  "The Newey-West long-run variance of xs with `lags` Bartlett-weighted
  autocovariances (each over n, as Newey & West)."
  ^double [xs ^long lags]
  (let [x (double-array xs)
        n (alength x)
        mu (/ (areduce x i s 0.0 (+ s (aget x i))) n)
        _ (dotimes [i n] (aset x i (- (aget x i) mu)))
        acov (fn [l] (/ (loop [i l s 0.0] (if (< i n) (recur (inc i) (+ s (* (aget x i) (aget x (- i l))))) s)) n))]
    (reduce (fn [s l] (+ s (* 2.0 (- 1.0 (/ l (+ lags 1.0))) (acov l)))) (acov 0) (range 1 (inc lags)))))

(defn- t-of-mean
  "The mean of xs over its Newey-West standard error; 0 when xs do not vary
  and are all zero (nothing to test), NaN when they do not vary otherwise."
  ^double [xs lags]
  (let [n (count xs)
        mu (/ (double (reduce + xs)) n)
        se (Math/sqrt (/ (double (long-run-variance xs lags)) n))]
    (cond (pos? se) (/ mu se)
          (zero? mu) 0.0
          :else Double/NaN)))

;; --- point forecasts --------------------------------------------------------------------

(defn oos-r2
  "Out-of-sample R² of forecasts f against benchmark forecasts b for outcomes y."
  ^double [y f b]
  (let [sq (fn [fs] (reduce + (map #(let [e (- %1 %2)] (* e e)) y fs)))]
    (- 1.0 (/ (sq f) (sq b)))))

(defn clark-west
  "Clark & West's adjusted MSPE test of forecasts f against the nested
  benchmark b: {:stat :p}, p one-sided (f better)."
  [y f b lags]
  (let [adj (map (fn [y f b] (- (* (- y b) (- y b)) (- (* (- y f) (- y f)) (* (- f b) (- f b))))) y f b)
        t (t-of-mean adj lags)]
    {:stat t :p (phi-sf t)}))

(defn diebold-mariano
  "Diebold & Mariano on losses la and lb: {:stat :p}, the statistic negative
  when a's losses are lower, p two-sided."
  [la lb lags]
  (let [t (t-of-mean (map - la lb) lags)]
    {:stat t :p (* 2.0 (phi-sf (Math/abs t)))}))

(defn pesaran-timmermann
  "Pesaran & Timmermann's test that forecasts f get the signs of y right
  more often than independent signs would: {:hit :stat :p}, p one-sided."
  [y f]
  (let [n (double (count y))
        sy (map pos? y) sf (map pos? f)
        hit (/ (count (filter true? (map = sy sf))) n)
        py (/ (count (filter true? sy)) n)
        px (/ (count (filter true? sf)) n)
        ps (+ (* py px) (* (- 1 py) (- 1 px)))
        v (/ (* ps (- 1 ps)) n)
        vs (+ (/ (* (Math/pow (- (* 2 py) 1) 2) px (- 1 px)) n)
              (/ (* (Math/pow (- (* 2 px) 1) 2) py (- 1 py)) n)
              (/ (* 4 py px (- 1 py) (- 1 px)) (* n n)))
        d (- v vs)]
    ;; with forecasts (or outcomes) of one sign the statistic is undefined:
    ;; there is no timing to test
    (if (> d (* 1e-12 v))
      (let [t (/ (- hit ps) (Math/sqrt d))] {:hit hit :stat t :p (phi-sf t)})
      {:hit hit :stat Double/NaN :p 1.0})))

(defn qlike
  "Patton's QLIKE loss of variance forecast h for realized variance rv."
  ^double [^double rv ^double h]
  (let [r (/ rv h)] (- r (Math/log r) 1.0)))

;; --- predictive distributions -------------------------------------------------------------

(defn crps-normal
  "CRPS of a Normal(mu, s) forecast at outcome x."
  ^double [^double mu ^double s ^double x]
  (let [z (/ (- x mu) s)]
    (* s (- (+ (* z (- (* 2.0 (phi z)) 1.0)) (* 2.0 (/ (Math/exp (* -0.5 z z)) (Math/sqrt (* 2.0 Math/PI)))))
            (/ 1.0 (Math/sqrt Math/PI))))))

(defn log-t
  "Log density at x of Student's t with location mu, scale s, nu dof."
  ^double [^double mu ^double s ^double nu ^double x]
  (let [z (/ (- x mu) s)]
    (- (m/log-gamma (/ (+ nu 1.0) 2.0)) (m/log-gamma (/ nu 2.0)) (* 0.5 (Math/log (* nu Math/PI))) (Math/log s)
       (* (/ (+ nu 1.0) 2.0) (Math/log1p (/ (* z z) nu))))))

(defn crps-t
  "CRPS of a Student t forecast (location mu, scale s, nu > 1 dof) at x, in
  closed form (Jordan, Krüger & Lerch 2019)."
  ^double [^double mu ^double s ^double nu ^double x]
  (let [z (/ (- x mu) s)
        dens (Math/exp (log-t 0.0 1.0 nu z))
        lb (fn [a b] (- (+ (m/log-gamma a) (m/log-gamma b)) (m/log-gamma (+ a b))))]
    (* s (- (+ (* z (- (* 2.0 (t-cdf z nu)) 1.0))
               (* 2.0 dens (/ (+ nu (* z z)) (- nu 1.0))))
            (/ (* 2.0 (Math/sqrt nu) (Math/exp (- (lb 0.5 (- nu 0.5)) (* 2.0 (lb 0.5 (/ nu 2.0))))))
               (- nu 1.0))))))

(defn- xlogy ^double [^double x ^double y] (if (zero? x) 0.0 (* x (Math/log y))))

(defn kupiec
  "Kupiec's test that `misses` of `n` outcomes outside an interval match its
  nominal miss rate p: {:stat :p}."
  [misses n p]
  (let [x (double misses) n (double n) ph (/ x n)
        lr (* -2.0 (- (+ (xlogy (- n x) (- 1.0 p)) (xlogy x p))
                      (+ (xlogy (- n x) (- 1.0 ph)) (xlogy x ph))))]
    {:stat lr :p (chi2-sf lr 1)}))

(defn christoffersen
  "Christoffersen's test that interval misses (a 0/1 sequence) come
  independently, not in clusters: {:stat :p}."
  [hits]
  (let [pairs (frequencies (map vector hits (rest hits)))
        c (fn [a b] (double (get pairs [a b] 0)))
        n00 (c 0 0) n01 (c 0 1) n10 (c 1 0) n11 (c 1 1)
        p01 (/ n01 (+ n00 n01)) p11 (if (pos? (+ n10 n11)) (/ n11 (+ n10 n11)) 0.0)
        p (/ (+ n01 n11) (+ n00 n01 n10 n11))
        l0 (+ (xlogy (+ n00 n10) (- 1.0 p)) (xlogy (+ n01 n11) p))
        l1 (+ (xlogy n00 (- 1.0 p01)) (xlogy n01 p01) (xlogy n10 (- 1.0 p11)) (xlogy n11 p11))
        lr (* -2.0 (- l0 l1))]
    {:stat lr :p (chi2-sf lr 1)}))

(defn berkowitz
  "Berkowitz's test that PIT values u are independent uniforms: their normal
  scores against an AR(1) with free mean, coefficient and variance
  (conditional likelihood), three degrees of freedom: {:stat :p}."
  [u]
  (let [z (mapv #(phi-inv (min (max (double %) 1e-12) (- 1.0 1e-12))) u)
        x (pop z) y (subvec z 1)
        n (double (count y))
        mx (/ (reduce + x) n) my (/ (reduce + y) n)
        sxy (reduce + (map #(* (- %1 mx) (- %2 my)) x y))
        sxx (reduce + (map #(* (- % mx) (- % mx)) x))
        rho (/ sxy sxx)
        c (- my (* rho mx))
        s2 (/ (reduce + (map #(let [e (- %2 c (* rho %1))] (* e e)) x y)) n)
        l1 (* -0.5 n (+ (Math/log (* 2.0 Math/PI s2)) 1.0))
        l0 (* -0.5 (+ (* n (Math/log (* 2.0 Math/PI))) (reduce + (map #(* % %) y))))
        lr (* -2.0 (- l0 l1))]
    {:stat lr :p (chi2-sf lr 3)}))

;; --- superior predictive ability ------------------------------------------------------------

(defn- spa*
  [cols n lags {:keys [block reps seed]}]
  (let [k (count cols)
        means (mapv #(/ (areduce ^doubles % i s 0.0 (+ s (aget ^doubles % i))) n) cols)
        omegas (mapv #(Math/sqrt (long-run-variance (seq %) lags)) cols)
        stat (max 0.0 (apply max (map #(/ (* (Math/sqrt n) %1) %2) means omegas)))
        ;; recentred under the null: means that could not be below zero by chance stay
        centre (mapv (fn [mu w] (if (>= mu (- (Math/sqrt (* (/ (* w w) n) 2.0 (Math/log (Math/log n)))))) mu 0.0))
                     means omegas)
        g (m/rng seed)
        q (/ 1.0 block)
        idx (int-array n)
        boot (fn []
               (loop [t 0 j (long (* n (m/next-double! g)))]
                 (when (< t n)
                   (aset idx t (int j))
                   (recur (inc t) (if (< (m/next-double! g) q) (long (* n (m/next-double! g))) (mod (inc j) n)))))
               (max 0.0 (apply max (for [c (range k)]
                                     (let [^doubles col (cols c)
                                           mu* (/ (areduce idx i s 0.0 (+ s (aget col (aget idx i)))) n)]
                                       (/ (* (Math/sqrt n) (- mu* (centre c))) (omegas c)))))))
        exceed (count (filter #(>= % stat) (repeatedly reps boot)))]
    {:stat stat :p (/ (inc exceed) (double (inc reps)))}))

(defn spa
  "Hansen's consistent test of superior predictive ability: `gains` holds,
  per model, its per-period gain over the benchmark (positive is better).
  H0: no model beats the benchmark. The stationary bootstrap (mean block
  length :block) resamples periods; {:stat :p}."
  [gains opts]
  (let [n (count (first gains))
        lags (nw-lags n)
        ;; a model whose gain never varies is the benchmark itself
        cols (filterv #(> (long-run-variance (seq %) lags) 1e-24) (map double-array gains))
        k (count cols)]
    (if (zero? k)
      {:stat 0.0 :p 1.0}
      (spa* cols n lags opts))))

