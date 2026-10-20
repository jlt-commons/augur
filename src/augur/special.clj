(ns augur.special
  "Special functions, held to scipy in the tests: the regularized incomplete
  gamma and beta functions and their uses (the chi-square tail, the normal
  cdf to double precision, Student's t cdf), the inverse incomplete beta,
  and the normal quantile refined to full precision."
  (:require [augur.math :as m]))

(defn log-gamma
  "ln Γ(x) (`augur.math/log-gamma`)."
  ^double [^double x]
  (m/log-gamma x))

(defn log-beta
  "ln B(a, b)."
  ^double [^double a ^double b]
  (m/log-beta a b))

(defn- gamma-series
  "P(a, x) by its series, for x < a + 1."
  ^double [^double a ^double x]
  (loop [n 1 term (/ 1.0 a) sum (/ 1.0 a)]
    (if (or (> n 1000) (< (Math/abs term) (* (Math/abs sum) 1e-16)))
      (* sum (Math/exp (- (* a (Math/log x)) x (m/log-gamma a))))
      (let [term (* term (/ x (+ a n)))] (recur (inc n) term (+ sum term))))))

(defn- gamma-cf
  "Q(a, x) by Lentz's continued fraction, for x >= a + 1."
  ^double [^double a ^double x]
  (let [tiny 1e-300]
    (loop [i 1 b (+ x 1.0 (- a)) c (/ 1.0 tiny) d (/ 1.0 (+ x 1.0 (- a))) h (/ 1.0 (+ x 1.0 (- a)))]
      (let [an (* (- i) (- i a))
            b (+ b 2.0)
            d (let [d (+ (* an d) b)] (/ 1.0 (if (< (Math/abs d) tiny) tiny d)))
            c (let [c (+ b (/ an c))] (if (< (Math/abs c) tiny) tiny c))
            del (* d c)
            h (* h del)]
        (if (or (> i 1000) (< (Math/abs (- del 1.0)) 1e-16))
          (* h (Math/exp (- (* a (Math/log x)) x (m/log-gamma a))))
          (recur (inc i) b c d h))))))

(defn gamma-p
  "The regularized lower incomplete gamma P(a, x): its series below a + 1,
  else one less the continued fraction for Q (Numerical Recipes 6.2)."
  ^double [^double a ^double x]
  (cond
    (<= x 0.0) 0.0
    (< x (+ a 1.0)) (gamma-series a x)
    :else (- 1.0 (gamma-cf a x))))

(defn gamma-q
  "The regularized upper incomplete gamma Q(a, x) = 1 - P(a, x), computed
  directly in the upper tail, where 1 - P would cancel."
  ^double [^double a ^double x]
  (cond
    (<= x 0.0) 1.0
    (< x (+ a 1.0)) (- 1.0 (gamma-series a x))
    :else (gamma-cf a x)))

(defn chi2-sf
  "P(X > x) for a chi-square with k degrees of freedom."
  ^double [^double x ^long k]
  (gamma-q (/ k 2.0) (/ x 2.0)))

(defn phi
  "The standard normal cdf, exact to double precision: through the
  incomplete gamma, Φ(x) = (1 ± P(1/2, x²/2)) / 2."
  ^double [^double x]
  (let [p (gamma-p 0.5 (/ (* x x) 2.0))]
    (if (neg? x) (* 0.5 (- 1.0 p)) (* 0.5 (+ 1.0 p)))))

(defn phi-sf
  "1 - Φ(x), without cancellation in the upper tail."
  ^double [^double x]
  (if (pos? x)
    (* 0.5 (gamma-q 0.5 (/ (* x x) 2.0)))
    (- 1.0 (phi x))))

(defn phi-inv
  "The standard normal quantile: Acklam's approximation and one Halley step."
  ^double [^double p]
  (let [x (m/normal-quantile p)
        e (- (phi x) p)
        u (* e (Math/sqrt (* 2.0 Math/PI)) (Math/exp (/ (* x x) 2.0)))]
    (- x (/ u (+ 1.0 (/ (* x u) 2.0))))))

(defn- beta-cf
  "The continued fraction of the regularized incomplete beta (NR 6.4)."
  ^double [^double a ^double b ^double x]
  (let [tiny 1e-300
        qab (+ a b) qap (+ a 1.0) qam (- a 1.0)
        fix (fn [^double v] (if (< (Math/abs v) tiny) tiny v))
        d0 (/ 1.0 (fix (- 1.0 (/ (* qab x) qap))))]
    (loop [mm 1 c 1.0 d d0 h d0]
      (let [m2 (* 2 mm)
            aa (/ (* mm (- b mm) x) (* (+ qam m2) (+ a m2)))
            d (/ 1.0 (fix (+ 1.0 (* aa d))))
            c (fix (+ 1.0 (/ aa c)))
            h (* h d c)
            aa (- (/ (* (+ a mm) (+ qab mm) x) (* (+ a m2) (+ qap m2))))
            d (/ 1.0 (fix (+ 1.0 (* aa d))))
            c (fix (+ 1.0 (/ aa c)))
            del (* d c)
            h (* h del)]
        (if (or (> mm 1000) (< (Math/abs (- del 1.0)) 1e-16)) h (recur (inc mm) c d h))))))

(defn beta-inc
  "The regularized incomplete beta I_x(a, b)."
  ^double [^double a ^double b ^double x]
  (cond
    (<= x 0.0) 0.0
    (>= x 1.0) 1.0
    :else
    (let [front (Math/exp (+ (- (m/log-gamma (+ a b)) (m/log-gamma a) (m/log-gamma b))
                             (* a (Math/log x)) (* b (Math/log1p (- x)))))]
      (if (< x (/ (+ a 1.0) (+ a b 2.0)))
        (/ (* front (beta-cf a b x)) a)
        (- 1.0 (/ (* front (beta-cf b a (- 1.0 x))) b))))))

(defn t-cdf
  "The cdf of Student's t with nu degrees of freedom."
  ^double [^double x ^double nu]
  (let [tail (* 0.5 (beta-inc (/ nu 2.0) 0.5 (/ nu (+ nu (* x x)))))]
    (if (pos? x) (- 1.0 tail) tail)))

(defn beta-inc-inv
  "The x with I_x(a, b) = p: bisection on [0, 1], which I_x is monotone
  over, to within 1e-15 of the root (60 halvings)."
  ^double [^double a ^double b ^double p]
  (cond
    (<= p 0.0) 0.0
    (>= p 1.0) 1.0
    :else (loop [lo 0.0 hi 1.0 i 0]
            (let [mid (* 0.5 (+ lo hi))]
              (if (>= i 60)
                mid
                (if (< (beta-inc a b mid) p) (recur mid hi (inc i)) (recur lo mid (inc i))))))))

(defn beta-pdf
  "The density of Beta(a, b) at x in (0, 1)."
  ^double [^double a ^double b ^double x]
  (Math/exp (- (+ (* (- a 1.0) (Math/log x)) (* (- b 1.0) (Math/log1p (- x)))) (m/log-beta a b))))

(defn gauss-legendre
  "[nodes weights] of the n-point Gauss-Legendre rule on [-1, 1]: each node
  a root of the Legendre polynomial P_n, found by Newton's method from
  Tricomi's estimate, P_n and its derivative by the three-term recurrence;
  the weight 2 / ((1 - x²) P_n'(x)²)."
  [n]
  (let [n (long n)
        legendre (fn [^double x]
                   ;; [P_n(x) P_n'(x)]
                   (loop [j 1 p0 1.0 p1 x]
                     (if (< j n)
                       (recur (inc j) p1 (/ (- (* (+ (* 2 j) 1) x p1) (* j p0)) (inc j)))
                       [p1 (/ (* n (- (* x p1) p0)) (- (* x x) 1.0))])))
        roots (for [i (range 1 (inc n))]
                (loop [x (Math/cos (/ (* Math/PI (- i 0.25)) (+ n 0.5))) it 0]
                  (let [[p dp] (legendre x)
                        x' (- x (/ p dp))]
                    (if (or (< (Math/abs (- x' x)) 1e-15) (> it 100))
                      (let [[_ dp] (legendre x')] [x' (/ 2.0 (* (- 1.0 (* x' x')) dp dp))])
                      (recur x' (inc it))))))]
    [(mapv first roots) (mapv second roots)]))
