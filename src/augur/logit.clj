(ns augur.logit
  "Weighted Bayesian logistic regression: Gaussian priors centred on zero
  (ridge) on standardized inputs, or a horseshoe-like reweighting of them,
  fit by damped Newton steps. Rows are {:x :y :w}: x the inputs (x[0] the
  intercept's 1) as a double array or a vector, y a boolean, w a weight.

  Fitting is for the end of a run or offline: it is O(rows x inputs^2) per
  step. `fit` returns a double array; `->state` makes it a vector of
  numbers for storing as EDN, and `predict` takes either.

  Options: :prior, the prior precision on the standardized scale, is
  required; :iterations (25) bounds the Newton steps; :start gives
  coefficients to start from."
  (:require [augur.math :as m]))

(set! *unchecked-math* true)

(defn- ->array ^doubles [x]
  (if (vector? x) (double-array x) x))

(defn ->state
  "Coefficients as plain data: a vector of doubles."
  [beta]
  (mapv double beta))

(defn <-state
  "Coefficients from their plain data, as `fit` returns them."
  ^doubles [state]
  (double-array state))

(defn- softplus ^double [^double e]
  (if (> e 0.0) (+ e (Math/log1p (Math/exp (- e)))) (Math/log1p (Math/exp e))))

(defn fit
  "Coefficients for `rows` ({:x double[k] :y bool :w weight}), x[0] the
  intercept, which is left free; the others have prior precision :prior on
  the scale of their standard deviation in the rows, so the prior does not
  depend on units. Damped Newton steps, each halved until the penalized
  likelihood improves, until they are small or :iterations (default 25) have
  run. :start gives coefficients to start from; :penalties, one multiplier
  of :prior per coefficient (`fit-shrunk`)."
  ^doubles [rows ^long k {:keys [prior start iterations penalties] :or {iterations 25}}]
  (when-not prior (throw (ex-info "augur.logit/fit needs :prior" {})))
  (let [start (some-> start ->array)
        rows (vec rows)
        n (count rows)
        prior (double prior)
        ;; the standardized inputs, row-major in one array
        ^doubles z (double-array (* n k))
        ^doubles ys (double-array n)
        ^doubles ws (double-array n)
        ^doubles mean (double-array k) ^doubles sd (double-array k 1.0)
        _ (dotimes [r n]
            (let [{:keys [x y w]} (rows r) ^doubles x (->array x)]
              (System/arraycopy x 0 z (* r k) k)
              (aset ys r (if y 1.0 0.0))
              (aset ws r (double w))))
        wsum (Math/max 1e-12 (loop [r 0 acc 0.0] (if (< r n) (recur (inc r) (+ acc (aget ws r))) acc)))
        _ (dotimes [r n]
            (let [w (aget ws r) o (* r k)]
              (dotimes [j k] (aset mean j (+ (aget mean j) (* w (aget z (+ o j))))))))
        _ (dotimes [j k] (aset mean j (/ (aget mean j) wsum)))
        _ (let [^doubles v (double-array k)]
            (dotimes [r n]
              (let [w (aget ws r) o (* r k)]
                (dotimes [j k] (let [d (- (aget z (+ o j)) (aget mean j))] (aset v j (+ (aget v j) (* w d d)))))))
            (dotimes [j k]
              (let [s (Math/sqrt (/ (aget v j) wsum))]
                (aset sd j (if (> s 1e-9) s 1.0)))))
        _ (do (aset mean 0 0.0) (aset sd 0 1.0))
        _ (dotimes [r n]
            (let [o (* r k)]
              (aset z o 1.0)
              (loop [j 1] (when (< j k) (aset z (+ o j) (/ (- (aget z (+ o j)) (aget mean j)) (aget sd j))) (recur (inc j))))))
        ;; the start, on the standardized scale
        ^doubles b (double-array k)
        _ (when start
            (let [^doubles st start]
              (aset b 0 (+ (aget st 0) (loop [j 1 acc 0.0] (if (< j k) (recur (inc j) (+ acc (* (aget st j) (aget mean j)))) acc))))
              (loop [j 1] (when (< j k) (aset b j (* (aget st j) (aget sd j))) (recur (inc j))))))
        ;; :penalties scales each coefficient's precision (on the
        ;; standardized scale); the intercept stays nearly free
        ^doubles pen (double-array (if penalties (map #(* prior (double %)) penalties) (repeat k prior)))
        _ (aset pen 0 1e-4)
        eta (fn ^double [^doubles bb ^long r]
              (let [o (* r k)]
                (loop [i 0 acc 0.0] (if (< i k) (recur (inc i) (+ acc (* (aget bb i) (aget z (+ o i))))) acc))))
        objective (fn ^double [^doubles bb]
                    (- (loop [r 0 acc 0.0]
                         (if (< r n)
                           (let [e (eta bb r)]
                             (recur (inc r) (+ acc (* (aget ws r) (- (* (aget ys r) e) (softplus e))))))
                           acc))
                       (* 0.5 (loop [j 0 acc 0.0] (if (< j k) (recur (inc j) (+ acc (* (aget pen j) (aget bb j) (aget bb j)))) acc)))))]
    (loop [it 0 obj (objective b)]
      (when (< it iterations)
        (let [^doubles h (double-array (* k k))
              ^doubles g (double-array k)]
          (dotimes [r n]
            (let [o (* r k)
                  p (m/expit (eta b r))
                  w (aget ws r)
                  rr (* w (- (aget ys r) p))
                  s (* w (Math/max 1e-9 (* p (- 1.0 p))))]
              (dotimes [i k]
                (let [sxi (* s (aget z (+ o i)))
                      hi (* i k)]
                  (aset g i (+ (aget g i) (* rr (aget z (+ o i)))))
                  ;; upper triangle only
                  (loop [j i]
                    (when (< j k)
                      (aset h (+ hi j) (+ (aget h (+ hi j)) (* sxi (aget z (+ o j)))))
                      (recur (inc j))))))))
          (dotimes [i k]
            (loop [j (inc i)] (when (< j k) (aset h (+ (* j k) i) (aget h (+ (* i k) j))) (recur (inc j))))
            (aset g i (- (aget g i) (* (aget pen i) (aget b i))))
            (aset h (+ (* i k) i) (+ (aget h (+ (* i k) i)) (aget pen i))))
          (let [^doubles step (m/solve-spd h g k)
                ;; halve the step until the objective improves
                [nb nobj] (loop [t 1.0 tries 0]
                            (let [^doubles cand (double-array k)]
                              (dotimes [i k] (aset cand i (+ (aget b i) (* t (aget step i)))))
                              (let [o (objective cand)]
                                (if (or (> o obj) (>= tries 30))
                                  [cand o]
                                  (recur (* 0.5 t) (inc tries))))))
                moved (loop [i 0 acc 0.0] (if (< i k) (recur (inc i) (Math/max acc (Math/abs (- (aget ^doubles nb i) (aget b i))))) acc))]
            (when (> nobj obj)
              (System/arraycopy nb 0 b 0 k))
            (when (and (> nobj obj) (> moved 1e-6))
              (recur (inc it) (Math/max obj nobj)))))))
    ;; back to the inputs' own scale
    (let [^doubles beta (double-array k)]
      (loop [j 1] (when (< j k) (aset beta j (/ (aget b j) (aget sd j))) (recur (inc j))))
      (aset beta 0 (- (aget b 0) (loop [j 1 acc 0.0] (if (< j k) (recur (inc j) (+ acc (* (aget beta j) (aget mean j)))) acc))))
      beta)))

(defn fit-shrunk
  "`fit` with a horseshoe-like penalty in place of the ridge: the log
  penalty log(b^2 + c), majorized by a ridge reweighted each round (Candes,
  Wakin & Boyd 2008), each coefficient's precision scaled by
  10c / (b^2 + c) with c the ridge's prior variance on the standardized
  scale: ten times the ridge's shrinkage near zero, the ridge's own at
  b^2 = 9c, less beyond, so small coefficients are shrunk harder and large
  ones freed, as the
  horseshoe does for many weak signals among a few strong ones (Carvalho,
  Polson & Scott 2010). `rounds` reweightings (3)."
  ^doubles [rows ^long k {:keys [prior rounds] :or {rounds 3} :as opts}]
  (when-not prior (throw (ex-info "augur.logit/fit-shrunk needs :prior" {})))
  (let [rows (vec rows)
        n (count rows)
        ;; each input's weighted spread in the rows, as `fit` standardizes,
        ;; to read coefficients on the scale the penalties act on
        wsum (max 1e-12 (reduce + (map #(double (:w %)) rows)))
        sd (vec (for [j (range k)]
                  (let [mu (/ (reduce + (map #(* (double (:w %)) (aget (->array (:x %)) j)) rows)) wsum)
                        s (Math/sqrt (/ (reduce + (map #(let [d (- (aget (->array (:x %)) j) mu)] (* (double (:w %)) d d)) rows))
                                        wsum))]
                    (if (> s 1e-9) s 1.0))))
        c (/ 1.0 (double prior))]
    (loop [r 0 beta (fit rows k opts)]
      (if (>= r rounds)
        beta
        (let [mult (vec (for [j (range k)]
                          (if (zero? j) 1.0 (let [b (* (aget ^doubles beta j) (sd j))] (/ (* 10.0 c) (+ (* b b) c))))))]
          (recur (inc r) (fit rows k (assoc opts :penalties mult :start beta))))))))

(defn predict
  "The probability for inputs x (x[0] the intercept's 1) under
  coefficients beta: arrays, or vectors as `->state` gives."
  ^double [beta x]
  (let [^doubles beta (->array beta) ^doubles x (->array x)]
    (m/expit (loop [i 0 acc 0.0] (if (< i (alength beta)) (recur (inc i) (+ acc (* (aget beta i) (aget x i)))) acc)))))
