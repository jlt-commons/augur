(ns augur.enkf
  "The ensemble Kalman filter (Evensen 1994, 2003): a distribution carried
  as an ensemble of state vectors, updated by an observation through the
  ensemble's own covariances, so no derivatives of the model are needed.
  With parameters appended to the state (the augmented state) the
  observations teach them too, through how they covary with what is
  observed. The update is the stochastic one, each member's observation
  perturbed by the observation noise (Burgers, van Leeuwen & Evensen
  1998), the perturbations centred. Repeated with the noise inflated (R
  times a, the inflations' reciprocals summing to one) it is the ensemble
  smoother with multiple data assimilation (Emerick & Reynolds 2013), the
  calibration of a deterministic model's parameters. Linear identities
  the members hold (accounting) are kept by projecting each member onto
  them after the update (project, update-constrained)."
  (:refer-clojure :exclude [update])
  (:require [augur.math :as m]))

(defn- cholesky
  "The lower factor of the symmetric k x k matrix `a` (row major); throws
  if it is not positive definite."
  ^doubles [^doubles a ^long k]
  (let [l (double-array (* k k))]
    (dotimes [i k]
      (dotimes [j (inc i)]
        (let [s (loop [p 0 s (aget a (+ (* i k) j))]
                  (if (< p j) (recur (inc p) (- s (* (aget l (+ (* i k) p)) (aget l (+ (* j k) p))))) s))]
          (if (= i j)
            (do (when-not (> s (* 1e-12 (Math/abs (aget a (+ (* i k) i)))))
                  (throw (ex-info "augur.enkf: the innovation covariance is not positive definite" {:row i})))
                (aset l (+ (* i k) j) (Math/sqrt s)))
            (aset l (+ (* i k) j) (/ s (aget l (+ (* j k) j))))))))
    l))

(defn- solve-chol
  "x with L L' x = b."
  ^doubles [^doubles l ^doubles b ^long k]
  (let [z (double-array k) x (double-array k)]
    (dotimes [i k]
      (aset z i (/ (loop [p 0 s (aget b i)] (if (< p i) (recur (inc p) (- s (* (aget l (+ (* i k) p)) (aget z p)))) s))
                   (aget l (+ (* i k) i)))))
    (loop [i (dec k)]
      (when (>= i 0)
        (aset x i (/ (loop [p (inc i) s (aget z i)] (if (< p k) (recur (inc p) (- s (* (aget l (+ (* p k) i)) (aget x p)))) s))
                     (aget l (+ (* i k) i))))
        (recur (dec i))))
    x))

(defn- finite? [x] (not (or (Double/isNaN (double x)) (Double/isInfinite (double x)))))

(defn update
  "The ensemble `ens` (a vector of n >= 2 state vectors of one length)
  after observing `y` (a vector of k) with noise variances `r` (k of
  them, finite and positive), each member's predicted observation (h
  member) (k values): each member moves by C_xh (C_hh + R)^-1 (y + e - h),
  e its centred perturbation; `g` an augur.math rng. A vector of n state
  vectors. opts :indexed, h called (h member i) with the member's index:
  an observation of a member's state against something of its own not in
  the state (a level observed as a ratio to the member's output), kept
  linear in the state."
  ([ens h y r g] (update ens h y r g {}))
  ([ens h y r g {:keys [indexed]}]
  (let [n (count ens) k (count y)]
    (when (< n 2) (throw (ex-info "augur.enkf/update: two members or more" {:n n})))
    (when (not= k (count r)) (throw (ex-info "augur.enkf/update: one variance per observation" {:y k :r (count r)})))
    (when-not (every? #(and (finite? %) (pos? (double %))) r)
      (throw (ex-info "augur.enkf/update: variances finite and positive" {:r r})))
    (when-not (apply = (map count ens)) (throw (ex-info "augur.enkf/update: members of one length" {})))
    (let [hs (if indexed (vec (map-indexed (fn [i x] (vec (h x i))) ens)) (mapv #(vec (h %)) ens))]
      (when-not (every? #(= k (count %)) hs) (throw (ex-info "augur.enkf/update: h gives one value per observation" {})))
      (when-not (every? #(every? finite? %) (concat ens hs))
        (throw (ex-info "augur.enkf/update: a member or its prediction is not finite" {})))
      (let [dim (count (first ens))
            mean-of (fn [vs] (let [nn (double (count vs))] (apply mapv (fn [& xs] (/ (reduce + xs) nn)) vs)))
            xm (mean-of ens) hm (mean-of hs)
            dx (mapv #(double-array (map - % xm)) ens)
            dh (mapv #(double-array (map - % hm)) hs)
            cov (fn [as i bs j] (/ (loop [p 0 s 0.0] (if (< p n) (recur (inc p) (+ s (* (aget ^doubles (as p) i) (aget ^doubles (bs p) j)))) s)) (dec n)))
            s (double-array (for [i (range k) j (range k)] (+ (cov dh i dh j) (if (= i j) (double (nth r i)) 0.0))))
            l (cholesky s k)
            cxh (vec (for [a (range dim)] (double-array (for [j (range k)] (cov dx a dh j)))))
            ;; the perturbations, centred (their n - 1 sample covariance has
            ;; expectation R, as the gain's covariances use n - 1)
            raw (vec (repeatedly n #(double-array (for [j (range k)] (* (Math/sqrt (double (nth r j))) (m/normal! g))))))
            em (mean-of (map vec raw))]
        (mapv (fn [x hx e]
                (let [innov (double-array (for [j (range k)] (+ (- (double (nth y j)) (double (nth hx j)))
                                                                 (- (aget ^doubles e j) (nth em j)))))
                      w (solve-chol l innov k)]
                  (mapv (fn [a xa] (+ xa (loop [j 0 acc 0.0] (if (< j k) (recur (inc j) (+ acc (* (aget ^doubles (cxh a) j) (aget w j)))) acc))))
                        (range dim) x)))
              ens hs raw))))))

;; -- linear equality constraints ---------------------------------------------------

(defn- basis
  "[[q c] ...]: the rows of `a` orthonormalized by modified Gram-Schmidt
  run twice, each q with its coefficients c over the rows (q = sum c_i
  a_i), so q's right-hand side for any b is c . b. A row that is a
  combination of earlier ones is dropped, with its combination
  (:dependent [[row c] ...]) to check b against."
  [a]
  (let [m (count a)]
    (reduce (fn [{:keys [qs] :as acc} [j row]]
              (let [r (double-array row)
                    c (double-array m)
                    _ (aset c j 1.0)
                    n0 (Math/sqrt (areduce r i s 0.0 (+ s (* (aget r i) (aget r i)))))]
                (dotimes [_ 2]
                  (doseq [[^doubles q ^doubles cq] qs]
                    (let [k (areduce r i s 0.0 (+ s (* (aget r i) (aget q i))))]
                      (dotimes [i (alength r)] (aset r i (- (aget r i) (* k (aget q i)))))
                      (dotimes [i m] (aset c i (- (aget c i) (* k (aget cq i))))))))
                (let [nr (Math/sqrt (areduce r i s 0.0 (+ s (* (aget r i) (aget r i)))))]
                  (if (<= nr (* 1e-10 n0))
                    (clojure.core/update acc :dependent conj [n0 c])
                    (do (dotimes [i (alength r)] (aset r i (/ (aget r i) nr)))
                        (dotimes [i m] (aset c i (/ (aget c i) nr)))
                        (clojure.core/update acc :qs conj [r c]))))))
            {:qs [] :dependent []} (map-indexed vector a))))

(defn- dot-c ^double [^doubles c b] (loop [i 0 s 0.0] (if (< i (alength c)) (recur (inc i) (+ s (* (aget c i) (double (nth b i))))) s)))

(defn- projector
  "(fn [x b]) -> x projected onto A x = b, from basis's factoring of `a`;
  throws when b contradicts a dependent row."
  [a]
  (let [{:keys [qs dependent]} (basis a)]
    (fn [x b]
      (doseq [[n0 c] dependent :let [res (dot-c c b)]]
        (when (> (Math/abs res) (* 1e-8 (max 1.0 n0 (reduce max 0.0 (map #(Math/abs (double %)) b)))))
          (throw (ex-info "augur.enkf/project: contradictory constraints" {:residual res}))))
      (let [x (double-array x)]
        (doseq [[^doubles q c] qs]
          (let [k (- (areduce q i s 0.0 (+ s (* (aget q i) (aget x i)))) (dot-c c b))]
            (dotimes [i (alength x)] (aset x i (- (aget x i) (* k (aget q i)))))))
        (vec x)))))

(defn project
  "The point nearest `x` (Euclidean) on {x : A x = b}, `a` the rows of A:
  x - A'(A A')^+ (A x - b), rows that repeat others allowed (Simon & Chia
  2002's projection of a Kalman estimate onto equality constraints)."
  [x a b]
  ((projector a) x b))

(defn update-constrained
  "update, then each member projected onto its linear constraints {:a rows
  :b right-hand side} or {:a :b-of (fn [index]) -> b} (member index's):
  identities every member holds stay held exactly, as the update alone
  keeps them only to rounding, and only when every member's b is the
  same. The projection is project's, Euclidean: where every b is the
  same it removes rounding alone (the update moves a member along the
  ensemble's deviations, which keep the identities); where they differ
  it is the least-squares correction, not one weighted by the ensemble's
  covariance (whose rank, members less one, is too small to meet many
  constraints)."
  [ens h y r g {:keys [a b b-of indexed]}]
  (let [post (update ens h y r g {:indexed indexed})
        p (projector a)]
    (vec (map-indexed (fn [i x] (p x (if b b (b-of i)))) post))))
