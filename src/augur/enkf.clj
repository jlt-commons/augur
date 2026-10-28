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
  vectors."
  [ens h y r g]
  (let [n (count ens) k (count y)]
    (when (< n 2) (throw (ex-info "augur.enkf/update: two members or more" {:n n})))
    (when (not= k (count r)) (throw (ex-info "augur.enkf/update: one variance per observation" {:y k :r (count r)})))
    (when-not (every? #(and (finite? %) (pos? (double %))) r)
      (throw (ex-info "augur.enkf/update: variances finite and positive" {:r r})))
    (when-not (apply = (map count ens)) (throw (ex-info "augur.enkf/update: members of one length" {})))
    (let [hs (mapv #(vec (h %)) ens)]
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
              ens hs raw)))))

;; -- linear equality constraints ---------------------------------------------------

(defn- orthonormal
  "[[q d] ...]: the rows of `a` with right-hand sides `b`, orthonormalized
  by modified Gram-Schmidt run twice (the rows' span, d carried along).
  A row that is a combination of earlier ones is dropped when its
  right-hand side agrees with theirs; throws when it does not."
  [a b]
  (reduce (fn [qs [row c]]
            (let [r (double-array row)
                  n0 (Math/sqrt (areduce r i s 0.0 (+ s (* (aget r i) (aget r i)))))
                  c0 (Math/abs (double c))
                  c (loop [pass 0 c (double c)]
                      (if (= pass 2)
                        c
                        (recur (inc pass)
                               (reduce (fn [c [^doubles q d]]
                                         (let [k (areduce r i s 0.0 (+ s (* (aget r i) (aget q i))))]
                                           (dotimes [i (alength r)] (aset r i (- (aget r i) (* k (aget q i)))))
                                           (- c (* k (double d)))))
                                       c qs))))
                  nr (Math/sqrt (areduce r i s 0.0 (+ s (* (aget r i) (aget r i)))))]
              (if (<= nr (* 1e-10 n0))
                (if (<= (Math/abs c) (* 1e-8 (max 1.0 c0 n0)))
                  qs
                  (throw (ex-info "augur.enkf/project: contradictory constraints" {:residual c})))
                (do (dotimes [i (alength r)] (aset r i (/ (aget r i) nr)))
                    (conj qs [r (/ c nr)])))))
          [] (map vector a b)))

(defn project
  "The point nearest `x` (Euclidean) on {x : A x = b}, `a` the rows of A:
  x - A'(A A')^+ (A x - b), rows that repeat others allowed (Simon & Chia
  2002's projection of a Kalman estimate onto equality constraints)."
  [x a b]
  (let [qs (orthonormal a b)
        x (double-array x)]
    (doseq [[^doubles q d] qs]
      (let [k (- (areduce q i s 0.0 (+ s (* (aget q i) (aget x i)))) (double d))]
        (dotimes [i (alength x)] (aset x i (- (aget x i) (* k (aget q i)))))))
    (vec x)))

(defn update-constrained
  "update, then each member projected onto its linear constraints {:a rows
  :b right-hand side} or {:a :b-of (fn [member index]) -> b}: identities
  every member holds stay held exactly, as the update alone keeps them
  only to rounding, and only when every member's b is the same."
  [ens h y r g {:keys [a b b-of]}]
  (let [post (update ens h y r g)]
    (if b
      (let [qs (orthonormal a b)]
        (mapv (fn [x] (let [x (double-array x)]
                        (doseq [[^doubles q d] qs]
                          (let [k (- (areduce q i s 0.0 (+ s (* (aget q i) (aget x i)))) (double d))]
                            (dotimes [i (alength x)] (aset x i (- (aget x i) (* k (aget q i)))))))
                        (vec x)))
              post))
      (vec (map-indexed (fn [i x] (project x a (b-of i))) post)))))
