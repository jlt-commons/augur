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
  calibration of a deterministic model's parameters."
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
            ;; the perturbations, centred and rescaled to the noise's variance
            raw (vec (repeatedly n #(double-array (for [j (range k)] (* (Math/sqrt (double (nth r j))) (m/normal! g))))))
            em (mean-of (map vec raw))
            scale (Math/sqrt (/ n (double (dec n))))]
        (mapv (fn [x hx e]
                (let [innov (double-array (for [j (range k)] (+ (- (double (nth y j)) (double (nth hx j)))
                                                                 (* scale (- (aget ^doubles e j) (nth em j))))))
                      w (solve-chol l innov k)]
                  (mapv (fn [a xa] (+ xa (loop [j 0 acc 0.0] (if (< j k) (recur (inc j) (+ acc (* (aget ^doubles (cxh a) j) (aget w j)))) acc))))
                        (range dim) x)))
              ens hs raw)))))
