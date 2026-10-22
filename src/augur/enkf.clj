(ns augur.enkf
  "The ensemble Kalman filter (Evensen 1994, 2003): a distribution carried
  as an ensemble of state vectors, updated by an observation through the
  ensemble's own covariances, so no derivatives of the model are needed.
  With parameters appended to the state (the augmented state) the
  observations teach them too, through how they covary with what is
  observed: the calibration of a simulation model (system dynamics,
  stocks and flows) from data. The update is the stochastic one, each
  member's observation perturbed by the observation noise (Burgers, van
  Leeuwen & Evensen 1998)."
  (:refer-clojure :exclude [update])
  (:require [augur.math :as m]))

(defn- mean-vec [vs]
  (let [n (double (count vs))]
    (apply mapv (fn [& xs] (/ (reduce + xs) n)) vs)))

(defn update
  "The ensemble `ens` (a vector of state vectors) after observing `y` (a
  vector) with noise variances `r` (a vector, one per observation), each
  member's predicted observation (h member); `g` an augur.math rng."
  [ens h y r g]
  (let [n (count ens)
        hs (mapv #(vec (h %)) ens)
        k (count y)
        xm (mean-vec ens) hm (mean-vec hs)
        dx (mapv #(mapv - % xm) ens)
        dh (mapv #(mapv - % hm) hs)
        cov (fn [as bs i j] (/ (reduce + (map #(* (nth %1 i) (nth %2 j)) as bs)) (dec n)))
        ;; C_hh + R, and C_xh
        shh (double-array (for [i (range k) j (range k)] (+ (cov dh dh i j) (if (= i j) (double (nth r i)) 0.0))))
        dim (count xm)
        cxh (vec (for [a (range dim)] (vec (for [j (range k)] (cov dx dh a j)))))]
    (mapv (fn [x hx]
            (let [innov (double-array (for [j (range k)] (+ (- (double (nth y j)) (double (nth hx j)))
                                                             (* (Math/sqrt (double (nth r j))) (m/normal! g)))))
                  w (m/solve-spd shh innov k)]
              (mapv (fn [a xa] (+ xa (reduce + (for [j (range k)] (* (get-in cxh [a j]) (aget ^doubles w j)))))) (range dim) x)))
          ens hs)))
