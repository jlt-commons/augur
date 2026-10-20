(ns augur.gamma
  "Gamma posteriors on a rate (events per unit of exposure: errors per
  turn, compactions per hundred turns), learned from counts, with old
  evidence fading back toward the prior.

  A posterior is a map {:kind :gamma :a :b :a0 :b0}: shape a and rate b
  (in exposure units), and the prior's, plain EDN. The mean rate is a / b.
  Its evidence is the exposure it holds beyond the prior, b - b0.
  Forgetting is by step count, as in `augur.beta`.

  Where a function takes a posterior, nil stands for the prior given as
  `:prior` in its options."
  (:require [augur.math :as m]
            [augur.special :as sp]))

(defn prior
  "The prior Gamma(shape a, rate b)."
  [a b]
  (let [a (double a) b (double b)]
    (when-not (and (pos? a) (pos? b))
      (throw (ex-info "augur.gamma/prior needs a > 0 and b > 0" {:a a :b b})))
    {:kind :gamma :a a :b b :a0 a :b0 b}))

(defn centred
  "A scale-free prior centred on `rate`, worth `exposure` units of
  observation: Gamma(rate x exposure, exposure)."
  [rate exposure]
  (prior (* rate exposure) exposure))

(defn- post-or-prior [post opts]
  (or post (:prior opts)
      (throw (ex-info "augur.gamma: a nil posterior needs :prior" {}))))

(defn decay
  "The posterior after `steps` steps of forgetting at rate f:
  a' = a0 + f^steps (a - a0), and likewise b."
  [post f steps]
  (let [{:keys [a b a0 b0]} post
        k (Math/pow (double f) (double steps))]
    (assoc post :a (+ a0 (* k (- a a0))) :b (+ b0 (* k (- b b0))))))

(defn observe
  "The posterior after `:count` events in `:exposure` units (1 when
  absent); :forget f decays the evidence by one step first; :prior stands
  in for a nil posterior."
  [post {:keys [count exposure forget] :or {exposure 1.0 forget 1.0} :as opts}]
  (when-not (and count (>= count 0) (pos? exposure))
    (throw (ex-info "augur.gamma/observe needs :count >= 0 and :exposure > 0" {:count count :exposure exposure})))
  (-> (cond-> (post-or-prior post opts) (< forget 1.0) (decay forget 1))
      (update :a + (double count))
      (update :b + (double exposure))))

(defn mean
  "The posterior mean rate a / b."
  ^double [{:keys [a b]}]
  (/ a b))

(defn evidence
  "The exposure the posterior holds beyond its prior: b - b0."
  ^double [{:keys [b b0]}]
  (- b b0))

(defn cdf
  "P(rate <= x)."
  ^double [{:keys [a b]} x]
  (sp/gamma-p a (* b x)))

(defn quantile
  "The q-quantile of the posterior: bisection on the cdf, bracketed by
  doubling from the mean, to within 1e-12 of the rate."
  ^double [{:keys [a b] :as post} q]
  (cond
    (<= q 0.0) 0.0
    (>= q 1.0) Double/POSITIVE_INFINITY
    :else
    (let [hi (loop [hi (max (/ a b) 1e-300)] (if (< (cdf post hi) q) (recur (* 2.0 hi)) hi))]
      (loop [lo 0.0 hi hi i 0]
        (let [mid (* 0.5 (+ lo hi))]
          (if (or (> i 200) (< (- hi lo) (* 1e-12 hi)))
            mid
            (if (< (cdf post mid) q) (recur mid hi (inc i)) (recur lo mid (inc i)))))))))

(defn interval
  "The central credible interval at `level`: [lo hi]."
  [post level]
  (let [t (/ (- 1.0 level) 2.0)]
    [(quantile post t) (quantile post (- 1.0 t))]))

(defn prob-above
  "P(rate > x)."
  ^double [post x]
  (- 1.0 (cdf post x)))

(defn sample
  "[rate rng']: a draw from the posterior from the generator state `rng`
  (`augur.math/rng-state`), and the state after it."
  [rng {:keys [a b]}]
  (m/with-rng rng #(m/gamma! % a (/ 1.0 b))))

(defn explain
  "{:mean :interval :evidence} at `level`."
  [post level]
  {:mean (mean post) :interval (interval post level) :evidence (evidence post)})
