(ns augur.beta
  "Beta posteriors on a probability, learned from outcomes, with old
  evidence fading back toward the prior.

  A posterior is a map {:kind :beta :a :b :a0 :b0}: the current parameters
  and the prior's, plain EDN. Its evidence is a + b - a0 - b0, the number of
  observations it holds after forgetting. Forgetting is by step count, so a
  posterior idle for five steps decays by f^5 when it is next read, with no
  write in between.

  Where a function takes a posterior, nil stands for the prior given as
  `:prior` in its options."
  (:require [augur.math :as m]
            [augur.special :as sp]))

(defn prior
  "The prior Beta(a, b)."
  [a b]
  (let [a (double a) b (double b)]
    (when-not (and (pos? a) (pos? b))
      (throw (ex-info "augur.beta/prior needs a > 0 and b > 0" {:a a :b b})))
    {:kind :beta :a a :b b :a0 a :b0 b}))

(defn jeffreys
  "Jeffreys' prior Beta(1/2, 1/2): it keeps the upper end of 0 in n open and
  the lower end of n in n, which a flat prior understates."
  []
  (prior 0.5 0.5))

(defn- post-or-prior [post opts]
  (or post (:prior opts)
      (throw (ex-info "augur.beta: a nil posterior needs :prior" {}))))

(defn from-counts
  "The posterior after k successes in n trials under `prior`."
  [k n prior]
  (when-not (<= 0 k n)
    (throw (ex-info "augur.beta/from-counts needs 0 <= k <= n" {:k k :n n})))
  (-> prior (update :a + k) (update :b + (- n k))))

(defn decay
  "The posterior after `steps` steps of forgetting at rate f (f^steps of
  its evidence kept): a' = a0 + f^steps (a - a0), and likewise b."
  [post f steps]
  (let [{:keys [a b a0 b0]} post
        k (Math/pow (double f) (double steps))]
    (assoc post :a (+ a0 (* k (- a a0))) :b (+ b0 (* k (- b b0))))))

(defn observe
  "The posterior after one outcome: true or false, or a number in [0 1]
  for a partial success. Options: :forget f decays the evidence by one step
  first (1, none, when absent); :weight w counts the outcome w times (1);
  :prior stands in for a nil posterior."
  [post outcome {:keys [forget weight] :or {forget 1.0 weight 1.0} :as opts}]
  (let [post (cond-> (post-or-prior post opts) (< forget 1.0) (decay forget 1))
        y (cond (true? outcome) 1.0 (false? outcome) 0.0 :else (double outcome))
        w (double weight)]
    (when-not (<= 0.0 y 1.0)
      (throw (ex-info "augur.beta/observe needs an outcome in [0 1]" {:outcome outcome})))
    (-> post (update :a + (* w y)) (update :b + (* w (- 1.0 y))))))

(defn mean
  "The posterior mean a / (a + b)."
  ^double [{:keys [a b]}]
  (/ a (+ a b)))

(defn evidence
  "The observations the posterior holds beyond its prior: a + b - a0 - b0."
  ^double [{:keys [a b a0 b0]}]
  (- (+ a b) a0 b0))

(defn cdf
  "P(p <= x)."
  ^double [{:keys [a b]} x]
  (sp/beta-inc a b x))

(defn quantile
  "The q-quantile of the posterior."
  ^double [{:keys [a b]} q]
  (sp/beta-inc-inv a b q))

(defn interval
  "The central credible interval at `level`: [lo hi]."
  [post level]
  (let [t (/ (- 1.0 level) 2.0)]
    [(quantile post t) (quantile post (- 1.0 t))]))

(defn prob-above
  "P(p > x)."
  ^double [post x]
  (- 1.0 (cdf post x)))

(defn prob-between
  "P(lo < p < hi)."
  ^double [post lo hi]
  (- (cdf post hi) (cdf post lo)))

(def ^:private rule (delay (sp/gauss-legendre 200)))

(defn prob-greater
  "P(p_A > p_B) for independent posteriors: the integral over [0 1] of A's
  density times B's cdf, by 200-point Gauss-Legendre. The nodes stay off
  the ends, where a posterior from Jeffreys' prior with k = 0 or k = n is
  singular but integrable."
  ^double [post-a post-b]
  (let [[xs ws] @rule
        {aa :a ba :b} post-a {ab :a bb :b} post-b]
    (reduce + 0.0 (map (fn [x w]
                         (let [t (/ (+ x 1.0) 2.0)]
                           (* (/ w 2.0) (sp/beta-pdf aa ba t) (sp/beta-inc ab bb t))))
                       xs ws))))

(defn sample
  "[p rng']: a draw from the posterior, for Thompson sampling, from the
  generator state `rng` (`augur.math/rng-state`), and the state after it."
  [rng {:keys [a b]}]
  (m/with-rng rng #(m/beta! % a b)))

(defn explain
  "What a person or a prompt reads: {:mean :interval :evidence} at `level`."
  [post level]
  {:mean (mean post) :interval (interval post level) :evidence (evidence post)})

;; --- pass@k and pass^k ----------------------------------------------------------------------

(defn pass-at-k
  "The unbiased pass@k from n runs with c passes: the chance at least one of
  k draws without replacement passes, 1 - C(n-c, k) / C(n, k), as a
  running product so the binomials never overflow. 0 with no runs or k 0."
  ^double [n c k]
  (cond
    (or (zero? n) (zero? k)) 0.0
    (< (- n c) k) 1.0
    :else (- 1.0 (reduce * 1.0 (map #(- 1.0 (/ (double k) %)) (range (inc (- n c)) (inc n)))))))

(defn pass-power-k
  "pass^k: the chance k independent runs all pass, (c/n)^k. Biased on
  purpose: the unbiased C(c, k) / C(n, k) is zero whenever c < k, which
  tells hard cases apart not at all. 0 with no runs."
  ^double [n c k]
  (if (zero? n) 0.0 (Math/pow (/ (double c) n) k)))
