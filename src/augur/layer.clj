(ns augur.layer
  "Shrinking toward a parent (empirical Bayes over a hierarchy): a child's
  prior is its parent's posterior mean at a fixed strength, so a new child
  starts from its parent's experience and earns its own estimate as its
  evidence comes in.

  Layers are posteriors from `augur.beta` or `augur.gamma`, all of one kind,
  child first: [project machine shipped]. Each holds its own evidence over
  its own prior; nil is a layer with no evidence. Folding from the top, each
  layer's prior is replaced by one made from the posterior above it, and
  its own evidence is added back."
  (:require [augur.beta :as beta]
            [augur.gamma :as gamma]))

(defn- ops [post]
  (case (:kind post)
    :beta {:mean beta/mean :evidence beta/evidence :interval beta/interval}
    :gamma {:mean gamma/mean :evidence gamma/evidence :interval gamma/interval}
    (throw (ex-info "augur.layer: a layer must be a :beta or :gamma posterior" {:kind (:kind post)}))))

(defn prior-from
  "A prior with the mean of `parent` (a posterior) and the given strength:
  for a Beta, a0 + b0 = strength; for a Gamma, b0 = strength units of
  exposure."
  [parent strength]
  (let [s (double strength)]
    (case (:kind parent)
      :beta (let [mu (beta/mean parent)] (beta/prior (* mu s) (* (- 1.0 mu) s)))
      :gamma (gamma/prior (* (gamma/mean parent) s) s)
      (throw (ex-info "augur.layer: the parent must be a :beta or :gamma posterior" {:kind (:kind parent)})))))

(defn- rebase
  "`post`'s own evidence (a - a0, b - b0) over `prior`; `prior` itself for
  a nil post."
  [post prior]
  (if post
    (assoc prior :a (+ (:a prior) (- (:a post) (:a0 post))) :b (+ (:b prior) (- (:b post) (:b0 post))))
    prior))

(defn- strength-at [strength i]
  (double (if (sequential? strength) (nth strength i) strength)))

(defn posterior
  "The child's posterior, folding [child parent grandparent ...] from the
  top: each layer's prior made from the posterior above it at :strength
  (one number, or one per layer, the entry for a layer setting how far it
  is drawn to the layer above; the top's is unused). The top layer keeps its
  own prior, and must not be nil."
  [layers {:keys [strength]}]
  (when (nil? strength) (throw (ex-info "augur.layer/posterior needs :strength" {})))
  (let [layers (vec layers)
        top (peek layers)]
    (when (nil? top) (throw (ex-info "augur.layer/posterior: the top layer cannot be nil" {})))
    (reduce (fn [above i] (rebase (layers i) (prior-from above (strength-at strength i))))
            top
            (range (- (count layers) 2) -1 -1))))

(defn explain
  "{:mean :interval :evidence :own :by-layer [{:mean :evidence} ...]} for
  the child under `posterior`'s fold: the folded mean and interval at
  :level, the child's own evidence and its share of the folded posterior's
  weight (:own: 0 for a fresh child, toward 1 as it learns), and each
  layer's own mean and evidence on its own prior, child first."
  [layers {:keys [level strength] :as opts}]
  (when (nil? level) (throw (ex-info "augur.layer/explain needs :level" {})))
  (let [post (posterior layers opts)
        {:keys [mean interval]} (ops post)
        own (if-let [c (first layers)] ((:evidence (ops c)) c) 0.0)
        s (strength-at strength 0)]
    {:mean (mean post)
     :interval (interval post level)
     :evidence own
     :own (/ own (+ own s))
     :by-layer (mapv (fn [l] (when l (let [{:keys [mean evidence]} (ops l)] {:mean (mean l) :evidence (evidence l)})))
                     layers)}))
