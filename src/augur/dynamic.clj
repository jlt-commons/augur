(ns augur.dynamic
  "Models that learn from experience: dynamic generalized linear models
  (West, Harrison & Migon 1985; West & Harrison 1997) for binary outcomes
  (logistic) and counts (Poisson), the dynamic linear model for
  continuous outcomes (linear: Gaussian, its observation variance unknown
  and learned with the coefficients, West & Harrison 1997 ch. 4), and
  dynamic model averaging over them (Raftery, Kárný & Ettler 2010;
  McCormick et al. 2012).

  A model's coefficients are a state, Gaussian with mean m and covariance
  C. Each period the state is discounted, C / delta, so old evidence fades
  and the coefficients can drift; then each of the period's rows updates
  it. The update is exact in the one dimension a row informs, the linear
  predictor x'b: its prior N(f, q) times the row's likelihood, a Laplace
  approximation at the mode, carried back to the coefficients by the
  linear Bayes rule. Nothing is refit: a model carries what it has learned
  forward.

  A model starts from what is known: :mean and :var give each coefficient's
  prior, a sign and size the literature gives and how far it is trusted, or
  zero and a wide variance for an input nobody has a view on. :discount is
  required: near 0.97 for a model that must follow fast changes, 1 for one
  whose coefficients should not move. :inputs picks the row's inputs the
  model reads (all of them by default), so models over different groups of
  inputs can be averaged on the same rows.

  Rows are {:x :y}, with an optional weight :w and offset :o (a fixed part
  of the linear predictor: another model's log odds, an exposure's log). A
  binary :y is true or a positive number; false, nil and 0 are no event.

  `averaged` combines models: each period their weights are flattened by
  :forget (weights to the power forget, renormalized), so a model's past
  record fades, then multiplied by how likely each made the period's
  outcomes. The weights through time are each model's share of the
  evidence: what has mattered, and when that changed. Averaging over the
  same inputs at several discounts tunes the discount from experience."
  (:require [augur.math :as m]))

(set! *unchecked-math* true)

(defn- need [f opts]
  (when-let [missing (seq (keep (fn [[k v]] (when (nil? v) k)) opts))]
    (throw (ex-info (str "augur.dynamic/" (name f) " needs " (vec missing)) {:missing (vec missing)}))))

(defn- model [kind {:keys [mean var discount inputs] :as opts}]
  (need kind {:mean mean :var var :discount discount})
  (when (not= (count mean) (count var))
    (throw (ex-info (str "augur.dynamic/" (name kind) " needs one variance per mean") opts)))
  (let [k (count mean)
        c (double-array (* k k))]
    (dotimes [i k] (aset c (+ (* i k) i) (double (nth var i))))
    {:type kind :k k :m (double-array (map double mean)) :c c
     :discount (double discount) :inputs (some-> inputs vec)}))

(defn logistic
  "A dynamic logistic regression for binary outcomes. Options above."
  [opts]
  (model :logistic opts))

(defn poisson
  "A dynamic Poisson regression (log link) for counts. Options above."
  [opts]
  (model :poisson opts))

(defn linear
  "A dynamic linear regression for continuous outcomes, y = x'b + noise of
  unknown variance V: the coefficients' update is exact (the Kalman
  filter), V's estimate :s carried with :n degrees of freedom (from :s0,
  default 1, and :n0, default 1) and updated by each row's standardized
  error, the coefficients' covariance rescaled with it; the outcome's
  predictive is Student t with n degrees of freedom, location f and scale
  q + s. :var is the coefficients' prior covariance at V = s0: as V's
  estimate moves the covariance moves with it, so a wrong s0 also widens
  or narrows that prior. A row's :w is a power on its likelihood (its
  precision scaled by w, the row counted w times), as in the other
  models. V does not drift (no variance discount): the noise is taken as
  constant while the coefficients move. Options above."
  [{:keys [s0 n0] :or {s0 1.0 n0 1.0} :as opts}]
  (when-not (and (pos? (double s0)) (pos? (double n0)))
    (throw (ex-info "augur.dynamic/linear needs a positive :s0 and :n0" {:s0 s0 :n0 n0})))
  (assoc (model :linear opts) :s (double s0) :n (double n0)))

(defn- pick ^doubles [{:keys [inputs k]} x]
  (let [out (double-array k)]
    (if inputs
      (dotimes [j k] (aset out j (double (nth x (inputs j)))))
      (dotimes [j k] (aset out j (double (nth x j)))))
    out))

(defn- moments
  "f = x'm (plus the offset) and q = x'Cx, the prior of the linear
  predictor, with C scaled by `scale`."
  [^doubles mm ^doubles c ^long k ^doubles x ^double offset ^double scale]
  (let [f (loop [i 0 acc offset] (if (< i k) (recur (inc i) (+ acc (* (aget mm i) (aget x i)))) acc))
        q (loop [i 0 acc 0.0]
            (if (< i k)
              (recur (inc i) (loop [j 0 acc acc]
                               (if (< j k) (recur (inc j) (+ acc (* (aget x i) (aget c (+ (* i k) j)) (aget x j)))) acc)))
              acc))]
    [f (* scale q)]))

(defn means "The coefficients' posterior means." [state] (vec (:m state)))

(defn- predict-one ^double [{:keys [type m c k discount]} x offset]
  (let [[f q] (moments m c k x (double (or offset 0.0)) (/ 1.0 discount))]
    (case type
      ;; the logistic-normal integral (MacKay 1992's probit approximation)
      :logistic (m/expit (/ f (Math/sqrt (+ 1.0 (/ (* Math/PI q) 8.0)))))
      ;; the mean of a lognormal rate
      :poisson (Math/exp (+ f (* 0.5 q)))
      :linear f)))

(defn- event?
  "A binary outcome: true, or a positive number (0 and false are none)."
  [y]
  (if (number? y) (pos? (double y)) (boolean y)))

(defn- loglik ^double [type ^double eta y ^double w]
  (case type
    :logistic (* w (- (* (if (event? y) 1.0 0.0) eta) (if (> eta 0.0) (+ eta (Math/log1p (Math/exp (- eta)))) (Math/log1p (Math/exp eta)))))
    :poisson (* w (- (* (double y) eta) (Math/exp eta)))))

(defn- grad-curv
  "The row's likelihood gradient and curvature in eta."
  [type ^double eta y ^double w]
  (case type
    :logistic (let [p (m/expit eta)] [(* w (- (if (event? y) 1.0 0.0) p)) (* w p (- 1.0 p))])
    :poisson (let [mu (Math/exp eta)] [(* w (- (double y) mu)) (* w mu)])))

(defn- update-row!
  "Updates mm and c in place with one row, c already discounted."
  [type ^doubles mm ^doubles c ^long k ^doubles x y ^double w ^double offset]
  (let [[f q] (moments mm c k x offset 1.0)
        f (double f) q (double q)]
    (when (> q 1e-12)
      ;; the mode of N(eta; f, q) times the row's likelihood, by damped Newton
      (let [eta (loop [i 0 eta f]
                  (let [[g h] (grad-curv type eta y w)
                        step (/ (+ (double g) (/ (- f eta) q)) (+ (double h) (/ 1.0 q)))
                        eta' (+ eta (Math/max -5.0 (Math/min 5.0 step)))]
                    (if (or (>= i 50) (< (Math/abs (- eta' eta)) 1e-10)) eta' (recur (inc i) eta'))))
            [_ h] (grad-curv type eta y w)
            q* (/ 1.0 (+ (double h) (/ 1.0 q)))
            ;; carried back by linear Bayes: m += Cx (eta - f)/q,
            ;; C -= Cx x'C (1 - q*/q)/q
            cx (double-array k)]
        (dotimes [i k]
          (aset cx i (loop [j 0 acc 0.0] (if (< j k) (recur (inc j) (+ acc (* (aget c (+ (* i k) j)) (aget x j)))) acc))))
        (let [a (/ (- eta f) q)
              shrink (/ (- 1.0 (/ q* q)) q)]
          (dotimes [i k] (aset mm i (+ (aget mm i) (* (aget cx i) a))))
          (dotimes [i k]
            (dotimes [j k]
              (aset c (+ (* i k) j) (- (aget c (+ (* i k) j)) (* (aget cx i) (aget cx j) shrink))))))))))

(declare step-averaged predict-averaged)

(defn predict
  "The probability of the outcome (logistic), the expected count
  (Poisson) or the expected value (linear) for inputs x in the coming
  period, the coefficients' uncertainty integrated over; `offset` as a
  row's :o."
  ([state x] (predict state x 0.0))
  ([state x offset]
   (if (= :averaged (:type state))
     (predict-averaged state x offset)
     (predict-one state (pick state x) offset))))

(declare predictive)

(defn- predictive-averaged [{:keys [models w kind]} x offset]
  (let [ms (for [[k mdl] models] [(w k) (predictive mdl x offset)])
        f (reduce + (for [[wk {:keys [f]}] ms] (* wk f)))]
    (cond-> {:f f
             ;; the mixture's variance: within the models plus between their means
             :q (- (reduce + (for [[wk {f' :f q' :q}] ms] (* wk (+ q' (* f' f'))))) (* f f))}
      ;; the noise beside it, the models' estimates weighted
      (= :linear kind) (assoc :s (reduce + (for [[wk {s' :s}] ms] (* wk s')))))))

(defn predictive
  "The linear predictor's predictive moments for inputs x in the coming
  period, {:f mean :q variance}, `offset` included in the mean; for a model
  average, the mixture's mean and variance. What predict integrates over.
  A linear model adds the noise's estimate :s and its degrees of freedom
  :n (the outcome Student t, scale q + s); an average of them, :s
  weighted."
  ([state x] (predictive state x 0.0))
  ([state x offset]
   (if (= :averaged (:type state))
     (predictive-averaged state x offset)
     (let [{:keys [m c k discount type s]} state
           [f q] (moments m c k (pick state x) (double (or offset 0.0)) (/ 1.0 discount))]
       (cond-> {:f f :q q} (= :linear type) (assoc :s s :n (:n state)))))))

(defn- linear-log-density
  "log of the Student t density of y with n degrees of freedom, location f
  and scale Q."
  ^double [^double y ^double f ^double big-q ^double n]
  (let [e (- y f)]
    (- (+ (m/log-gamma (* 0.5 (+ n 1.0))) (- (m/log-gamma (* 0.5 n))))
       (* 0.5 (Math/log (* n Math/PI big-q)))
       (* 0.5 (+ n 1.0) (Math/log1p (/ (* e e) (* n big-q)))))))

(defn- poisson-log-predictive
  "log P(y) for a count whose log rate is N(f, q): the negative binomial
  with the lognormal rate's mean mu = exp(f + q/2) and variance mu + mu^2
  (e^q - 1), moment matched, whose size is then r = 1 / (e^q - 1) whatever
  mu is; the Poisson when r is huge (q negligible). Computed in logs, so
  a wildly uncertain model scores low, not NaN."
  ^double [^double f ^double q ^double y]
  (let [log-mu (Math/min 700.0 (+ f (* 0.5 q)))
        r (/ 1.0 (Math/max 1e-300 (Math/expm1 (Math/min 700.0 q))))]
    (if (> r 1e8)
      (- (* y log-mu) (Math/exp log-mu) (m/log-gamma (+ 1.0 y)))
      ;; log(r + mu), stably
      (let [log-r (Math/log r)
            log-sum (if (> log-mu log-r)
                      (+ log-mu (Math/log1p (Math/exp (- log-r log-mu))))
                      (+ log-r (Math/log1p (Math/exp (- log-mu log-r)))))]
        (+ (m/log-gamma (+ y r)) (- (m/log-gamma r)) (- (m/log-gamma (+ 1.0 y)))
           (* r (- log-r log-sum)) (* y (- log-mu log-sum)))))))

(defn log-score
  "The log predictive probability of `rows` under the state before it
  learns from them (a period's evidence for the state), the coefficients'
  uncertainty integrated over; the rows scored independently against that
  state. (An average weighs its linear models on their sequential
  evidence, each row given the ones before it.)"
  [state rows]
  (reduce + (for [{:keys [x y w o]} rows
                  :let [w (double (or w 1.0))
                        o (double (or o 0.0))]]
              (if (= :averaged (:type state))
                (let [p (when-not (= :linear (:kind state)) (predict state x o))]
                  (case (:kind state)
                    ;; the mixture's density
                    :linear (let [{:keys [models] ws :w} state
                                  ls (for [[k mdl] models]
                                       (+ (Math/log (Math/max 1e-300 (double (ws k))))
                                          (log-score mdl [{:x x :y y :o o}])))
                                  top (apply max ls)]
                              (* w (+ top (Math/log (reduce + (map #(Math/exp (- % top)) ls))))))
                    :logistic (* w (Math/log (Math/max 1e-300 (if (event? y) p (- 1.0 p)))))
                    ;; the mixture's mean only: a period's evidence is scored per model
                    :poisson (* w (- (* (double y) (Math/log (Math/max 1e-300 p))) p (m/log-gamma (+ 1.0 (double y)))))))
                (case (:type state)
                  :logistic (let [p (predict state x o)]
                              (* w (Math/log (Math/max 1e-300 (if (event? y) p (- 1.0 p))))))
                  :poisson (let [[f q] (moments (:m state) (:c state) (:k state) (pick state x) o (/ 1.0 (:discount state)))]
                             (* w (poisson-log-predictive f q (double y))))
                  :linear (let [[f q] (moments (:m state) (:c state) (:k state) (pick state x) o (/ 1.0 (:discount state)))]
                            (* w (linear-log-density (double y) f (+ q (:s state)) (:n state)))))))))

(defn- linear-step
  "[state log-evidence] of a linear model after `rows`, its mean `mm` and
  covariance `c` (already discounted) updated in place: the Kalman update,
  V's estimate learned with it (West & Harrison 4.6), and each row's log
  predictive given the rows before it."
  [state ^doubles mm ^doubles c rows]
  (let [k (long (:k state))]
    (loop [rows (seq rows) s (double (:s state)) n (double (:n state)) ll 0.0]
      (if-not rows
        [(assoc state :m mm :c c :s s :n n) ll]
        (let [{:keys [x y w o]} (first rows)
              x (pick state x) w (double (or w 1.0))
              [f q] (moments mm c k x (double (or o 0.0)) 1.0)
              e (- (double y) (double f))
              ll (+ ll (* w (linear-log-density (double y) (double f) (+ (double q) s) n)))
              big-q (+ (double q) (/ s w))
              cx (double-array k)]
          (dotimes [i k]
            (aset cx i (loop [j 0 acc 0.0] (if (< j k) (recur (inc j) (+ acc (* (aget c (+ (* i k) j)) (aget x j)))) acc))))
          ;; the likelihood to the power w: shape + w/2, rate + s e^2/(2Q)
          (let [n' (+ n w)
                s' (Math/max 1e-12 (+ s (* (/ s n') (- (/ (* e e) big-q) w))))
                r (/ s' s)]
            (dotimes [i k] (aset mm i (+ (aget mm i) (/ (* (aget cx i) e) big-q))))
            (dotimes [i k]
              (dotimes [j k]
                (aset c (+ (* i k) j) (* r (- (aget c (+ (* i k) j)) (/ (* (aget cx i) (aget cx j)) big-q))))))
            (recur (next rows) s' n' ll)))))))

(defn- discounted [state]
  (let [{:keys [k discount]} state
        mm (aclone ^doubles (:m state))
        ^doubles c (aclone ^doubles (:c state))]
    (dotimes [i (* k k)] (aset c i (/ (aget c i) discount)))
    [mm c]))

(defn step
  "The state after a period: discounted, then updated with each of `rows`."
  [state rows]
  (if (= :averaged (:type state))
    (step-averaged state rows)
    (let [{:keys [type k]} state
          [mm c] (discounted state)]
      (if (= :linear type)
        (first (linear-step state mm c rows))
        (do (doseq [{:keys [x y w o]} rows]
              (update-row! type mm c k (pick state x) y (double (or w 1.0)) (double (or o 0.0))))
            (assoc state :m mm :c c))))))

(defn averaged
  "Dynamic model averaging over `models` ({name state}, all logistic,
  all Poisson or all linear), starting from equal weights; :forget
  (required) flattens the weights each period, so a model's past record
  fades."
  [models {:keys [forget]}]
  (need :averaged {:forget forget})
  (let [kinds (set (map :type (vals models)))]
    (when (not= 1 (count kinds))
      (throw (ex-info "augur.dynamic/averaged needs models of one kind" {:kinds kinds})))
    {:type :averaged :kind (first kinds) :models models :forget (double forget)
     :w (into {} (for [k (keys models)] [k (/ 1.0 (count models))]))}))

(defn weights "Each model's weight after the last period." [avg] (:w avg))

(defn- predict-averaged [{:keys [models w]} x offset]
  (reduce + (for [[k mdl] models] (* (w k) (predict mdl x offset)))))

(defn- step-averaged [{:keys [models w forget] :as avg} rows]
  (let [flat (into {} (for [[k v] w] [k (Math/pow v forget)]))
        z (reduce + (vals flat))
        prior (update-vals flat #(/ % z))
        ;; each model's evidence: how likely it made the period's outcomes
        ;; a linear model's is exact and sequential: each row given the
        ;; ones before it (its step computes it)
        stepped (into {} (for [[k mdl] models]
                           [k (if (= :linear (:type mdl))
                                (let [[mm c] (discounted mdl)] (linear-step mdl mm c rows))
                                [nil (log-score mdl rows)])]))
        ll (update-vals stepped second)
        top (apply max (vals ll))
        post (into {} (for [[k p] prior] [k (* p (Math/exp (- (ll k) top)))]))
        z (reduce + (vals post))
        ;; a small share spread evenly, so no weight underflows to zero and
        ;; stays there (0 to any power is 0): a model can always come back
        eps 1e-6
        n (count post)
        post (update-vals post #(+ (* (- 1.0 eps) (/ % z)) (/ eps n)))
        z 1.0]
    (assoc avg
           :w (update-vals post #(/ % z))
           :models (into {} (for [[k mdl] models] [k (or (first (stepped k)) (step mdl rows))])))))

(defn couple
  "Two models of one layout learning from each other's experience: each
  coefficient at `:indices` moves toward the other's by w = s p' / (p + s
  p'), p and p' the two precisions and s the :strength (0 none, 1 as if
  the other's belief were one more of one's own). Only the means move:
  repeated coupling must not count the same evidence twice, so the
  variances stay. [a' b'], each moved by the other's belief before the
  coupling."
  [a b {:keys [indices strength]}]
  (need :couple {:indices indices :strength strength})
  (let [k (:k a)
        ma (aclone ^doubles (:m a)) mb (aclone ^doubles (:m b))
        ^doubles ca (:c a) ^doubles cb (:c b)
        s (double strength)]
    (doseq [i indices]
      (let [i (long i)
            pa (/ 1.0 (Math/max 1e-12 (aget ca (+ (* i k) i))))
            pb (/ 1.0 (Math/max 1e-12 (aget cb (+ (* i k) i))))
            a0 (aget ^doubles (:m a) i) b0 (aget ^doubles (:m b) i)]
        (aset ma i (+ a0 (* (/ (* s pb) (+ pa (* s pb))) (- b0 a0))))
        (aset mb i (+ b0 (* (/ (* s pa) (+ pb (* s pa))) (- a0 b0))))))
    [(assoc a :m ma) (assoc b :m mb)]))
