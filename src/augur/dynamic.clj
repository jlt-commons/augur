(ns augur.dynamic
  "Models that learn from experience: dynamic generalized linear models
  (West, Harrison & Migon 1985; West & Harrison 1997) for binary outcomes
  (logistic) and counts (Poisson), and dynamic model averaging over them
  (Raftery, Kárný & Ettler 2010; McCormick et al. 2012).

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
      :poisson (Math/exp (+ f (* 0.5 q))))))

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
  "The probability of the outcome (logistic) or the expected count
  (Poisson) for inputs x in the coming period, the coefficients'
  uncertainty integrated over; `offset` as a row's :o."
  ([state x] (predict state x 0.0))
  ([state x offset]
   (if (= :averaged (:type state))
     (predict-averaged state x offset)
     (predict-one state (pick state x) offset))))

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
  uncertainty integrated over."
  [state rows]
  (reduce + (for [{:keys [x y w o]} rows
                  :let [w (double (or w 1.0))
                        o (double (or o 0.0))]]
              (if (= :averaged (:type state))
                (let [p (predict state x o)]
                  (case (:kind state)
                    :logistic (* w (Math/log (Math/max 1e-300 (if (event? y) p (- 1.0 p)))))
                    ;; the mixture's mean only: a period's evidence is scored per model
                    :poisson (* w (- (* (double y) (Math/log (Math/max 1e-300 p))) p (m/log-gamma (+ 1.0 (double y)))))))
                (case (:type state)
                  :logistic (let [p (predict state x o)]
                              (* w (Math/log (Math/max 1e-300 (if (event? y) p (- 1.0 p))))))
                  :poisson (let [[f q] (moments (:m state) (:c state) (:k state) (pick state x) o (/ 1.0 (:discount state)))]
                             (* w (poisson-log-predictive f q (double y)))))))))

(defn step
  "The state after a period: discounted, then updated with each of `rows`."
  [state rows]
  (if (= :averaged (:type state))
    (step-averaged state rows)
    (let [{:keys [type k discount]} state
          mm (aclone ^doubles (:m state))
          ^doubles c (aclone ^doubles (:c state))]
      (dotimes [i (* k k)] (aset c i (/ (aget c i) discount)))
      (doseq [{:keys [x y w o]} rows]
        (update-row! type mm c k (pick state x) y (double (or w 1.0)) (double (or o 0.0))))
      (assoc state :m mm :c c))))

(defn averaged
  "Dynamic model averaging over `models` ({name state}, all logistic or all
  Poisson), starting from equal weights; :forget (required) flattens the
  weights each period, so a model's past record fades."
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
        ll (into {} (for [[k mdl] models] [k (log-score mdl rows)]))
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
           :models (update-vals models #(step % rows)))))

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
