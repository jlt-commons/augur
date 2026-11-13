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
  (:require [augur.hier :as hier]
            [augur.math :as m]))

(set! *unchecked-math* true)

(defn- need [f opts]
  (when-let [missing (seq (keep (fn [[k v]] (when (nil? v) k)) opts))]
    (throw (ex-info (str "augur.dynamic/" (name f) " needs " (vec missing)) {:missing (vec missing)}))))

(defn- by-layout
  "The pooled-slopes layout of a model of k coefficients: {:index :groups
  :slopes :tau :base k}, checked."
  [k {:keys [index groups slopes tau] :as by}]
  (need :by {:index index :groups groups :slopes slopes :tau tau})
  (let [slopes (vec slopes)
        taus (if (number? tau) (vec (repeat (count slopes) (double tau))) (mapv double tau))]
    (when-not (and (pos? (long groups)) (seq slopes) (every? #(< -1 (long %) k) slopes)
                   (= (count taus) (count slopes)) (every? pos? taus))
      (throw (ex-info "augur.dynamic: :by needs groups > 0, slopes among the model's coefficients and a positive tau each" by)))
    {:index (long index) :groups (long groups) :slopes slopes :tau taus :base k}))

(defn- model [kind {:keys [mean var discount inputs by hold group-shrink] :as opts}]
  (need kind {:mean mean :var var :discount discount})
  (when group-shrink
    (let [{:keys [groups tau rate]} group-shrink]
      (when-not (and (number? tau) (pos? (double tau))
                     (every? (fn [g] (and (seq g) (every? #(< -1 (long %) (count mean)) g))) groups)
                     (or (nil? rate) (< 0.0 (double rate) 1.000001)))
        (throw (ex-info (str "augur.dynamic/" (name kind) ": :group-shrink needs a positive :tau, groups of the model's coefficients and a :rate in (0, 1]") group-shrink)))))
  (when (not= (count mean) (count var))
    (throw (ex-info (str "augur.dynamic/" (name kind) " needs one variance per mean") opts)))
  (when-not (every? #(< -1 (long %) (count mean)) hold)
    (throw (ex-info (str "augur.dynamic/" (name kind) ": :hold names a coefficient the model does not have") opts)))
  (let [k0 (count mean)
        by (some->> by (by-layout k0))
        ;; pooled slopes: each group's deviation from the common slope, N(0, tau^2)
        devs (when by (for [_ (range (:groups by)) t (:tau by)] t))
        mean (concat (map double mean) (map (constantly 0.0) devs))
        var (concat (map double var) (map #(* % %) devs))
        k (count mean)
        c (double-array (* k k))]
    (dotimes [i k] (aset c (+ (* i k) i) (double (nth var i))))
    (cond-> {:type kind :k k :m (double-array mean) :c c
             :discount (double discount) :inputs (some-> inputs vec)}
      by (assoc :by by)
      group-shrink (assoc :gshrink {:groups (mapv #(mapv long %) (:groups group-shrink))
                                    :tau (double (:tau group-shrink))
                                    :rate (double (or (:rate group-shrink) (- 1.0 (double discount))))})
      ;; per coefficient, the discount's scale: the deviations of pooled
      ;; slopes are held (a discount on them would dissolve the pooling),
      ;; as is every coefficient in :hold
      (or by (seq hold))
      (assoc :sv (let [held (set (map long hold))]
                   (double-array (for [i (range k)]
                                   (if (or (>= i k0) (held i)) 1.0 (/ 1.0 (Math/sqrt (double discount)))))))))))

(defn logistic
  "A dynamic logistic regression for binary outcomes. Options above, and
  :group-shrink {:groups [[i ...] ...] :tau t :rate r}: each group of
  coefficients shrunk toward zero by its own horseshoe-type scale (see
  `group-scales`), so a group that carries nothing shrinks and one that
  carries a signal escapes."
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

(defn- pick ^doubles [{:keys [inputs k by]} x]
  (let [k0 (if by (:base by) k)
        out (double-array k)]
    (if inputs
      (dotimes [j k0] (aset out j (double (nth x (inputs j)))))
      (dotimes [j k0] (aset out j (double (nth x j)))))
    (when by
      (let [{:keys [index groups slopes]} by
            g (nth x index)
            g (when (number? g) (long g))]
        (when-not (and g (< -1 g groups))
          (throw (ex-info "augur.dynamic: a row's group is not one of the model's" {:group (nth x index) :groups groups})))
        (let [from (+ k0 (* g (count slopes)))]
          (dotimes [s (count slopes)] (aset out (+ from s) (aget out (long (slopes s))))))))
    out))

(defn- moments*
  "As moments, with C's entry ij further scaled by sv_i sv_j when sv is given."
  [^doubles mm ^doubles c ^long k ^doubles x ^double offset ^double scale sv]
  (let [x0 x
        x (if sv (let [^doubles sv sv o (double-array k)] (dotimes [i k] (aset o i (* (aget x i) (aget sv i)))) o) x)
        ;; the mean from the inputs themselves: sv scales the covariance only
        f (loop [i 0 acc offset] (if (< i k) (recur (inc i) (+ acc (* (aget mm i) (aget ^doubles x0 i)))) acc))
        q (loop [i 0 acc 0.0]
            (if (< i k)
              (recur (inc i) (loop [j 0 acc acc]
                               (if (< j k) (recur (inc j) (+ acc (* (aget x i) (aget c (+ (* i k) j)) (aget x j)))) acc)))
              acc))]
    [f (* scale q)]))

(defn- moments
  "f = x'm (plus the offset) and q = x'Cx, the prior of the linear
  predictor, with C scaled by `scale`."
  [^doubles mm ^doubles c ^long k ^doubles x ^double offset ^double scale]
  (moments* mm c k x offset scale nil))

(defn- ahead
  "[f q]: the linear predictor's moments for the picked inputs x in the
  coming period, the state's covariance discounted as `step` will."
  [{:keys [m c k discount sv]} x offset]
  (if sv
    (moments* m c k x (double offset) 1.0 sv)
    (moments m c k x (double offset) (/ 1.0 (double discount)))))

(defn means "The coefficients' posterior means." [state] (vec (:m state)))

(defn slopes
  "A model with pooled slopes (:by): each group's slopes, the common part
  plus the group's deviation, in the order of :slopes."
  [{:keys [by] :as state}]
  (let [{:keys [base groups] ss :slopes} by
        mm (:m state)]
    (vec (for [g (range groups)]
           (vec (for [[i j] (map-indexed vector ss)]
                  (+ (aget ^doubles mm (long j)) (aget ^doubles mm (+ base (* g (count ss)) i)))))))))

(defn- predict-one ^double [{:keys [type] :as state} x offset]
  (let [[f q] (ahead state x (double (or offset 0.0)))]
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

(declare step-averaged step-averaged* predict-averaged step-posterior pooled-moments shared-score)

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

(defn- predictive-averaged [{:keys [models w kind] :as avg} x offset]
  (if (#{:evidence :synth} (:rule (:weighting avg)))
    (pooled-moments avg x offset)
  (let [ms (for [[k mdl] models] [(w k) (predictive mdl x offset)])
        f (reduce + (for [[wk {:keys [f]}] ms] (* wk f)))]
    (cond-> {:f f
             ;; the mixture's variance: within the models plus between their means
             :q (- (reduce + (for [[wk {f' :f q' :q}] ms] (* wk (+ q' (* f' f'))))) (* f f))}
      ;; the noise beside it, the models' estimates weighted
      (= :linear kind) (assoc :s (reduce + (for [[wk {s' :s}] ms] (* wk s'))))))))

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
     (let [{:keys [type s]} state
           [f q] (ahead state (pick state x) (double (or offset 0.0)))]
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
  (if (and (not= :averaged (:type state)) (some :shared rows))
    (shared-score (:type state) rows
                  #(log-score state [(dissoc % :shared)])
                  (fn [{:keys [x o]}] (let [[f q] (ahead state (pick state x) (double (or o 0.0)))] {:f f :q q})))
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
                  :poisson (let [[f q] (ahead state (pick state x) o)]
                             (* w (poisson-log-predictive f q (double y))))
                  :linear (let [[f q] (ahead state (pick state x) o)]
                            (* w (linear-log-density (double y) f (+ q (:s state)) (:n state))))))))))

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
  (let [{:keys [k discount sv]} state
        k (long k)
        mm (aclone ^doubles (:m state))
        ^doubles c (aclone ^doubles (:c state))]
    (if sv
      (let [^doubles sv sv]
        (dotimes [i k] (dotimes [j k] (aset c (+ (* i k) j) (* (aget c (+ (* i k) j)) (aget sv i) (aget sv j))))))
      (dotimes [i (* k k)] (aset c i (/ (aget c i) discount))))
    [mm c]))

(defn- half-cauchy-scale
  "The mode of a group's local scale u = lambda^2 under a half-Cauchy
  prior on lambda, its k coefficients N(0, tau^2 u) with sum of squares s
  tau^2 (Carvalho, Polson & Scott 2010's horseshoe, by group): the root of
  (k + 3) u^2 - (s - k - 1) u - s = 0. Near s / (k + 3) for a group that
  carries a signal, near zero for one that carries none."
  ^double [^double s ^long k]
  (let [a (+ k 3.0) b (- s k 1.0)]
    (/ (+ b (Math/sqrt (+ (* b b) (* 4.0 a s)))) (* 2.0 a))))

(defn- pseudo-zero!
  "Updates mm and c in place with a pseudo-observation that coefficient i
  is zero, with variance r (the exact Gaussian update)."
  [^doubles mm ^doubles c ^long k ^long i ^double r]
  (let [big-q (+ (aget c (+ (* i k) i)) r)
        ci (double-array k)]
    (dotimes [j k] (aset ci j (aget c (+ (* j k) i))))
    (let [e (/ (- (aget mm i)) big-q)]
      (dotimes [j k] (aset mm j (+ (aget mm j) (* (aget ci j) e)))))
    (dotimes [a k]
      (dotimes [b k]
        (aset c (+ (* a k) b) (- (aget c (+ (* a k) b)) (/ (* (aget ci a) (aget ci b)) big-q)))))))

(defn- shrink-groups!
  "Shrinks each group of `gshrink` in place: its local scale u from its
  coefficients' second moments (m^2 + C, after the discount), then a
  pseudo-observation of each coefficient at zero with variance tau^2 u /
  rate. The rate is what the discount forgets each period (1 - discount by
  default), so the shrinkage prior is refreshed as fast as it fades and
  holds at about its full weight however long the model runs. The scales."
  [^doubles mm ^doubles c ^long k {:keys [groups tau rate]}]
  (let [t2 (* (double tau) (double tau))]
    (mapv (fn [g]
            (let [s (/ (reduce + 0.0 (map (fn [i] (let [i (long i)] (+ (* (aget mm i) (aget mm i)) (aget c (+ (* i k) i))))) g)) t2)
                  u (half-cauchy-scale s (count g))
                  r (/ (* t2 (Math/max 1e-12 u)) (double rate))]
              (doseq [i g] (pseudo-zero! mm c k (long i) r))
              u))
          groups)))

(defn group-scales
  "A model with :group-shrink: each group's local scale (lambda^2, in units
  of tau^2) at its last period, in the order of its groups; nil before."
  [state]
  (:scales state))

(defn- shared-groups
  "`rows` split by their :shared effect: [[nil rows-without] [[id mean var]
  rows] ...], the groups in order of first appearance; each group's
  effect checked to be one [id mean var] with var >= 0."
  [rows]
  (let [order (distinct (keep #(some-> (:shared %) first) rows))
        by (group-by #(some-> (:shared %) first) rows)]
    (into [[nil (vec (by nil))]]
          (for [id order
                :let [rs (by id)
                      effs (distinct (map :shared rs))
                      [_ mean var :as eff] (first effs)]]
            (do (when-not (and (= 1 (count effs)) (= 3 (count eff))
                               (if (sequential? mean)
                                 ;; several dimensions: d means, a d x d variance, each row its :load
                                 (let [d (count mean)]
                                   (and (pos? d) (every? number? mean) (sequential? var) (= d (count var))
                                        (every? #(and (sequential? %) (= d (count %)) (every? number? %)) var)
                                        (every? #(>= (double (get-in var [% %])) 0.0) (range d))
                                        (every? #(let [a (:load %)] (and (sequential? a) (= d (count a)) (every? number? a))) rs)))
                                 (and (number? mean) (number? var) (>= (double var) 0.0))))
                  (throw (ex-info "augur.dynamic: a shared effect is [id mean variance], one per id, the variance at least 0; with d means, a d x d variance and each row's :load of d numbers" {:shared (vec effs)})))
                [eff rs])))))

(defn- step-shared!
  "Updates mm and c in place with rows some of which share an effect
  (:shared [id mean var]: a term N(mean, var) common to every row with
  that id, added to their linear predictors). The rows without one update
  as `step` does; each group updates the state augmented by its effect,
  which is then integrated out (the Gaussian marginal: the block of the
  coefficients). A coefficient whose input is the same at every row of a
  group (an intercept, an input common to the period) is confounded with
  the effect within it, so it learns from the group about what one
  observation of variance var would teach, not once per row."
  [{:keys [type k] :as state} ^doubles mm ^doubles c rows]
  (let [k (long k)
        k1 (inc k)]
    (doseq [[eff rs] (shared-groups rows)]
      (if (nil? eff)
        (doseq [{:keys [x y w o]} rs]
          (update-row! type mm c k (pick state x) y (double (or w 1.0)) (double (or o 0.0))))
        (let [[_ mean var] eff]
          (if (sequential? mean)
            ;; a state of d dimensions, each row adding its :load times it
            (let [d (count mean)
                  kd (+ k d)
                  m1 (double-array kd)
                  c1 (double-array (* kd kd))]
              (dotimes [i k] (aset m1 i (aget mm i)))
              (dotimes [a d] (aset m1 (+ k a) (double (nth mean a))))
              (dotimes [i k] (dotimes [j k] (aset c1 (+ (* i kd) j) (aget c (+ (* i k) j)))))
              (dotimes [a d] (dotimes [b d] (aset c1 (+ (* (+ k a) kd) (+ k b)) (double (nth (nth var a) b)))))
              (doseq [{:keys [x y w o load]} rs]
                (let [x0 (pick state x)
                      x1 (double-array kd)]
                  (dotimes [i k] (aset x1 i (aget x0 i)))
                  (dotimes [a d] (aset x1 (+ k a) (double (nth load a))))
                  (update-row! type m1 c1 kd x1 y (double (or w 1.0)) (double (or o 0.0)))))
              (dotimes [i k] (aset mm i (aget m1 i)))
              (dotimes [i k] (dotimes [j k] (aset c (+ (* i k) j) (aget c1 (+ (* i kd) j))))))
            (let [m1 (double-array k1)
                  c1 (double-array (* k1 k1))]
              (dotimes [i k] (aset m1 i (aget mm i)))
              (aset m1 k (double mean))
              (dotimes [i k] (dotimes [j k] (aset c1 (+ (* i k1) j) (aget c (+ (* i k) j)))))
              (aset c1 (+ (* k k1) k) (double var))
              (doseq [{:keys [x y w o]} rs]
                (let [x0 (pick state x)
                      x1 (double-array k1)]
                  (dotimes [i k] (aset x1 i (aget x0 i)))
                  (aset x1 k 1.0)
                  (update-row! type m1 c1 k1 x1 y (double (or w 1.0)) (double (or o 0.0)))))
              (dotimes [i k] (aset mm i (aget m1 i)))
              (dotimes [i k] (dotimes [j k] (aset c (+ (* i k) j) (aget c1 (+ (* i k1) j))))))))))))

(defn- shared-score
  "The log score of `rows`, some sharing an effect: the rows without one
  by `one` (fn [row] -> log probability), each group by its rows' log
  marginal with the effect integrated (augur.hier/shared-marginal), each
  row's predictive moments from `moments` (fn [row] -> {:f :q})."
  [kind rows one moments]
  (when (= :linear kind)
    (throw (ex-info "augur.dynamic: a shared effect needs a logistic or Poisson model" {:kind kind})))
  (reduce + 0.0 (for [[eff rs] (shared-groups rows)]
                  (if (or (nil? eff) (and (not (sequential? (nth eff 1))) (zero? (double (nth eff 2))) (zero? (double (nth eff 1)))))
                    ;; no effect, or one fixed at 0: the rows scored as they are
                    (reduce + 0.0 (map one rs))
                    (let [[_ mean var] eff]
                      (:ll (hier/shared-marginal kind (for [r rs] (cond-> (assoc (moments r) :y (:y r) :w (double (or (:w r) 1.0)))
                                                                    (sequential? mean) (assoc :load (:load r))))
                                                 mean var)))))))

(defn step
  "The state after a period: discounted, then updated with each of `rows`.
  A row's :shared [id mean var] is a term common to every row of the
  period with that id, N(mean, var) on the linear predictor (a period's
  common shock): those rows are learned jointly with it and it is then
  integrated out (logistic and Poisson models). With mean a vector of d
  and var a d x d matrix the term has d dimensions and each row adds its
  :load (d numbers) times it: a common shock and a slope on each row's
  exposure, with loads [1 e], so an exposure-weighted input is taught
  about one observation a period too."
  [state rows]
  (if (= :averaged (:type state))
    (step-averaged state rows)
    (let [{:keys [type k gshrink]} state
          [mm c] (discounted state)
          scales (when gshrink (shrink-groups! mm c (long k) gshrink))
          state (cond-> state gshrink (assoc :scales scales))]
      (cond
        (and (= :linear type) (some :shared rows))
        (throw (ex-info "augur.dynamic: a shared effect needs a logistic or Poisson model" {:type type}))
        (= :linear type)
        (first (linear-step state mm c rows))
        (some :shared rows)
        (do (step-shared! state mm c rows)
            (assoc state :m mm :c c))
        :else
        (do (doseq [{:keys [x y w o]} rows]
              (update-row! type mm c k (pick state x) y (double (or w 1.0)) (double (or o 0.0))))
            (assoc state :m mm :c c))))))

(defn averaged
  "Dynamic model averaging over `models` ({name state}, all logistic,
  all Poisson or all linear), starting from equal weights; :forget
  (required) flattens the weights each period, so a model's past record
  fades.

  :weighting {:rule :temper :power t} raises each period's likelihoods to
  the power t (0 < t <= 1, tempered or fractional posterior weights;
  Grunwald's safe Bayes): the evidence is the same, counted as if the
  period held fewer independent rows, so the weights move more slowly and
  select less sharply where the models fit about as well. t = 1 is the
  posterior.

  :weighting {:rule :stack :rate eta :floor phi} sets the weights by
  stacking instead of by each model's posterior: each row moves them by
  exponentiated gradient on the mixture's log score (w_k times exp(eta
  p_k / p), p_k the model's predictive probability of the row and p the
  mixture's; Helmbold et al. 1998), and each period they are mixed with
  phi spread evenly, so no weight falls below phi / n. Posterior weights
  multiply a period's likelihoods over all its rows and select one model
  where several fit about as well; stacked weights blend them. Blending
  is not ignoring: a model whose forecasts only add noise keeps a share
  of the weight and adds its noise to the average.

  :weighting {:rule :evidence :base b :scale c :parents {name parent}
  :cap true} ranks the models by their proven increment instead
  (bnm-56c.2): each model's discounted log score (the period's log
  predictive before it learns, discounted by :forget), less its parent's
  (b unless :parents names another: a model nested in another is set
  against it), per discounted row, is its gain g. Each model's weight is
  g+ / (c + the sum of every g+), the base's c / (c + the sum): c, nats per
  row, is the prior's evidence for the base, so a model must prove a gain
  to earn weight and weighs in proportion to it. The forecast's linear
  predictor is the base's plus each model's increment over its parent
  times its weight (a log pool, residual layers ordered by evidence); with
  :cap (the default), taken strongest first, no increment moves it further
  than the largest of the stronger ones. It starts on the base.

  :weighting {:rule :synth :discount d :var v} is a simple dynamic
  predictive synthesis (after McAlinn & West 2019): a dynamic model of
  the same kind whose inputs are an intercept and each model's predictive
  linear predictor, its coefficients starting at 1/n (an even log pool)
  with variance v (the intercept's 1) and discounted by d. Its intercept
  and the coefficients' sum calibrate the pool; the coefficients are the
  pool's time-varying weights (`weights`: their positive parts, shares).

  :evidence-at :issue scores each row, for every rule, by the predictive
  each model issued for it (the row's :issued, from `issued` at the
  forecast origin) rather than by the model's state when the row becomes
  known (:learn, the default). Where outcomes are learned long after
  their forecast, a model scored at learn time has meanwhile learned the
  rows around it, and its evidence is hindsight fit, not forecast skill
  (dynamic model averaging's evidence is the one-step-ahead predictive,
  Raftery, Karny & Ettler 2010); every row then needs its :issued.

  :evidence-at :sequential scores a period's rows at learn time each
  given the period's rows before it (`sequential-score`), as a linear
  model's evidence always is: the log marginal of the period's rows with
  the coefficients' uncertainty, and any effect the rows share, counted
  once for the period. Scored independently, each row integrates the
  coefficients on its own, so a model unsure of an input that differs
  across a period's many rows is paid or rewarded for that uncertainty
  once per row.

  :evidence-at :issue-joint scores what each model issued, jointly: per
  origin (the rows carrying one :issue-state, from `issue-state` when
  their forecasts were issued), the log marginal of the origin's rows
  under the model's coefficients as they stood then and any effect the
  rows share as issued, integrated over all the rows together
  (`joint-score`). So the evidence is the issued forecast's prequential
  score, with no hindsight (:issue) and no uncertainty paid once per row
  (:sequential, but at issue). Rows need :issued and :issue-state. The
  :stack and :synth rules, which read each row on its own, read :issued
  as under :issue.

  :prior {name share} starts the weights from those shares instead of
  equal ones, the models not named sharing the rest evenly, and the
  forgetting flattens the weights toward them (w^forget prior^(1 -
  forget)) rather than toward equal weights: where the models fit alike
  the weights stay at the prior, and evidence moves them off it as
  before."
  [models {:keys [forget weighting prior evidence-at]}]
  (need :averaged {:forget forget})
  (when-not (contains? #{nil :learn :issue :sequential :issue-joint} evidence-at)
    (throw (ex-info "augur.dynamic/averaged: :evidence-at must be :learn, :issue, :sequential or :issue-joint" {:evidence-at evidence-at})))
  (when-let [bad (seq (remove (set (keys models)) (keys prior)))]
    (throw (ex-info "augur.dynamic/averaged: :prior names a model that is not averaged" {:models (vec bad)})))
  (when (or (some #(neg? (double %)) (vals prior)) (> (reduce + 0.0 (vals prior)) 1.000001))
    (throw (ex-info "augur.dynamic/averaged: :prior's shares must be at least 0 and sum to at most 1" prior)))
  (when weighting
    (case (:rule weighting)
      :evidence (do (need :averaged {:base (:base weighting) :scale (:scale weighting)})
                    (when-not (and (contains? models (:base weighting)) (pos? (double (:scale weighting)))
                                   (every? #(and (contains? models (key %)) (contains? models (val %))) (:parents weighting)))
                      (throw (ex-info "augur.dynamic/averaged: :evidence needs a :base and :parents among the models and a positive :scale" weighting))))
      :synth (do (need :averaged {:discount (:discount weighting) :var (:var weighting)})
                 (when (some #(= :linear (:type %)) (vals models))
                   (throw (ex-info "augur.dynamic/averaged: :synth needs logistic or Poisson models" weighting))))
      :stack (need :averaged {:rate (:rate weighting) :floor (:floor weighting)})
      :temper (do (need :averaged {:power (:power weighting)})
                  (when-not (< 0.0 (double (:power weighting)) 1.000001)
                    (throw (ex-info "augur.dynamic/averaged: :temper's :power must be in (0, 1]" weighting))))
      (throw (ex-info "augur.dynamic/averaged: :weighting's :rule must be :stack, :temper, :evidence or :synth" weighting))))
  (let [kinds (set (map :type (vals models)))]
    (when (not= 1 (count kinds))
      (throw (ex-info "augur.dynamic/averaged needs models of one kind" {:kinds kinds})))
    (cond-> {:type :averaged :kind (first kinds) :models models :forget (double forget)
             :w (into {} (for [k (keys models)] [k (/ 1.0 (count models))]))}
      (= :issue evidence-at) (assoc :evidence-at :issue)
      (= :sequential evidence-at) (assoc :evidence-at :sequential)
      (= :issue-joint evidence-at) (assoc :evidence-at :issue-joint)
      (seq prior) (as-> avg (let [named (reduce + 0.0 (map double (vals prior)))
                                  others (- (count models) (count prior))
                                  pi (into {} (for [k (keys models)]
                                                [k (if (contains? prior k)
                                                     (double (prior k))
                                                     (/ (- 1.0 named) (double others)))]))]
                              (assoc avg :prior pi :w pi)))
      weighting (assoc :weighting (case (:rule weighting)
                                    :stack {:rule :stack :rate (double (:rate weighting)) :floor (double (:floor weighting))}
                                    :temper {:rule :temper :power (double (:power weighting))}
                                    :evidence {:rule :evidence :base (:base weighting) :scale (double (:scale weighting))
                                               :parents (or (:parents weighting) {}) :cap (get weighting :cap true)}
                                    :synth {:rule :synth :discount (double (:discount weighting)) :var (double (:var weighting))}))
      (= :evidence (:rule weighting))
      (assoc :ev (zipmap (keys models) (repeat 0.0)) :n 0.0 :gains (zipmap (keys models) (repeat 0.0))
             :w (into {} (for [k (keys models)] [k (if (= k (:base weighting)) 1.0 0.0)])))
      (= :synth (:rule weighting))
      (as-> avg (let [order (vec (sort-by str (keys models)))
                      n (count order)]
                  (assoc avg :order order
                         :synth (model (first kinds) {:mean (into [0.0] (repeat n (/ 1.0 n)))
                                                      :var (into [1.0] (repeat n (double (:var weighting))))
                                                      :discount (:discount weighting)})))))))

(defn weights "Each model's weight after the last period." [avg] (:w avg))

(defn issued
  "Each averaged model's predictive for inputs x and `offset` now, {name
  (predictive model x offset)}: what a row carries as :issued, recorded
  when its forecast is issued, for :evidence-at :issue."
  [{:keys [models]} x offset]
  (into {} (for [[k mdl] models] [k (predictive mdl x offset)])))

(defn- issued-score
  "The log probability of outcome y under an issued predictive (the
  linear predictor's moments, offset included), times the row's weight w."
  ^double [kind {:keys [f q s n]} y ^double w]
  (* w (case kind
         :logistic (let [p (m/expit (/ (double f) (Math/sqrt (+ 1.0 (/ (* Math/PI (double q)) 8.0)))))]
                     (Math/log (Math/max 1e-300 (if (event? y) p (- 1.0 p)))))
         :poisson (poisson-log-predictive (double f) (double q) (double y))
         :linear (linear-log-density (double y) (double f) (+ (double q) (double s)) (double n)))))

(defn- issued-of
  "A row's issued predictive for model k; throws when it has none."
  [row k]
  (or (get-in row [:issued k])
      (throw (ex-info "augur.dynamic: :evidence-at :issue needs each row's :issued predictive for every model"
                      {:model k :row (dissoc row :issued)}))))

(defn- issue? [avg] (contains? #{:issue :issue-joint} (:evidence-at avg)))

(defn- score-row!
  "The weighted log predictive of one row under (mm, c), c already
  discounted; the row is then learned (`update-row!`)."
  ^double [type ^doubles mm ^doubles c ^long k ^doubles x y ^double w ^double o]
  (let [[f q] (moments mm c k x o 1.0)
        f (double f) q (double q)
        lp (case type
             :logistic (let [p (m/expit (/ f (Math/sqrt (+ 1.0 (/ (* Math/PI q) 8.0)))))]
                         (Math/log (Math/max 1e-300 (if (event? y) p (- 1.0 p)))))
             :poisson (poisson-log-predictive f q (double y)))]
    (update-row! type mm c k x y w o)
    (* w lp)))

(defn sequential-score
  "The log probability of a period's `rows` under a logistic or Poisson
  state, each row given the period's rows before it: the state discounted
  as `step` does, then each row scored by its predictive and learned. Rows
  sharing an effect (:shared, as `step`) are scored and learned with the
  state augmented by it, which is integrated out after them. The chain
  rule's log marginal of the period's rows, so the coefficients'
  uncertainty and a shared effect are each counted once for the period,
  not once per row (a linear model's evidence, `linear-step`, is already
  this). Rows in the order given."
  [{:keys [type k] :as state} rows]
  (when-not (#{:logistic :poisson} type)
    (throw (ex-info "augur.dynamic/sequential-score: a logistic or Poisson model" {:type type})))
  (let [k (long k)
        [mm c] (discounted state)]
    (reduce
     + 0.0
     (for [[eff rs] (shared-groups rows)]
       (if (nil? eff)
         (reduce + 0.0 (for [{:keys [x y w o]} rs]
                         (score-row! type mm c k (pick state x) y (double (or w 1.0)) (double (or o 0.0)))))
         (let [[_ mean var] eff
               vec? (sequential? mean)
               d (if vec? (count mean) 1)
               kd (+ k d)
               m1 (double-array kd)
               c1 (double-array (* kd kd))]
           (dotimes [i k] (aset m1 i (aget ^doubles mm i)))
           (dotimes [i k] (dotimes [j k] (aset c1 (+ (* i kd) j) (aget ^doubles c (+ (* i k) j)))))
           (dotimes [a d]
             (aset m1 (+ k a) (double (if vec? (nth mean a) mean)))
             (dotimes [b d] (aset c1 (+ (* (+ k a) kd) (+ k b)) (double (if vec? (nth (nth var a) b) var)))))
           (let [ll (reduce + 0.0 (for [{:keys [x y w o load]} rs]
                                    (let [x0 (pick state x)
                                          x1 (double-array kd)]
                                      (dotimes [i k] (aset x1 i (aget ^doubles x0 i)))
                                      (dotimes [a d] (aset x1 (+ k a) (double (if vec? (nth load a) 1.0))))
                                      (score-row! type m1 c1 kd x1 y (double (or w 1.0)) (double (or o 0.0))))))]
             (dotimes [i k] (aset ^doubles mm i (aget m1 i)))
             (dotimes [i k] (dotimes [j k] (aset ^doubles c (+ (* i k) j) (aget c1 (+ (* i kd) j)))))
             ll)))))))

(defn- chol-psd!
  "Overwrites the p x p symmetric positive semi-definite `a` (row major)
  with its lower Cholesky factor L, a L L' = a. A pivot at or below 1e-13
  of the largest diagonal (a direction of no variance) zeroes its column,
  so a singular a is factored rather than refused."
  [^doubles a ^long p]
  (let [top (loop [i 0 mx 0.0] (if (< i p) (recur (inc i) (Math/max mx (aget a (+ (* i p) i)))) mx))
        eps (* 1e-13 (Math/max 1e-300 top))]
    (dotimes [j p]
      (let [d (loop [s 0 acc (aget a (+ (* j p) j))]
                (if (< s j) (recur (inc s) (- acc (* (aget a (+ (* j p) s)) (aget a (+ (* j p) s))))) acc))]
        (if (> d eps)
          (let [ljj (Math/sqrt d)]
            (aset a (+ (* j p) j) ljj)
            (loop [i (inc j)]
              (when (< i p)
                (let [v (loop [s 0 acc (aget a (+ (* i p) j))]
                          (if (< s j) (recur (inc s) (- acc (* (aget a (+ (* i p) s)) (aget a (+ (* j p) s))))) acc))]
                  (aset a (+ (* i p) j) (/ v ljj)))
                (recur (inc i)))))
          (loop [i j] (when (< i p) (aset a (+ (* i p) j) 0.0) (recur (inc i))))))
      ;; the upper triangle zeroed
      (loop [i 0] (when (< i j) (aset a (+ (* i p) j) 0.0) (recur (inc i)))))
    a))

(defn- spd-solve-logdet
  "[x log|a|] for a symmetric positive definite p x p `a` (row major, left
  unchanged) and b."
  [^doubles a ^long p ^doubles b]
  (let [l (chol-psd! (aclone a) p)
        y (double-array p)
        x (double-array p)]
    (dotimes [i p]
      (aset y i (/ (loop [s 0 acc (aget b i)] (if (< s i) (recur (inc s) (- acc (* (aget l (+ (* i p) s)) (aget y s)))) acc))
                   (aget l (+ (* i p) i)))))
    (loop [i (dec p)]
      (when (>= i 0)
        (aset x i (/ (loop [s (inc i) acc (aget y i)] (if (< s p) (recur (inc s) (- acc (* (aget l (+ (* s p) i)) (aget x s)))) acc))
                     (aget l (+ (* i p) i))))
        (recur (dec i))))
    [x (* 2.0 (loop [i 0 acc 0.0] (if (< i p) (recur (inc i) (+ acc (Math/log (aget l (+ (* i p) i))))) acc)))]))

(defn- row-terms
  "[log p, d/deta, -d2/deta2] of one row's outcome y at linear predictor
  eta (a logistic or Poisson likelihood), times its weight w."
  [type ^double eta y ^double w]
  (case type
    :logistic (let [yy (if (event? y) 1.0 0.0)
                    p (m/expit eta)]
                [(loglik :logistic eta y w) (* w (- yy p)) (* w p (- 1.0 p))])
    :poisson (let [mu (Math/exp (Math/min 700.0 eta))
                   yy (double y)]
               [(* w (- (* yy eta) mu (m/log-gamma (+ 1.0 yy)))) (* w (- yy mu)) (* w mu)])))

(defn joint-score
  "The log marginal probability of one origin's `rows` under the joint
  predictive a logistic or Poisson `state` issued for them: the
  coefficients N(m, C), discounted as `step` will (the state as it stood
  when the rows' forecasts were issued), and any shared effect each row
  carries (:shared [id mean var], with :load for one of several
  dimensions, as `step`), integrated jointly over all the rows, not row by
  row. One origin is then one draw of the coefficients and of each shared
  effect, so their uncertainty is paid or rewarded once for the origin
  however many rows it has.

  Laplace's approximation in the coefficients and the shared effects
  together, theta ~ N(mu, S) with S = diag(C, var...): with S = L L' and
  theta = mu + L v, the rows' log likelihood less v'v/2 is maximized in v
  by damped Newton, and the log marginal is its maximum less half log|I +
  Z'WZ| (Z the rows' inputs times L, W the likelihood's curvature at the
  mode). It costs a p x p system a step, p the coefficients and shared
  dimensions, whatever the rows; a singular S (a dimension known exactly)
  needs no inverse.

  Rows {:x :y :w :o}, and the issued moments :f :q of each row's linear
  predictor (offset included, shared effect excluded) when the forecast
  was issued at something other than x'm + o: each row's mean is then the
  one at which its predictive under the coefficients' variance x'Cx gives
  the probability (logistic, MacKay's probit approximation) or the
  expected count (Poisson) the issued moments do, so a row issued over a
  mixture of offsets (:q 0, :f the mixture's log odds) is scored at what
  was issued, and a row issued at x'm + o is scored at f exactly."
  [{:keys [type k] :as state} rows]
  (when-not (#{:logistic :poisson} type)
    (throw (ex-info "augur.dynamic/joint-score: a logistic or Poisson model" {:type type})))
  (let [rows (vec rows)
        n (count rows)
        k (long k)
        [mm c] (discounted state)
        ^doubles mm mm ^doubles c c
        ;; the shared effects: each id's block of dimensions after the coefficients
        groups (rest (shared-groups rows))
        blocks (loop [gs groups at k out {}]
                 (if-let [[[id mean var] _] (first gs)]
                   (let [d (if (sequential? mean) (count mean) 1)]
                     (recur (rest gs) (+ at d) (assoc out id [at d mean var])))
                   out))
        p (long (+ k (reduce + 0 (map (fn [[_ d]] d) (vals blocks)))))
        s (double-array (* p p))]
    (dotimes [i k] (dotimes [j k] (aset s (+ (* i p) j) (aget c (+ (* i k) j)))))
    (doseq [[_ [at d _ var]] blocks]
      (let [at (long at) d (long d)]
        (dotimes [a d]
          (dotimes [b d]
            (aset s (+ (* (+ at a) p) (+ at b))
                  (double (if (sequential? var) (nth (nth var a) b) var)))))))
    (let [l (chol-psd! s p)
          z (double-array (* n p))
          big-f (double-array n)
          ws (double-array n)]
      (dotimes [r n]
        (let [{:keys [x o w f q shared load]} (nth rows r)
              x0 (pick state x)
              xt (double-array p)
              _ (dotimes [i k] (aset xt i (aget ^doubles x0 i)))
              shift (if shared
                      (let [[at d mean] (blocks (first shared))
                            at (long at) d (long d)]
                        (loop [a 0 acc 0.0]
                          (if (< a d)
                            (let [ld (if (sequential? mean) (double (nth load a)) 1.0)]
                              (aset xt (+ at a) ld)
                              (recur (inc a) (+ acc (* ld (double (if (sequential? mean) (nth mean a) mean))))))
                            acc)))
                      0.0)
              [f0 qc] (moments mm c k x0 (double (or o 0.0)) 1.0)
              f0 (double f0) qc (double qc)
              fr (if (nil? f)
                   f0
                   (let [f (double f) q (double (or q qc))]
                     (if (<= (Math/abs (- q qc)) (* 1e-9 (+ 1.0 q)))
                       f
                       (case type
                         :logistic (* f (Math/sqrt (/ (+ 1.0 (/ (* Math/PI qc) 8.0)) (+ 1.0 (/ (* Math/PI q) 8.0)))))
                         :poisson (+ f (* 0.5 (- q qc)))))))]
          (aset big-f r (+ (double fr) (double shift)))
          (aset ws r (double (or w 1.0)))
          ;; z_r = L' xt
          (dotimes [a p]
            (aset z (+ (* r p) a)
                  (loop [b a acc 0.0] (if (< b p) (recur (inc b) (+ acc (* (aget xt b) (aget l (+ (* b p) a))))) acc))))))
      (let [eval-at (fn [^doubles v]
                      ;; [log lik less v'v/2, its gradient in v, I + Z'WZ]
                      (let [g (double-array p)
                            a (double-array (* p p))]
                        (dotimes [i p] (aset a (+ (* i p) i) 1.0))
                        (let [ll (loop [r 0 acc 0.0]
                                   (if (< r n)
                                     (let [off (* r p)
                                           eta (+ (aget big-f r) (loop [i 0 e 0.0] (if (< i p) (recur (inc i) (+ e (* (aget z (+ off i)) (aget v i)))) e)))
                                           [lr gr hr] (row-terms type eta (:y (nth rows r)) (aget ws r))
                                           gr (double gr) hr (double hr)]
                                       (dotimes [i p]
                                         (let [zi (aget z (+ off i))]
                                           (when-not (zero? zi)
                                             (aset g i (+ (aget g i) (* gr zi)))
                                             (let [hz (* hr zi)]
                                               (loop [j 0]
                                                 (when (<= j i)
                                                   (aset a (+ (* i p) j) (+ (aget a (+ (* i p) j)) (* hz (aget z (+ off j)))))
                                                   (recur (inc j))))))))
                                       (recur (inc r) (+ acc (double lr))))
                                     acc))]
                          (dotimes [i p]
                            (aset g i (- (aget g i) (aget v i)))
                            (loop [j 0] (when (< j i) (aset a (+ (* j p) i) (aget a (+ (* i p) j))) (recur (inc j)))))
                          [(- ll (* 0.5 (loop [i 0 acc 0.0] (if (< i p) (recur (inc i) (+ acc (* (aget v i) (aget v i)))) acc)))) g a])))
            obj (fn [^doubles v]
                  (- (loop [r 0 acc 0.0]
                       (if (< r n)
                         (let [off (* r p)
                               eta (+ (aget big-f r) (loop [i 0 e 0.0] (if (< i p) (recur (inc i) (+ e (* (aget z (+ off i)) (aget v i)))) e)))]
                           (recur (inc r) (+ acc (double (first (row-terms type eta (:y (nth rows r)) (aget ws r)))))))
                         acc))
                     (* 0.5 (loop [i 0 acc 0.0] (if (< i p) (recur (inc i) (+ acc (* (aget v i) (aget v i)))) acc)))))
            ;; damped Newton: each step cut to a length of 5 and halved until
            ;; the objective does not fall (it is concave, so this converges)
            v (loop [it 0 v (double-array p)]
                (let [[f0 g a] (eval-at v)
                      [dx _] (spd-solve-logdet a p g)
                      ^doubles dx dx
                      big (loop [i 0 mx 0.0] (if (< i p) (recur (inc i) (Math/max mx (Math/abs (aget dx i)))) mx))
                      cut (if (> big 5.0) (/ 5.0 big) 1.0)
                      f0 (double f0)
                      tol (* 1e-12 (+ 1.0 (Math/abs f0)))
                      step (loop [t cut]
                             (let [v' (double-array p)]
                               (dotimes [i p] (aset v' i (+ (aget v i) (* t (aget dx i)))))
                               (cond (>= (double (obj v')) (- f0 tol)) [t v']
                                     (< t 1e-8) nil
                                     :else (recur (* 0.5 t)))))]
                  (if-let [[t v'] step]
                    (if (or (>= it 100) (< (* (double t) big) 1e-10)) v' (recur (inc it) v'))
                    v)))
            [f* _ a] (eval-at v)
            [_ logdet] (spd-solve-logdet a p (double-array p))]
        (- (double f*) (* 0.5 (double logdet)))))))

(defn issue-state
  "What an average's models are when a forecast is issued, {name state}:
  each row carries it as :issue-state, beside its :issued predictives,
  for :evidence-at :issue-joint (the same value for every row of an
  origin)."
  [avg]
  (:models avg))

(defn- by-origin
  "`rows` grouped by their :issue-state, one group per state (compared by
  identity), in order of first appearance."
  [rows]
  (reduce (fn [groups r]
            (let [st (or (:issue-state r)
                         (throw (ex-info "augur.dynamic: :evidence-at :issue-joint needs each row's :issue-state" {:row (dissoc r :issued)})))
                  i (first (keep-indexed (fn [i [s _]] (when (identical? s st) i)) groups))]
              (if i (update-in groups [i 1] conj r) (conj groups [st [r]]))))
          [] rows))

(defn- joint-evidence
  "Each model's evidence under :evidence-at :issue-joint: per origin, the
  joint log marginal of its rows under the state the model issued them
  from, each row's mean at what it issued."
  [{:keys [models]} rows]
  (let [origins (by-origin rows)]
    (into {} (for [k (keys models)]
               [k (reduce + 0.0 (for [[st rs] origins]
                                  (joint-score (or (get st k) (throw (ex-info "augur.dynamic: an :issue-state without the model" {:model k})))
                                               (for [r rs] (let [{:keys [f q]} (issued-of r k)] (assoc r :f f :q q))))))]))))

(defn- learn-score
  "A model's evidence from a period's rows at learn time: each row given
  the ones before it under :evidence-at :sequential (a logistic or
  Poisson model), else `log-score`."
  [avg mdl rows]
  (if (and (= :sequential (:evidence-at avg)) (#{:logistic :poisson} (:type mdl)))
    (sequential-score mdl rows)
    (log-score mdl rows)))

(defn evidence
  "Each averaged model's evidence from `rows` before it learns them, {name
  log probability}: the log score of each row by the predictive the model
  issued for it (:evidence-at :issue), else by its state now (a linear
  model's sequentially, each row given the ones before it, and under
  :evidence-at :sequential every model's). What the averaging weighs the
  models by."
  [{:keys [models kind] :as avg} rows]
  (cond
    (= :issue-joint (:evidence-at avg))
    (joint-evidence avg rows)
    (issue? avg)
    (into {} (for [k (keys models)]
                [k (if (some :shared rows)
                     (shared-score kind rows
                                   #(issued-score kind (issued-of % k) (:y %) (double (or (:w %) 1.0)))
                                   #(select-keys (issued-of % k) [:f :q]))
                     (reduce + 0.0 (for [r rows] (issued-score kind (issued-of r k) (:y r) (double (or (:w r) 1.0))))))]))
    :else
    (into {} (for [[k mdl] models]
               [k (if (= :linear (:type mdl))
                    (let [[mm c] (discounted mdl)] (second (linear-step mdl mm c rows)))
                    (learn-score avg mdl rows))]))))



(defn- from-moments
  "The predictive mean of an outcome whose linear predictor is N(f, q)."
  ^double [kind ^double f ^double q]
  (case kind
    :logistic (m/expit (/ f (Math/sqrt (+ 1.0 (/ (* Math/PI q) 8.0)))))
    :poisson (Math/exp (+ f (* 0.5 q)))
    :linear f))

(defn- pooled? [avg] (#{:evidence :synth} (:rule (:weighting avg))))

(defn- predict-averaged [{:keys [models w kind] :as avg} x offset]
  (if (pooled? avg)
    (let [{:keys [f q]} (pooled-moments avg x offset)] (from-moments kind f q))
    (reduce + (for [[k mdl] models] (* (w k) (predict mdl x offset))))))

(defn- synth-inputs
  "The synthesis' inputs for x: an intercept and each model's predictive
  moments, in the synthesis' order. [inputs qs]"
  [{:keys [models order]} x offset]
  (let [ps (mapv #(predictive (models %) x offset) order)]
    [(into [1.0] (map :f ps)) (mapv :q ps)]))

(defn- pooled-moments
  "The linear predictor's moments under the :evidence or :synth rule."
  [{:keys [models w weighting gains synth] :as avg} x offset]
  (case (:rule weighting)
    :evidence
    (let [{:keys [base parents cap]} weighting
          pm (into {} (for [[k mdl] models] [k (predictive mdl x offset)]))
          parent (fn [k] (get parents k base))
          order (sort-by (fn [k] [(- (double (gains k 0.0))) (str k)])
                         (filter #(and (not= base %) (pos? (double (w % 0.0)))) (keys models)))
          [f _] (reduce (fn [[f top] k]
                          (let [c (* (double (w k)) (- (double (:f (pm k))) (double (:f (pm (parent k))))))
                                c (if (and cap top) (Math/max (- (double top)) (Math/min (double top) c)) c)]
                            [(+ f c) (Math/max (double (or top 0.0)) (Math/abs c))]))
                        [(double (:f (pm base))) nil] order)]
      {:f f :q (reduce + 0.0 (for [[k v] w] (* (double v) (double (:q (pm k))))))})
    :synth
    (let [[xs qs] (synth-inputs avg x offset)
          theta (:m synth)
          {:keys [f q]} (predictive synth xs 0.0)]
      {:f f :q (+ (double q) (reduce + 0.0 (map-indexed (fn [i qk] (let [t (aget ^doubles theta (inc i))] (* t t (double qk)))) qs)))})))

(defn- flattened
  "The weights `w` flattened by `forget`, toward the prior's shares when
  there is one, else toward equal weights; normalized."
  [w forget prior]
  (let [flat (if prior
               (into {} (for [[k v] w] [k (* (Math/pow v forget) (Math/pow (prior k) (- 1.0 forget)))]))
               (into {} (for [[k v] w] [k (Math/pow v forget)])))
        z (reduce + (vals flat))]
    (update-vals flat #(/ % z))))

(defn- stacked-weights
  "The weights after `rows` by exponentiated gradient on the mixture's log
  score, each model's predictive from its state before the period, then
  floored."
  [{:keys [models kind] :as avg} w {:keys [rate floor]} rows]
  (let [ks (vec (keys models))
        n (count ks)
        w (loop [rows (seq rows) w w]
            (if-not rows
              w
              (let [r (first rows)
                    r1 (dissoc r :w)
                    ps (into {} (for [k ks] [k (Math/exp (if (issue? avg)
                                                          (issued-score kind (issued-of r k) (:y r) 1.0)
                                                          (log-score (models k) [r1])))]))
                    mix (Math/max 1e-300 (reduce + (for [k ks] (* (w k) (ps k)))))
                    eta (* rate (double (or (:w r) 1.0)))
                    ;; centred on the best ratio so the exponent stays bounded
                    top (apply max (for [k ks] (/ (ps k) mix)))
                    w (into {} (for [k ks] [k (* (w k) (Math/exp (* eta (- (/ (ps k) mix) top))))]))
                    z (reduce + (vals w))]
                (recur (next rows) (update-vals w #(/ % z))))))]
    (update-vals w #(+ (* (- 1.0 floor) %) (/ floor n)))))

(defn- evidence-weights
  "The :evidence rule's gains and weights from the discounted log scores
  `ev` over `n` discounted rows."
  [ev n {:keys [base scale parents]}]
  (let [gains (into {} (for [[k e] ev :when (not= k base)]
                         [k (if (pos? n) (/ (- (double e) (double (ev (get parents k base)))) n) 0.0)]))
        pos (update-vals gains #(Math/max 0.0 (double %)))
        z (+ (double scale) (reduce + 0.0 (vals pos)))]
    [gains (assoc (update-vals pos #(/ % z)) base (/ (double scale) z))]))

(defn- step-evidence [{:keys [models forget weighting ev n] :as avg} rows]
  (let [ll (if (issue? avg) (evidence avg rows) (into {} (for [[k mdl] models] [k (learn-score avg mdl rows)])))
        ev (into {} (for [[k e] ev] [k (+ (* forget (double e)) (double (ll k)))]))
        n (+ (* forget (double n)) (reduce + 0.0 (map #(double (or (:w %) 1.0)) rows)))
        [gains w] (evidence-weights ev n weighting)]
    (assoc avg :ev ev :n n :gains gains :w w
           :models (into {} (for [[k mdl] models] [k (step mdl rows)])))))

(defn- step-synth [{:keys [models order synth] :as avg} rows]
  (let [srows (vec (for [{:keys [x y w o] :as r} rows]
                     (cond-> {:x (if (issue? avg)
                                   ;; the models' linear predictors as issued
                                   (into [1.0] (map #(:f (issued-of r %)) order))
                                   (first (synth-inputs avg x (double (or o 0.0)))))
                              :y y}
                       w (assoc :w w))))
        synth (step synth srows)
        theta (mapv #(Math/max 0.0 (double %)) (rest (means synth)))
        z (reduce + 0.0 theta)]
    (assoc avg :synth synth
           :w (zipmap order (if (pos? z) (map #(/ % z) theta) (repeat (/ 1.0 (count order)))))
           :models (into {} (for [[k mdl] models] [k (step mdl rows)])))))

(defn- step-averaged [{:keys [models w forget weighting] :as avg} rows]
  (case (:rule weighting)
    :evidence (step-evidence avg rows)
    :synth (step-synth avg rows)
    (step-averaged* avg rows)))

(defn- step-averaged* [{:keys [models w forget weighting] :as avg} rows]
  (if (= :stack (:rule weighting))
    (let [flat (flattened w forget (:prior avg))]
      (assoc avg
             :w (stacked-weights avg flat weighting rows)
             :models (into {} (for [[k mdl] models] [k (step mdl rows)]))))
    (step-posterior avg rows)))

(defn- step-posterior [{:keys [models w forget] :as avg} rows]
  (let [prior (flattened w forget (:prior avg))
        ;; each model's evidence: how likely it made the period's outcomes
        ;; a linear model's is exact and sequential: each row given the
        ;; ones before it (its step computes it)
        stepped (if (issue? avg)
                  (into {} (for [[k e] (evidence avg rows)] [k [nil e]]))
                  (into {} (for [[k mdl] models]
                             [k (if (= :linear (:type mdl))
                                  (let [[mm c] (discounted mdl)] (linear-step mdl mm c rows))
                                  [nil (learn-score avg mdl rows)])])))
        power (if (= :temper (:rule (:weighting avg))) (:power (:weighting avg)) 1.0)
        ll (update-vals stepped #(* power (second %)))
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
