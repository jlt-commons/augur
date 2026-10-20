(ns augur.changepoint
  "Bayesian online changepoint detection (Adams & MacKay 2007) for a series
  whose level shifts: after each observation, the posterior over the run
  length since the last break, with a constant break hazard and a
  Normal-Gamma prior on each run's mean and precision, so a run's
  predictive is a Student t. For each step: the chance the level broke in
  the last `:recent` steps, and the expected run length.

  Incremental: `init` gives a state, `step` adds one observation and
  returns the new state with that step's reading. The state is EDN, and
  bounded: the posterior keeps run lengths below `:max-run` only. `run`
  steps through a whole series.

  Options, all required:
    :hazard     the chance of a break at each step
    :recent     how many steps back counts as a recent break
    :prior-var  the prior's scale for a run's variance (1 for standardized
                inputs)
    :max-run    the number of run lengths tracked"
  (:require [augur.math :as m]))

(defn- log-t
  "Log density of a Student t at x: location mu, squared scale s2, dof nu."
  ^double [^double x ^double mu ^double s2 ^double nu]
  (let [z (/ (* (- x mu) (- x mu)) (* nu s2))]
    (- (+ (m/log-gamma (/ (+ nu 1.0) 2.0)) (* -0.5 (Math/log (* nu Math/PI s2))))
       (+ (m/log-gamma (/ nu 2.0)) (* (/ (+ nu 1.0) 2.0) (Math/log1p z))))))

(defn init
  "A detector before any observation. The prior is centred on the first
  value it sees, with the series' own spread unknown (kappa 1, alpha 1,
  beta from :prior-var)."
  [{:keys [hazard recent prior-var max-run] :as opts}]
  (when-let [missing (seq (remove #(some? (get opts %)) [:hazard :recent :prior-var :max-run]))]
    (throw (ex-info (str "augur.changepoint/init needs " (vec missing)) {:missing (vec missing)})))
  {:hazard (double hazard) :recent (long recent) :prior-var (double prior-var) :max-run (long max-run)
   :mu0 nil :ps [] :mus [] :kappas [] :alphas [] :betas []})

(defn step
  "[state' {:recent :run}]: the detector after observing x, with the chance
  of a break in the last :recent steps and the expected run length."
  [{:keys [hazard recent prior-var max-run] :as state} x]
  (let [x (double x)
        h (double hazard)
        {:keys [mu0 ps mus kappas alphas betas]}
        (if (nil? (:mu0 state))
          (assoc state :mu0 x :ps [1.0] :mus [x] :kappas [1.0] :alphas [1.0] :betas [(double prior-var)])
          state)
        n (count ps)
        pred (mapv (fn [i]
                     (let [mu (mus i) k (kappas i) a (alphas i) b (betas i)]
                       (Math/exp (log-t x mu (/ (* b (+ k 1.0)) (* a k)) (* 2.0 a)))))
                   (range n))
        grow (mapv (fn [i] (* (ps i) (pred i) (- 1.0 h))) (range n))
        brk (reduce + (map (fn [i] (* (ps i) (pred i) h)) (range n)))
        joint (into [brk] grow)
        z (reduce + joint)
        post (if (pos? z) (mapv #(/ % z) joint) (vec (cons 1.0 (repeat n 0.0))))
        ;; each run's parameters after x; a new run starts from the prior
        upd (fn [i] (let [mu (mus i) k (kappas i) a (alphas i) b (betas i)]
                      [(/ (+ (* k mu) x) (+ k 1.0)) (+ k 1.0) (+ a 0.5)
                       (+ b (/ (* k (- x mu) (- x mu)) (* 2.0 (+ k 1.0))))]))
        grown (mapv upd (range n))
        keep (min (inc n) max-run)
        post (subvec post 0 keep)
        s (reduce + post)
        post (mapv #(/ % s) post)
        params (vec (take keep (cons [mu0 1.0 1.0 (double prior-var)] grown)))]
    [(assoc state :mu0 mu0 :ps post
            :mus (mapv #(% 0) params) :kappas (mapv #(% 1) params)
            :alphas (mapv #(% 2) params) :betas (mapv #(% 3) params))
     {:recent (reduce + (take (inc recent) post))
      :run (reduce + (map-indexed (fn [r p] (* r p)) post))}]))

(defn run
  "[{:recent :run}] for each value of `xs`, one detector stepped through
  them (options as `init`)."
  [xs opts]
  (loop [xs (seq xs) state (init opts) out (transient [])]
    (if xs
      (let [[state r] (step state (first xs))]
        (recur (next xs) state (conj! out r)))
      (persistent! out))))
