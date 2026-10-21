(ns augur.hier
  "Hierarchical effects over a tree of units (say countries in subregions in
  regions in the world), each unit's effect the sum of those of the nodes
  on its path, so a unit with little data of its own is pulled toward its
  parents.

  `intercepts`: static nested random intercepts for a logistic model whose
  other terms are a fixed offset per row (Gelman & Hill 2007, ch. 12-13):
  each level's effects Gaussian around zero with a variance estimated from
  the data (empirical Bayes, Laplace approximation), so the data say how
  much a country differs from its subregion and a subregion from its
  region.

  `states`: dynamic effects for counts, a latent AR(1) state per node and
  period on the log scale of a Poisson rate, e.g. the onsets that came
  against those a model expected (the dynamic frailty of Duffie, Eckner,
  Horel & Saita 2009, over a tree). A node's evidence is its leaves' counts
  summed (bottom up) and its state scales the rates of every leaf under it
  (top down). The states are filtered by a Kalman filter on the Laplace
  approximation (Fahrmeir & Wagenpfeil 1997), and each level's persistence
  and size chosen from grids by the filter's own predictive likelihood of
  the counts at the horizon the states are used at (prequential, Dawid
  1984): a level whose states
  do not help forecast the next period's counts gets size zero and drops
  out. EM for these variances stalls where a level is weakly identified,
  which the grid does not. States are defined so the multiplier
  exp(state) has mean one in the long run: a forecast that was right on
  average stays right on average, and a state only moves it while the
  evidence says its node is running hot or cold.

  Paths list node ids from the top down to the leaf, one per level."
  (:require [augur.math :as m]))

(set! *unchecked-math* true)

;; --- static nested intercepts ---------------------------------------------

(defn effect
  "The sum of the effects of the nodes on `path` (a node never fit adds 0,
  so a new unit takes its parents' effect)."
  ^double [fit path]
  (let [fx (:effects fit)]
    (reduce + 0.0 (map #(double (get fx % 0.0)) path))))

(defn- softplus ^double [^double e]
  (if (> e 0.0) (+ e (Math/log1p (Math/exp (- e)))) (Math/log1p (Math/exp e))))

(defn intercepts
  "Nested random intercepts for logistic rows {:path :o :y :w}: o a fixed
  offset (the rest of the model's log odds), path the row's nodes from the
  top. Each node's effect ~ N(0, tau2 of its level); effects found by
  coordinate Newton steps (backfitting), and after each sweep each level's
  tau2 set to the mean of its effects' squares and posterior variances
  (the Laplace EM step), unless :fixed. Options: :tau2, one starting
  variance per level, and :sweeps, both required; :start, effects to start
  from; :sd-prior, one scale per level of a half-normal prior on the
  level's sd (Gelman 2006), which makes the variance step a posterior mode
  and keeps nodes whose rows all go one way from inflating it. {:effects {id effect} :tau2 [...]}."
  [rows {:keys [tau2 sweeps fixed start sd-prior]}]
  (when-not (and tau2 sweeps) (throw (ex-info "augur.hier/intercepts needs :tau2 and :sweeps" {})))
  (when (and sd-prior (not= (count sd-prior) (count tau2)))
    (throw (ex-info "augur.hier/intercepts needs one :sd-prior scale per level" {:sd-prior sd-prior :tau2 tau2})))
  (let [rows (vec rows)
        n (count rows)
        L (count tau2)
        ^doubles lin (double-array n)
        ^doubles ys (double-array n)
        ^doubles ws (double-array n)
        ;; per level, node -> the indices of its rows
        members (vec (for [l (range L)]
                       (let [acc (java.util.HashMap.)]
                         (dotimes [r n]
                           (when-let [id (nth (:path (rows r)) l nil)]
                             (.add ^java.util.ArrayList (or (.get acc id) (let [a (java.util.ArrayList.)] (.put acc id a) a)) r)))
                         (into {} (for [[id rs] acc] [id (int-array rs)])))))
        fx (java.util.HashMap.)
        vars (java.util.HashMap.)]
    (dotimes [r n]
      (let [{:keys [o y w path]} (rows r)]
        (aset ys r (if y 1.0 0.0))
        (aset ws r (double w))
        (aset lin r (+ (double (or o 0.0)) (reduce + 0.0 (map #(double (get start % 0.0)) path))))))
    (doseq [l (range L) id (keys (members l))] (.put fx id (double (get start id 0.0))))
    (loop [sweep 0 tau2 (mapv double tau2)]
      (if (>= sweep sweeps)
        {:effects (into {} fx) :tau2 tau2}
        (do
          (doseq [l (range L) [id ^ints rs] (members l)]
            (let [prec (/ 1.0 (double (tau2 l)))
                  a0 (double (.get fx id))]
              ;; damped Newton: each step halved until the node's penalized
              ;; log likelihood improves (a node with few rows has little
              ;; curvature, where full steps overshoot and oscillate)
              (let [objective (fn ^double [^double a]
                                (loop [j 0 acc (* -0.5 prec a a)]
                                  (if (< j (alength rs))
                                    (let [r (aget rs j)
                                          e (+ (aget lin r) (- a a0))]
                                      (recur (inc j) (+ acc (* (aget ws r) (- (* (aget ys r) e) (softplus e))))))
                                    acc)))]
                (loop [i 0 a a0 obj (objective a0)]
                  (let [[g h] (loop [j 0 g (- (* prec a)) h prec]
                                (if (< j (alength rs))
                                  (let [r (aget rs j)
                                        p (m/expit (+ (aget lin r) (- a a0)))
                                        w (aget ws r)]
                                    (recur (inc j) (+ g (* w (- (aget ys r) p))) (+ h (* w p (- 1.0 p)))))
                                  [g h]))
                        step (/ (double g) (double h))
                        ;; no worse than rounding: near the mode a full step's
                        ;; gain is below the sum's precision
                        tol (* 1e-12 (+ 1.0 (Math/abs obj)))
                        [a' obj'] (loop [t 1.0 tries 0]
                                    (let [c (+ a (* t step)) o (objective c)]
                                      (if (or (>= o (- obj tol)) (>= tries 30)) [c o] (recur (* 0.5 t) (inc tries)))))
                        [a' obj'] (if (>= obj' (- obj tol)) [a' obj'] [a obj])]
                    (if (and (< i 50) (> (Math/abs (- (double a') a)) 1e-9))
                      (recur (inc i) a' obj')
                      (let [h (loop [j 0 h prec]
                                (if (< j (alength rs))
                                  (let [r (aget rs j) p (m/expit (+ (aget lin r) (- (double a') a0)))]
                                    (recur (inc j) (+ h (* (aget ws r) p (- 1.0 p)))))
                                  h))]
                        (.put fx id a')
                        (.put vars id (/ 1.0 h))
                        (let [d (- (double a') a0)]
                          (dotimes [j (alength rs)] (let [r (aget rs j)] (aset lin r (+ (aget lin r) d))))))))))))
          (recur (inc sweep)
                 (if fixed
                   tau2
                   (vec (for [l (range L)]
                          (let [ids (keys (members l))]
                            (if (empty? ids)
                              (tau2 l)
                              (let [ss (reduce + (map #(let [a (double (.get fx %))] (+ (* a a) (double (.get vars %)))) ids))
                                    m (double (count ids))]
                                (max 1e-6
                                     (if sd-prior
                                       ;; the mode under a half-normal(0, s) prior on the sd:
                                       ;; v^2 + m s^2 v - ss s^2 = 0, ss/m as s grows
                                       (let [s2 (let [x (double (sd-prior l))] (* x x))]
                                         (/ (* 2.0 ss s2) (+ (* m s2) (Math/sqrt (+ (* m m s2 s2) (* 4.0 ss s2))))))
                                       (/ ss m)))))))))))))))

;; --- dynamic states --------------------------------------------------------

(defn- stationary ^double [^double rho ^double q] (/ q (max 1e-12 (- 1.0 (* rho rho)))))

(defn- run-filter
  "The forward pass: per node, arrays over t0..t1 of the filtered means and
  variances, and :ll, the sum over periods of the log predictive
  probability of the period's counts (less the log factorials) given the
  periods up to `horizon` before it, the states filtered then and run
  forward: the horizon a forecast made from these states looks ahead.
  It is taken jointly over the states the leaves share (scoring each leaf
  alone counts a shared state's uncertainty once per leaf, and favours
  large noisy states), by Laplace's approximation at the mode, its log
  determinant node by node. At each period every state is predicted, then
  each level in turn, from the top, is updated by Newton steps on its
  leaves' counts given the other levels (two Gauss-Seidel passes). A level
  of size zero stays at zero."
  [{:keys [rho q nodes ^ints lvl obs-at]} t0 t1 horizon]
  (let [T (inc (- (long t1) (long t0)))
        N (count nodes)
        L (count rho)
        h (long horizon)
        ^doubles rr (double-array rho)
        ^doubles qq (double-array q)
        ^doubles mf (double-array (* N T))
        ^doubles pf (double-array (* N T))
        ^doubles cur (double-array N)
        ^doubles cv (double-array N)
        ^doubles hv (double-array N)
        ^doubles st (double-array N)
        ;; per node, the sums of its leaves' counts and rates at a period
        ^doubles sy (double-array N)
        ^doubles se (double-array N)
        ^doubles m0 (double-array N)
        ^doubles p0 (double-array N)
        ;; the states as seen from `horizon` periods before, for scoring
        ^doubles hm (double-array N)
        ^doubles hp (double-array N)
        ^doubles tc (double-array N)
        ^doubles tv (double-array N)
        ll (double-array 1)
        ;; the mode of the states given a period's counts under prior (pm, pv),
        ;; into (sc, sv), starting from sc
        mode! (fn [[^objects paths ^doubles ys ^doubles es] ^doubles pm ^doubles pv ^doubles sc ^doubles sv]
                (let [n (alength ys)
                      lin (fn ^double [^ints path]
                            (loop [j 0 acc 0.0] (if (< j (alength path)) (let [v (aget path j)] (recur (inc j) (+ acc (- (aget sc v) (aget hv v))))) acc)))]
                  (dotimes [_ 2]
                    (dotimes [l L]
                      (when (pos? (aget qq l))
                        (java.util.Arrays/fill sy 0.0) (java.util.Arrays/fill se 0.0)
                        (dotimes [k n]
                          (let [^ints path (aget paths k)
                                v (aget path l)]
                            (aset sy v (+ (aget sy v) (aget ys k)))
                            ;; the rate without v's own state
                            (aset se v (+ (aget se v) (* (aget es k) (Math/exp (- (double (lin path)) (aget sc v))))))))
                        (dotimes [v N]
                          (when (and (= l (aget lvl v)) (pos? (aget se v)))
                            (let [y (aget sy v) e (aget se v) mm (aget pm v) pp (aget pv v)
                                  s (loop [k 0 s (aget sc v)]
                                      (let [r (* e (Math/exp s))
                                            g (- y r (/ (- s mm) pp))
                                            hh (+ r (/ 1.0 pp))
                                            s' (+ s (/ g hh))]
                                        (if (and (< k 20) (> (Math/abs (- s' s)) 1e-10)) (recur (inc k) s') s')))]
                              (aset sc v s)
                              (aset sv v (/ 1.0 (+ (* e (Math/exp s)) (/ 1.0 pp))))))))))))
        ;; the log predictive of a period's counts under prior (pm, pv), at its mode sc
        score (fn ^double [[^objects paths ^doubles ys ^doubles es] ^doubles pm ^doubles pv ^doubles sc]
                (java.util.Arrays/fill se 0.0)
                (let [n (alength ys)
                      acc (double-array 1)]
                  (dotimes [k n]
                    (let [^ints path (aget paths k)
                          mu (+ (Math/log (aget es k))
                                (loop [j 0 a 0.0] (if (< j (alength path)) (let [v (aget path j)] (recur (inc j) (+ a (- (aget sc v) (aget hv v))))) a)))
                          r (Math/exp mu)]
                      (aset acc 0 (+ (aget acc 0) (- (* (aget ys k) mu) r)))
                      (dotimes [j (alength path)]
                        (let [v (aget path j)] (aset se v (+ (aget se v) r))))))
                  (dotimes [v N]
                    (when (and (pos? (aget qq (aget lvl v))) (pos? (aget se v)))
                      (let [d (- (aget sc v) (aget pm v)) pp (aget pv v)]
                        (aset acc 0 (- (aget acc 0) (/ (* d d) (* 2.0 pp)) (* 0.5 (Math/log1p (* pp (aget se v)))))))))
                  (aget acc 0)))]
    (dotimes [v N]
      (let [l (aget lvl v) s (stationary (aget rr l) (aget qq l))]
        (aset st v s) (aset hv v (* 0.5 s)) (aset cv v s)))
    (dotimes [i T]
      (let [here (get obs-at (+ (long t0) i))]
        ;; predict
        (dotimes [v N]
          (let [l (aget lvl v) r (aget rr l)]
            (aset cur v (* r (aget cur v)))
            (aset cv v (+ (* r r (aget cv v)) (aget qq l)))
            (aset m0 v (aget cur v)) (aset p0 v (aget cv v))))
        (when here
          (if (= h 1)
            (do (mode! here m0 p0 cur cv)
                (aset ll 0 (+ (aget ll 0) (double (score here m0 p0 cur)))))
            (do
              ;; the states filtered `horizon` periods before, run forward
              (dotimes [v N]
                (let [r (aget rr (aget lvl v))
                      rh (Math/pow r h)
                      j (- i h)
                      [m p] (if (neg? j) [0.0 (aget st v)] [(aget mf (+ (* v T) j)) (aget pf (+ (* v T) j))])]
                  (aset hm v (* rh (double m)))
                  (aset hp v (+ (* rh rh (double p)) (* (aget st v) (- 1.0 (* rh rh)))))
                  (aset tc v (aget hm v)) (aset tv v (aget hp v))))
              (mode! here hm hp tc tv)
              (aset ll 0 (+ (aget ll 0) (double (score here hm hp tc))))
              (mode! here m0 p0 cur cv))))
        (dotimes [v N]
          (aset mf (+ (* v T) i) (aget cur v))
          (aset pf (+ (* v T) i) (aget cv v)))))
    {:mf (into {} (map-indexed (fn [v id] [id (java.util.Arrays/copyOfRange mf (int (* v T)) (int (* (inc v) T)))]) nodes))
     :pf (into {} (map-indexed (fn [v id] [id (java.util.Arrays/copyOfRange pf (int (* v T)) (int (* (inc v) T)))]) nodes))
     :ll (aget ll 0)}))

(defn- structure [obs paths]
  (let [nodes (vec (distinct (mapcat val paths)))
        idx (into {} (map-indexed (fn [i v] [v i]) nodes))
        level (into {} (for [[_ path] paths [l v] (map-indexed vector path)] [v l]))
        lvl (int-array (map level nodes))
        ipath (into {} (for [[leaf path] paths] [leaf (int-array (map idx path))]))
        by-t (reduce-kv (fn [acc [leaf t] [y e]]
                          (if (and (contains? paths leaf) (pos? (double e))) (update acc t (fnil conj []) [leaf y e]) acc))
                        {} obs)
        obs-at (into {} (for [[t xs] by-t]
                          [t [(object-array (map #(ipath (first %)) xs))
                              (double-array (map #(double (second %)) xs))
                              (double-array (map #(double (nth % 2)) xs))]]))]
    {:nodes nodes :level level :lvl lvl :paths paths :obs-at obs-at}))

(defn- q-of [rho sd] (mapv (fn [r s] (* (double s) (double s) (- 1.0 (* (double r) (double r))))) rho sd))

(defn states
  "Dynamic states for counts `obs` ({[leaf t] [count expected]}) over
  leaves with `paths` ({leaf [node ... leaf]}), periods t0..t1: the count
  is Poisson with mean expected x exp(sum of the path's states, each less
  half its level's stationary variance). Each state is AR(1) around zero,
  with its level's persistence rho and stationary sd. Each level's pair is
  chosen in turn, from the top, `:passes` times, from the grids `:rhos`
  and `:sds` (and the current pair), by the predictive log likelihood of
  the counts `:horizon` periods ahead (the horizon the states will be used
  at: choosing persistence on one step ahead favours states that chase
  the last wave), less :penalty for each level whose states have
  a size (2.0, Akaike's price of its two parameters, makes a level earn
  its place). Options, all required: :rho and :sd, the starting pairs per
  level, :rhos, :sds, :passes, :penalty and :horizon. Returns the fit, its
  :rho, :sd and :q (innovation variances) and :ll, filtered (every state
  uses only the periods up to its own)."
  [obs paths [t0 t1] {:keys [rho sd rhos sds passes penalty horizon]}]
  (when-not (and rho sd rhos sds passes penalty horizon)
    (throw (ex-info "augur.hier/states needs :rho :sd :rhos :sds :passes :penalty :horizon" {})))
  (let [st (structure obs paths)
        L (count rho)
        run (fn [rho sd] (let [cfg (assoc st :rho rho :sd sd :q (q-of rho sd) :horizon horizon)
                               f (run-filter cfg t0 t1 horizon)]
                           (merge cfg f {:t0 t0 :t1 t1
                                         :score (- (double (:ll f)) (* (double penalty) (count (filter pos? sd))))})))]
    (loop [pass 0 best (run (mapv double rho) (mapv double sd))]
      (if (>= pass passes)
        best
        (recur (inc pass)
               (reduce (fn [best l]
                         (reduce (fn [best [r s]]
                                   (let [cand (run (assoc (:rho best) l (double r)) (assoc (:sd best) l (double s)))]
                                     (if (> (double (:score cand)) (double (:score best))) cand best)))
                                 best (for [r rhos s sds] [r s])))
                       best (range L)))))))

(defn refilter
  "The filter of `fit`'s hyperparameters run over other counts."
  [fit obs [t0 t1]]
  (let [st (structure obs (:paths fit))
        cfg (merge st (select-keys fit [:rho :sd :q :horizon]))]
    (merge cfg (run-filter cfg t0 t1 (:horizon cfg 1)) {:t0 t0 :t1 t1})))

(defn state-mean
  "Node v's filtered state at t."
  ^double [fit v t]
  (aget ^doubles ((:mf fit) v) (int (- t (:t0 fit)))))

(defn- node-at
  "[mean var steps] of node v's state as last filtered at or before t, and
  how many periods after it t is; the stationary prior before t0 or for an
  unknown node."
  [fit v t]
  (let [{:keys [t0 t1 rho q level]} fit]
    (if-let [l (level v)]
      (let [vs (stationary (rho l) (q l))]
        (if (< t t0)
          [0.0 vs 0]
          (let [u (min t t1) i (int (- u t0))]
            [(aget ^doubles ((:mf fit) v) i) (aget ^doubles ((:pf fit) v) i) (- t u)])))
      nil)))

(defn multiplier
  "The expected scaling of a leaf's rate (path `path`) over periods
  t+k1..t+k2 given the states filtered by t: the mean over those periods
  of E exp(sum of the path's states less their half variances). One far
  from any evidence."
  ^double [fit path t [k1 k2]]
  (let [{:keys [rho q level]} fit
        at (keep (fn [v] (when-let [x (node-at fit v t)] (conj x v))) path)]
    (/ (reduce + (for [k (range k1 (inc k2))]
                   (Math/exp
                    (reduce + 0.0
                            (for [[mean var steps v] at
                                  :let [l (level v) r (double (rho l)) qq (double (q l))
                                        h (+ (long steps) (long k))
                                        rh (Math/pow r h)
                                        vs (stationary r qq)
                                        mk (* rh (double mean))
                                        vk (+ (* rh rh (double var)) (* vs (- 1.0 (* rh rh))))]]
                              (- (+ mk (* 0.5 vk)) (* 0.5 vs)))))))
       (double (inc (- k2 k1))))))

(defn adjust
  "A window probability p scaled by a multiplier of its hazard."
  ^double [^double p ^double mult]
  (- 1.0 (Math/pow (- 1.0 (min p (- 1.0 1e-12))) mult)))

(defn any-probability
  "The chance of at least one event among leaves `paths` ({leaf path}) with
  window probabilities `ps` ({leaf p}) over periods t+k1..t+k2, their
  shared states drawn from the filter by t and run forward (:draws paths,
  :seed). Shared states make events come together: the chance of any is
  not the independent product's."
  ^double [fit paths ps t [k1 k2] {:keys [draws seed]}]
  (let [{:keys [rho q level]} fit
        g (m/rng seed)
        nodes (distinct (mapcat val paths))
        start (into {} (keep (fn [v] (when-let [x (node-at fit v t)] [v x])) nodes))
        width (inc (- k2 k1))
        haz (into {} (for [[leaf p] ps] [leaf (- (Math/log (- 1.0 (min (double p) (- 1.0 1e-12)))))]))
        survive (for [_ (range draws)]
                  (let [;; each node's state at each period of the window
                        path-of (into {} (for [[v [mean var steps]] start
                                               :let [l (level v) r (double (rho l)) qq (double (q l))
                                                     s0 (+ (double mean) (* (Math/sqrt (double var)) (m/normal! g)))
                                                     xs (loop [h 1 s s0 out []]
                                                          (if (> h (+ (long steps) k2))
                                                            out
                                                            (let [s (+ (* r s) (* (Math/sqrt qq) (m/normal! g)))]
                                                              (recur (inc h) s (if (>= h (+ (long steps) k1)) (conj out s) out)))))]]
                                           [v [xs (* 0.5 (stationary r qq))]]))
                        total (reduce + (for [[leaf path] paths]
                                          (* (double (haz leaf))
                                             (/ (reduce + (for [k (range width)]
                                                            (Math/exp (reduce + 0.0 (for [v path :let [[xs hv] (path-of v)] :when xs]
                                                                                      (- (double (xs k)) (double hv)))))))
                                                width))))]
                    (Math/exp (- total))))]
    (- 1.0 (/ (reduce + survive) (double draws)))))
