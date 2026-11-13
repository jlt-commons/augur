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

;; --- grouped slopes ---------------------------------------------------------

(defn- cholesky!
  "The lower Cholesky factor of the n x n row-major matrix `a`, in place of
  a fresh array (pivots clamped as in augur.math/solve-spd)."
  ^doubles [^doubles a ^long n]
  (let [^doubles l (double-array (* n n))]
    (dotimes [i n]
      (dotimes [j (inc i)]
        (let [s (loop [k 0 s (aget a (+ (* i n) j))]
                  (if (< k j) (recur (inc k) (- s (* (aget l (+ (* i n) k)) (aget l (+ (* j n) k))))) s))]
          (aset l (+ (* i n) j) (if (= i j) (Math/sqrt (Math/max s 1e-300)) (/ s (aget l (+ (* j n) j))))))))
    l))

(defn- chol-solve
  "x with l l' x = b."
  ^doubles [^doubles l ^doubles b ^long n]
  (let [^doubles x (aclone b)]
    (dotimes [i n]
      (aset x i (/ (loop [k 0 s (aget x i)] (if (< k i) (recur (inc k) (- s (* (aget l (+ (* i n) k)) (aget x k)))) s))
                   (aget l (+ (* i n) i)))))
    (loop [i (dec n)]
      (when (>= i 0)
        (aset x i (/ (loop [k (inc i) s (aget x i)] (if (< k n) (recur (inc k) (- s (* (aget l (+ (* k n) i)) (aget x k)))) s))
                     (aget l (+ (* i n) i))))
        (recur (dec i))))
    x))

(defn- inverse-diagonal
  "The diagonal entries `idx` of (l l')^-1: ||l^-1 e_i||^2, a forward solve
  from i."
  [^doubles l ^long n idx]
  (into {} (for [i idx]
             (let [^doubles y (double-array n)]
               (aset y i (/ 1.0 (aget l (+ (* i n) i))))
               (loop [r (inc i)]
                 (when (< r n)
                   (aset y r (/ (- (loop [k i s 0.0] (if (< k r) (recur (inc k) (+ s (* (aget l (+ (* r n) k)) (aget y k)))) s)))
                                (aget l (+ (* r n) r))))
                   (recur (inc r))))
               [i (loop [r i s 0.0] (if (< r n) (recur (inc r) (+ s (* (aget y r) (aget y r)))) s))]))))

(defn grouped-slopes
  "A logistic model with slopes shared by every group and each group's
  deviations from them, partially pooled: rows {:x [1 x1 ...] :y :w :g},
  x of length k with x[0] the intercept's 1, g the row's group (nil: the
  shared part only). log odds = sum_j (b_j + d_gj) x_j on inputs
  standardized over the rows; b_0 free, b_j ~ N(0, 1/:prior), d_gj ~ N(0,
  tau2_j), one variance per coefficient (the intercept's too) shared by
  the groups. Newton steps on all coefficients at once (the Hessian has
  one k x k block per group, so a row costs k^2), each halved until the
  penalized likelihood improves; after each sweep each tau2_j is set by
  the Laplace EM step, the mean over groups of d^2 plus its posterior
  variance, made a posterior mode by a half-normal prior on tau_j of
  scale (:sd-prior j) as `intercepts` does (Gelman 2006), so the data
  say how far a group's form departs from the shared one. Options:
  :prior, :tau2 (k starting variances), :sd-prior (k scales) and :sweeps
  are required; :iterations (25) Newton steps a sweep. Returns {:beta [k]
  :dev {g [k]} :tau2 [k] :groups [g ...]}, beta and dev on the inputs'
  own scale (tau2 on the standardized one): a group's coefficients are
  beta + its dev, and a group never seen takes beta (`grouped-predict`)."
  [rows ^long k {:keys [prior tau2 sd-prior sweeps iterations] :or {iterations 25}}]
  (when-not (and prior tau2 sd-prior sweeps)
    (throw (ex-info "augur.hier/grouped-slopes needs :prior, :tau2, :sd-prior and :sweeps" {})))
  (when-not (= k (count tau2) (count sd-prior))
    (throw (ex-info "augur.hier/grouped-slopes needs one :tau2 and :sd-prior per coefficient" {:k k})))
  (let [rows (vec rows)
        n (count rows)
        groups (vec (sort-by str (distinct (keep :g rows))))
        G (count groups)
        gi (into {} (map-indexed (fn [i g] [g i]) groups))
        P (* k (inc G))
        ^doubles z (double-array (* n k))
        ^doubles ys (double-array n)
        ^doubles ws (double-array n)
        ^ints gs (int-array n)
        ^doubles mean (double-array k) ^doubles sd (double-array k 1.0)
        _ (dotimes [r n]
            (let [{:keys [x y w g]} (rows r)]
              (dotimes [j k] (aset z (+ (* r k) j) (double (nth x j))))
              (aset ys r (if y 1.0 0.0))
              (aset ws r (double (or w 1.0)))
              (aset gs r (int (if (some? g) (gi g) -1)))))
        wsum (Math/max 1e-12 (loop [r 0 a 0.0] (if (< r n) (recur (inc r) (+ a (aget ws r))) a)))
        _ (dotimes [r n] (dotimes [j k] (aset mean j (+ (aget mean j) (* (aget ws r) (aget z (+ (* r k) j)))))))
        _ (dotimes [j k] (aset mean j (/ (aget mean j) wsum)))
        _ (let [^doubles v (double-array k)]
            (dotimes [r n] (dotimes [j k] (let [d (- (aget z (+ (* r k) j)) (aget mean j))] (aset v j (+ (aget v j) (* (aget ws r) d d))))))
            (dotimes [j k] (let [s (Math/sqrt (/ (aget v j) wsum))] (aset sd j (if (> s 1e-9) s 1.0)))))
        _ (do (aset mean 0 0.0) (aset sd 0 1.0))
        _ (dotimes [r n]
            (aset z (* r k) 1.0)
            (loop [j 1] (when (< j k) (aset z (+ (* r k) j) (/ (- (aget z (+ (* r k) j)) (aget mean j)) (aget sd j))) (recur (inc j)))))
        ^doubles th (double-array P)
        eta (fn ^double [^doubles t ^long r]
              (let [o (* r k) g (aget gs r) go (* k (inc g))]
                (loop [j 0 acc 0.0]
                  (if (< j k)
                    (recur (inc j) (+ acc (* (+ (aget t j) (if (>= g 0) (aget t (+ go j)) 0.0)) (aget z (+ o j)))))
                    acc))))
        pen-of (fn ^doubles [tau2]
                 (let [^doubles p (double-array P)]
                   (aset p 0 1e-4)
                   (loop [j 1] (when (< j k) (aset p j (double prior)) (recur (inc j))))
                   (dotimes [g G] (dotimes [j k] (aset p (+ (* k (inc g)) j) (/ 1.0 (double (nth tau2 j))))))
                   p))
        objective (fn ^double [^doubles t ^doubles pen]
                    (- (loop [r 0 acc 0.0]
                         (if (< r n)
                           (let [e (eta t r)] (recur (inc r) (+ acc (* (aget ws r) (- (* (aget ys r) e) (softplus e))))))
                           acc))
                       (* 0.5 (loop [i 0 acc 0.0] (if (< i P) (recur (inc i) (+ acc (* (aget pen i) (aget t i) (aget t i)))) acc)))))
        ;; the penalized Hessian and gradient at t
        hessian (fn [^doubles t ^doubles pen]
                  (let [^doubles h (double-array (* P P))
                        ^doubles grad (double-array P)
                        blocks (vec (repeatedly (inc G) #(double-array (* k k))))
                        rs (vec (repeatedly (inc G) #(double-array k)))]
                    (dotimes [r n]
                      (let [o (* r k) g (aget gs r)
                            ^doubles s (blocks (inc g)) ^doubles rg (rs (inc g))
                            p (m/expit (eta t r))
                            w (aget ws r)
                            res (* w (- (aget ys r) p))
                            c (* w (Math/max 1e-9 (* p (- 1.0 p))))]
                        (dotimes [i k]
                          (let [zi (aget z (+ o i)) ci (* c zi) hi (* i k)]
                            (aset rg i (+ (aget rg i) (* res zi)))
                            (loop [j i] (when (< j k) (aset s (+ hi j) (+ (aget s (+ hi j)) (* ci (aget z (+ o j))))) (recur (inc j))))))))
                    (doseq [^doubles s blocks]
                      (dotimes [i k] (loop [j (inc i)] (when (< j k) (aset s (+ (* j k) i) (aget s (+ (* i k) j))) (recur (inc j))))))
                    (dotimes [b (inc G)]
                      (let [^doubles s (blocks b) ^doubles rg (rs b) g (dec b)]
                        (dotimes [i k]
                          (aset grad i (+ (aget grad i) (aget rg i)))
                          (dotimes [j k] (aset h (+ (* i P) j) (+ (aget h (+ (* i P) j)) (aget s (+ (* i k) j))))))
                        (when (>= g 0)
                          (let [go (* k (inc g))]
                            (dotimes [i k]
                              (aset grad (+ go i) (+ (aget grad (+ go i)) (aget rg i)))
                              (dotimes [j k]
                                (let [v (aget s (+ (* i k) j))]
                                  (aset h (+ (* (+ go i) P) (+ go j)) v)
                                  (aset h (+ (* i P) (+ go j)) v)
                                  (aset h (+ (* (+ go i) P) j) v))))))))
                    (dotimes [i P]
                      (aset grad i (- (aget grad i) (* (aget pen i) (aget t i))))
                      (aset h (+ (* i P) i) (+ (aget h (+ (* i P) i)) (aget pen i))))
                    [h grad]))
        newton (fn [^doubles pen]
                 (loop [it 0 obj (objective th pen)]
                   (when (< it iterations)
                     (let [[h grad] (hessian th pen)
                           ^doubles step (chol-solve (cholesky! h P) grad P)
                           [^doubles nt nobj] (loop [s 1.0 tries 0]
                                                (let [^doubles c (double-array P)]
                                                  (dotimes [i P] (aset c i (+ (aget th i) (* s (aget step i)))))
                                                  (let [o (objective c pen)]
                                                    (if (or (> o obj) (>= tries 30)) [c o] (recur (* 0.5 s) (inc tries))))))
                           moved (loop [i 0 a 0.0] (if (< i P) (recur (inc i) (Math/max a (Math/abs (- (aget nt i) (aget th i))))) a))]
                       (when (> nobj obj) (System/arraycopy nt 0 th 0 P))
                       (when (and (> nobj obj) (> moved 1e-7)) (recur (inc it) nobj))))))
        tau2 (loop [sweep 0 tau2 (mapv double tau2)]
               (let [pen (pen-of tau2)]
                 (newton pen)
                 (if (or (>= sweep sweeps) (zero? G))
                   tau2
                   (let [[h _] (hessian th pen)
                         l (cholesky! h P)
                         vars (inverse-diagonal l P (for [g (range G) j (range k)] (+ (* k (inc g)) j)))
                         tau2' (vec (for [j (range k)]
                                      (let [ss (reduce + (for [g (range G) :let [i (+ (* k (inc g)) j)]]
                                                           (+ (* (aget th i) (aget th i)) (double (vars i)))))
                                            s2 (let [x (double (nth sd-prior j))] (* x x))
                                            mm (double G)]
                                        (max 1e-6 (/ (* 2.0 ss s2) (+ (* mm s2) (Math/sqrt (+ (* mm mm s2 s2) (* 4.0 ss s2)))))))))
                         done (every? #(< (Math/abs (Math/log (/ (double (tau2' %)) (double (tau2 %))))) 1e-3) (range k))]
                     (if done
                       (do (newton (pen-of tau2')) tau2')
                       (recur (inc sweep) tau2'))))))
        unscale (fn [^doubles t off]
                  (let [b (vec (for [j (range k)] (if (zero? j) 0.0 (/ (aget t (+ off j)) (aget sd j)))))]
                    (assoc b 0 (- (aget t off) (reduce + (map #(* (b %) (aget mean %)) (range 1 k)))))))]
    {:beta (unscale th 0)
     :dev (into {} (for [g groups] [g (unscale th (* k (inc (gi g))))]))
     :tau2 tau2
     :groups groups}))

(defn grouped-predict
  "The probability for inputs x (x[0] the intercept's 1) under a
  `grouped-slopes` fit for group g: beta plus g's deviations, beta alone
  for a group the fit never saw (or nil)."
  ^double [fit x g]
  (let [b (:beta fit) d (get-in fit [:dev g])]
    (m/expit (reduce + 0.0 (map-indexed (fn [j xj] (* (double xj) (+ (double (b j)) (if d (double (d j)) 0.0)))) x)))))

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

;; --- one shared state, online ------------------------------------------------

(defn- shared-terms
  "[log p, d/dz, -d2/dz2] of one row at shift z: a logistic row's outcome
  under the predictive N(f, q) of its log odds, shifted by z (MacKay's
  probit approximation, as augur.dynamic predicts), or a Poisson count
  whose log rate is N(f + z, q)."
  [kind {:keys [f q y w]} ^double z]
  (let [f (double f) q (double (or q 0.0)) w (double (or w 1.0))]
    (case kind
      :logistic (let [k (/ 1.0 (Math/sqrt (+ 1.0 (/ (* Math/PI q) 8.0))))
                      e (* k (+ f z))
                      yy (if (if (number? y) (pos? (double y)) (boolean y)) 1.0 0.0)
                      p (m/expit e)
                      lp (- (* yy e) (if (> e 0.0) (+ e (Math/log1p (Math/exp (- e)))) (Math/log1p (Math/exp e))))]
                  [(* w lp) (* w k (- yy p)) (* w k k p (- 1.0 p))])
      :poisson (let [eta (+ f z (* 0.5 q))
                     mu (Math/exp (Math/min 700.0 eta))
                     yy (double y)]
                 [(* w (- (* yy eta) mu (m/log-gamma (+ 1.0 yy)))) (* w (- yy mu)) (* w mu)])
      (throw (ex-info "augur.hier: a shared state reads :logistic or :poisson rows" {:kind kind})))))

(defn- solve
  "x with a x = b, a a small square matrix [[...] ...], b a vector
  (Gaussian elimination with partial pivoting)."
  [a b]
  (let [n (count b)
        aug (into-array (map (fn [row bi] (double-array (conj (vec row) bi))) a b))]
    (dotimes [col n]
      (let [piv (apply max-key #(Math/abs (aget ^doubles (aget aug %) col)) (range col n))
            tmp (aget aug col)]
        (aset aug col (aget aug piv))
        (aset aug piv tmp)
        (let [^doubles pr (aget aug col) d (aget pr col)]
          (doseq [r (range (inc col) n)]
            (let [^doubles rr (aget aug r) f (/ (aget rr col) d)]
              (dotimes [j (inc n)] (aset rr j (- (aget rr j) (* f (aget pr j))))))))))
    (let [x (double-array n)]
      (doseq [i (reverse (range n))]
        (let [^doubles ri (aget aug i)]
          (aset x i (/ (- (aget ri n) (loop [j (inc i) acc 0.0] (if (< j n) (recur (inc j) (+ acc (* (aget ri j) (aget x j)))) acc)))
                       (aget ri i)))))
      (vec x))))

(defn- log-det
  "The log determinant of a small matrix with a positive determinant (LU)."
  ^double [a]
  (let [n (count a)
        m (into-array (map double-array a))]
    (loop [col 0 acc 0.0]
      (if (= col n)
        acc
        (let [piv (apply max-key #(Math/abs (aget ^doubles (aget m %) col)) (range col n))
              tmp (aget m col)]
          (aset m col (aget m piv))
          (aset m piv tmp)
          (let [^doubles pr (aget m col) d (aget pr col)]
            (doseq [r (range (inc col) n)]
              (let [^doubles rr (aget m r) f (/ (aget rr col) d)]
                (dotimes [j n] (aset rr j (- (aget rr j) (* f (aget pr j)))))))
            (recur (inc col) (+ acc (Math/log (Math/abs d))))))))))

(defn- mat-vec [a x] (mapv (fn [row] (reduce + 0.0 (map * row x))) a))
(defn- mat-mul [a b] (let [bt (apply mapv vector b)] (mapv (fn [row] (mapv #(reduce + 0.0 (map * row %)) bt)) a)))

(defn- check-shared-n
  "A state of several dimensions: m a vector of d means, v a d x d
  variance with no negative diagonal, every row a :load of d numbers."
  [rows m v]
  (let [d (count m)]
    (when-not (and (sequential? v) (= d (count v)) (every? #(and (sequential? %) (= d (count %))) v)
                   (every? number? m) (every? #(every? number? %) v))
      (throw (ex-info "augur.hier/shared-marginal: a state of d dimensions takes d means and a d x d variance" {:m m :v v})))
    (when (some #(neg? (double (get-in v [% %]))) (range d))
      (throw (ex-info "augur.hier/shared-marginal: a negative variance" {:v v})))
    (when-let [bad (first (remove #(and (sequential? (:load %)) (= d (count (:load %))) (every? number? (:load %))) rows))]
      (throw (ex-info "augur.hier/shared-marginal: each row loads on the state with d numbers (:load)" {:row bad :d d})))))

(defn- shared-marginal-n
  "shared-marginal for a state s ~ N(m, v) of several dimensions, each row
  adding its :load times s. Newton in the form (I + V W) delta = V g - (s -
  m), which needs no inverse of V, so a dimension known exactly (variance
  0) stays at its mean."
  [kind rows m v]
  (check-shared-n rows m v)
  (let [d (count m)
        rows (vec rows)
        n (count rows)
        loads (object-array (map #(double-array (map double (:load %))) rows))
        m (mapv double m)
        v (mapv #(mapv double %) v)
        ;; [l g w] at s: the rows' log likelihood, its gradient and minus its Hessian
        sum (fn [s]
              (let [^doubles sa (double-array s)
                    g (double-array d)
                    w (double-array (* d d))]
                (loop [i 0 l 0.0]
                  (if (= i n)
                    [l (vec g) (vec (for [a (range d)] (vec (for [b (range d)] (aget w (+ (* a d) b))))))]
                    (let [^doubles a (aget loads i)
                          z (loop [j 0 acc 0.0] (if (< j d) (recur (inc j) (+ acc (* (aget a j) (aget sa j)))) acc))
                          [x y u] (shared-terms kind (nth rows i) z)
                          y (double y) u (double u)]
                      (dotimes [j d]
                        (aset g j (+ (aget g j) (* y (aget a j))))
                        (dotimes [k d]
                          (aset w (+ (* j d) k) (+ (aget w (+ (* j d) k)) (* u (aget a j) (aget a k))))))
                      (recur (inc i) (+ l (double x))))))))
        eye (vec (for [i (range d)] (vec (for [j (range d)] (if (= i j) 1.0 0.0)))))
        ipvw (fn [w] (mapv (fn [r e] (mapv + r e)) (mat-mul v w) eye))
        ;; Newton alone can cycle from a prior far from the mode, so each step
        ;; is cut to a length of 5 and halved until the log posterior does not
        ;; fall. The prior's quadratic is u'Vu with s - m = V u, u carried
        ;; along the steps (du solves (I + W V) du = g - u), so V is never
        ;; inverted. A full step is taken as it was without the search.
        ipwv (fn [w] (mapv (fn [r e] (mapv + r e)) (mat-mul w v) eye))
        post-ll (fn [l u] (- (double l) (* 0.5 (reduce + 0.0 (map * u (mat-vec v u))))))
        s (loop [i 0 s m u (vec (repeat d 0.0))]
            (let [[l g w] (sum s)
                  delta (solve (ipvw w) (mapv - (mat-vec v g) (mapv - s m)))
                  big (reduce max 0.0 (map #(Math/abs (double %)) delta))
                  c (if (> big 5.0) (/ 5.0 big) 1.0)
                  delta (if (< c 1.0) (mapv #(* c (double %)) delta) delta)
                  du (mapv #(* c (double %)) (solve (ipwv w) (mapv - g u)))
                  f0 (post-ll l u)
                  tol (* 1e-10 (+ 1.0 (Math/abs f0)))
                  step (loop [t 1.0]
                         (let [s' (mapv + s (map #(* t (double %)) delta))
                               u' (mapv + u (map #(* t (double %)) du))]
                           (cond (>= (post-ll (first (sum s')) u') (- f0 tol)) [t s' u']
                                 (< t 1e-6) nil
                                 :else (recur (* 0.5 t)))))]
              (if-let [[t s' u'] step]
                (if (or (>= i 100) (< (* t (min big 5.0)) 1e-10)) s' (recur (inc i) s' u'))
                s)))
        [l g w] (sum s)
        a (ipvw w)
        ;; the posterior variance (V^-1 + W)^-1 = (I + V W)^-1 V, column by
        ;; column, symmetrized; a dimension known exactly keeps variance 0
        ;; (its roundoff can fall a hair below)
        post (let [cols (mapv (fn [j] (solve a (mapv #(nth % j) v))) (range d))]
               (vec (for [i (range d)]
                      (vec (for [j (range d)
                                 :let [x (* 0.5 (+ (nth (nth cols j) i) (nth (nth cols i) j)))]]
                             (if (= i j) (Math/max 0.0 x) x))))))]
    ;; at the mode V^-1 (s - m) = g, so the prior's quadratic is (s - m).g
    {:z s :v post
     :ll (- (double l) (* 0.5 (reduce + 0.0 (map * (mapv - s m) g))) (* 0.5 (log-det a)))}))

(defn shared-marginal
  "One period's rows sharing a state z ~ N(m, v), added to each row's
  linear predictor: {:z the posterior mode, :v its variance, :ll the log
  marginal likelihood of the rows with z integrated out} by Laplace's
  approximation (Newton to the mode). Rows {:f :q :y :w}: the predictive
  moments of each row's linear predictor without z, its outcome and
  weight; `kind` :logistic or :poisson. With v = 0 the rows are scored at
  z = m, independently. Scored jointly, a shock the rows share is paid for
  once, through z, rather than once per row.

  With m a vector of d means and v a d x d variance the state has d
  dimensions and each row adds its :load (d numbers) times it: a common
  shock and a slope on each row's exposure, say, with loads [1 e]. :z is
  then the mode's vector and :v its variance matrix."
  [kind rows m v]
  (if (sequential? m)
    (shared-marginal-n kind rows m v)
  (let [m (double m) v (double v)
        sum (fn [z] (reduce (fn [[a b c] r] (let [[x y u] (shared-terms kind r z)] [(+ a x) (+ b y) (+ c u)]))
                            [0.0 0.0 0.0] rows))]
    (when (neg? v) (throw (ex-info "augur.hier/shared-marginal: a negative variance" {:v v})))
    (if (or (zero? v) (empty? rows))
      {:z m :v v :ll (double (first (sum m)))}
      (let [post-ll (fn [l z] (- (double l) (/ (* (- z m) (- z m)) (* 2.0 v))))
            ;; Newton, its step cut to 5 and halved until the log posterior
            ;; does not fall (Newton alone can cycle from a distant prior)
            z (loop [i 0 z m]
                (let [[l g h] (sum z)
                      g (- (double g) (/ (- z m) v))
                      h (+ (double h) (/ 1.0 v))
                      step (Math/max -5.0 (Math/min 5.0 (/ g h)))
                      f0 (post-ll l z)
                      tol (* 1e-10 (+ 1.0 (Math/abs f0)))
                      z' (loop [t 1.0]
                           (let [z' (+ z (* t step))]
                             (cond (>= (post-ll (first (sum z')) z') (- f0 tol)) z'
                                   (< t 1e-6) nil
                                   :else (recur (* 0.5 t)))))]
                  (cond (nil? z') z
                        (or (>= i 100) (< (Math/abs (- z' z)) 1e-10)) z'
                        :else (recur (inc i) z'))))
            [l _ h] (sum z)
            d (- z m)]
        {:z z :v (/ 1.0 (+ (double h) (/ 1.0 v)))
         :ll (- (double l) (/ (* d d) (* 2.0 v)) (* 0.5 (Math/log1p (* v (double h)))))})))))

(defn states-predict
  "A single AR(1) state {:m :v} run h periods forward under {:rho :sd}
  (sd the stationary sd): mean rho^h m, variance rho^2h v + sd^2 (1 -
  rho^2h)."
  [{:keys [m v]} {:keys [rho sd]} h]
  (let [rh (Math/pow (double rho) (double h))
        s2 (* (double sd) (double sd))]
    {:m (* rh (double m)) :v (+ (* rh rh (double v)) (* s2 (- 1.0 (* rh rh))))}))

(defn states-step
  "One online step of a single shared state (the dynamic frailty of
  `states`, one node and one period at a time, its rho and sd fixed): the
  state's prior for the period {:m :v} (from `states-predict`) updated by
  the period's rows (see `shared-marginal`; :kind :logistic or :poisson).
  A state of several dimensions ({:m [..] :v [[..] ..]}, each row's :load)
  is updated the same way.
  {:m :v :ll}, :ll the rows' log marginal under the prior: the period's
  prequential score of the prior."
  [{:keys [m v]} rows {:keys [kind] :or {kind :logistic}}]
  (let [{z :z vz :v ll :ll} (shared-marginal kind rows m v)]
    {:m z :v vz :ll ll}))
