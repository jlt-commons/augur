(ns augur.pool
  "Combining forecasters of a binary outcome: a linear pool weighted by EM
  on the log score (or to its maximum with SQUAREM), exponential weights by
  each forecaster's record (Hedge), the median on the log-odds scale, and a
  log-linear stack. Each takes `ps`, per row the forecasters'
  probabilities, `ys` the outcomes and `ws` the rows' weights. Weights come
  back as double arrays; `vec` makes them EDN."
  (:require [augur.math :as m]
            [augur.logit :as logit]))

(set! *unchecked-math* true)

(defn- likelihoods
  "Each forecaster's likelihood of each row's outcome, row-major."
  ^doubles [ps ys]
  (let [ps (vec ps) ys (vec ys) n (count ps) m (count (first ps))
        lik (double-array (* n m))]
    (dotimes [r n]
      (let [row (ps r) y (ys r)]
        (dotimes [j m]
          (let [p (double (nth row j))]
            (aset lik (+ (* r m) j) (if y p (- 1.0 p)))))))
    lik))

(defn- em-step
  "One EM update of mixture weights `w` (m of them) over n rows."
  ^doubles [^doubles lik ^doubles wr tot n m ^doubles w]
  (let [n (long n) m (long m) tot (double tot)
        acc (double-array m)]
    (dotimes [r n]
      (let [o (* r m)
            mix (loop [j 0 s 0.0] (if (< j m) (recur (inc j) (+ s (* (aget w j) (aget lik (+ o j))))) s))]
        (when (> mix 0.0)
          (let [x (aget wr r)]
            (dotimes [j m] (aset acc j (+ (aget acc j) (/ (* x (aget w j) (aget lik (+ o j))) mix))))))))
    (dotimes [j m] (aset acc j (/ (aget acc j) tot)))
    acc))

(defn- mix-loglik ^double [^doubles lik ^doubles wr n m ^doubles w]
  (let [n (long n) m (long m)]
    (loop [r 0 s 0.0]
      (if (< r n)
        (let [o (* r m)
              mix (loop [j 0 t 0.0] (if (< j m) (recur (inc j) (+ t (* (aget w j) (aget lik (+ o j))))) t))]
          (recur (inc r) (+ s (* (aget wr r) (Math/log (max mix 1e-300))))))
        s))))

(defn pool-weights
  "Weights of a linear pool of forecasters by EM on the weighted log score:
  `ps` gives each row's probabilities, one per forecaster. A fixed number of
  steps from equal weights, which with many similar members stops well
  short of the maximum and so shrinks toward equal weights
  (`pool-weights-ml` goes to the maximum)."
  ^doubles [ps ys ws & {:keys [iterations] :or {iterations 200}}]
  (let [n (count ps) m (count (first ps))
        lik (likelihoods ps ys)
        ^doubles wr (double-array (map double ws))
        tot (areduce wr i s 0.0 (+ s (aget wr i)))]
    (loop [k 0 w (double-array m (/ 1.0 m))]
      (if (< k iterations) (recur (inc k) (em-step lik wr tot n m w)) w))))

(defn pool-weights-ml
  "The weights of a linear pool that maximize the weighted log score, by EM
  accelerated with SQUAREM (Varadhan & Roland 2008): two EM steps give a
  direction, an extrapolated step along it is kept when it raises the
  likelihood. Stops when a round gains less than `tol` of the log score
  (1e-6 lands within a tenth of a nat of the maximum, a thousand times
  sooner than 1e-8)."
  ^doubles [ps ys ws & {:keys [tol max-rounds] :or {tol 1e-6 max-rounds 1000}}]
  (let [n (count ps) m (count (first ps))
        lik (likelihoods ps ys)
        ^doubles wr (double-array (map double ws))
        tot (areduce wr i s 0.0 (+ s (aget wr i)))
        step #(em-step lik wr tot n m %)
        ll #(mix-loglik lik wr n m %)]
    (loop [k 0 ^doubles w (double-array m (/ 1.0 m)) l (ll (double-array m (/ 1.0 m)))]
      (let [^doubles w1 (step w) ^doubles w2 (step w1)
            r (double-array m) v (double-array m)
            _ (dotimes [j m]
                (aset r j (- (aget w1 j) (aget w j)))
                (aset v j (- (aget w2 j) (* 2.0 (aget w1 j)) (- (aget w j)))))
            nr (Math/sqrt (areduce r i s 0.0 (+ s (* (aget r i) (aget r i)))))
            nv (Math/sqrt (areduce v i s 0.0 (+ s (* (aget v i) (aget v i)))))
            a (if (pos? nv) (min -1.0 (- (/ nr nv))) -1.0)
            ;; the extrapolated point, kept on the simplex, then one EM step to stabilise
            x (double-array m)
            _ (dotimes [j m] (aset x j (max 0.0 (+ (aget w j) (* -2.0 a (aget r j)) (* a a (aget v j))))))
            sx (areduce x i s 0.0 (+ s (aget x i)))
            _ (dotimes [j m] (aset x j (/ (aget x j) sx)))
            x (step x)
            lx (ll x) l2 (ll w2)
            [w' l'] (if (>= lx l2) [x lx] [w2 l2])]
        (if (or (>= k max-rounds) (< (- l' l) (* tol (max 1.0 (Math/abs l)))))
          (if (>= l' l) w' w)
          (recur (inc k) w' l'))))))

(defn hedge-weights
  "Exponential weights of forecasters by their own record (Hedge; the online
  expert weighting of Fouliard, Howell, Rey & Stavrakeva for crisis
  forecasts): each weight proportional to exp(eta x its weighted log score
  per weighted outcome), so a forecaster a nat per onset ahead of another
  gets e^eta times its weight, however many rows there are."
  ^doubles [ps ys ws eta]
  (let [m (count (first ps))
        ls (double-array m)
        pos (reduce + (map (fn [y w] (if y (double w) 0.0)) ys ws))]
    (doseq [[row y w] (map vector ps ys ws)
            j (range m)
            :let [p (min (max (double (nth row j)) 1e-6) (- 1.0 1e-6))]]
      (aset ls j (+ (aget ls j) (* (double w) (Math/log (if y p (- 1.0 p)))))))
    (let [scale (/ (double eta) (max pos 1.0))
          top (apply max (seq ls))
          e (double-array (map #(Math/exp (* scale (- % top))) ls))
          tot (reduce + (seq e))]
      (double-array (map #(/ % tot) e)))))

(defn median-pool
  "The members' median forecast, on the log-odds scale (an even count takes
  the mean of the middle two): the untrained combination that matched or
  beat trained weights in the epidemic forecast hubs (Cramer et al., PNAS
  2022; Sherratt et al., eLife 2023), where weights fit on few outcomes
  chase noise."
  ^double [ps]
  (let [ls (vec (sort (map #(m/logit (min (max (double %) 1e-9) (- 1.0 1e-9))) ps)))
        n (count ls)]
    (m/expit (if (odd? n) (ls (quot n 2)) (* 0.5 (+ (ls (dec (quot n 2))) (ls (quot n 2))))))))

(defn pool
  "The linear pool of probabilities ps with weights w."
  ^double [w ps]
  (let [^doubles w (if (vector? w) (double-array w) w)]
    (loop [j 0 s 0.0 [p & more] (seq ps)]
      (if p (recur (inc j) (+ s (* (aget w j) (double p))) more) s))))

(defn- clipped-logit ^double [^double p]
  (let [p (Math/min (Math/max p 1e-6) (- 1.0 1e-6))]
    (Math/log (/ p (- 1.0 p)))))

(defn- stack-x ^doubles [ps]
  (double-array (cons 1.0 (map #(clipped-logit (double %)) ps))))

(defn fit-stack
  "A log-linear stack: a logistic regression on the forecasters' logits,
  which can combine one that knows where with one that knows when. `opts`
  as `augur.logit/fit` takes them (:prior required)."
  ^doubles [ps ys ws opts]
  (let [rows (map (fn [p y w] {:x (stack-x p) :y y :w w}) ps ys ws)]
    (logit/fit rows (inc (count (first ps))) opts)))

(defn predict-stack
  "The stack's probability for the forecasters' probabilities ps."
  ^double [beta ps]
  (logit/predict beta (stack-x ps)))
