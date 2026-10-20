(ns augur.trees
  "Shallow gradient-boosted trees for onset probabilities: what a logit
  can't learn without hand-written terms, such as an economy's shock
  mattering only under some regimes. Logistic loss, second-order leaf
  values with an L2 penalty (as XGBoost), splits from histograms over each
  input's quantiles in the training rows, shrinkage. Trees of depth 2 or 3
  learn interactions while staying close to additive: deep trees overfit
  rare onsets (Wang, Political Analysis 2019).")

(def defaults
  "The values bayesian-news-model uses: for reference, not applied. `fit`
  needs every one."
  {:rounds 100 :depth 2 :rate 0.1 :bins 16 :lambda 1.0 :min-weight 2.0})

(defn- sigmoid ^double [^double z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))

(defn cuts
  "Cut points of column j: midway between each pair of neighbouring distinct
  values when there are at most `bins` (a flag, however rare, gets its
  cut), else midway above each of the bins-1 quantiles. Missing values
  (NaN) are left out, and go left of every cut."
  [xs j bins]
  (let [v (vec (sort (remove #(Double/isNaN %) (map #(aget ^doubles % (int j)) xs))))
        d (vec (dedupe v))
        mid (fn [a b] (* 0.5 (+ (double a) (double b))))]
    (if (<= (count d) bins)
      (mapv mid d (rest d))
      (let [n (count v)
            ds (apply sorted-set d)
            above (fn [x] (first (subseq ds > x)))]
        (vec (distinct (for [k (range 1 bins)
                             :let [x (v (min (dec n) (quot (* k n) bins)))
                                   y (above x)]
                             :when y]
                         (mid x y))))))))

(defn- bin-of ^long [cuts ^double x]
  ;; the number of cuts below x
  (loop [lo 0 hi (count cuts)]
    (if (< lo hi)
      (let [mid (quot (+ lo hi) 2)]
        (if (< (double (nth cuts mid)) x) (recur (inc mid) hi) (recur lo mid)))
      lo)))

(defn- best-split
  "The split of rows `idx` with the largest gain, or nil: [gain j cut-index]."
  [^objects binned cuts ^doubles g ^doubles h idx {:keys [lambda min-weight]}]
  (let [nf (alength binned)
        gt (reduce + (map #(aget g (int %)) idx))
        ht (reduce + (map #(aget h (int %)) idx))
        parent (/ (* gt gt) (+ ht lambda))]
    (reduce
     (fn [best j]
       (let [nc (count (nth cuts j))]
         (if (zero? nc)
           best
           (let [^longs bj (aget binned j)
                 gs (double-array (inc nc)) hs (double-array (inc nc))]
             (doseq [i idx]
               (let [b (aget bj (int i))]
                 (aset gs b (+ (aget gs b) (aget g (int i))))
                 (aset hs b (+ (aget hs b) (aget h (int i))))))
             (loop [c 0 gl 0.0 hl 0.0 best best]
               (if (< c nc)
                 (let [gl (+ gl (aget gs c)) hl (+ hl (aget hs c))
                       gr (- gt gl) hr (- ht hl)
                       gain (- (+ (/ (* gl gl) (+ hl lambda)) (/ (* gr gr) (+ hr lambda))) parent)]
                   (recur (inc c) gl hl
                          (if (and (>= hl min-weight) (>= hr min-weight) (> gain 1e-12)
                                   (or (nil? best) (> gain (double (first best)))))
                            [gain j c]
                            best)))
                 best))))))
     nil (range nf))))

(defn- grow
  [binned cuts g h idx depth {:keys [lambda rate] :as opts}]
  (let [leaf (fn [] {:leaf (* rate (- (/ (reduce + (map #(aget ^doubles g (int %)) idx))
                                          (+ (reduce + (map #(aget ^doubles h (int %)) idx)) lambda))))})]
    (if (zero? depth)
      (leaf)
      (if-let [[_ j c] (best-split binned cuts g h idx opts)]
        (let [^longs bj (aget ^objects binned (int j))
              {left true right false} (group-by #(<= (aget bj (int %)) (long c)) idx)]
          {:feature j :cut (nth (nth cuts j) c)
           :left (grow binned cuts g h left (dec depth) opts)
           :right (grow binned cuts g h right (dec depth) opts)})
        (leaf)))))

(defn- tree-value ^double [tree ^doubles x]
  (if-let [v (:leaf tree)]
    v
    (let [v (aget x (int (:feature tree)))]
      ;; missing goes left, as in fitting (bin-of puts NaN below every cut)
      (recur (if (or (Double/isNaN v) (<= v (double (:cut tree)))) (:left tree) (:right tree)) x))))

(defn score ^double [{:keys [base trees]} ^doubles x]
  (reduce (fn [s t] (+ s (tree-value t x))) base trees))

(defn predict ^double [model ^doubles x] (sigmoid (score model x)))

(defn fit
  "Boosted trees from rows {:x double[] :y bool :w weight}: {:base :trees},
  plain EDN. Options: :rounds of boosting, tree :depth, shrinkage :rate,
  histogram :bins per input, the leaves' L2 penalty :lambda, and the
  :min-weight (Hessian) of a child; all required, see `defaults`."
  [rows opts]
  (when-let [missing (seq (remove #(some? (get opts %)) (keys defaults)))]
    (throw (ex-info (str "augur.trees/fit needs " (vec missing)) {:missing (vec missing)})))
  (let [{:keys [rounds depth bins]} opts
        xs (mapv :x rows) n (count rows)
        ys (double-array (map #(if (:y %) 1.0 0.0) rows))
        ws (double-array (map #(double (:w %)) rows))
        nf (alength ^doubles (first xs))
        cuts (mapv #(cuts xs % bins) (range nf))
        binned (object-array (for [j (range nf)]
                               (long-array (map #(bin-of (cuts j) (aget ^doubles % (int j))) xs))))
        wy (reduce + (map * ys ws)) wt (reduce + ws)
        rate (min (max (/ wy wt) 1e-6) (- 1.0 1e-6))
        base (Math/log (/ rate (- 1.0 rate)))
        f (double-array n base)
        g (double-array n) h (double-array n)
        idx (vec (range n))]
    (loop [r 0 trees []]
      (if (< r rounds)
        (do (dotimes [i n]
              (let [p (sigmoid (aget f i)) w (aget ws i)]
                (aset g i (* w (- p (aget ys i))))
                (aset h i (* w (max (* p (- 1.0 p)) 1e-6)))))
            (let [t (grow binned cuts g h idx depth opts)]
              (dotimes [i n] (aset f i (+ (aget f i) (tree-value t (xs i)))))
              (recur (inc r) (conj trees t))))
        {:base base :trees trees}))))
