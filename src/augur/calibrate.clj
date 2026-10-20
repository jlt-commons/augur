(ns augur.calibrate
  "Turning raw probabilities into calibrated ones, learned from how past
  forecasts turned out.

  Online: raw probabilities are binned per key (any value), and each bin
  keeps forgetting-weighted counts of how often the outcome followed. The
  calibrated probability is a Beta posterior mean, the bin's observed rate
  shrunk toward the raw probability when the bin has seen little, made
  nondecreasing across bins. The table is plain EDN.

  In bulk, from past forecasts and outcomes: the isotonic fit (CORP) or a
  logistic fit on the logit (Cox 1958), each as a function of a new
  forecast.

  The options, every one required by the functions that use it:
    :edges   upper edges of the probability bins, ascending, the last 1.0
    :shrink  pseudo-observations at the raw probability each bin starts with
    :forget  the discount applied to old counts each time a batch is learned")

(def defaults
  "The values bayesian-news-model uses: for reference, not applied."
  {:edges [0.02 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0]
   :shrink 10.0
   :forget 0.9})

(defn- bin-of [edges p]
  (or (first (keep-indexed (fn [i e] (when (<= p e) i)) edges))
      (dec (count edges))))

(defn isotonic
  "The weighted least-squares nondecreasing fit to ys: pool adjacent
  violators, in one pass with backtracking (a port of scikit-learn's
  _inplace_contiguous_isotonic_regression). The CORP reliability diagram
  (Dimitriadis, Gneiting & Jordan 2021) is this fit of outcomes on forecasts."
  [ys ws]
  (let [n (count ys)
        y (double-array ys) w (double-array ws)
        target (long-array (range n))]
    (loop [i 0]
      (when (< i n)
        (let [k (inc (aget target i))]
          (when (< k n)
            (if (< (aget y i) (aget y k))
              (recur k)
              (let [i' (loop [k k swy (* (aget w i) (aget y i)) sw (aget w i)]
                         (let [prev (aget y k)
                               swy (+ swy (* (aget w k) (aget y k)))
                               sw (+ sw (aget w k))
                               k (inc (aget target k))]
                           (if (or (= k n) (< prev (aget y k)))
                             (do (aset y i (/ swy sw))
                                 (aset w i sw)
                                 (aset target i (dec k))
                                 (aset target (dec k) i)
                                 (if (pos? i) (aget target (dec i)) i))
                             (recur k swy sw))))]
                (recur i')))))))
    (loop [i 0]
      (when (< i n)
        (let [k (inc (aget target i))]
          (doseq [j (range (inc i) k)] (aset y j (aget y i)))
          (recur k))))
    (vec y)))

(defn- bin-center [edges i]
  (let [lo (if (zero? i) 0.0 (nth edges (dec i)))]
    (/ (+ lo (nth edges i)) 2.0)))

(defn calibrate
  "The calibrated probability for raw `p` under key `k` of table `cal`: each
  bin's observed rate shrunk toward its raw probability (`p` itself for p's
  bin), then made nondecreasing across bins by pooling adjacent violators,
  so a higher forecast is never calibrated lower. `p` itself for a key not
  in the table."
  [cal k p {:keys [edges shrink]}]
  (if-let [{:keys [pos n]} (get cal k)]
    (let [b (bin-of edges p)
          raw (fn [i] (if (= i b) p (bin-center edges i)))
          est (mapv (fn [i] (/ (+ (nth pos i) (* shrink (raw i))) (+ (nth n i) shrink))) (range (count edges)))
          ws (mapv #(+ % shrink) n)]
      (nth (isotonic est ws) b))
    p))

(defn learn
  "Table `cal` (nil for none) updated with a batch of resolved forecasts
  {:key :p :outcome}: the old counts discounted by :forget once, then the
  batch added."
  [cal outcomes {:keys [edges forget]}]
  (let [nb (count edges)
        empty-bins {:pos (vec (repeat nb 0.0)) :n (vec (repeat nb 0.0))}
        decayed (into {} (for [[k b] cal]
                           [k {:pos (mapv #(* forget %) (:pos b))
                               :n (mapv #(* forget %) (:n b))}]))]
    (reduce (fn [cal {:keys [key p outcome]}]
              (let [i (bin-of edges p)]
                (-> cal
                    (update key #(or % empty-bins))
                    (update-in [key :n i] + 1.0)
                    (update-in [key :pos i] + (if outcome 1.0 0.0)))))
            decayed outcomes)))

(defn isotonic-map
  "Recalibration learned from past forecasts ps and their outcomes ys: the
  isotonic fit of outcome on forecast (CORP), as a step function of a new
  forecast (the step at or below it; the lowest step below the range)."
  [ps ys]
  (let [pairs (sort-by first (map vector ps ys))
        xs (double-array (map first pairs))
        fit (double-array (isotonic (map #(if (second %) 1.0 0.0) pairs) (repeat (count pairs) 1.0)))
        n (alength xs)]
    (fn [p]
      ;; the last index with x <= p, by bisection
      (let [j (loop [lo -1 hi n]
                (if (<= (- hi lo) 1)
                  lo
                  (let [mid (quot (+ lo hi) 2)]
                    (if (<= (aget xs mid) p) (recur mid hi) (recur lo mid)))))]
        (aget fit (max 0 (min (dec n) j)))))))

(defn logistic-map
  "Recalibration learned from past forecasts ps and outcomes ys: a logistic
  regression of the outcome on the forecast's logit (Cox 1958), so a
  forecast too extreme is pulled in by a slope below one. Strictly
  monotone, so it keeps the forecasts' order, where the isotonic fit's
  flat steps tie them."
  [ps ys]
  (let [logit (fn [p] (let [p (min (max p 1e-6) (- 1.0 1e-6))] (Math/log (/ p (- 1.0 p)))))
        ;; Newton's method on two parameters, from the identity map
        rows (map (fn [p y] [(logit p) (if y 1.0 0.0)]) ps ys)]
    (loop [a 0.0 b 1.0 it 0]
      (let [[ga gb haa hab hbb]
            (reduce (fn [[ga gb haa hab hbb] [x y]]
                      (let [q (/ 1.0 (+ 1.0 (Math/exp (- (+ a (* b x))))))
                            r (- y q) v (* q (- 1.0 q))]
                        [(+ ga r) (+ gb (* r x)) (+ haa v) (+ hab (* v x)) (+ hbb (* v x x))]))
                    [0.0 0.0 1e-6 0.0 1e-6] rows)
            det (- (* haa hbb) (* hab hab))
            da (/ (- (* hbb ga) (* hab gb)) det)
            db (/ (- (* haa gb) (* hab ga)) det)]
        (if (or (>= it 50) (< (+ (Math/abs da) (Math/abs db)) 1e-9))
          (fn [p] (/ 1.0 (+ 1.0 (Math/exp (- (+ a (* b (logit p))))))))
          (recur (+ a da) (+ b db) (inc it)))))))
