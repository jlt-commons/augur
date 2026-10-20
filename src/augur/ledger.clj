(ns augur.ledger
  "Forecasts registered, resolved, scored and used to recalibrate: a
  forecast is made now, its outcome is known later, and the record of past
  forecasts under a key recalibrates new ones.

  A ledger is plain EDN. Keys are any EDN value ([:gate :no-edits \"glm\"]);
  ids are any EDN value unique among the open forecasts. Times (:at) are
  whatever the caller counts in (a turn, a run, a bin): the ledger only
  orders by them.

  Everything that grows is capped by an option, and what falls off is kept
  as counts: open forecasts beyond :max-open are expired oldest first,
  resolved rows beyond :max-rows per key are dropped from the rows (their
  scores are already in the key's sums), and keys beyond :max-keys are
  folded, least recently used first, into the sums under ::other.

  Scores are sums decayed by :forget at each resolution of their key, so
  `record` weighs recent forecasts more. Brier and log scores are kept for
  the forecast and for each benchmark registered with it.

  Options to `empty`, all required:
    :max-open     open forecasts kept
    :max-rows     resolved rows kept per key, for refitting
    :max-keys     keys kept
    :forget       the discount of a key's sums at each of its resolutions
    :calibration  `augur.calibrate` options (:edges :shrink :forget); each
                  resolution is learned as a batch of one, so its :forget
                  applies per resolution of the key"
  (:refer-clojure :exclude [empty keys resolve])
  (:require [augur.calibrate :as cal]))

(def ^:private eps
  "Probabilities are kept this far inside (0, 1) for the log score."
  1e-12)

(defn empty
  "A ledger with nothing registered."
  [opts]
  (when-let [missing (seq (remove #(some? (get opts %)) [:max-open :max-rows :max-keys :forget :calibration]))]
    (throw (ex-info (str "augur.ledger/empty needs " (vec missing)) {:missing (vec missing)})))
  {:opts opts :open {} :by-key {} :unknown 0 :dropped-keys 0})

(defn- need-ledger [L]
  (or L (throw (ex-info "augur.ledger: a nil ledger cannot be written; start from ledger/empty" {}))))

(defn- scores
  "{:brier :log} of probability p for outcome y."
  [p y]
  (let [p (double p) o (if y 1.0 0.0)
        q (min (max p eps) (- 1.0 eps))]
    {:brier (* (- p o) (- p o)) :log (Math/log (if y q (- 1.0 q)))}))

(defn- add-sums
  "Sums s (nil for none) decayed by f, plus one forecast's scores."
  [s f sc]
  (let [{:keys [n brier log]} (or s {:n 0.0 :brier 0.0 :log 0.0})]
    {:n (+ (* f n) 1.0) :brier (+ (* f brier) (:brier sc)) :log (+ (* f log) (:log sc))}))

(defn- fold-sums [a b]
  (merge-with (fn [x y] (if (map? x) (merge-with fold-sums x y) (+ x y))) a b))

(defn- evict-keys
  "L with keys beyond :max-keys folded into ::other, least recent first."
  [{:keys [opts by-key] :as L}]
  (let [max-keys (:max-keys opts)
        live (dissoc by-key ::other)]
    (if (<= (count live) max-keys)
      L
      (let [victims (take (- (count live) max-keys) (sort-by (comp :last val) live))]
        (reduce (fn [L [k {:keys [sums bench expired]}]]
                  (-> L
                      (update :by-key dissoc k)
                      (update-in [:by-key ::other] (fn [o] (-> (or o {})
                                                               (update :sums #(if % (fold-sums % sums) sums))
                                                               (update :bench #(fold-sums (or % {}) (or bench {})))
                                                               (update :expired (fnil + 0) (or expired 0)))))
                      (update :dropped-keys inc)))
                L victims)))))

(defn- expire-ids
  "L with open forecasts `ids` dropped and counted as expired under their keys."
  [L ids at]
  (reduce (fn [L id]
            (let [{:keys [key]} (get-in L [:open id])]
              (-> L
                  (update :open dissoc id)
                  (update-in [:by-key key :expired] (fnil inc 0))
                  (update-in [:by-key key :last] #(if (some? at) at %)))))
          L ids))

(defn register
  "L with forecast {:id :key :p :at}, optionally :bench {name p} (benchmark
  forecasts, scored beside it) and :x (features, kept with its row for
  refitting). An id already open is an error. Beyond :max-open, the oldest
  open forecasts expire."
  [L {:keys [id key p at bench x] :as forecast}]
  (let [L (need-ledger L)]
    (when (contains? (:open L) id)
      (throw (ex-info "augur.ledger/register: id already open" {:id id})))
    (when-not (and (some? id) (number? p) (<= 0.0 p 1.0))
      (throw (ex-info "augur.ledger/register needs an :id and a :p in [0 1]" {:id id :p p})))
    (let [L (assoc-in L [:open id] (cond-> {:key key :p (double p) :at at}
                                     bench (assoc :bench bench)
                                     (some? x) (assoc :x x)))
          over (- (count (:open L)) (get-in L [:opts :max-open]))]
      (if (pos? over)
        (expire-ids L (map first (take over (sort-by (comp :at val) (:open L)))) at)
        L))))

(defn resolve
  "L with forecast `id` resolved by `outcome` (true or false): scored with
  its benchmarks into its key's decayed sums, learned by the key's
  calibration, and kept as a row (up to :max-rows). An id not open is
  counted under :unknown and otherwise ignored. Options: :at, the time of
  resolution."
  [L id outcome {:keys [at]}]
  (let [L (need-ledger L)]
    (if-let [{:keys [key p bench x] made :at} (get-in L [:open id])]
      (let [{:keys [forget max-rows calibration]} (:opts L)
            y (boolean outcome)]
        (-> L
            (update :open dissoc id)
            (update-in [:by-key key] (fn [e]
                                       (let [e (or e {})]
                                         (-> e
                                             (assoc :sums (add-sums (:sums e) forget (scores p y)))
                                             (assoc :bench (reduce-kv (fn [acc b pb] (assoc acc b (add-sums (get acc b) forget (scores pb y))))
                                                                      (or (:bench e) {}) (or bench {})))
                                             (assoc :cal (get (cal/learn (if-let [c (:cal e)] {key c} {}) [{:key key :p p :outcome y}] calibration) key))
                                             (assoc :rows (let [rows (conj (vec (:rows e)) (cond-> {:id id :p p :y y :at made :resolved at}
                                                                                                    (some? x) (assoc :x x)))]
                                                            (if (> (count rows) max-rows) (subvec rows (- (count rows) max-rows)) rows)))
                                             (assoc :last (if (some? at) at made))))))
            evict-keys))
      (update L :unknown inc))))

(defn expire
  "L with the open forecasts that satisfy (pred forecast) dropped, never to
  be scored (their run ended without an outcome), counted per key. Each
  forecast is passed with its :id. Options: :at, now."
  [L pred {:keys [at]}]
  (let [L (need-ledger L)]
    (expire-ids L (keep (fn [[id f]] (when (pred (assoc f :id id)) id)) (:open L)) at)))

(defn calibrated
  "Probability p recalibrated by key k's record (`augur.calibrate`), or p
  itself for a key with none."
  [L k p]
  (if-let [table (get-in L [:by-key k :cal])]
    (cal/calibrate {k table} k p (get-in L [:opts :calibration]))
    p))

(defn- reliability [L table]
  (let [edges (get-in L [:opts :calibration :edges])]
    (vec (for [i (range (count edges))
               :let [n (get-in table [:n i] 0.0)]
               :when (pos? n)]
           [(/ (+ (if (zero? i) 0.0 (nth edges (dec i))) (nth edges i)) 2.0)
            (/ (get-in table [:pos i]) n) n]))))

(defn record
  "Key k's record: {:n the decayed count of resolved forecasts, :brier and
  :log their decayed mean scores, :skill {bench Brier skill against it},
  :reliability [[bin-centre rate n] ...], :open, :expired}. k may be
  ::other, the keys folded away. nil scores without resolved forecasts."
  [L k]
  (let [{:keys [sums bench cal expired]} (get-in L [:by-key k])
        n (:n sums 0.0)
        mean #(when (pos? n) (/ % n))]
    {:n n
     :brier (some-> sums :brier mean)
     :log (some-> sums :log mean)
     :skill (into {} (for [[b s] bench
                           :let [bb (:brier s)]
                           :when (and (pos? n) (pos? bb))]
                       [b (- 1.0 (/ (:brier sums) bb))]))
     :reliability (if cal (reliability L cal) [])
     :open (count (filter #(= k (:key %)) (vals (:open L))))
     :expired (or expired 0)}))

(defn rows
  "Key k's resolved rows {:id :p :y :x :at :resolved}, oldest first."
  [L k]
  (get-in L [:by-key k :rows] []))

(defn keys
  "Every key with a record or an open forecast."
  [L]
  (vec (distinct (concat (clojure.core/keys (:by-key L)) (map :key (vals (:open L)))))))

(defn stats
  "{:open :keys :unknown :dropped-keys}: how full the ledger is, and how
  many resolutions named no open forecast."
  [L]
  {:open (count (:open L)) :keys (count (:by-key L)) :unknown (:unknown L 0) :dropped-keys (:dropped-keys L 0)})
