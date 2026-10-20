(ns augur.analog
  "Analog forecasts (Delle Monache et al., MWR 2013, for weather): the
  chance of an onset is the weighted share of onsets among the k most
  similar past rows, in inputs standardized over the training rows, shrunk
  toward the base rate by `prior` pseudo-rows. No shape is assumed, so a
  kind of crisis the coefficients learned on another driver miss can still
  be seen in the places it resembles.")

(defn fit
  "A model from rows {:x double[] :y bool :w weight} and {:k :prior}."
  [rows {:keys [k prior]}]
  (let [xs (mapv :x rows)
        n (count xs)
        width (alength ^doubles (first xs))
        mean (double-array width) sd (double-array width)]
    (dotimes [j width]
      (let [vs (map #(aget ^doubles % j) xs)
            m (/ (reduce + vs) n)]
        (aset mean j m)
        (aset sd j (Math/sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) vs)) n)))))
    (let [;; standardized inputs, flat; constant columns (sd 0) dropped
          cols (vec (filter #(pos? (aget sd %)) (range width)))
          d (count cols)
          z (double-array (* n d))]
      (dotimes [i n]
        (let [^doubles x (xs i)]
          (dotimes [c d]
            (let [j (cols c)] (aset z (+ (* i d) c) (/ (- (aget x j) (aget mean j)) (aget sd j)))))))
      {:k k :prior prior :n n :d d :cols cols :mean mean :sd sd :z z
       :y (boolean-array (map :y rows)) :w (double-array (map #(double (:w %)) rows))
       :base (let [wt (reduce + (map :w rows))]
               (/ (reduce + (map #(if (:y %) (:w %) 0.0) rows)) wt))})))

(defn predict
  "The analog forecast for inputs x."
  ^double [{:keys [k prior n d cols ^doubles mean ^doubles sd ^doubles z ^booleans y ^doubles w base]} ^doubles x]
  (let [n (long n) d (long d)
        q (double-array d)
        _ (dotimes [c d] (let [j (cols c)] (aset q c (/ (- (aget x j) (aget mean j)) (aget sd j)))))
        dist (double-array n)]
    (dotimes [i n]
      (let [o (* i d)]
        (aset dist i (loop [c 0 s 0.0]
                       (if (< c d) (let [v (- (aget z (+ o c)) (aget q c))] (recur (inc c) (+ s (* v v)))) s)))))
    ;; the k nearest: the k-th smallest distance from a primitive sort of a
    ;; copy, then every row within it (ties at the cut all count)
    (let [kk (min (long k) n)
          sorted (aclone dist)
          _ (java.util.Arrays/sort sorted)
          cut (aget sorted (dec kk))
          [wt hits] (loop [i 0 wt 0.0 hits 0.0]
                      (if (< i n)
                        (if (<= (aget dist i) cut)
                          (recur (inc i) (+ wt (aget w i)) (if (aget y i) (+ hits (aget w i)) hits))
                          (recur (inc i) wt hits))
                        [wt hits]))]
      (/ (+ hits (* prior base)) (+ wt prior)))))
