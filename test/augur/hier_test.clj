(ns augur.hier-test
  (:require [augur.hier :as h]
            [augur.math :as m]
            [clojure.test :refer [deftest is testing]]))

;; a tree of 8 regions x 4 subregions x 4 leaves
(def leaves
  (vec (for [r (range 8) s (range 4) l (range 4)]
         [(str "r" r) (str "r" r "s" s) (str "r" r "s" s "l" l)])))

(defn- loss [rows f]
  (/ (reduce + (for [{:keys [y] :as r} rows
                     :let [p (f r)]]
                 (- (Math/log (if y p (- 1.0 p))))))
     (count rows)))

(deftest nested-intercepts-pool-toward-the-parent
  (let [g (m/rng 3)
        truth (into {} (for [path leaves id path]
                         [id (* (case (count id) 2 1.0 4 0.5 0.3) (m/normal! g))]))
        eta (fn [path] (+ -2.0 (reduce + (map truth path))))
        draw (fn [n] (vec (for [path leaves _ (range n)]
                            {:path path :o -2.0 :y (< (m/next-double! g) (m/expit (eta path))) :w 1.0})))
        train (draw 30)
        test (draw 30)
        fit (h/intercepts train {:tau2 [0.1 0.1 0.1] :sweeps 30})]
    (testing "the effects predict held-out rows better than none"
      (is (< (loss test #(m/expit (+ -2.0 (h/effect fit (:path %)))))
             (loss test (constantly (/ (count (filter :y train)) (double (count train))))))))
    (testing "the variances are estimated, largest at the top"
      (is (> (first (:tau2 fit)) (last (:tau2 fit))) (:tau2 fit)))
    (testing "a leaf never seen takes its subregion's effect"
      (let [path ["r0" "r0s0" "new"]]
        (is (= (h/effect fit path) (h/effect fit ["r0" "r0s0"])))))))

(deftest nested-intercepts-respect-offsets-and-weights
  (testing "with no structure beyond the offset the effects stay near zero"
    (let [g (m/rng 5)
          rows (vec (for [path leaves _ (range 40)
                          :let [o (m/normal! g)]]
                      {:path path :o o :y (< (m/next-double! g) (m/expit o)) :w 1.0}))
          fit (h/intercepts rows {:tau2 [0.1 0.1 0.1] :sweeps 30})]
      (is (every? #(< (Math/abs (double %)) 0.25) (vals (:effects fit))))))
  (testing "doubling every weight is twice the data"
    (let [rows [{:path ["a" "b"] :o 0.0 :y true :w 1.0} {:path ["a" "b"] :o 0.0 :y false :w 1.0}
                {:path ["a" "b"] :o 0.0 :y true :w 1.0}]
          once (h/intercepts rows {:tau2 [1.0 1.0] :sweeps 50 :fixed true})
          twice (h/intercepts (concat rows rows) {:tau2 [1.0 1.0] :sweeps 50 :fixed true})
          heavy (h/intercepts (map #(assoc % :w 2.0) rows) {:tau2 [1.0 1.0] :sweeps 50 :fixed true})]
      (is (< (Math/abs (- (h/effect twice ["a" "b"]) (h/effect heavy ["a" "b"]))) 1e-9))
      (is (> (h/effect twice ["a" "b"]) (h/effect once ["a" "b"]))))))

;; --- dynamic states --------------------------------------------------------

(defn- simulate
  "Counts per leaf and quarter from a world and region AR(1) state."
  [seed T rho q]
  (let [g (m/rng seed)
        regions (distinct (map first leaves))
        step (fn [s] (+ (* rho s) (* (Math/sqrt q) (m/normal! g))))
        world (vec (take T (iterate step 0.0)))
        region (into {} (for [r regions] [r (vec (take T (iterate step 0.0)))]))
        e 0.05
        obs (into {} (for [t (range T) [r _ leaf] leaves
                           :let [lam (* e (Math/exp (+ (world t) ((region r) t))))]]
                       [[leaf t] [(double (m/poisson! g lam)) e]]))]
    {:obs obs :world world :region region}))

(def paths (into {} (for [[r s l] leaves] [l ["world" r s l]])))

(def opts {:rho [0.5 0.5 0.5 0.5] :sd [0.2 0.2 0.2 0.2] :rhos [0.5 0.8 0.95] :sds [0.0 0.1 0.2 0.4 0.8] :passes 2 :penalty 2.0 :horizon 1})

(defn- corr [xs ys]
  (let [n (count xs) mx (/ (reduce + xs) n) my (/ (reduce + ys) n)
        sxy (reduce + (map #(* (- %1 mx) (- %2 my)) xs ys))
        sxx (reduce + (map #(let [d (- % mx)] (* d d)) xs))
        syy (reduce + (map #(let [d (- % my)] (* d d)) ys))]
    (/ sxy (Math/sqrt (* sxx syy)))))

(deftest dynamic-states-learn-their-persistence
  (let [{:keys [obs world region]} (simulate 11 160 0.95 0.04)
        fit (h/states obs paths [0 159] opts)]
    (testing "the world and region levels learn high persistence"
      (is (< 0.75 (first (:rho fit)) 1.0) (:rho fit))
      (is (< 0.75 (second (:rho fit)) 1.0) (:rho fit)))
    (testing "the filtered world state follows the true one"
      (is (> (corr (map #(h/state-mean fit "world" %) (range 160)) world) 0.7)))
    (testing "a region's filtered state follows its own"
      (is (> (corr (map #(h/state-mean fit "r1" %) (range 160)) (region "r1")) 0.6)))
    (testing "the leaves, with no state of their own, get less than the world"
      (is (< (last (:sd fit)) (first (:sd fit))) (:sd fit)))))

(deftest filtered-states-never-look-ahead
  (let [{:keys [obs]} (simulate 2 120 0.9 0.05)
        fit (h/states obs paths [0 119] opts)
        refilter (h/refilter fit (into {} (filter #(<= (second (key %)) 80) obs)) [0 119])]
    (testing "with the same hyperparameters, the state at 80 ignores what came after"
      (doseq [node ["world" "r0" "r2s3"]]
        (is (< (Math/abs (- (h/state-mean fit node 80) (h/state-mean refilter node 80))) 1e-12) node)))))

(deftest multipliers-decay-to-one
  (let [{:keys [obs]} (simulate 4 120 0.9 0.08)
        fit (h/states obs paths [0 119] opts)
        path (paths "r0s0l0")
        near (h/multiplier fit path 119 [1 4])
        far (h/multiplier fit path 119 [400 403])]
    (is (< (Math/abs (- far 1.0)) 1e-3) far)
    (is (pos? near))
    (testing "a hot region's leaves are lifted"
      (let [hot (assoc obs ["r0s0l0" 119] [5.0 0.05] ["r0s1l2" 119] [4.0 0.05] ["r0s2l1" 119] [4.0 0.05])
            fit2 (h/refilter fit hot [0 119])]
        (is (> (h/multiplier fit2 path 119 [1 4]) near))))))

(deftest groups-are-at-least-as-likely-as-independence-says
  ;; shared states make onsets come together: the chance of any onset in a
  ;; group, once states are uncertain, differs from the independent product
  (let [{:keys [obs]} (simulate 6 120 0.9 0.1)
        fit (h/states obs paths [0 119] opts)
        members (for [[r s l] leaves :when (= s "r1s2")] l)
        ps (into {} (for [l members] [l 0.05]))
        indep (- 1.0 (reduce * (map #(- 1.0 %) (vals ps))))
        any (h/any-probability fit (select-keys paths members) ps 119 [1 4] {:draws 2000 :seed 1})]
    (is (< 0.0 any 1.0))
    (testing "with states far from now, the mean multiplier is one and sharing only adds variance"
      (let [far (h/any-probability fit (select-keys paths members) ps 119 [400 403] {:draws 4000 :seed 1})]
        (is (< far (+ indep 0.01)) [far indep])))))

(deftest states-are-chosen-at-the-horizon-they-serve
  ;; the settings chosen to forecast eight periods out forecast eight
  ;; periods out at least as well as those chosen for the next period
  (let [{:keys [obs]} (simulate 5 200 0.5 0.6)
        near (h/states obs paths [0 199] opts)
        far (h/states obs paths [0 199] (assoc opts :horizon 8))
        near-at-8 (h/states obs paths [0 199] (assoc opts :horizon 8 :rho (:rho near) :sd (:sd near) :passes 0))]
    (is (>= (:score far) (:score near-at-8)))
    (testing "the horizon changes what is scored"
      (is (not= (:ll near) (:ll near-at-8))))))

(deftest nested-intercepts-reach-the-mode-with-few-rows
  ;; a leaf with a few rows and a loose prior has little curvature, where
  ;; an undamped Newton step overshoots and oscillates
  (let [rows (vec (concat (for [w [4.0 4.4 5.6]] {:path ["few"] :o -3.77 :y false :w w})
                          (repeat 4 {:path ["few"] :o -3.77 :y true :w 0.9})))
        fit (h/intercepts rows {:tau2 [3.4] :sweeps 2 :fixed true :start {"few" -0.65}})
        a (get (:effects fit) "few")
        grad (+ (- (/ a 3.4))
                (reduce + (for [{:keys [o y w]} rows
                                :let [p (m/expit (+ o a))]]
                            (* w (- (if y 1.0 0.0) p)))))]
    (testing "the leaf sits at its mode"
      (is (< (Math/abs (double grad)) 1e-4) (:effects fit))))
  (testing "the variances stay finite when estimated"
    (let [rows (vec (for [l (range 30) i (range 7)]
                      {:path ["a" (str "l" l)] :o -3.35 :y (< i 4) :w (if (< i 4) 0.9 4.5)}))
          fit (h/intercepts rows {:tau2 [0.7 3.4] :sweeps 12})]
      (is (every? #(< % 100.0) (:tau2 fit)) (:tau2 fit)))))

(deftest a-prior-on-the-spread-reins-in-separated-nodes
  ;; leaves with all their rows one way pull their effects outward and the
  ;; EM reads that as spread; a half-normal prior on each level's sd (Gelman
  ;; 2006; Chung et al. 2013) shrinks it unless the data insist
  (let [rows (vec (for [l (range 40) i (range 12)]
                    {:path [(str "l" l)] :o -3.0 :y (and (< l 4) (< i 8)) :w 1.0}))
        free (h/intercepts rows {:tau2 [1.0] :sweeps 40})
        tight (h/intercepts rows {:tau2 [1.0] :sweeps 40 :sd-prior [0.5]})
        loose (h/intercepts rows {:tau2 [1.0] :sweeps 40 :sd-prior [1e6]})]
    (is (< (first (:tau2 tight)) (* 0.8 (first (:tau2 free)))) [(:tau2 tight) (:tau2 free)])
    (testing "a very wide prior is no prior"
      (is (< (Math/abs (- (first (:tau2 loose)) (first (:tau2 free)))) 1e-3) [(:tau2 loose) (:tau2 free)])))
  (is (thrown? Exception (h/intercepts [{:path ["a"] :o 0.0 :y true :w 1.0}] {:tau2 [1.0 1.0] :sweeps 1 :sd-prior [1.0]}))
      "one scale per level"))
