# W02 — Decision Trees (ID3, C4.5, CART)

Slides: `Decision_Tree.pdf`. Code: [decision_tree.py](../../src/from_scratch/decision_tree.py), [w02_decision_tree.py](../../experiments/part1_pre_midterm/w02_decision_tree.py)

## A. Concept capsule
- **Model:** A flowchart of feature tests. Internal nodes test one feature (for example $x_j \le T$), and leaves predict the majority class (or the mean target for regression). The tree is grown greedily, top-down, by choosing the split that reduces impurity the most.
- **Key assumptions:** None about the data distribution (non-parametric). Decision boundaries are **axis-aligned**, since each test uses one feature. Greedy local choices are assumed to give a good tree.
- **Objective / split criteria:**
  - Entropy: $H(S) = -\sum_i p_i \log_2 p_i$, used by ID3 through the information gain $IG(S, A) = H(S) - \sum_v \frac{|S_v|}{|S|} H(S_v)$.
  - Gain ratio (C4.5): $GR = IG / \mathrm{SplitInfo}$.
  - Gini (CART): $G(t) = 1 - \sum_k p(k \mid t)^2$, maximising $\Delta G(s, t) = G(t) - \frac{N_L}{N_t} G(t_L) - \frac{N_R}{N_t} G(t_R)$.
  - Overfitting is controlled by pruning: pre-pruning (max depth, minimum samples) or post-pruning.

## B. Worked example
Split Play Tennis on **Humidity** (the slide exercise). $S$ has 9 Yes / 5 No, so $H(S) = 0.940$ and $G(S) = 0.459$.
- High: 3 Yes / 4 No, so $H = 0.985$ and $G = 0.490$.
- Normal: 6 Yes / 1 No, so $H = 0.592$ and $G = 0.245$.
- Weighted entropy: $\frac{7}{14} \cdot 0.985 + \frac{7}{14} \cdot 0.592 = 0.789$, so $IG = 0.940 - 0.789 = 0.152$.
- Weighted Gini: $0.367$, so $\Delta G = 0.092$.
- Outlook has the higher $IG = 0.247$, so ID3 puts **Outlook** at the root. My `id3()` produces the slide tree: Overcast → Yes, Sunny → Humidity, Rain → Wind.

## C. Code-to-theory trace
- **Entropy and Gini:** `entropy()` at lines 14–18 and `gini()` at lines 21–25 in `decision_tree.py`.
- **Information gain, SplitInfo and gain ratio:** `information_gain()`, `split_info()` and `gain_ratio()`, lines 28–44.
- **ID3 greedy choice (argmax IG) and its stopping rules** (pure node, no attribute left): `id3()`, lines 52–69. The choice itself is at line 66.
- **Continuous thresholds (midpoints between sorted values):** `_best_split()`, lines 106–125. Lines 114–115 build the left/right class counts for every threshold with a cumulative sum, and line 116 keeps only cuts between distinct values.
- **Split quality $\Delta G(s, t)$:** lines 120–121.
- **Pre-pruning** (pure, too few samples, `max_depth`, `min_impurity_decrease`): `_grow()`, lines 129–133.
- **Leaf prediction (majority class):** `Node.prediction`, line 86.
- **Reduced-error post-pruning** (collapse a subtree if the leaf error is not worse): `prune()`, lines 158–172. The collapse happens at lines 166–167.
- The slide numbers are reproduced in [test_decision_tree.py](../../tests/test_decision_tree.py): $H = 0.94$, $G = 0.46$, IG $= 0.25 / 0.15 / 0.05 / 0.03$, $GR = 0.16$.

## D. Controlled experiment
- **Question:** How does the maximum depth (pre-pruning) affect a Gini tree on HAR?
- **Setup:** Grow the full tree once (depth 17, 129 leaves), then evaluate it truncated at depth 1–15. This is identical to training with `max_depth`, because growth is greedy. The same was repeated with entropy.
- **Result:** For Gini, train macro-F1 rises monotonically (0.23 → 1.00). Validation macro-F1 peaks at **0.841 at depth 4** and then drops to 0.722 for the full tree. The selected tree (Gini, depth 4) reaches **test macro-F1 = 0.8359**, exactly the same as sklearn's `DecisionTreeClassifier` with the same settings. The full tree scores 0.8068. Reduced-error post-pruning shrinks it from 129 to 18 leaves and raises it to 0.8324.
- **Conclusion:** A deeper tree memorises the training subjects. Pruning (pre or post) is what makes a tree generalise.

## E. Failure
- **Observed error:** The walking classes are the weakest (F1 0.75–0.81), while LAYING is perfect (1.00). The tree is about 10 F1 points below my W03/W04 models (0.937 and 0.947).
- **Cause:** Axis-aligned splits must separate the classes one feature at a time. The walking classes differ only through combinations of many correlated features, which a linear or MLP model captures directly. LAYING is isolated by a single gravity feature (`tGravityAcc-min()-X`), which is exactly the root split.
- **Common misconception:** "Entropy and Gini give very different trees." Here they choose different splits from depth 2 onward, but their best validation scores are close (0.841 vs 0.824).

## F. Written-exam capsule
A decision tree predicts by routing a sample from the root through a sequence of feature tests to a leaf, which outputs the majority class of the training samples that reached it. The tree is built greedily from the top. At each node, the algorithm picks the test that most reduces impurity: information gain based on entropy (ID3), gain ratio to avoid favouring many-valued attributes (C4.5), or the Gini decrease with binary splits (CART). Continuous features are handled by testing thresholds at the midpoints between sorted values. Because a fully grown tree can fit every training point, trees overfit easily and are controlled by pre-pruning (depth or size limits) or post-pruning (reduced-error, pessimistic-error or cost-complexity pruning).

## G. Reflection
- **Understood:** Why a greedy tree truncated at depth $d$ is the same as a tree trained with `max_depth = d`, and why the cumulative-count trick makes the threshold search $O(n \log n)$ per feature.
- **Still unclear / next:** Cost-complexity pruning ($R_\alpha(T) = R(T) + \alpha |T|$) with cross-validation, and whether a random forest closes the gap to the MLP.

## H. Inquiry trail
<!-- To be filled in by me (AI use disclosure). -->
