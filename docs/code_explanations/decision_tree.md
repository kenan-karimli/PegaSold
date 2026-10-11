# `src/decision_tree.py` — detailed walkthrough

This is the from-scratch NumPy tree. The same recursive structure serves two tasks: classification predicts a category and regression predicts a number. The estimator stores split rules as `TreeNode` objects and exposes `fit`, `predict`, and classification-only probability estimates.

## Imports, labels, and tree-node representation

The only dependency is NumPy. All candidate split statistics, class counts, means, and predictions are computed directly in arrays.

`TreeNode` describes one node, without storing the full dataset subset:

- `value` is a leaf prediction. It is `None` at an internal split.
- `feature_index` identifies the input column used at a split. `None` marks a leaf and is the universal leaf check.
- `threshold` is the numeric boundary. Rows with `x[feature_index] <= threshold` go left; rows above go right.
- `left` and `right` reference child nodes.
- `impurity_decrease` stores the local gain for feature importance.
- `n_samples` stores how many training rows reached the node.

Classification leaves additionally receive `class_counts` when built. Counts allow `predict_proba` and visualization without retaining every leaf training label. Internal nodes need no label frequencies for prediction.

## `DecisionTree.__init__`: settings and validation

`task` must be one of two explicit modes. If `criterion=None`, classification defaults to Gini and regression to MSE. Classification accepts `gini` or `entropy`; regression accepts only `mse`. Invalid settings fail immediately rather than silently fitting a different model.

- `max_depth=None` means no explicit depth cap; zero is a valid cap and gives only a root leaf.
- `min_samples_split` is the number of examples needed before considering a split.
- `min_samples_leaf` is a constraint on the size of each child. It is more restrictive than just checking parent size.
- `min_impurity_decrease` is a gain floor for accepting a split.

`root=None` marks an unfitted estimator. Fit-time values like `classes_`, `n_features_in_`, and `feature_importances_` are only set after successful fitting.

## `fit(X, y)`: validation, recursive build, and importance

Inputs are normalized to arrays. X must have two axes; y must have one. Their row counts must match and neither samples nor features can be empty. Input X must be numeric and finite; the algorithm cannot sort NaN because NaN has no ordinary order. Regression y is converted to float and also checked finite. Classification can use string, integer, or other comparable labels.

`n_features_in_` is retained so later prediction can reject a matrix with the wrong number of columns. `classes_ = np.unique(y)` stores sorted class labels; this fixes the meaning/order of probability columns. Regression has no class vocabulary.

`_build_tree(X,y,depth=0)` returns the root. Feature importance then traverses the structure: for each internal node, add `node.n_samples * node.impurity_decrease` to the split feature. Dividing all sums by their total yields normalized scores that add to one. If there were no useful splits, the all-zero vector is returned. Weighting by samples means a root split affects more observations than an equally sized gain deep in a small subtree.

## Impurity math

For class proportions $p_k$:

- Gini: $G=1-\sum_k p_k^2$. It is zero when a node is pure.
- Entropy: $H=-\sum_{k:p_k>0}p_k\log_2(p_k)$. The code removes zero-probability terms so it never calculates `log(0)`.

Regression impurity is population variance/MSE: $I=\frac{1}{n}\sum_i(y_i-\bar y)^2$. Each split gain is parent impurity minus child impurity weighted by sample fraction:

`gain = parent - (n_left/n)*impurity(left) - (n_right/n)*impurity(right)`.

The gain is in target units squared for regression, and dimensionless for classification.

## `_build_tree`: how each recursive node is built

1. `_leaf_value(y)` is calculated first. It gives a safe fallback prediction even if the node cannot split.
2. `_should_stop` checks the basic stopping rules. If true, `_make_leaf` retains the prediction and its class counts.
3. `_best_split` searches for the strongest legal split. No candidate means the node becomes a leaf.
4. A gain below `min_impurity_decrease` also becomes a leaf.
5. The selected feature/threshold creates one Boolean mask; its complement creates the right mask. X and y use the same masks so feature/target alignment is maintained.
6. An empty child is guarded against, though min-leaf candidate filtering should prevent it. The fallback makes recursion robust to future split-search changes.
7. Both children recurse at `depth+1`; the parent stores selected feature, threshold, child references, sample count, and gain.

The tree is binary. Every internal node has exactly two children, which keeps prediction and plotting logic simple.

## `_make_leaf`, `_should_stop`, `_leaf_value`

`_make_leaf` records `n_samples`. For classification it uses `np.unique(..., return_counts=True)` and stores a dictionary label → count. For regression only the mean prediction is needed.

`_should_stop` checks, in order: too few rows to split, configured maximum depth reached, a single unique class, or a constant regression target. A classification node can have nonzero impurity but still stop if it reaches a user-specified depth/size boundary.

`_leaf_value` uses majority class (`argmax` of unique-label counts) or numeric mean. With ties, NumPy's sorted unique-label order makes the chosen class deterministic.

## `_best_split`: the central optimized routine

Let `n_samples,n_features=X.shape`. Parent impurity is calculated once. `best_gain=-inf` ensures the first valid candidate can win. For classification, targets are converted to integer positions in sorted unique classes with `np.unique(return_inverse=True)`. `onehot` gives each row a one-hot label and `total` gives whole-node class counts. For regression, `yy`, `total_sum`, and `total_sq` are precomputed once.

For each feature:

1. `np.argsort(..., kind="mergesort")` gives a stable row order. Stable order plus a deterministic feature loop makes ties reproducible.
2. `values=...` is the sorted feature vector. `values[:-1] < values[1:]` marks locations between distinct adjacent values; equal values cannot be divided by any threshold without splitting identical feature values inconsistently.
3. `positions` converts a boundary into the number of rows assigned left. Boundaries that violate `min_samples_leaf` on either side are removed before statistics are computed.
4. `thresholds` places each split halfway between adjacent distinct values. The subtraction-first formula `a+(b-a)/2` avoids an unnecessary large `a+b` sum.

### Classification cumulative counts

`np.cumsum(onehot[order], axis=0)` accumulates class counts down the sorted feature. At boundary `position`, the previous row is left; subtracting its prefix counts from `total` gives right counts. `nl/nr` are vectorized sizes for all candidate positions.

Gini child impurity is computed directly from counts: $1-\sum_k count_k^2/n^2$. Entropy divides counts by child size, and `np.maximum(probs,1e-300)` prevents `log2(0)` from producing warnings; `np.where` ensures zero terms contribute zero.

### Regression cumulative sums

Targets follow the feature sort. Prefix sums and squared sums allow each candidate child mean/variance to be calculated in constant time after sorting:

`variance = sum(y²)/n - (sum(y)/n)²`.

Small negative values can occur through floating-point cancellation even though variance is mathematically nonnegative, so `np.maximum(0, variance)` clips round-off noise to zero.

### Candidate comparison

All candidate gains for a feature are vectorized. `np.argmax` chooses the first local maximum; the outer feature loop and strict `gain > best_gain` keep ties deterministic. A candidate also must meet `min_impurity_decrease`. If no split qualifies, return `None`; otherwise return `(feature_index, threshold, gain)`.

The `_split_score(y_left,y_right,parent_impurity)` utility computes the same weighted-gain definition directly and is kept as a readable reference; the fast `_best_split` uses cumulative statistics so it does not create child arrays for every candidate.

## Prediction and probabilities

`predict(X)` requires a fitted tree, 2-D numeric finite X, and exactly the fitted feature count. It calls `_predict_one` per row and returns one-dimensional labels/numbers.

`_predict_one(row,node)` loops down the tree until `feature_index is None`; `<= threshold` follows left. The loop avoids Python recursion for every prediction, though each row still traverses one node per tree depth.

`predict_proba` is classification-only and validates shape/type/finiteness. `_predict_proba_one` follows the same path, then divides leaf class counts by their total. The list comprehension iterates `self.classes_` rather than dictionary order, ensuring probability columns align with `classes_`. The row probabilities sum to one. A pure leaf gives one-hot probabilities; a mixed leaf gives its observed training proportions.

## Implementation caveats

- All split features are numeric; categories have already been one-hot encoded in `data_prep.py`.
- A high-cardinality one-hot block increases split-search work because every feature is considered at every node.
- Greedy split search is locally optimal, not a globally optimal tree search.
- Importance can favor correlated features that happen to be selected first and is not causal evidence.
- Exact equivalence with scikit-learn is not guaranteed: float thresholds, tie handling, and stopping details can vary even with matched hyperparameters.
- This estimator's fit input must already be finite; missing-value handling is the preparation module's responsibility.
