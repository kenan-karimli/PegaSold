# `src/tree_viz.py` — detailed tree inspection and plotting walkthrough

This module translates the fitted custom `DecisionTree` structure into text, Graphviz DOT, or a Matplotlib diagram. It is deliberately separate from the estimator: drawing a tree must not change its fit, thresholds, or predictions.

## `_is_leaf(node)`

The tree's sentinel for a leaf is `feature_index is None`. Both the model's prediction code and visualization module use that definition. There is no need to infer a leaf from missing children; the explicit split field is authoritative.

## `_collect_stats(tree,X,y)`

Tree nodes store prediction/split state but not all training rows. When a caller supplies X and y, this helper reconstructs node summaries by routing row indices through the tree.

1. If either X or y is absent, return `{}`. Rendering still works but omits sample/impurity information that cannot be recovered.
2. Convert inputs to arrays. Cast regression targets to float.
3. Recursive `walk(node,idx)` selects `ys=y[idx]`, computes this subset's size and impurity using the estimator's own `_impurity`, then stores it in `stats` keyed by `id(node)`.
4. At split nodes, compare the chosen feature values against threshold; recurse into left and right index subsets.

Using `id(node)` avoids adding visualization-only state to the estimator. The helper recomputes data-dependent summaries only when requested; it does not mutate the model. Memory holds only row-index subsets during recursion and y slices in the returned stats.

## Name validation and node labels

`_names` generates fallback names `x[0]`, `x[1]`, etc., or validates the provided feature-name count against the model's fitted feature count. This prevents an apparently polished diagram from labeling a split with the wrong source column.

`_class_names` returns `None` for regression. For classification it stringifies `tree.classes_` if no custom names are supplied, or checks the custom list length. The model's sorted `classes_` order governs all class-count/probability columns.

`_node_summary(tree,node,stats,cnames)` provides the shared display values `(sample_count, impurity, class_counts, prediction_text)` to text/DOT/plot outputs:

- For a classification node with supplied data, reconstruct class counts from its training target subset, then choose the `argmax` class as text prediction.
- For a leaf without data, stored `node.class_counts` still gives counts/prediction.
- For an internal classifier without `X,y`, node training counts are unavailable, so it returns `None` for summaries rather than inventing counts.
- For regression with data, sample count, impurity and mean target are computed from routed rows. A leaf without data still shows its saved value.
- Internal regression nodes without data have no stored prediction and omit it.

## `export_text(...)`

It rejects an unfitted tree, resolves feature/class labels, and optionally routes X/y for summaries. The recursive local `rec(node,depth)` builds a list of lines with indentation proportional to depth. Internal nodes display one `<=` branch and its recursive subtree, then a `>` branch and right subtree. Leaf labels say `class` or `value`; optional counts appear in parentheses. `decimals` formats thresholds/regression values. Finally `"\n".join(lines)` returns text rather than printing, so callers can display or save it.

## `to_dot(...)`

DOT is a text graph specification for Graphviz. The function requires a fitted model and uses the same name/stat helpers. `counter=[0]` is a mutable integer holder so nested recursive calls can allocate sequential node IDs. For each node it composes a multiline label containing split, criterion impurity, samples, class counts, and prediction when available. Leaves get a light green fill, internal nodes white. Recursive child IDs are connected by True/False edges. The result is a single string; it does not launch external software.

One practical caveat: user-provided feature names are inserted into DOT labels; avoid quotes/backslashes in custom labels unless escaped. The project feature vocabulary is safe.

## Matplotlib layout

### `_layout(root)`

The recursive layout gives terminal leaves x positions from left to right. An internal node's x is the average of its children's x positions, centering parents over subtrees. The y coordinate is negative depth, so root is highest and deeper nodes lower. Positions are keyed by node identity. This keeps the tree readable without requiring Graphviz.

### `_tree_depth(node)`

Returns zero for a leaf and one plus the largest child depth otherwise. It determines plot height based on number of levels.

### `plot_tree(...)`

It lazily imports Matplotlib so importing the project doesn't require plot initialization. It validates fit and labels, calculates optional data summaries, positions, leaf count, and depth. If no Axes was passed, it creates a figure; default width scales with leaf count and height with depth. A supplied Axes lets another plot own layout/size. Axes are hidden because coordinates/scale are visual structure only.

Color rules:

- Classification uses a `tab10` palette. The dominant class selects hue, and node purity determines how strongly that hue is mixed with white. Without recoverable class counts, the node is white.
- Regression gathers leaf values, creates a value normalization range, and maps predictions through `viridis`. If every leaf predicts the same value, `hi+1` avoids a zero-width color interval.

Nested `draw(node)` first draws child edges behind the nodes. Edge labels True/False match the `<=` condition's left/right routing. It then assembles node text and places a rounded box at the precomputed location. `zorder` keeps boxes in front of lines. `set_xlim/ylim` adds padding around leaf extrema and root/deepest level. When `save_path` is supplied, `bbox_inches="tight"` trims external whitespace without cropping node text. The function returns `ax` so callers can customize/save it further.

Very large trees create many labels and an extremely wide figure; the project therefore exports a deliberately shallow depth-3 example rather than attempting to draw the entire final tree.

## `render_dot(dot_string,out_path)`

This is the only function that needs external Graphviz. It checks `shutil.which("dot")`; if absent it raises with an install hint. Output type is inferred from the suffix, then `subprocess.run` invokes Graphviz with argument-list (not shell) execution and feeds the DOT bytes on stdin. `check=True` propagates render errors. The output path is returned on success.

## Choosing an output

- Use `export_text` for diffs, terminal inspection, or README snippets.
- Use `plot_tree` for the project's standard portable Matplotlib figure.
- Use `to_dot` + `render_dot` when Graphviz styling or layout is needed.

All three renderings expose the same fitted split logic but differ in legibility for large trees. Importance and split diagrams describe predictive partitioning, not causal effects.
