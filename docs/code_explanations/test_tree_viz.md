# `tests/test_tree_viz.py` — visualization contract tests

`iris_tree()` supplies one fitted multiclass model for reusable text/DOT/plot tests. Regression, stump, and unfitted variants are built locally where their special conditions matter. `tmp_path` is pytest's isolated temporary directory, so generated PNG checks never clutter the project.

- `test_export_text_structure` checks readable branch indentation, feature conditions, and leaves.
- Default-name/no-data behavior confirms the API can render `x[index]` labels and works when node training summaries were not supplied.
- Regression export verifies that leaves display numeric `value` rather than class names.
- DOT test checks the graph wrapper and edges without depending on Graphviz installation.
- PNG save tests assert an artifact is created; regression verifies continuous node output.
- Single-leaf plotting checks the degenerate depth-zero layout and value normalization.
- Name-length validation prevents mislabeling a feature index with the wrong display name.
- Unfitted-tree test expects a deliberate error instead of a confusing attribute failure.

The tests do not pixel-compare images, since fonts/backends differ across systems. They verify structure and artifact creation; visual aesthetics remain a manual review step.
