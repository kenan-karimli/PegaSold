# `tests/test_decision_tree.py` — tree correctness suite

This test module tests the estimator's observable contract and selected internal invariants. It uses small deterministic NumPy arrays and sklearn's built-in Iris/reference trees; no Bina.az file is required.

## Fixture/helper functions

- `depth(node)`, `leaves(node)`, and `count_nodes(node)` recursively inspect model shape, used to verify depth/structure constraints.
- `leaf_sizes(tree,X)` routes rows to leaves and counts how many reach each leaf; this confirms minimum-leaf guarantees, not just a stored hyperparameter value.
- `iris()` loads a small canonical multiclass dataset.
- `clf_data()` and `reg_data()` make repeatable classification/regression arrays with known boundaries/noise.

## Test groups and what failures mean

### `TestTreeNode`

Checks that a default node is leaf-like and that constructor values are stored without alteration. A failure means the representation contract changed and recursion/rendering may no longer agree on node state.

### `TestInit`

Checks default Gini/MSE selection, accepted criteria, invalid task/criterion/depth/sample/gain settings, and `max_depth=0`. These tests ensure bad configurations fail early and default choices are task-appropriate.

### `TestFitValidation`

Exercises one-dimensional X, two-dimensional y, mismatched/empty rows, zero columns, nonnumeric/nonfinite X, and invalid regression targets. The estimator should reject invalid matrices clearly instead of learning from silently coerced values.

### Impurity and split-score tests

Check Gini, entropy, and MSE against hand-computed values, pure/constant impurity zero, weighted gain calculation, and nonnegative gains in expected examples. These are the math-level tests: if they fail, every recursive tree result can be wrong even if predictions look plausible.

### Leaf values and stopping conditions

Verify classification majority, deterministic tie choice, string labels, regression means, and stops for max depth, too few rows, pure classes, and constant y. The cases distinguish a sensible leaf from recursion that needlessly searches thresholds.

### `_best_split` behavior

Small examples assert midpoint thresholds, informative feature selection, no split on constants, expected regression splits, minimum leaf enforcement, zero-gain XOR handling, and the globally best candidate. XOR is especially useful: zero immediate gain can be permitted because a deeper tree may resolve interactions.

### Classification and structure

Tests include Iris quality, train memorization for unique data, holdout behavior, XOR depth needs, threshold equality routing left, edge-leaf extrapolation, string/bool/float labels, single-class/single-row/constant feature inputs, deterministic ties, row permutation, and feature scaling invariance. Structural checks require two children at every internal node, in-range feature indices, conserved leaf counts, majority-valued leaves, and a root split at least as good as alternatives.

### Hyperparameters and probabilities

Depth, minimum split/leaf counts and `min_impurity_decrease` are checked against built trees. `predict_proba` is verified for shape, row-sum-one, sorted class-column order, argmax consistency, pure leaves, and mixed-leaf frequencies. These tests protect downstream ROC-AUC interpretation.

### Regression behavior

Checks step-function approximation, mean-valued leaves, full-grown fit, output range on covered regions, depth-related training MSE, constant target, integer target coercion, float prediction dtype, holdout R², and minimum leaf counts.

### scikit-learn comparisons

On deterministic examples, compare root feature/threshold or end prediction behavior. Close agreement gives an independent reference for split search; intentional tie/stopping differences should be judged using tolerance rather than requiring byte-for-byte trees.

## What this suite does not test

It does not prove behavior for every pathological floating-point scale or very deep recursion. It also does not replace real-data model selection or validate current-market generalization; those belong to full-run diagnostics and external data validation.
