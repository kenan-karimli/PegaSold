"""Starter structure for implementing a decision tree from scratch with NumPy."""

import numpy as np


class TreeNode:
    """One point in the tree: either a prediction leaf or a split node.

    A leaf stores its prediction. A split node stores which feature to inspect,
    the threshold to compare against, and references to its left and right
    child nodes.
    """

    def __init__(
        self,
        value=None,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
    ):
        """Store the information needed to represent this node.

        Args:
            value: Prediction to return if this node is a leaf.
            feature_index: Column of X used by this node's split; None for leaf.
            threshold: Rows <= this value go left; other rows go right.
            left: Left child TreeNode.
            right: Right child TreeNode.
        """
        self.value = value
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.impurity_decrease = 0.0
        self.n_samples = 0


class DecisionTree:
    """Decision tree interface for classification and regression.

    Classification predicts a category (such as a price tier). Regression
    predicts a numeric value (such as a house price). X should be a 2D NumPy
    array of numeric features; y should be a 1D array of targets.
    """

    def __init__(
        self,
        task="classification",
        criterion=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
    ):
        """Save model settings and initialize attributes populated by fit.

        Args:
            task: Either "classification" or "regression".
            criterion: Classification split measure ("gini" or "entropy") or
                regression measure ("mse"). Choose a sensible default from task
                when this is None.
            max_depth: Maximum number of split levels; None means no depth cap.
            min_samples_split: Minimum number of rows required to try splitting
                a node.
        """
        if task not in {"classification", "regression"}:
            raise ValueError(
                "task must be either 'classification' or 'regression'."
            )
        if criterion is None:
            criterion = "gini" if task == "classification" else "mse"
        if task == "classification":
            if criterion not in {"gini", "entropy"}:
                raise ValueError(
                    "classification criterion must be either 'gini' or 'entropy'."
                )
        else:
            if criterion != "mse":
                raise ValueError(
                    "regression criterion must be either 'mse'."
                )
        if max_depth is not None and max_depth < 0:
            raise ValueError(
                "max_depth cannot be negative."
            )
        if min_samples_split < 2:
            raise ValueError(
                "min_samples_split must be at least 2."
            )
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease cannot be negative.")
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        # This will contain the root TreeNode after fit()
        self.root = None

    def fit(self, X, y):
        """Learn the tree from training data and return this model.

        X contains one row per example and one column per feature. y contains
        the target corresponding to each row. This method should validate the
        inputs, build the root node, and save it on the model.
        """

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        if X.shape[0] == 0:
            raise ValueError("X and y cannot be empty")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature")

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numeric values")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if self.task == "regression":
            try:
                y = y.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Regression targets must be numeric") from exc
            if not np.isfinite(y).all():
                raise ValueError("Regression targets must be finite")

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y) if self.task == "classification" else None
        self.root = self._build_tree(X, y, depth=0)
        importances = np.zeros(self.n_features_in_, dtype=float)
        def collect_importance(node):
            if node.feature_index is not None:
                importances[node.feature_index] += node.n_samples * node.impurity_decrease
                collect_importance(node.left)
                collect_importance(node.right)
        collect_importance(self.root)
        total_importance = importances.sum()
        self.feature_importances_ = importances / total_importance if total_importance > 0 else importances

        return self


    def predict(self, X):
        """Predict a target for every row in X.

        Returns a one-dimensional NumPy array. Classification predictions
        should use the original target labels; regression predictions should
        be numeric. fit must have been called first.
        """
        if self.root is None:
            raise ValueError("fit must be called before predict")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numeric values")
        if X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("X must have the fitted feature count and finite values")

        return np.asarray([self._predict_one(row, self.root) for row in X])

    def predict_proba(self, X):
        """Return class frequencies in each reached leaf."""
        if self.task != "classification":
            raise ValueError("predict_proba is available only for classification")
        if self.root is None:
            raise ValueError("fit must be called before predict_proba")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must be 2D with the fitted feature count")
        if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
            raise ValueError("X must contain finite numeric values")
        return np.vstack([self._predict_proba_one(row, self.root) for row in X])

    def _build_tree(self, X, y, depth):
        """Build one node recursively and return its TreeNode.

        First determine the prediction this node would use as a leaf. If a
        stopping condition applies, return a leaf. Otherwise find the best
        split, divide X and y into left/right groups, recursively build both
        children, and return a split node.
        """
        prediction = self._leaf_value(y)

        if self._should_stop(y, depth):
            return self._make_leaf(y, prediction)

        best_split = self._best_split(X, y)

        if best_split is None:
            return self._make_leaf(y, prediction)
        if best_split[2] < self.min_impurity_decrease:
            return self._make_leaf(y, prediction)

        feature_index, threshold, _ = best_split

        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        x_left = X[left_mask]
        y_left = y[left_mask]

        x_right = X[right_mask]
        y_right = y[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            return self._make_leaf(y, prediction)

        left_child = self._build_tree(x_left, y_left, depth + 1)
        right_child = self._build_tree(x_right, y_right, depth + 1)

        node = TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )
        node.n_samples = len(y)
        node.impurity_decrease = float(best_split[2])
        return node

    def _make_leaf(self, y, prediction):
        node = TreeNode(value=prediction)
        node.n_samples = len(y)
        if self.task == "classification":
            values, counts = np.unique(y, return_counts=True)
            node.class_counts = dict(zip(values.tolist(), counts.tolist()))
        return node

    def _should_stop(self, y, depth):
        """Decide whether the current node should become a leaf.

        Possible conditions include reaching max_depth, having fewer than
        min_samples_split rows, having only one class (classification), or
        finding no split that improves the score. Return True or False.
        """
        if len(y) < self.min_samples_split:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if self.task == "classification" and len(np.unique(y)) == 1:
            return True
        return self.task == "regression" and bool(np.all(y == y[0]))

    def _leaf_value(self, y):
        """Calculate the prediction for a leaf containing targets y.

        For classification, return the most frequent label. For regression,
        return the mean target value.
        """
        if self.task == "classification":
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
        return np.mean(y)

    def _impurity(self, y):
        """Measure how mixed or spread out the targets y are.

        Use Gini or entropy for classification, according to criterion. Use
        mean squared error (MSE) for regression. A pure classification node or
        a regression node with identical targets should have impurity zero.
        """
        if self.task == "classification":
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / counts.sum()
            if self.criterion == "gini":
                return float(1.0 - np.sum(probabilities ** 2))
            nonzero = probabilities > 0
            return float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))
        return float(np.mean((y - np.mean(y)) ** 2))

    def _best_split(self, X, y):
        """Search features and thresholds; return the best useful split.

        For numeric features, candidate thresholds can be placed between
        neighboring distinct feature values. Evaluate each split with
        _split_score. Return (feature_index, threshold, gain), or None when no
        split improves the parent node.
        """
        n_samples, n_features = X.shape

        best_gain, best_feature, best_threshold = -np.inf, None, None
        n = len(y)
        if self.task == "classification":
            _, encoded = np.unique(y, return_inverse=True)
            k = encoded.max() + 1
            onehot = np.eye(k, dtype=float)[encoded]
            total = onehot.sum(axis=0)
        else:
            # Center the target before prefix-summing squares. Regression
            # targets here are log(AZN), typically around 12; computing
            # E[y^2] - E[y]^2 from those large offsets loses precision for
            # small child nodes and can change which near-tied split wins.
            # Variance and every MSE split gain are translation invariant.
            yy = y.astype(float)
            yy = yy - yy.mean()
            total_sum, total_sq = yy.sum(), np.dot(yy, yy)
        parent_score = self._impurity(yy) if self.task == "regression" else self._impurity(y)

        for feature_index in range(n_features):
            order = np.argsort(X[:, feature_index], kind="mergesort")
            values = X[order, feature_index]
            # A boundary is possible only between unequal adjacent feature values.
            positions = np.flatnonzero(values[:-1] < values[1:]) + 1
            positions = positions[(positions >= self.min_samples_leaf) &
                                  (n - positions >= self.min_samples_leaf)]
            if not positions.size:
                continue
            thresholds = values[positions - 1] + (values[positions] - values[positions - 1]) / 2
            if self.task == "classification":
                left_counts = np.cumsum(onehot[order], axis=0)[positions - 1]
                right_counts = total - left_counts
                nl, nr = positions.astype(float), (n - positions).astype(float)
                if self.criterion == "gini":
                    il = 1.0 - np.sum(left_counts ** 2, axis=1) / nl ** 2
                    ir = 1.0 - np.sum(right_counts ** 2, axis=1) / nr ** 2
                else:
                    def entropy(counts, sizes):
                        probs = counts / sizes[:, None]
                        return -np.sum(np.where(probs > 0, probs * np.log2(np.maximum(probs, 1e-300)), 0), axis=1)
                    il, ir = entropy(left_counts, nl), entropy(right_counts, nr)
            else:
                sorted_y = yy[order]
                prefix = np.cumsum(sorted_y)
                prefix_sq = np.cumsum(sorted_y ** 2)
                nl, nr = positions.astype(float), (n - positions).astype(float)
                sl, sql = prefix[positions - 1], prefix_sq[positions - 1]
                sr, sqr = total_sum - sl, total_sq - sql
                il = np.maximum(0.0, sql / nl - (sl / nl) ** 2)
                ir = np.maximum(0.0, sqr / nr - (sr / nr) ** 2)
            gains = parent_score - (positions / n * il + (n - positions) / n * ir)
            local = int(np.argmax(gains))
            gain = float(gains[local])
            if gain > best_gain and gain >= self.min_impurity_decrease:
                best_gain, best_feature, best_threshold = gain, feature_index, float(thresholds[local])

        if best_feature is None:
            return None

        return best_feature, best_threshold, best_gain

    def _split_score(self, y_left, y_right, parent_impurity):
        """Calculate impurity reduction for a proposed split.

        Compute the child impurity as a size-weighted average of the left and
        right impurities. Return parent_impurity minus that weighted average;
        larger positive gain means a better split.
        """
        total = len(y_left) + len(y_right)
        child_impurity = (
            len(y_left) / total * self._impurity(y_left)
            + len(y_right) / total * self._impurity(y_right)
        )
        return parent_impurity - child_impurity

    def _predict_one(self, row, node):
        """Return one prediction by following a row from node to a leaf.

        At a split node, compare row[feature_index] with threshold and follow
        left for <= threshold, otherwise right. At a leaf, return its value.
        """
        while node.feature_index is not None:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _predict_proba_one(self, row, node):
        while node.feature_index is not None:
            node = node.left if row[node.feature_index] <= node.threshold else node.right
        counts = node.class_counts
        return np.asarray([counts.get(label, 0) for label in self.classes_], dtype=float) / sum(counts.values())
