# `src/evaluate.py` — metrics, cross-validation, comparisons, and plots

Source: [`src/evaluate.py`](../../src/evaluate.py). This module implements evaluation statistics in NumPy and supplies reusable experiments/visualizations. It does not implement the models themselves. Package-relative imports let it run through `python -m src.run_all`; guarded fallback imports support direct `src/`-on-path use.

## Classification metrics, lines/functions 1–130

`accuracy(y_true,y_pred)` casts both inputs to arrays and returns the mean equality indicator.

`confusion_matrix()` creates a class-to-index mapping, then increments `[true_index,predicted_index]` for every paired row. If labels are omitted it uses the sorted union of observed true/predicted labels. Returning labels alongside the matrix prevents row/column order ambiguity. Supplying explicit `[0,1]` gives the project's `[[TN,FP],[FN,TP]]` convention.

`per_class_prf()` derives true positives from the diagonal, predicted counts from column sums, and true counts from row sums. `np.divide(...,where=...)` defines zero precision/recall as zero instead of dividing by zero. F1 is `2PR/(P+R)`, also guarded when both are zero. It returns class support so macro and weighted summaries can be calculated transparently.

`roc_auc()` implements the rank statistic. It sorts scores, assigns average ranks for ties, and computes the Mann–Whitney form `(positive_rank_sum - n_pos(n_pos+1)/2)/(n_pos*n_neg)`. If the input has no positive or no negative examples, it returns `NaN`: AUC is undefined in that case. This implementation expects a binary positive label and a one-dimensional score per row.

`roc_curve()` sorts descending by score, records cumulative true/false positives at each distinct score, and adds the origin. It returns false-positive and true-positive rates, not thresholds. Plot/report callers needing threshold values use sklearn's standard helper instead.

`classification_report_dict()` packages accuracy, macro precision/recall/F1, weighted F1, and optional ROC-AUC. The experiment-facing `classification_metrics()` adds positive-class precision/recall/F1 and a fixed 0/1 confusion matrix, matching `run_all.py`/report expectations. This wrapper uses label `1` as premium. Scores must be real-valued model outputs (tree probability or SVM decision margin), not hard labels.

## Regression metrics, lines/functions 110–165

`mae()` averages absolute residuals; `mse()` averages squared residuals; `rmse()` square-roots MSE so it returns to the target's units. `r2_score()` uses `1 - residual_sum_squares / total_sum_squares`; constant targets return `NaN` because the denominator is zero. `mape()` excludes zero targets, since percentage division by zero is undefined, and returns percent units. `regression_report_dict()` groups these values; the compatibility `regression_metrics()` calls it.

The runner evaluates log-price separately from AZN. An RMSE of 0.25 in `ln(price)` is dimensionless; an RMSE in AZN is currency. They must not be compared or labeled interchangeably.

## Fitted-model adapters and comparison

`_get_scores()` calls `decision_function()` for binary classifiers and rejects multi-column score output; this maintains one scalar ranking score per example. `evaluate_model()` predicts once, returns metric dictionary/predictions/scores, and switches to regression metrics when `task='regression'`.

`compare_models()` deep-copies each prototype, times fit and inference, evaluates holdout scores, and records training accuracy/R² as a diagnostic. A `ValueError` becomes a `skipped` status (useful for binary SVM on multiclass labels) rather than aborting every candidate. If `out_dir` is set, `save_results()` writes CSV and JSON. In the project runner, tuning/selection is validation-first; this generic helper should not be handed the final test set during model selection.

## Cross-validation helpers

`kfold_indices()` creates seeded folds. In stratified mode it independently shuffles indices from each class and round-robins them across folds, so each fold has approximately the same class mix. It then sorts validation and training indices for stable output. Non-stratified mode distributes a single seeded permutation round-robin.

`cross_validate()` deep-copies and fits the model per fold, evaluates each validation fold, and appends mean/std rows. The `std(ddof=1)` is sample standard deviation across folds. For the final project, `run_all.py` uses sklearn splitters and its own matched-model loops; it intentionally notes that shared preprocessing was fitted before CV folds, so the result is not a fully nested pipeline CV.

## Pegasos regularization sweep

`svm_lambda_study()` fits one binary Pegasos model per lambda on train, evaluates validation metrics, and records train accuracy, validation accuracy/F1/AUC, final objective, number of support vectors, and margin width. The best lambda must be chosen from validation results, then the final model refit. The helper assumes the supplied class implements `lambda_param`, `epochs`, `random_state`, `objective_history_`, `support_vector_mask_`, and `margin_width_`.

## Serialization and plotting

`save_results()` ensures the destination directory exists and writes both a CSV and JSON list of records. `_plt()` lazily imports Matplotlib with the noninteractive backend; `_finish()` optionally creates the parent directory and writes a tight PNG. The plotting functions build metric bars, normalized/count confusion matrices, ROC curves, Pegasos objective curves, lambda-vs-validation curves, and actual/predicted regression charts. They accept explicit paths so callers choose where artifacts live.

Confusion plots make class order visible; ROC plots use continuous scores; objective plots show training optimization progress, not generalization. `plot_regression_fit()` is intended for numeric regression targets. None of these plots replaces metric tables, and none implies causal feature impact.

## `default_models()`

For regression it returns our `DecisionTree` and sklearn tree plus a Ridge reference. For binary classification it returns our Gini and entropy trees and sklearn tree; with `binary=True`, it adds our Pegasos and sklearn SGD hinge/LinearSVC baselines. Package-safe imports are used for the scratch tree and SVM. The project’s core models remain NumPy-built; sklearn estimators are comparison baselines only.

## Main correctness checks

- Compare accuracy, precision, recall, F1, confusion matrix, ROC-AUC, MAE, RMSE and R² against scikit-learn on deterministic toy inputs.
- Include score ties in AUC checks and verify `NaN` when a class is absent.
- Check matrix orientation explicitly using a hand-counted binary example.
- Ensure all test metrics use predictions from the untouched test partition after configuration selection.
