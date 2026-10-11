# `src/run_all.py` — full experiment orchestration guide

Source: [`src/run_all.py`](../../src/run_all.py). This is the one-command entry point that connects EDA, cleaning, tuning, final model fitting, evaluation, plots, and report generation. Run it from the repository root with `python -m src.run_all`.

## Imports and paths

The runner sets Matplotlib's noninteractive backend; imports NumPy/Pandas-compatible utilities plus scikit-learn splitters, scaling and comparison estimators. It imports the new cleaning API (`RANDOM_SEED` alias `SEED`, `prepare_data`), `explore_raw`, the project decision tree, evaluation wrappers, scratch Pegasos/RFF Pegasos, tree visualization, and report generation. `ROOT`, `REPORTS`, and `FIGURES` are resolved from `__file__`, so report destinations do not depend on the caller's working directory (the source CSV default still assumes the repository root).

## Shared helpers

### `_score_regression()`

Thin delegation to the single regression metric implementation. This prevents different loops from using slightly different error formulas.

### `_tree()` and `_sk_tree()`

Construct scratch and sklearn trees with the same task, split criterion, depth, and leaf size. `min_samples_split` is set to at least two and otherwise twice `min_samples_leaf`, making a split impossible when it could not leave enough examples in both children. The sklearn regression criterion is `squared_error`, matching the scratch MSE objective. Fixed seed stabilizes tie-breaking/reference outcomes.

### `_fit_time()`

Measures wall-clock fit duration around `.fit()` and returns both model and seconds. These measurements depend on hardware/load; they are report context, not a portable speed guarantee.

### `_cv_tree()`

Seed-selects at most 6,000 training rows, then makes five shuffled stratified folds for classification or shuffled KFold for regression. For each fold, scratch and sklearn receive the identical training/validation indices and hyperparameters. Accuracy is recorded for classification; RMSE in log-target space for regression. Mean and population standard deviation summarize fold scores. The feature matrix's fitted preprocessing state was learned before these folds, so this is not fully nested preprocessing CV.

### `_cv_svm()`

Uses the same seeded 6,000-row cap and stratified five-fold split. Each fold fits its own StandardScaler using that fold's training rows, then scratch Pegasos and a pipeline containing LinearSVC are compared on the fold's raw validation matrix. For the sklearn reference, `C=1/(lambda*n_train)` matches the normalized regularization scale as closely as this objective convention allows. Twelve epochs are used for the CV Pegasos estimate; this is a stated compute/uncertainty setting, not the final refit epoch count.

## Plot helpers

### `_plot_tuning(curves)`

Draws validation log-RMSE across regression tree candidates and validation F1 across classification tree candidates. Candidate labels include the tuned depth/leaf/criterion. All candidate runs are available as JSON tables; the chart is an at-a-glance view.

### `_learning_curves(d,best_reg,best_cls)`

Samples increasing training sizes with a fixed RNG, refits selected tree settings at each size, and records training versus full validation error. It exposes whether more examples narrow a train/validation gap. The regression y-axis is log-target RMSE; classification uses accuracy. The plotted values are diagnostics, not additional tuning.

### `_make_figures(d,results,final_models)`

Creates the model report's images from cleaned split arrays: cleaning/price distribution; price-vs-area scatter; tier balance; missing numeric values; rooms and location summaries; Pegasos objective and signed scores; regression actual/predicted and residual plots; decision-tree impurity importances; and a compact three-level decision-tree diagram. Samples are seeded where point clouds would otherwise be too dense. Importance is impurity reduction in these fitted trees, not a causal effect. Output PNG names are consumed by both Markdown and LaTeX reports and by the 15-slide deck.

## `_with_roc()`

Calls shared classification metrics and sklearn's `roc_curve`; stores at most about 200 evenly spaced ROC points to keep `results.json` manageable. It always includes the final curve point. AUC itself comes from the NumPy evaluation module, providing a useful independent metric implementation.

## `run()` — execution sequence

1. Create report folders; run full raw EDA into `reports/eda`.
2. Call `prepare_data()` to write split artifacts and build in-memory arrays. Print final count.
3. **Task A tuning:** fit scratch regression trees over depths 4/8/12/unlimited and leaves 5/20; select minimum validation RMSE on `ln(price)`.
4. **Task B tuning:** sweep Gini/entropy, depths 4/8/12, and leaves 5/20; select maximum validation positive-class F1.
5. **Linear SVM tuning:** fit train-only scaler; select a seeded train subset capped at 12,000; sweep lambda values and epochs/config; score only on validation. Refit the selected setting on all training rows later.
6. **RFF tuning:** use a seeded training subset capped at 8,000, tune gamma for a 128-component random Fourier mapping and score validation. The RBF sklearn baseline uses the same subset.
7. Fit selected scratch models and matched sklearn references on their declared train sets. No test metric is used to select the settings.
8. Predict held-out test data, calculate classification and log/AZN regression metrics, fit/predict times, SVM norm/margin/support-vector counts, and store predictions needed for diagnostic figures.
9. Run five-fold sample CV; capture explicit validation comparisons, price-quartile/category error slices, feature importance, numeric correlations, location summaries, and learning curves.
10. Save `reports/results.json`; call `write_reports()` to produce full Markdown, LaTeX/PDF, and slides.

## Important evaluation details

- Regression target is `ln(AZN)`. `np.exp(pred_log)` transforms it back for AZN errors; clipping at `[0,25]` prevents overflow. No smearing correction is applied, so AZN predictions should not be interpreted as unbiased conditional means.
- Task B's cutpoint comes from train-only `prepare_data`. The split itself used coarse global-median strata only to distribute price bands; that distinction is documented in preprocessing rules/report.
- Validation selects settings; test is held for the concluding comparison. The report identifies that source scope/window choices were made after full-dataset EDA and that shared preprocessing precedes CV.
- Feature correlations and location summaries are descriptive. Correlation can reflect confounding; tree feature importance can distribute credit among correlated columns.
- The RFF run is intentionally a finite random approximation (128 components) and uses a smaller train subset for runtime. Its scores are conditional on this seed/settings.

## File outputs

- `data/processed/`: train/validation/test `.npz`, `.npy`, readable CSVs, ordered feature names, preprocessing audit and exact fitted rules.
- `reports/eda/`: complete raw EDA report and figures.
- `reports/figures/`: cleaned-data, tuning, error, model and comparison plots.
- `reports/results.json`: complete machine-readable scores, all sweep rows, CV values, diagnostics and data audit.
- `reports/final_report.md`: results plus embedded plots.
- `report/report.tex` and `report/report.pdf`: compact typeset report.
- `presentation/slides.tex` and `presentation/slides.pdf`: 15-slide Beamer deck in the Colombia-inspired yellow/blue/red design.

## Tricky but important limits

This is a random split across distinct normalized advert IDs, not a future-time split. Listings posted more than once under different item IDs can still be ambiguous; the pipeline reports such candidate groups and avoids deleting them on a weak key. Advertised prices are not transaction prices. Data cleaning bounds were set after seeing full source distributions, so this is a transparent course benchmark rather than a blinded prospective estimate.
