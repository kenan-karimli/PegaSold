"""Generate the detailed Markdown and two-column PDF report from experiment outputs."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def _f(value, digits=4):
    return f"{value:.{digits}f}" if value is not None else "—"


def _table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


TEAM = ["Emiliya Abdulhalimova", "Aygun Asvarova", "Murad Bagirzade", "Kanan Karimli"]
TEAM_LINE = ", ".join(TEAM)


def _markdown(results, data, reports, figures):
    audit, test = results["data_audit"], results["test_results"]
    reg = test["decision_tree_regression"]
    clas = test["decision_tree_classification"]
    svm = test["pegasos_svm"]
    tree_rows = []
    for name in ("decision_tree_regression", "sklearn_tree_regression"):
        m = test[name]
        tree_rows.append([name, _f(m["rmse"]), _f(m["mae"]), _f(m["r2"]), f"{m['rmse_AZN']:,.0f}", f"{m['mae_AZN']:,.0f}", f"{m['median_absolute_error_AZN']:,.0f}", f"{m['max_absolute_error_AZN']:,.0f}", _f(m["r2_AZN"]), f"{m['fit_seconds']:.2f}", f"{m['predict_seconds']:.3f}"])
    cls_rows = []
    for name in ("decision_tree_classification", "sklearn_tree_classification", "pegasos_svm", "linear_svc", "rff_pegasos_svm", "rbf_svc"):
        m = test[name]
        cls_rows.append([name, _f(m["accuracy"]), _f(m["precision"]), _f(m["recall"]), _f(m["f1"]), _f(m["roc_auc"]), f"{m['fit_seconds']:.2f}", f"{m['predict_seconds']:.3f}"])
    reg_cv = results["cross_validation"]["decision_tree_regression"]
    cls_cv = results["cross_validation"]["decision_tree_classification"]
    svm_cv = results["cross_validation"]["pegasos_svm"]
    test_log_path = reports / "test_results.txt"
    test_log = test_log_path.read_text(encoding="utf-8") if test_log_path.exists() else ""
    test_summary = next((line.strip().strip("=").strip() for line in reversed(test_log.splitlines())
                         if " passed" in line or " failed" in line),
                        "Not run yet; execute `pytest -vv` from the repository root.")
    out = [
        "# Bina.az Housing Price Prediction — Final Project Report", "",
        "**MLE-AI-201 · Machine Learning · Final Project V3**  ",
        f"**Team PegaSold:** {TEAM_LINE}.  ",
        "**Dataset:** Bina.az sale listings from the course-provided `bina_az_sale.csv`; scrape records span October–November 2024. The report does not claim transaction prices or a current market sample.", "",
        "> This report is generated from the checked-in NumPy implementations and local dataset by `python -m src.run_all`. The test split is held back until hyperparameters are selected on validation. It reports both log-price and AZN-space errors because a strong log score can hide large currency errors.", "",
        "## Executive summary", "",
        f"Domain cleaning retains **{audit['final_rows']:,}** Baku residential listings from {audit['raw_rows']:,} raw records. A train-only log-price MAD fence removes **{audit['log_price_outlier_filter']['train_outliers_removed']:,} training rows**; all **{results['split']['validation']:,} validation** and **{results['split']['test']:,} test** rows remain, including price tails. The premium/standard boundary is the post-filter training median (**{results['tier_threshold_AZN']:,.0f} AZN**). On the full held-out test split, our NumPy decision tree achieved RMSE **{reg['rmse']:.3f}** in ln(AZN) and **{reg['rmse_AZN']:,.0f} AZN** in raw price, with MAE **{reg['mae_AZN']:,.0f} AZN**. Its tier classifier reached **{clas['accuracy']:.1%} accuracy / {clas['f1']:.3f} F1**. Pegasos reached **{svm['accuracy']:.1%} accuracy / {svm['f1']:.3f} F1**. The matched scikit-learn trees provide reference results; nonlinear RFF Pegasos is included to satisfy the nonlinear SVM requirement.", "",
        "## 1. Problem and dataset", "",
        "Task A predicts sale price. Because prices have a long upper tail, the tree is trained on `ln(price)`; we report errors in both log space and AZN space after exponentiating predictions. Task B classifies a listing as standard (0) or premium (1), with premium defined as price greater than or equal to the median price computed from the training partition only. This is a relative price tier, not an appraisal or a market valuation.", "",
        "The data are a scraped [bina.az sale listing dataset](https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project). Azerbaijani measurement columns arrive as strings and location/categorical columns are sparse. The source contains repeat scrapes, missing measurements and non-residential listing categories. The raw data file is kept locally and is not embedded in this report.", "",
        "### Cleaning and leakage control", "",
        _table(["Stage", "Rows", "Action"], [["Raw CSV", f"{audit['raw_rows']:,}", "Read source; it is never overwritten"], ["One row per advert ID", f"{audit['after_dedup']:,}", "Normalize URL to numeric ID; keep last scrape date"], ["Baku residential scope", f"{audit['after_scope']:,}", "Keep Baku and the three residential categories"], ["Domain-valid rows", f"{audit['final_rows']:,}", "Fixed price, area, price/m² and nonmissing room/floor bounds"], ["Train after MAD", f"{results['split']['train']:,}", "Train-derived log-price fence; validation/test are unchanged"]]), "",
        f"Scope removed `{audit['after_dedup']-audit['after_scope']:,}` rows. Fixed domain filters removed `{audit['after_scope']-audit['final_rows']:,}` more. Then a two-sided modified-z MAD rule on `log(price)` (cutoff 3.5) estimated its center and scale from training data and learned the fence **{audit['log_price_outlier_filter']['price_fence_AZN'][0]:,.0f}–{audit['log_price_outlier_filter']['price_fence_AZN'][1]:,.0f} AZN**. It removed **{audit['log_price_outlier_filter']['train_outliers_removed']:,} rows from training only**. Validation/test were not filtered by their target values, preserving all **{results['split']['validation']:,}/{results['split']['test']:,}** allocated rows for tuning/evaluation; their errors include the difficult price tails. There are **{audit['ambiguous_repost_candidate_rows']:,} rows** in coordinate/area/room/price candidate-match groups; these are reported, not silently discarded, because the key does not prove property identity. Missing training cells before median imputation: **{audit['missing_cells_before_imputation']:,}**.", "",
        "We explicitly exclude the target `price`; derived `total_price` and `unit_price`; free-text `description` and `attributes`; scrape/listing IDs (`id_x`, `id_y`, `estate_id`); person/agency fields (`owner_name`, `shop_name`); and raw address. They either encode the answer, identify/repeat an item, or can expose private/high-cardinality location text. We keep coordinates and coarse `location` as explicit spatial predictors. The train-derived MAD fence removes training tails only; all held-out test tail observations remain represented in the reported errors.", "",
        f"The model has {results['numeric_feature_count']} numeric/binary inputs: log area, rooms, floor, building height, log plot size, coordinates, distance to Baku centre, floor ratio, area per room, log land ratio, floor-availability/ground/top flags, repair/title/mortgage, VIP/featured, and owner-listed flags. Category and location are one-hot encoded; location levels with at least 100 training examples receive columns and all others use `other`. The final matrix has **{results['feature_count']}** columns. Train-only medians fill rooms/coordinates; train-only means and population standard deviations scale continuous fields. No clipping is used. Split allocation is stratified by whole-cleaned-data median price bins; the actual premium threshold is independently computed from the training median only (seed {results['seed']}). Repeated scrapes sharing normalized advert IDs cannot cross partitions.", "",
        "![Data cleaning attrition and training price histogram](figures/data_and_prices.png)", "*Figure 1. Cleaning stages and skewed training price distribution.*", "",
        "![Price area relationship](figures/price_vs_area.png)", "*Figure 2. Price and area on log axes; points are colored by listing category.*", "",
        "### Exploratory analysis", "",
        "The raw EDA report is in [`reports/eda/eda_report.txt`](eda/eda_report.txt), with its 16 source-level plots in `reports/eda/figures/`. The model-oriented plots below use cleaned training data. Location medians require at least 100 training listings and are descriptive rather than causal. Numeric correlations describe pairwise association, not independent feature effects.", "",
        f"Tier shares (premium): train {results['exploratory_analysis']['tier_prevalence']['train']:.3f}, validation {results['exploratory_analysis']['tier_prevalence']['validation']:.3f}, test {results['exploratory_analysis']['tier_prevalence']['test']:.3f}.", "",
        _table(["Numeric field", "Missing before imputation (%)"], [[k, f"{v:.2f}"] for k, v in results['exploratory_analysis']['missing_numeric_percent'].items()]), "",
        _table(["Numeric field", "Pearson r with ln(price)"], [[x['feature'], f"{x['pearson_r_with_log_price']:.3f}"] for x in results['exploratory_analysis']['numeric_correlations_with_log_price']]), "",
        _table(["Location (min 100 rows)", "Train n", "Median advertised price (AZN)"], [[x['location'], x['n'], f"{x['median_price_AZN']:,.0f}"] for x in results['exploratory_analysis']['locations_min_100_train_rows_top_median']]), "",
        "![Tier balance](figures/class_balance.png)", "![Missingness](figures/missingness.png)", "![Price by rooms](figures/price_vs_rooms.png)", "![Location median prices](figures/location_prices.png)", "*Figure 3. Additional EDA: class balance, missingness, room-count and location associations.*", "",
        "## 2. Models built from scratch", "",
        "### Decision tree", "",
        "The NumPy tree supports both classification and regression. At each node it sorts each feature once and evaluates all legal boundaries between distinct values using cumulative class counts or target sums/squared sums. Classification split gain is parent impurity minus weighted child impurity, using Gini or entropy. Regression uses MSE reduction (equivalent to variance reduction). Leaves predict the majority class or the mean target. Growth is bounded by maximum depth and minimum split/leaf counts; normalized feature importance sums weighted impurity decrease over internal nodes.", "",
        "Tree selection uses only validation: Task A sweeps depth `{4, 8, 12, None}` × minimum leaf `{5, 20}`, minimizing log-price RMSE. Task B sweeps Gini/entropy × depth `{4, 8, 12}` × minimum leaf `{5, 20}`, maximizing F1. The selected parameter dictionaries and every validation score are in `results.json`; the full sweep is also tabulated below.", "",
        "![Tree tuning curves](figures/tree_tuning.png)", "*Figure 4. Validation scores for every tree configuration; selection uses validation only.*", "",
        "![Tree learning curves](figures/tree_learning_curves.png)", "*Figure 5. Training versus validation performance by sample size; the gap illustrates variance/overfit behaviour.*", "",
        "### Linear and nonlinear SVM", "",
        r"We optimize the regularized hinge-loss primal with Pegasos: `J(w,b) = (lambda/2)||w||² + mean(max(0, 1 − yᵢ(w·xᵢ+b)))`, mapping the two tier labels to −1/+1 internally. At step t, `eta = 1/(lambda*t)`; weights shrink by `(1−eta*lambda)` and violating points add `eta*y*x`. The intercept is unregularized and its per-update step is clipped to `min(eta, 1)` for stability; this is a bounded-bias Pegasos variant, and the difference from an unbounded augmented-vector update is a limitation. Features are standardized using training data. We tune lambda in `{0.001, 0.01, 0.1}` on a seeded 12,000-row training sample and validation F1, then refit the selected linear model on the full training split. We match the scikit-learn LinearSVC using objective-matched `C=1/(lambda*n_train)`.", "",
        "The required nonlinear alternative is random Fourier feature Pegasos. We draw Gaussian frequencies and uniform phases to map rows to `sqrt(2/D) cos(xW+b)`, which approximates an RBF kernel, then optimize the same primal in that mapped space. We tune RBF gamma `{0.01, 0.1, 1.0}` on a seeded 8,000-row train sample and validation F1. Its reference is sklearn RBF SVC trained on that same subset. This approximation depends on random features and is expected to vary with component count; we fix the seed and use 128 components.", "",
        "The learned linear margin is `2/||w||`; support vectors are training points with signed margin at most 1. The full norm, margin width and support-vector count are reported in the JSON and below.", "",
        "![Pegasos objective](figures/pegasos_objective.png)", "*Figure 6. Regularized hinge objective over epochs for the final linear Pegasos fit.*", "",
        "![Margin scores](figures/svm_margin_scores.png)", "*Figure 7. Signed decision scores; zero is the separating hyperplane and ±1 bounds the margin.*", "",
        "## 3. Experimental design", "",
        "Model hyperparameters are selected using validation only, and the test partition is scored after model settings are fixed. The data scope and fixed price/area/price-per-m² bounds were chosen after looking at the full source distribution before training; this is disclosed as a selection limitation, and those bounds were not tuned to test model scores. Tree baselines use scikit-learn `DecisionTreeRegressor`/`DecisionTreeClassifier` on the same full training matrix and matched depth/leaf/criterion settings. The sklearn LinearSVC baseline uses a scaler inside a pipeline, all training rows and objective-matched C. The nonlinear RBF SVC uses the RFF model's tuning subset, matched gamma and corresponding C. These library models are references only; the submitted tree and Pegasos optimization are NumPy implementations.", "",
        f"For uncertainty, we run five-fold CV on all {reg_cv['sample_n']:,} post-cleaning training rows (each validation fold is about one fifth; each fold trains on the other four fifths). The selected tree and Pegasos configurations are held fixed in folds. An additional SVM scaler is fitted within each fold. The shared feature matrix's numeric imputation/scaling statistics and categorical vocabulary were fitted on the full training partition before CV; this unsupervised preprocessing means CV dispersion is indicative rather than a fully nested estimate. The independent test partition is excluded from CV and remains the final held-out evaluation.", "",
        "Regression reports RMSE/MAE/R² in log space and AZN space. Classification reports accuracy, precision, recall, F1, ROC-AUC, confusion matrices and ROC curves. Fit/predict times are wall-clock seconds on the current machine and are informational, not cross-machine guarantees.", "",
        "## 4. Held-out results", "",
        "### Task A — regression", "",
        _table(["Model", "RMSE ln(AZN)", "MAE ln(AZN)", "R² ln", "RMSE AZN", "MAE AZN", "Median abs error AZN", "Max abs error AZN", "R² AZN", "Fit s", "Predict s"], tree_rows), "",
        f"The scratch tree's **{reg['mae_AZN']:,.0f} AZN MAE is an average absolute error, not a maximum**. Median absolute error is **{reg['median_absolute_error_AZN']:,.0f} AZN**; the largest test error was **{reg['max_absolute_error_AZN']:,.0f} AZN**: a {reg['max_error_case']['area_m2']:,.0f} m², {reg['max_error_case']['rooms']:g}-room {reg['max_error_case']['category']} listing in {reg['max_error_case']['location']} priced at {reg['max_error_case']['actual_AZN']:,.0f} AZN was predicted at {reg['max_error_case']['predicted_AZN']:,.0f} AZN. This is one high-price-tail miss; RMSE penalizes it strongly.", "",
        f"Selected scratch-tree setting: `{json.dumps(results['selected']['tree_regression'])}`. Five-fold CV on all {reg_cv['sample_n']:,} training rows gives scratch log-RMSE **{reg_cv['scratch']['mean']:.4f} ± {reg_cv['scratch']['std']:.4f}** and sklearn **{reg_cv['sklearn']['mean']:.4f} ± {reg_cv['sklearn']['std']:.4f}**. In currency space, scratch-tree MAE is **{reg_cv['azn_metrics']['scratch_mae']['mean']:,.0f} ± {reg_cv['azn_metrics']['scratch_mae']['std']:,.0f} AZN** across folds; matched sklearn MAE is **{reg_cv['azn_metrics']['sklearn_mae']['mean']:,.0f} ± {reg_cv['azn_metrics']['sklearn_mae']['std']:,.0f} AZN**. CV MAE is lower than test MAE in part because the folds draw from the MAD-filtered training partition, while the untouched test retains extreme-price listings. These fold averages describe stability; they do not replace the held-out test MAE.", "",
        "![Predictions and residuals](figures/regression_error_analysis.png)", "*Figure 8. Held-out predicted versus observed price and residuals in log space.*", "",
        "### Task B — price-tier classification", "",
        _table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Fit s", "Predict s"], cls_rows), "",
        f"Selected scratch tree: `{json.dumps(results['selected']['tree_classification'])}`. Selected linear lambda: `{results['selected']['linear_svm']['lambda']}`. Selected RFF gamma: `{results['selected']['rff_svm']['gamma']}`. Tier cutoff: **{results['tier_threshold_AZN']:,.0f} AZN** (training median).", "",
        f"Five-fold CV uses all {cls_cv['sample_n']:,} training rows: scratch tree accuracy **{cls_cv['scratch']['mean']:.4f} ± {cls_cv['scratch']['std']:.4f}**, matched sklearn tree **{cls_cv['sklearn']['mean']:.4f} ± {cls_cv['sklearn']['std']:.4f}**; Pegasos **{svm_cv['scratch']['mean']:.4f} ± {svm_cv['scratch']['std']:.4f}**, LinearSVC **{svm_cv['sklearn']['mean']:.4f} ± {svm_cv['sklearn']['std']:.4f}**. Each fold trains on four fifths and validates on the remaining fifth. Imputation/scaling/category levels were fit on all training rows before folds, so this CV is not fully nested.", "",
        "![Classification score comparison](figures/classification_comparison.png)", "*Figure 9. Accuracy and F1 on the held-out test set.*", "",
        "![ROC curves](figures/roc_curves.png)", "*Figure 10. Tier ROC curves on untouched test rows.*", "",
        "![Scratch tree confusion matrix](figures/tree_confusion_matrix.png)", "![Pegasos confusion matrix](figures/pegasos_confusion_matrix.png)", "![RFF confusion matrix](figures/rff_confusion_matrix.png)", "*Figure 11. Test confusion matrices (true rows, predicted columns).*", "",
        "### Confusion matrices", "",
    ]
    for name in ("decision_tree_classification", "sklearn_tree_classification", "pegasos_svm", "linear_svc", "rff_pegasos_svm", "rbf_svc"):
        m = test[name]
        out.append(f"- **{name}:** `[[TN={m['confusion_matrix'][0][0]}, FP={m['confusion_matrix'][0][1]}], [FN={m['confusion_matrix'][1][0]}, TP={m['confusion_matrix'][1][1]}]]`; support vectors/points: {m.get('support_vector_count', 'n/a')}; margin width: {m.get('margin_width', 'n/a') if m.get('margin_width') is not None else 'n/a'}.")
    out += ["", "### Selected tree feature importance", ""]
    for task in ("regression", "classification"):
        out += [f"#### {task.title()}", "", _table(["Rank", "Feature", "Normalized impurity importance"], [[i+1, x['feature'], f"{x['importance']:.5f}"] for i, x in enumerate(results['feature_importance'][task][:15])]), ""]
        out.append(f"![{task.title()} feature importance](figures/feature_importance_{task}.png)")
    out += ["", "### Regression error slices", "", _table(["Slice", "n", "Median actual AZN", "MAE AZN", "MAE ln(price)"], [[x['slice'], x['n'], f"{x.get('median_actual_AZN', float('nan')):,.0f}" if 'median_actual_AZN' in x else "—", f"{x['mae_AZN']:,.0f}", f"{x['mae_log']:.4f}"] for x in results['regression_error_slices']]), "",
        "The price-quartile slices use training-set price cutpoints applied to test rows. Property-category slices are shown separately because their prices and sample sizes differ. Currency MAE can be dominated by expensive listings; log MAE shows proportional error more evenly. The plot and slice table should be read together, not as evidence of causal effects.", "",
        "## 5. Validation sweeps", "",
        "### Regression tree", "",
        _table(["Depth", "min leaf", "Validation RMSE log", "Validation MAE log", "Fit s"], [[x['depth_label'], x['min_samples_leaf'], f"{x['validation_rmse_log']:.5f}", f"{x['validation_mae_log']:.5f}", f"{x['fit_seconds']:.2f}"] for x in results['tuning']['tree_regression']]), "",
        "### Classification tree", "",
        _table(["Criterion", "Depth", "min leaf", "Accuracy", "F1", "ROC-AUC", "Fit s"], [[x['criterion'], x['depth_label'], x['min_samples_leaf'], f"{x['validation_accuracy']:.4f}", f"{x['validation_f1']:.4f}", f"{x['validation_roc_auc']:.4f}", f"{x['fit_seconds']:.2f}"] for x in results['tuning']['tree_classification']]), "",
        "### Pegasos and RFF", "",
        _table(["Model sweep", "Hyperparameter", "Accuracy", "F1", "ROC-AUC", "Train subset", "Fit s"], [["Linear Pegasos", f"lambda={x['lambda']}", f"{x['validation_accuracy']:.4f}", f"{x['validation_f1']:.4f}", f"{x['validation_roc_auc']:.4f}", x['tuning_train_n'], f"{x['fit_seconds']:.2f}"] for x in results['tuning']['svm_linear']] + [["RFF Pegasos", f"gamma={x['gamma']}", f"{x['validation_accuracy']:.4f}", f"{x['validation_f1']:.4f}", f"{x['validation_roc_auc']:.4f}", x['tuning_train_n'], f"{x['fit_seconds']:.2f}"] for x in results['tuning']['svm_rff']]), "",
        "## 6. Discussion: what went wrong and what worked", "",
        f"- **Performance is useful but not appraisal-grade.** The scratch tree's log-price MAE is `{reg['mae']:.3f}`; its raw-price MAE is `{reg['mae_AZN']:,.0f} AZN` and its largest absolute test error is `{reg['max_absolute_error_AZN']:,.0f} AZN`. These are distinct statistics. Test R² in raw AZN is `{reg['r2_AZN']:.3f}`; the high-price tail remains difficult even with log training.",
        f"- **The tier problem is easier than exact valuation.** The tree and Pegasos both achieve F1 near `{max(clas['f1'], svm['f1']):.2f}` against a balanced median label. Because area, property type, location and price are correlated, this should not be interpreted as causal price explanation.",
        "- **A modest scratch/reference gap is expected.** The reference trees and SVC use the same split and matched hyperparameters, but tie-breaking, floating-point boundaries, and implementation details can differ. We report those differences instead of claiming equality.",
        "- **Bias and variance are visible in learning curves.** Small training sets have larger train/validation gaps. Depth/min-leaf tuning controls variance; unlimited depth is intentionally excluded from the candidate set to reduce overfit risk.",
        "- **Sparse locations and asking-price labels limit transfer.** Rare neighborhoods share an `other` level; these models predict advertised prices, not completed transactions.",
        "- **The RFF result is approximate.** It uses 128 seeded random components and an 8,000-row training subset for runtime. It is a genuine nonlinear NumPy Pegasos pipeline but should not be mistaken for exact RBF optimization.",
        "- **Potential selection bias remains.** This is a scraped listing dataset, not verified closing transactions; advertised prices can differ from sale prices. The single snapshot also reflects the 2024 market and cannot support present-day valuation without fresh data.", "",
        "## 7. Automated tests", "",
        f"The complete `pytest -vv` run finished with: **{test_summary}**. The full test-by-test terminal output is in [`reports/test_results.txt`](test_results.txt). The suite covers both tree tasks and split criteria, Pegasos input/learning/objective behavior, the nonlinear RFF model, visualization exports, metric parity, measurement parsing, advert-ID deduplication, scope/domain filtering, saved processed artifacts, finite prepared matrices and normalized tree feature importances.", "",
        "## 8. Reproducibility and files", "",
        "From the repository root, install pinned requirements, place the dataset at `data/bina_az_sale.csv`, then run `python -m src.run_all`. It regenerates `reports/results.json`, all report figures, this Markdown report, `report/report.tex`, and the two-column `report/report.pdf`. Run `pytest -q` to execute the model, visualization and data-preparation tests. The data archive is excluded from version control.", "",
        "## 9. Four-person contribution and presentation plan", "",
        _table(["Speaker", "Owned deliverables", "Presentation scope", "Effort target"], [
            ["Emiliya Abdulhalimova — data engineering", "Raw EDA, cleaning decisions, leakage audit, processed splits and feature engineering", "Problem, dataset, EDA evidence, preprocessing and split", "25%"],
            ["Aygun Asvarova — decision trees", "NumPy classification/regression trees, Gini/entropy/MSE, tuning and tree diagnostics", "Tree math, implementation, validation sweep and feature interpretation", "25%"],
            ["Murad Bagirzade — SVM", "NumPy Pegasos, hinge objective, scaling, margins, support vectors and RFF extension", "SVM updates, convergence evidence, linear/nonlinear comparison", "25%"],
            ["Kanan Karimli — evaluation/reproducibility", "Matched baselines, metrics, CV, error analysis, tests, reports and run instructions", "Fair benchmark, results, limitations and reproducible handoff", "25%"]]), "",
        "The plan targets equal effort and equal speaking time (about 3–4 minutes each in a 14-minute presentation). Each speaker owns a substantive technical workstream with analysis/results, rather than assigning one person only formatting. These are proposed presentation/work allocations; record actual completed contributions honestly in `contribution_report.md`. Cross-review another member’s module so all four can explain the end-to-end pipeline.", "",
        "## 10. AI-assistance disclosure", "",
        "AI was used as a coding assistant for suggestions, debugging, corrections, review, tests, visualizations, and documentation. The PegaSold team is responsible for checking the final code, results, and explanations. The decision-tree and Pegasos algorithms are implemented in the project’s NumPy code; scikit-learn is used for comparison models and standard utilities. The dataset was supplied separately by the course/team and is not included in the repository.", "",
        "## 11. Figures", "",
    ]
    for path in sorted(figures.glob("*.png")):
        out.append(f"- `{path.name}`")
    out += ["", "## Appendix A — Raw-data exploration figures", "",
            "These 16 figures accompany the complete raw-source analysis in Appendix B. They are descriptive plots; the model results above come from the processed train/validation/test partitions.", ""]
    eda_figures = sorted((reports / "eda" / "figures").glob("*.png"))
    for image in eda_figures:
        out.append(f"![Raw EDA: {image.stem}](eda/figures/{image.name})")
        out.append("")
    eda_report = reports / "eda" / "eda_report.txt"
    if eda_report.exists():
        out += ["## Appendix B — Complete raw-data EDA transcript", "",
                "```text", eda_report.read_text(encoding="utf-8").rstrip(), "```", ""]
    if test_log:
        out += ["## Appendix C — Full automated test output", "", "```text",
                test_log.rstrip(), "```", ""]
    out += ["## Appendix D — Full source code and line-by-line guides", "",
            "The complete, executable source remains in `src/`; each module's code-specific guide maps the functions and tricky implementation lines to their purpose:", "",
            "- [`data_prep.py` walkthrough](../docs/code_explanations/data_prep.md)",
            "- [`explore.py` walkthrough](../docs/code_explanations/explore.md)",
            "- [`decision_tree.py` walkthrough](../docs/code_explanations/decision_tree.md)",
            "- [`svm.py` walkthrough](../docs/code_explanations/svm.md)",
            "- [`evaluate.py` walkthrough](../docs/code_explanations/evaluate.md)",
            "- [`run_all.py` walkthrough](../docs/code_explanations/run_all.md)",
            "- [`reporting.py` walkthrough](../docs/code_explanations/reporting.md)",
            "- [`site_data.py` and website walkthrough](../docs/code_explanations/website.md)",
            "- [`tree_viz.py` walkthrough](../docs/code_explanations/tree_viz.md)",
            "- Test-by-test notes: `docs/code_explanations/test_*.md`", ""]
    (reports / "final_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def _latex(results, report_dir):
    audit, test = results["data_audit"], results["test_results"]
    reg, cls, svm = test["decision_tree_regression"], test["decision_tree_classification"], test["pegasos_svm"]
    reg_cv = results["cross_validation"]["decision_tree_regression"]
    baseline_reg, baseline_cls = test["sklearn_tree_regression"], test["sklearn_tree_classification"]
    body = rf'''\documentclass[10pt,twocolumn,a4paper]{{article}}
\usepackage[a4paper,margin=.65in,columnsep=.25in]{{geometry}}
\usepackage[T1]{{fontenc}}\usepackage[utf8]{{inputenc}}\usepackage{{mathptmx}}
\usepackage{{graphicx,amsmath,booktabs,hyperref}}
\title{{Bina.az Listing Price and Tier Prediction}}
\author{{MLE-AI-201 Final Project V3\\[2pt]\small PegaSold: Emiliya Abdulhalimova, Aygun Asvarova, Murad Bagirzade, Kanan Karimli}}
\date{{October 2026}}
\begin{{document}}\maketitle
\begin{{abstract}}
We implement a NumPy decision tree for log-price regression and median-tier classification, and a NumPy Pegasos SVM with a random Fourier feature alternative. Of {audit['raw_rows']:,} raw rows, {audit['final_rows']:,} remain after normalized advert-ID deduplication, Baku residential scope selection, and declared domain bounds. The training-only tier cut is {results['tier_threshold_AZN']:,.0f} AZN. The scratch tree test RMSE is {reg['rmse']:.3f} in log-price space ({reg['rmse_AZN']:,.0f} AZN); scratch tree and Pegasos test F1 are {cls['f1']:.3f} and {svm['f1']:.3f}. Asking prices are not verified transactions.
\end{{abstract}}
\section{{Data and preprocessing}}
The source contains Azerbaijani text fields, mixed missingness, repeat scrapes, and categories with different measurement units. The pipeline keeps Baku listings from the three residential categories. It normalizes numeric IDs from advert URLs and retains each ID's last dated scrape. Candidate reposts sharing coordinates, area, rooms, and price are reported but not deleted because the key does not establish that two ads describe one property. Fixed domain checks require price 10,000--5,000,000 AZN, area 10--2,000 m$^2$, unit price 200--20,000 AZN/m$^2$, plausible room counts, and valid stated floors. After splitting, a two-sided modified-z MAD fence is estimated from training `log(price)` only (cutoff 3.5) and applied only to training. All validation/test rows are retained, so the test metrics include outlier tails. The processed split is {results['split']['train']:,}/{results['split']['validation']:,}/{results['split']['test']:,} (seed {results['seed']}). Split allocation uses global-median price bins; the premium threshold is separately learned from the remaining training rows only.

The {results['feature_count']}-column matrix contains {results['numeric_feature_count']} continuous/binary features and one-hot category/location columns. Train-only medians fill rooms/coordinates; train-only population means/stds scale continuous features. IDs, free text, views, `unit\_price`, and `total\_price` are excluded from model inputs. The raw file is never overwritten.
\begin{{figure}}[t]\centering\includegraphics[width=\linewidth]{{../reports/figures/data_and_prices.png}}\caption{{Row counts through cleaning and training price distribution.}}\end{{figure}}
\section{{Models and evaluation}}
The NumPy tree tests numeric threshold midpoints. Gini/entropy use cumulative class counts; regression uses variance/MSE reduction via cumulative sums and squared sums. Leaves predict a majority class or target mean. Depth and minimum leaf settings are chosen from validation evidence.

Pegasos minimizes $J(w,b)=\frac{{\lambda}}{{2}}\|w\|^2+\frac{{1}}{{n}}\sum_i\max(0,1-y_i(w^Tx_i+b))$. It uses $\eta_t=1/(\lambda t)$, weight shrinkage, and hinge-loss subgradients. The intercept update is bounded by $\min(\eta_t,1)$ and is unregularized. The nonlinear option maps inputs to $\sqrt{{2/D}}\cos(xW+\phi)$ with 128 seeded random Fourier features. Library trees and SVMs are benchmark references; the course models remain project NumPy implementations.

We select model settings on validation, then report the held-out test once. Five-fold CV uses all {reg_cv['sample_n']:,} post-cleaning training rows and reports fold mean and standard deviation; each fold trains on four fifths and validates on one fifth. Regression CV also reports AZN-space MAE. Shared feature preprocessing was fitted before fold creation, so it is not fully nested CV. Validation and test data are not part of CV. Regression metrics are reported in both $\ln(\mathrm{{AZN}})$ and AZN. Classification reports accuracy, precision, recall, F1, ROC-AUC and confusion matrices.
\section{{Held-out results}}
\begin{{table}}[h]\centering\small\begin{{tabular}}{{lrrr}}\toprule Model & RMSE log & RMSE AZN & $R^2$ AZN\\\midrule
Scratch tree & {reg['rmse']:.3f} & {reg['rmse_AZN']:,.0f} & {reg['r2_AZN']:.3f}\\
sklearn tree & {baseline_reg['rmse']:.3f} & {baseline_reg['rmse_AZN']:,.0f} & {baseline_reg['r2_AZN']:.3f}\\\bottomrule\end{{tabular}}\caption{{Held-out Task A results.}}\end{{table}}
\begin{{table}}[h]\centering\small\begin{{tabular}}{{lrrr}}\toprule Model & Accuracy & F1 & ROC-AUC\\\midrule
Scratch tree & {cls['accuracy']:.3f} & {cls['f1']:.3f} & {cls['roc_auc']:.3f}\\
sklearn tree & {baseline_cls['accuracy']:.3f} & {baseline_cls['f1']:.3f} & {baseline_cls['roc_auc']:.3f}\\
Pegasos & {svm['accuracy']:.3f} & {svm['f1']:.3f} & {svm['roc_auc']:.3f}\\\bottomrule\end{{tabular}}\caption{{Held-out Task B results.}}\end{{table}}
\begin{{figure}}[t]\centering\includegraphics[width=\linewidth]{{../reports/figures/tree_tuning.png}}\caption{{Validation-based tree sweeps.}}\end{{figure}}
\begin{{figure}}[t]\centering\includegraphics[width=\linewidth]{{../reports/figures/regression_error_analysis.png}}\caption{{Predictions and residual analysis.}}\end{{figure}}
\begin{{figure}}[t]\centering\includegraphics[width=\linewidth]{{../reports/figures/roc_curves.png}}\caption{{ROC curves on held-out rows.}}\end{{figure}}
\section{{Limitations}}
The target is advertised price rather than transaction price; the data are a 2024 scrape and do not describe today's market. Fixed value bounds were selected after full-source EDA. Ambiguous duplicate advertisements under different IDs may remain. CV shares preprocessing learned before folds. Impurity importance/correlation are descriptive rather than causal. Log-space prediction transformed with an exponential does not include a smearing correction for arithmetic conditional means.
\section*{{AI-assistance disclosure}}
AI assisted with coding suggestions, debugging, corrections, review, tests, visualizations, and documentation. The team is responsible for checking the final code and results. The decision tree and Pegasos algorithms are implemented in project NumPy code; scikit-learn supplies comparison models and standard utilities.
\end{{document}}
'''
    (report_dir / "report.tex").write_text(body, encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix="pega-tex-") as tmp:
            for _ in range(2):
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tmp}", "report.tex"], cwd=report_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            shutil.copy2(Path(tmp) / "report.pdf", report_dir / "report.pdf")
            error_file = report_dir / "latex_build_error.txt"
            if error_file.exists():
                error_file.unlink()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = exc.stdout[-2000:] if isinstance(exc, subprocess.CalledProcessError) and exc.stdout else str(exc)
        (report_dir / "latex_build_error.txt").write_text(detail, encoding="utf-8")


def _slides(results, slide_dir):
    t = results["test_results"]
    r, c, s = t["decision_tree_regression"], t["decision_tree_classification"], t["pegasos_svm"]
    sr, sc = t["sklearn_tree_regression"], t["sklearn_tree_classification"]
    body = rf'''\documentclass[aspectratio=169]{{beamer}}
\usetheme{{default}}
\usepackage{{graphicx,booktabs,array}}
\definecolor{{PegaNavy}}{{HTML}}{{003893}}
\definecolor{{PegaLime}}{{HTML}}{{FCD116}}
\definecolor{{PegaRed}}{{HTML}}{{CE1126}}
\definecolor{{PegaMist}}{{HTML}}{{F4F7FC}}
\setbeamercolor{{normal text}}{{fg=PegaNavy,bg=white}}
\setbeamercolor{{structure}}{{fg=PegaNavy}}
\setbeamercolor{{frametitle}}{{fg=PegaNavy,bg=white}}
\setbeamercolor{{title}}{{fg=white,bg=PegaNavy}}
\setbeamercolor{{block title}}{{fg=white,bg=PegaNavy}}
\setbeamercolor{{block body}}{{fg=PegaNavy,bg=PegaMist}}
\setbeamertemplate{{navigation symbols}}{{}}
\setbeamertemplate{{headline}}{{}}
\setbeamertemplate{{itemize item}}{{\color{{PegaRed}}\raisebox{{.15ex}}{{\tiny\rule{{.7ex}}{{.7ex}}}}}}
\setbeamertemplate{{itemize subitem}}{{\color{{PegaNavy}}\scriptsize$\bullet$}}
\setbeamertemplate{{footline}}{{%
  \leavevmode\hbox to \paperwidth{{\color{{PegaLime}}\rule{{.18\paperwidth}}{{1.4pt}}\color{{PegaNavy}}\rule{{.68\paperwidth}}{{1.4pt}}\color{{PegaRed}}\rule{{.14\paperwidth}}{{1.4pt}}}}\par\nointerlineskip
  \leavevmode\hbox{{%
  \begin{{beamercolorbox}}[wd=.22\paperwidth,ht=2.8ex,dp=1.1ex,leftskip=1em]{{author in head/foot}}\color{{white}}\bfseries PegaSold\end{{beamercolorbox}}%
  \begin{{beamercolorbox}}[wd=.64\paperwidth,ht=2.8ex,dp=1.1ex,center]{{title in head/foot}}\color{{white}}MLE-AI-201  ·  Bina.az listing project\end{{beamercolorbox}}%
  \begin{{beamercolorbox}}[wd=.14\paperwidth,ht=2.8ex,dp=1.1ex,rightskip=1em plus 1fil]{{date in head/foot}}\color{{white}}\insertframenumber/\inserttotalframenumber\end{{beamercolorbox}}}}
  \vskip0pt}}
\setbeamercolor{{author in head/foot}}{{bg=PegaNavy}}
\setbeamercolor{{title in head/foot}}{{bg=PegaNavy}}
\setbeamercolor{{date in head/foot}}{{bg=PegaNavy}}
\setbeamertemplate{{frametitle}}{{\vspace{{.2em}}\insertframetitle\par\vspace{{.15em}}\color{{PegaLime}}\rule{{\textwidth}}{{1.2pt}}\vspace{{.25em}}}}
\setlength{{\leftmargini}}{{1.3em}}
\title{{Bina.az Housing Price Prediction}}
\subtitle{{NumPy decision trees and Pegasos SVM}}
\author[PegaSold]{{PegaSold\\[.7em]Emiliya Abdulhalimova · Aygun Asvarova · Murad Bagirzade · Kanan Karimli}}
\date{{October 2026}}
\begin{{document}}
\begin{{frame}}
\vspace*{{.7cm}}
\begin{{beamercolorbox}}[wd=\textwidth,sep=1.2em,center,rounded=true]{{title}}
{{\LARGE\inserttitle\par}}\vspace{{.35em}}{{\large\insertsubtitle\par}}
\end{{beamercolorbox}}
\vspace{{1.1cm}}{{\large\bfseries PegaSold\par}}
\vspace{{.25cm}}{{\normalsize Emiliya Abdulhalimova · Aygun Asvarova · Murad Bagirzade · Kanan Karimli\par}}
\vfill{{\large\insertdate\par}}\vspace{{.5cm}}
\end{{frame}}
\begin{{frame}}{{Project summary and learning tasks}}
\begin{{columns}}\begin{{column}}{{.61\textwidth}}
\begin{{block}}{{Task A — regression}}
Predict advertised sale price as $\ln(\mathrm{{price}})$. Report log-scale and AZN-scale errors; these measure proportional and currency error respectively.
\end{{block}}
\begin{{block}}{{Task B — binary classification}}
Predict standard vs premium. Premium means price at or above the training-only median ({results['tier_threshold_AZN']:,.0f} AZN).
\end{{block}}
\end{{column}}\begin{{column}}{{.35\textwidth}}
\centering\Large\textbf{{{results['data_audit']['raw_rows']:,}}}\\[-.1em]\normalsize raw rows\\[.8em]
\Large\textbf{{{results['data_audit']['final_rows']:,}}}\\[-.1em]\normalsize clean listings\\[.8em]
\Large\textbf{{{results['feature_count']}}}\\[-.1em]\normalsize feature columns
\end{{column}}\end{{columns}}
\vfill
\small Models built from NumPy; scikit-learn supplies matched references and standard utilities.
\end{{frame}}
\begin{{frame}}{{Dataset and source exploration}}
\begin{{columns}}\begin{{column}}{{.45\textwidth}}
\begin{{itemize}}
\item Real Bina.az sale-listing scrape; source snapshot is from October 2024.
\item Raw fields mix Azerbaijani text, numeric strings, currencies, categories, locations, and repeated scrapes.
\item The EDA inventories missingness, categories, duplicates, leakage candidates, and price quantiles.
\item Raw exploration is descriptive only; it does not choose model settings.
\end{{itemize}}
\end{{column}}\begin{{column}}{{.52\textwidth}}
\centering\includegraphics[width=\linewidth,height=4.4cm,keepaspectratio]{{../reports/eda/figures/fig01_missing_values.png}}
\end{{column}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Cleaning, deduplication, and leakage control}}
\small
\begin{{columns}}\begin{{column}}{{.47\textwidth}}
\begin{{itemize}}
\item Sort repeated listing URLs by scrape time and retain the latest row before splitting.
\item Keep Baku residential types; apply declared price/area/price-per-m² and room/floor bounds.
\item Invalid coordinates become missing and are imputed from training rows, not used to drop a listing.
\item Final data: {results['data_audit']['final_rows']:,} listings; raw CSV is never overwritten.
\item Exclude price-derived fields, IDs, free text, owner/shop names, and raw address.
\end{{itemize}}
\end{{column}}\begin{{column}}{{.49\textwidth}}
\centering\includegraphics[width=\linewidth,height=4.3cm,keepaspectratio]{{../reports/figures/data_and_prices.png}}
\end{{column}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Features, split, and exploratory patterns}}
\small
\begin{{columns}}\begin{{column}}{{.51\textwidth}}
\begin{{itemize}}
\item Numeric: area, rooms, floors, land, coordinates, engineered distance/ratios, and property flags.
\item Categories: property type, location, and repair state; one-hot vocabulary is learned from train.
\item Impute, standardize, and learn frequent location columns from train only; rare/unseen locations map to `other`.
\item Seed {results['seed']}; split {results['split']['train']:,}/{results['split']['validation']:,}/{results['split']['test']:,} (70/15/15).
\item Training median defines the premium threshold; validation and test do not define it.
\item Area and room count show positive price associations; location differences are descriptive, not causal.
\end{{itemize}}
\end{{column}}\begin{{column}}{{.47\textwidth}}
\centering\includegraphics[width=\linewidth,height=4.4cm,keepaspectratio]{{../reports/figures/price_vs_area.png}}
\par\scriptsize Price and area on log axes.
\end{{column}}\end{{columns}}
\vspace{{.2em}}\small The long price tail motivates log-price fitting; AZN metrics still expose currency-scale errors.
\end{{frame}}
\begin{{frame}}{{Decision tree: split search from first principles}}
\begin{{columns}}\begin{{column}}{{.52\textwidth}}
\begin{{itemize}}
\item Sort each numeric feature and test boundaries between distinct values.
\item Cumulative class counts give Gini/entropy gains; cumulative sums and squared sums give MSE reduction.
\item Recurse on the best legal split; leaves predict a majority label or mean target.
\item Stop on depth, sample/leaf constraints, pure targets, or no positive gain.
\item Importance sums sample-weighted impurity decrease.
\end{{itemize}}
\end{{column}}\begin{{column}}{{.43\textwidth}}
\centering\includegraphics[width=\linewidth,height=4.25cm,keepaspectratio]{{../reports/figures/decision_tree.png}}
\end{{column}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Linear Pegasos: objective, updates, and margin}}
\small
\begin{{columns}}\begin{{column}}{{.53\textwidth}}
\begin{{block}}{{Objective}}
$J(w,b)=\frac{{\lambda}}{{2}}\|w\|^2+\frac{{1}}{{n}}\sum_i\max(0,1-y_i(w^Tx_i+b))$, where $y_i\in\{{-1,+1\}}$.
\end{{block}}
\begin{{itemize}}
\item Standardize features from training rows; shuffle examples each epoch.
\item Use $\eta_t=1/(\lambda t)$; shrink weights, then add a hinge-loss subgradient for margin violations.
\item The intercept is unregularized and its per-update step is bounded for stability.
\item Margin width $2/\|w\|={s['margin_width']:.4f}$; {s['support_vector_count']:,} training points lie on/inside the margin.
\end{{itemize}}
\end{{column}}\begin{{column}}{{.43\textwidth}}
\begin{{block}}{{Selected fit}}
$\lambda={results['selected']['linear_svm']['lambda']}$\\
Accuracy {s['accuracy']:.3f}\\F1 {s['f1']:.3f}\\ROC-AUC {s['roc_auc']:.3f}
\end{{block}}
\centering\includegraphics[width=\linewidth,height=2.9cm,keepaspectratio]{{../reports/figures/pegasos_objective.png}}\\[-.2em]\scriptsize Objective by epoch; signed-margin plot and diagnostics are in the report.
\end{{column}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Nonlinear SVM: random Fourier features}}
\begin{{columns}}\begin{{column}}{{.56\textwidth}}
\begin{{itemize}}
\item Map each scaled row to $z(x)=\sqrt{{2/D}}\cos(xW+b)$; seeded Gaussian frequencies approximate an RBF kernel.
\item Train the same NumPy Pegasos solver on the mapped features.
\item $D=128$, selected $\gamma={results['selected']['rff_svm']['gamma']}$, and {results['svm_diagnostics']['rff_train_sample_count']:,} training rows.
\item RFF accuracy {t['rff_pegasos_svm']['accuracy']:.3f}, F1 {t['rff_pegasos_svm']['f1']:.3f}; RBF SVC reference accuracy {t['rbf_svc']['accuracy']:.3f}.
\end{{itemize}}
\end{{column}}\begin{{column}}{{.40\textwidth}}
\centering\includegraphics[width=\linewidth,height=4.2cm,keepaspectratio]{{../reports/figures/rff_confusion_matrix.png}}
\end{{column}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Experimental design and uncertainty}}
\begin{{columns}}\begin{{column}}{{.54\textwidth}}
\begin{{enumerate}}
\item Fit candidates on train and choose settings using validation only.
\item Refit selected scratch and reference models on training rows.
\item Evaluate the held-out test set after selection is finished.
\item Run seeded five-fold CV on all {results['cross_validation']['decision_tree_regression']['sample_n']:,} training rows for dispersion.
\end{{enumerate}}
\end{{column}}\begin{{column}}{{.42\textwidth}}
\begin{{block}}{{Matched references}}
DecisionTreeRegressor/Classifier use corresponding settings. LinearSVC uses hinge loss and $C=1/(\lambda n)$ with fold-local scaling. RBF SVC uses the RFF training subset.
\end{{block}}
\end{{column}}\end{{columns}}
\small CV: tree classification accuracy {results['cross_validation']['decision_tree_classification']['scratch']['mean']:.3f} $\pm$ {results['cross_validation']['decision_tree_classification']['scratch']['std']:.3f}; Pegasos {results['cross_validation']['pegasos_svm']['scratch']['mean']:.3f} $\pm$ {results['cross_validation']['pegasos_svm']['scratch']['std']:.3f}. Regression AZN MAE: {results['cross_validation']['decision_tree_regression']['azn_metrics']['scratch_mae']['mean']:,.0f} $\pm$ {results['cross_validation']['decision_tree_regression']['azn_metrics']['scratch_mae']['std']:,.0f}. Preprocessing vocabulary/imputation precedes folds, so this is not fully nested CV.
\end{{frame}}
\begin{{frame}}{{Task A: held-out price regression}}
\begin{{columns}}\begin{{column}}{{.39\textwidth}}
\begin{{block}}{{Scratch NumPy tree}}
Log-price RMSE: {r['rmse']:.3f}\\Log-price MAE: {r['mae']:.3f}\\$R^2$ (log): {r['r2']:.3f}\\[.3em]
AZN RMSE: {r['rmse_AZN']:,.0f}\\AZN MAE (average): {r['mae_AZN']:,.0f}\\Median absolute error: {r['median_absolute_error_AZN']:,.0f}\\Maximum absolute error: {r['max_absolute_error_AZN']:,.0f}\\$R^2$ (AZN): {r['r2_AZN']:.3f}
\end{{block}}
\small Matched sklearn tree AZN RMSE: {sr['rmse_AZN']:,.0f}.
\end{{column}}\begin{{column}}{{.57\textwidth}}
\centering\includegraphics[width=\linewidth,height=4.55cm,keepaspectratio]{{../reports/figures/regression_error_analysis.png}}
\end{{column}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Task B: held-out price-tier classification}}
\scriptsize
\begin{{tabular}}{{lrrrr}}
\toprule Model & Accuracy & Precision & F1 & ROC-AUC\\\midrule
NumPy tree & {c['accuracy']:.3f} & {c['precision']:.3f} & {c['f1']:.3f} & {c['roc_auc']:.3f}\\
sklearn tree & {sc['accuracy']:.3f} & {sc['precision']:.3f} & {sc['f1']:.3f} & {sc['roc_auc']:.3f}\\
NumPy Pegasos & {s['accuracy']:.3f} & {s['precision']:.3f} & {s['f1']:.3f} & {s['roc_auc']:.3f}\\
LinearSVC & {t['linear_svc']['accuracy']:.3f} & {t['linear_svc']['precision']:.3f} & {t['linear_svc']['f1']:.3f} & {t['linear_svc']['roc_auc']:.3f}\\
RFF Pegasos & {t['rff_pegasos_svm']['accuracy']:.3f} & {t['rff_pegasos_svm']['precision']:.3f} & {t['rff_pegasos_svm']['f1']:.3f} & {t['rff_pegasos_svm']['roc_auc']:.3f}\\
RBF SVC & {t['rbf_svc']['accuracy']:.3f} & {t['rbf_svc']['precision']:.3f} & {t['rbf_svc']['f1']:.3f} & {t['rbf_svc']['roc_auc']:.3f}\\\bottomrule\end{{tabular}}
\vspace{{.3em}}
\begin{{columns}}\begin{{column}}{{.49\textwidth}}\centering\includegraphics[width=\linewidth,height=3.55cm,keepaspectratio]{{../reports/figures/classification_comparison.png}}\end{{column}}
\begin{{column}}{{.49\textwidth}}\centering\includegraphics[width=\linewidth,height=3.55cm,keepaspectratio]{{../reports/figures/roc_curves.png}}\end{{column}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Validation sweeps and learning behavior}}
\begin{{columns}}\begin{{column}}{{.49\textwidth}}
\centering\includegraphics[width=\linewidth,height=4.3cm,keepaspectratio]{{../reports/figures/tree_tuning.png}}\\[-.1em]\scriptsize Validation sweep: log-RMSE and F1.
\end{{column}}\begin{{column}}{{.49\textwidth}}
\centering\includegraphics[width=\linewidth,height=4.3cm,keepaspectratio]{{../reports/figures/tree_learning_curves.png}}\\[-.1em]\scriptsize Training and validation behavior.
\end{{column}}\end{{columns}}
\vspace{{.15em}}\small Select minimum Task A log-RMSE and maximum Task B validation F1; final models are refit on all training rows.
\end{{frame}}
\begin{{frame}}{{Model interpretation and error analysis}}
\begin{{columns}}\begin{{column}}{{.49\textwidth}}
\centering\includegraphics[width=\linewidth,height=4.3cm,keepaspectratio]{{../reports/figures/feature_importance_regression.png}}\\[-.1em]\scriptsize Regression tree importance.
\end{{column}}\begin{{column}}{{.49\textwidth}}
\centering\includegraphics[width=\linewidth,height=4.3cm,keepaspectratio]{{../reports/figures/feature_importance_classification.png}}\\[-.1em]\scriptsize Tier tree importance.
\end{{column}}\end{{columns}}
\vspace{{.15em}}\small Importance summarizes predictive split reductions; it is not causal. Price-quartile errors and category slices are in the report; the luxury tail drives large AZN-space errors.
\end{{frame}}
\begin{{frame}}{{What worked, what did not, and limits}}
\begin{{columns}}\begin{{column}}{{.49\textwidth}}
\begin{{block}}{{Worked well}}
Leakage-aware split and feature design; scratch trees closely track matched tree references; median-tier prediction is easier than exact price prediction.
\end{{block}}
\begin{{block}}{{Did not work as well}}
The 128-feature RFF model underperformed the linear SVM here; increasing nonlinear complexity did not improve this run.
\end{{block}}
\end{{column}}\begin{{column}}{{.47\textwidth}}
\begin{{itemize}}
\item Asking prices are not verified sale prices.
\item One 2024 scrape does not represent today's market.
\item Sparse districts and views may not generalize.
\item Log errors and AZN errors describe different costs.
\end{{itemize}}
\end{{column}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Reproducibility and project handoff}}
\begin{{columns}}\begin{{column}}{{.57\textwidth}}
\begin{{block}}{{One-command run}}
\texttt{{python -m src.run\_all}}
\end{{block}}
\begin{{itemize}}
\item Raw exploration: \texttt{{src/explore.py}}; saved splits: \texttt{{data/processed/}}.
\item Metrics, plots, and limitations: \texttt{{reports/final\_report.md}}.
\item Detailed module guides: \texttt{{docs/code\_explanations/}}.
\item Tests: run the complete suite with \texttt{{pytest -q}}.
\end{{itemize}}
\end{{column}}\begin{{column}}{{.39\textwidth}}
\begin{{block}}{{AI-assistance disclosure}}
AI assisted with coding suggestions, debugging, corrections, and review. The team is responsible for checking the final code and results.
\end{{block}}
\end{{column}}\end{{columns}}
\vfill\small Team PegaSold · MLE-AI-201 Final Project
\end{{frame}}
\end{{document}}
'''
    slide_dir.mkdir(exist_ok=True)
    (slide_dir / "slides.tex").write_text(body, encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix="pega-slides-") as tmp:
            for _ in range(2):
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tmp}", "slides.tex"], cwd=slide_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            shutil.copy2(Path(tmp) / "slides.pdf", slide_dir / "slides.pdf")
            error_file = slide_dir / "latex_build_error.txt"
            if error_file.exists(): error_file.unlink()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = exc.stdout[-2000:] if isinstance(exc, subprocess.CalledProcessError) and exc.stdout else str(exc)
        (slide_dir / "latex_build_error.txt").write_text(detail, encoding="utf-8")


def _slides_madrid(results, slide_dir):
    """Build the 15-slide Madrid/Colombian-palette presentation."""
    t = results["test_results"]
    reg, tree, svm = t["decision_tree_regression"], t["decision_tree_classification"], t["pegasos_svm"]
    baseline_reg, baseline_tree = t["sklearn_tree_regression"], t["sklearn_tree_classification"]
    team = "Emiliya Abdulhalimova · Aygun Asvarova · Murad Bagirzade · Kanan Karimli"
    body = rf'''\documentclass[aspectratio=169]{{beamer}}
\usetheme{{Madrid}}
\usepackage{{graphicx,booktabs,array}}
\definecolor{{PegaBlue}}{{HTML}}{{003893}}
\definecolor{{PegaYellow}}{{HTML}}{{FCD116}}
\definecolor{{PegaRed}}{{HTML}}{{CE1126}}
\setbeamercolor{{structure}}{{fg=PegaBlue}}
\setbeamercolor{{frametitle}}{{fg=PegaBlue,bg=white}}
\setbeamercolor{{title}}{{fg=white,bg=PegaBlue}}
\setbeamercolor{{block title}}{{fg=white,bg=PegaBlue}}
\setbeamercolor{{block body}}{{fg=black,bg=blue!4}}
\setbeamertemplate{{navigation symbols}}{{}}
\setbeamertemplate{{itemize item}}{{\color{{PegaRed}}\small$\blacktriangleright$}}
\setbeamertemplate{{footline}}{{%
  \leavevmode\hbox to \paperwidth{{\color{{PegaYellow}}\rule{{.50\paperwidth}}{{2pt}}\color{{PegaBlue}}\rule{{.25\paperwidth}}{{2pt}}\color{{PegaRed}}\rule{{.25\paperwidth}}{{2pt}}}}\par\nointerlineskip
  \hbox{{\begin{{beamercolorbox}}[wd=.82\paperwidth,ht=2.6ex,dp=1ex,leftskip=1em]{{author in head/foot}}\color{{white}}PegaSold\hfill MLE-AI-201 · Bina.az Housing Price Prediction\end{{beamercolorbox}}\begin{{beamercolorbox}}[wd=.18\paperwidth,ht=2.6ex,dp=1ex,center]{{date in head/foot}}\color{{white}}\insertframenumber/\inserttotalframenumber\end{{beamercolorbox}}}}}}
\setbeamercolor{{author in head/foot}}{{bg=PegaBlue}}
\setbeamercolor{{date in head/foot}}{{bg=PegaBlue}}
\title[Bina.az Price Prediction]{{Bina.az Housing Price Prediction}}
\subtitle{{From-scratch decision trees and Pegasos SVM}}
\author[PegaSold]{{MLE-AI-201 Final Project V3\\[.4em]PegaSold\\[.25em]{team}}}
\date{{October 2026}}
\begin{{document}}
\begin{{frame}}\titlepage\end{{frame}}
\begin{{frame}}{{Project question — Speaker 1: data and task}}
\begin{{itemize}}
\item Bina.az asking-price scrape: {results['data_audit']['raw_rows']:,} raw rows; October–November 2024 records.
\item Task A predicts $\ln(\mathrm{{price}})$; errors are reported in log space and AZN.
\item Task B predicts standard/premium; the train-only median cutoff is {results['tier_threshold_AZN']:,.0f} AZN.
\item Decision tree and Pegasos training are NumPy implementations; sklearn supplies comparison baselines.
\end{{itemize}}
\end{{frame}}
\begin{{frame}}{{Cleaning and leakage control — Speaker 1}}
\small
\begin{{columns}}[T]\column{{.47\textwidth}}
\begin{{itemize}}
\item Normalize advert URLs to numeric IDs; keep each advert's latest scrape.
\item Keep Baku's three residential categories; fixed bounds check price, floor area, AZN/m$^2$, rooms and floors.
\item Train-derived log-price MAD fence: {results['data_audit']['log_price_outlier_filter']['price_fence_AZN'][0]:,.0f}--{results['data_audit']['log_price_outlier_filter']['price_fence_AZN'][1]:,.0f} AZN; removed {results['data_audit']['log_price_outlier_filter']['train_outliers_removed']:,} from train only.
\item Splits after train filtering: {results['split']['train']:,}/{results['split']['validation']:,}/{results['split']['test']:,}; validation/test retain all rows.
\item Exclude target-derived values, IDs, free text, owner/agency fields and raw address.
\item Imputation, scaling and location vocabulary are fitted on train; raw CSV stays unchanged.
\end{{itemize}}
\column{{.53\textwidth}}\centering\includegraphics[width=\linewidth,height=.77\textheight,keepaspectratio]{{../reports/figures/data_and_prices.png}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Exploration and feature design — Speaker 1}}
\small
\begin{{columns}}[T]\column{{.48\textwidth}}
\begin{{itemize}}
\item Raw EDA measures missingness, repeat scrapes, leakage candidates, categories, geography and price tails.
\item Parse Azerbaijani numeric strings; represent yes/no flags as 0/1 and invalid coordinates as missing.
\item Use log area/land, location/type indicators and engineered distance/floor/area ratios.
\item EDA describes this 2024 scrape; it is not causal evidence or today's market estimate.
\end{{itemize}}
\column{{.50\textwidth}}\centering\includegraphics[width=\linewidth,height=.76\textheight,keepaspectratio]{{../reports/figures/price_vs_area.png}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Split, targets, and validation protocol — Speaker 1}}
\begin{{itemize}}
\item Seed {results['seed']}: {results['split']['train']:,} train, {results['split']['validation']:,} validation, {results['split']['test']:,} test rows.
\item Repeated advert IDs are kept in one partition; split allocation balances price bins.
\item Premium threshold is independently learned from training prices only: {results['tier_threshold_AZN']:,.0f} AZN.
\item Select hyperparameters on validation, refit on train, and evaluate test after selection.
\item Five-fold CV uses all {results['cross_validation']['decision_tree_regression']['sample_n']:,} training rows; shared preprocessing precedes folds, so CV is not fully nested.
\end{{itemize}}
\end{{frame}}
\begin{{frame}}{{Decision tree from scratch — Speaker 2}}
\small
\begin{{columns}}[T]\column{{.49\textwidth}}
\begin{{itemize}}
\item Sort numeric values and evaluate boundaries between distinct adjacent values.
\item Cumulative class counts compute Gini/entropy gain; target sums and squared sums compute MSE reduction.
\item Recurse on the best legal split; leaves predict majority class or mean target.
\item Depth, minimum leaf and split constraints control model growth.
\item Importance sums sample-weighted impurity decreases; it does not establish causation.
\end{{itemize}}
\column{{.49\textwidth}}\centering\includegraphics[width=\linewidth,height=.73\textheight,keepaspectratio]{{../reports/figures/decision_tree.png}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Tree results and generalization — Speaker 2}}
\centering\small
\begin{{tabular*}}{{.92\linewidth}}{{@{{\extracolsep{{\fill}}}}lrlr}}
\toprule Metric & Score & Metric & Score\\\midrule
Task A log RMSE & {reg['rmse']:.3f} & Task B accuracy & {tree['accuracy']:.3f}\\
Task A AZN RMSE & {reg['rmse_AZN']:,.0f} & Task B F1 / AUC & {tree['f1']:.3f} / {tree['roc_auc']:.3f}\\
Reference tree log RMSE & {baseline_reg['rmse']:.3f} & Reference accuracy & {baseline_tree['accuracy']:.3f}\\
\bottomrule\end{{tabular*}}
\vspace{{.15cm}}\includegraphics[width=.88\linewidth,height=.53\textheight,keepaspectratio]{{../reports/figures/tree_learning_curves.png}}
\end{{frame}}
\begin{{frame}}{{Interpreting the tree's AZN price error — Speaker 2}}
\small
\begin{{columns}}[T]
\column{{.43\textwidth}}
\begin{{itemize}}
\item The tree learns mean squared error on $\ln(\mathrm{{price}})$, then exponentiates its output to AZN.
\item Test MAE: {reg['mae_AZN']:,.0f} AZN (average); median absolute error: {reg['median_absolute_error_AZN']:,.0f} AZN.
\item Test RMSE: {reg['rmse_AZN']:,.0f} AZN; maximum absolute error: {reg['max_absolute_error_AZN']:,.0f} AZN.
\item The maximum test error is {reg['max_absolute_error_AZN']:,.0f} AZN: one {reg['max_error_case']['actual_AZN']:,.0f} AZN home was estimated at {reg['max_error_case']['predicted_AZN']:,.0f} AZN.
\item Validation/test retain prices outside the fence; luxury-tail errors remain large and visible.
\item This is a historical asking-price estimate, not a transaction appraisal.
\end{{itemize}}
\column{{.55\textwidth}}\centering
\includegraphics[width=\linewidth,height=.72\textheight,keepaspectratio]{{../reports/figures/regression_error_analysis.png}}
\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Linear Pegasos SVM — Speaker 3}}
\small
\begin{{columns}}[T]\column{{.53\textwidth}}
\begin{{itemize}}
\item Minimize $\lambda\|w\|^2/2+$ mean hinge loss using stochastic subgradient updates.
\item Standardize from training data; labels map to $-1,+1$ internally.
\item Selected $\lambda={results['selected']['linear_svm']['lambda']}$; margin width $2/\|w\|={svm['margin_width']:.4f}$.
\item Support vectors (signed margin $\leq1$): {svm['support_vector_count']:,} training rows.
\item Test: accuracy {svm['accuracy']:.3f}, F1 {svm['f1']:.3f}, ROC-AUC {svm['roc_auc']:.3f}.
\end{{itemize}}
\column{{.47\textwidth}}\centering\includegraphics[width=\linewidth,height=.76\textheight,keepaspectratio]{{../reports/figures/pegasos_objective.png}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Nonlinear SVM and decision margin — Speaker 3}}
\small
\begin{{columns}}[T]\column{{.48\textwidth}}
\begin{{itemize}}
\item Random Fourier Features use seeded cosine projections to approximate an RBF kernel; NumPy Pegasos trains the mapped features.
\item 128 components; selected $\gamma={results['selected']['rff_svm']['gamma']}$; tuned on an 8,000-row training subset.
\item RFF test: accuracy {t['rff_pegasos_svm']['accuracy']:.3f}, F1 {t['rff_pegasos_svm']['f1']:.3f}, AUC {t['rff_pegasos_svm']['roc_auc']:.3f}.
\item Linear model performed better on this dataset/run; nonlinear mapping did not guarantee a gain.
\end{{itemize}}
\column{{.50\textwidth}}\centering\includegraphics[width=\linewidth,height=.75\textheight,keepaspectratio]{{../reports/figures/svm_margin_scores.png}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Fair comparison and uncertainty — Speaker 4}}
\small
\begin{{columns}}[T]\column{{.50\textwidth}}
\begin{{itemize}}
\item Tree references use matching criteria/depth/leaf settings on the same input matrix.
\item LinearSVC uses hinge loss, train-only scaling pipeline and objective-matched $C=1/(\lambda n)$.
\item Five-fold CV is on a seeded train sample; report means and standard deviations.
\item Test metrics are held out once; test results do not drive model selection.
\end{{itemize}}
\column{{.48\textwidth}}\centering\includegraphics[width=\linewidth,height=.76\textheight,keepaspectratio]{{../reports/figures/roc_curves.png}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Held-out classification comparison — Speaker 4}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lrrrr}}
\toprule Model & Accuracy & F1 & AUC & Fit sec\\\midrule
Scratch tree & {tree['accuracy']:.3f} & {tree['f1']:.3f} & {tree['roc_auc']:.3f} & {tree['fit_seconds']:.2f}\\
sklearn tree & {baseline_tree['accuracy']:.3f} & {baseline_tree['f1']:.3f} & {baseline_tree['roc_auc']:.3f} & {baseline_tree['fit_seconds']:.2f}\\
Pegasos & {svm['accuracy']:.3f} & {svm['f1']:.3f} & {svm['roc_auc']:.3f} & {svm['fit_seconds']:.2f}\\
LinearSVC & {t['linear_svc']['accuracy']:.3f} & {t['linear_svc']['f1']:.3f} & {t['linear_svc']['roc_auc']:.3f} & {t['linear_svc']['fit_seconds']:.2f}\\
RFF Pegasos & {t['rff_pegasos_svm']['accuracy']:.3f} & {t['rff_pegasos_svm']['f1']:.3f} & {t['rff_pegasos_svm']['roc_auc']:.3f} & {t['rff_pegasos_svm']['fit_seconds']:.2f}\\
\bottomrule\end{{tabular*}}
\vspace{{.18cm}}\begin{{columns}}\column{{.5\textwidth}}\includegraphics[width=\linewidth,height=.53\textheight,keepaspectratio]{{../reports/figures/classification_comparison.png}}\column{{.5\textwidth}}\includegraphics[width=\linewidth,height=.53\textheight,keepaspectratio]{{../reports/figures/regression_error_analysis.png}}\end{{columns}}
\end{{frame}}
\begin{{frame}}{{Validation sweeps and model interpretation — Speaker 4}}
\begin{{columns}}[T]\column{{.50\textwidth}}\centering\includegraphics[width=\linewidth,height=.65\textheight,keepaspectratio]{{../reports/figures/tree_tuning.png}}\\Validation parameter sweeps.
\column{{.48\textwidth}}\centering\includegraphics[width=\linewidth,height=.65\textheight,keepaspectratio]{{../reports/figures/feature_importance_regression.png}}\\Regression tree importance.
\end{{columns}}
\small Validation chooses settings; importance measures split utility and is not causal evidence.
\end{{frame}}
\begin{{frame}}{{Conclusions and limitations — All}}
\begin{{itemize}}
\item Median price-tier classification is easier than exact advertised-price prediction; AZN error exposes expensive-property misses.
\item Asking prices are not verified transaction prices; one 2024 scrape is not today's market.
\item Fixed cleaning bounds were chosen after source EDA; ambiguous cross-ID duplicate candidates may remain.
\item Sparse locations and the non-nested CV preprocessing can limit generalization claims.
\item Reproduce with \texttt{{python -m src.run\_all}}; full results and caveats are in \texttt{{reports/final\_report.md}}.
\end{{itemize}}
\end{{frame}}
\begin{{frame}}{{Reproducibility and four-part handoff — All}}
\small
\begin{{columns}}[T]\column{{.55\textwidth}}
\begin{{enumerate}}
\item Speaker 1: raw EDA, cleaning, targets, splits and leakage checks.
\item Speaker 2: NumPy decision tree, criteria, tuning and tree results.
\item Speaker 3: Pegasos, RFF mapping, margin and SVM findings.
\item Speaker 4: baselines, cross-validation, metrics, error analysis and reproducibility.
\end{{enumerate}}
\column{{.42\textwidth}}
\begin{{block}}{{Run and verify}}
\texttt{{python -m src.run\_all}}\\
\texttt{{pytest -q}}\\[.4em]
The automated test summary and outputs are in \texttt{{reports/}} and \texttt{{data/processed/}}.
\end{{block}}
\end{{columns}}
AI-assisted coding suggestions, debugging, corrections and review were used; the team checked the final code and results.
\end{{frame}}
\end{{document}}
'''
    slide_dir.mkdir(exist_ok=True)
    (slide_dir / "slides.tex").write_text(body, encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix="pega-slides-") as tmp:
            for _ in range(2):
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tmp}", "slides.tex"], cwd=slide_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            shutil.copy2(Path(tmp) / "slides.pdf", slide_dir / "slides.pdf")
            error_file = slide_dir / "latex_build_error.txt"
            if error_file.exists(): error_file.unlink()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = exc.stdout[-2000:] if isinstance(exc, subprocess.CalledProcessError) and exc.stdout else str(exc)
        (slide_dir / "latex_build_error.txt").write_text(detail, encoding="utf-8")


def write_reports(results, data, reports, figures):
    report_dir = reports.parent / "report"
    report_dir.mkdir(exist_ok=True)
    _markdown(results, data, reports, figures)
    _latex(results, report_dir)
    _slides_madrid(results, reports.parent / "presentation")
    from .site_data import write_site_data
    log_path = reports / "test_results.txt"
    write_site_data(results, reports.parent,
                    log_path.read_text(encoding="utf-8") if log_path.exists() else "")
