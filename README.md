# MLE-AI-201 — Bina.az machine-learning final project

This repository implements the course brief's two core models from scratch in NumPy: a decision tree for both regression and classification, and a linear soft-margin SVM optimized with Pegasos. It uses a real Bina.az residential-listing dataset and compares against matched scikit-learn references. A seeded random-Fourier-feature mapping supplies the required nonlinear Pegasos alternative.

## Tasks

- **Task A — regression:** predict `ln(price)` to handle the right-skewed advertised-price distribution. Report RMSE, MAE and R² in log space and in AZN space.
- **Task B — binary classification:** label listings premium when price is at or above the median computed from training rows only. Report accuracy, precision, recall, F1, ROC-AUC and confusion matrices.

## Reproduce the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Download the Kaggle CSV to data/bina_az_sale.csv; details in data/README.md.
python -m src.explore
python -m src.run_all
pytest -q
```

`src.explore` inspects the untouched CSV and saves raw missingness, category, leakage, duplicate and price summaries. Then `src.run_all` cleans the data, writes reusable model-ready splits under `data/processed/`, runs validation-only hyperparameter searches, matched baselines, five-fold train-sample uncertainty estimates, final held-out scoring, error analysis, visualization and report generation. The source CSV is never edited. Processed artifacts include `train.npz`, `validation.npz`, `test.npz`, per-split `.npy` arrays and readable `*_clean.csv` tables, `feature_names.csv`, `preprocessing_report.json`, and `preprocessing_rules.json`.

The experiment runner also creates:

- `reports/final_report.md` — detailed report with embedded figures and complete experiment tables.
- `report/report.pdf` and `report/report.tex` — generated two-column report and its source.
- `presentation/slides.pdf` and `presentation/slides.tex` — a 15-slide Madrid-themed deck with team members on the title slide and Colombian tricolor footer accents.
- `index.html` and `report.html` — English price estimator and experiment report in the supplied minimal black, lime and white design.
- `reports/results.json` — parameters, scores, split sizes, feature importances, error slices and model diagnostics.
- `reports/figures/` — regenerated plots for data quality, sweeps, learning curves, scores, margins, confusion matrices, ROC curves and errors.

Imputation, continuous-feature scaling, location vocabulary and the premium cutpoint come from the training partition only. Re-scraped items are deduplicated by numeric ID parsed from their URL. Ambiguous same-coordinate/size/price groups are reported as candidates and retained because those fields do not prove two ads describe the same home. Price/area bounds are fixed before splitting, but were selected after inspecting the source data; the report discloses this choice. Model hyperparameters are selected on validation, and test metrics are reported after selection.

## Project code

- `src/data_prep.py` — data loading, cleaning, split, features and tier label.
- `src/explore.py` — raw-data checks and initial plots before preprocessing.
- `src/decision_tree.py` — NumPy tree with Gini, entropy and MSE criteria.
- `src/svm.py` — NumPy Pegasos and random Fourier feature Pegasos.
- `src/evaluate.py` — shared metrics.
- `src/run_all.py` — end-to-end experiment entry point.
- `src/reporting.py` — Markdown and LaTeX report generation.
- `tests/` — model, visualization, and pipeline tests.

Detailed module-by-module explanations, including the tricky implementation choices, are in [`docs/code_explanations/`](docs/code_explanations/).

## Run the interactive price demo

```bash
python -m src.web_demo
```

Open `http://127.0.0.1:8765`. The server reads the saved preprocessing rules and training arrays, fits the validation-selected NumPy regression and classification trees, then serves predictions from those models. The form accepts Baku location, property category, area, rooms, floor, optional house plot size, repair and deed flags. `report.html` remains available for experiment results and figures. This is a local demonstration of historical asking-price data, not a current-market appraisal.

## Team presentation

**Team PegaSold:** Emiliya Abdulhalimova, Aygun Asvarova, Murad Bagirzade, and Kanan Karimli. See [the balanced four-person presentation plan](presentation/README.md) and [contribution report](contribution_report.md); update the contribution descriptions to match work actually performed.

## Submission items

The course brief asks for a tagged repository link, an IEEE-style report PDF, and slides PDF. The generated report and slides name PegaSold and its four members. Review the metrics, AI-assistance disclosure, actual contribution statements, and repository tag before submitting.
