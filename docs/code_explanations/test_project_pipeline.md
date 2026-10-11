# `tests/test_project_pipeline.py` — integration and evaluation tests

Source: [`tests/test_project_pipeline.py`](../../tests/test_project_pipeline.py). These tests lock the interface between parsing, cleaning, model arrays, reportable metrics, and visualization-oriented metadata.

## Measurement parser test

The parser receives a vector containing residential square metres, sot, and a missing cell. It must extract the first numeric token without trying to guess whether sot is square metres; the caller handles units. `pd.isna` is used because nullable pandas values do not always support Python's `np.isnan` boolean conversion.

## Prepared-data integration test

The test writes an 80-row synthetic CSV with every required Azerbaijani raw column, all three accepted property categories, Baku labels, numeric-string fields, and one normalized advert ID repeated twice. It calls `prepare_data()` with a temporary output directory, so the test never rewrites repository data.

Assertions verify raw count, last-sighting ID deduplication, Baku/category scope, exclusion of description/unit price from features, finite train data, equal feature width across partitions, and the fact that the tier cutoff equals the training price median. It then opens each `.npz`, checks feature width, confirms the readable clean split CSV exists, and checks that advert IDs are disjoint across train/validation/test. Finally it verifies both the preprocessing audit and the exact fitted rules are saved. This catches the common failure where source code changes but docs/tests still expect a prior function/file schema.

## EDA parser non-mutation test

`build_exploration_frame()` receives a tiny raw frame, adds parsed numeric/floor/date/item-ID columns to its returned copy, and leaves the input byte-for-byte equivalent at the DataFrame level. The test checks URL normalization and the area extraction. Full EDA rendering uses the production dataset and is exercised when `run_all` is run.

## Metric definition test

A hand-countable binary example asserts the exact `[[TN,FP],[FN,TP]]` matrix, positive-class precision, recall, F1, accuracy, and bounded AUC. A separate numeric example checks MAE, RMSE and R². These known-value assertions catch swapped matrix axes and accidental metric-unit changes.

## Other test modules

`test_decision_tree.py` covers constructor and input validation, classification/regression behavior, criteria, stopping settings, probabilities, importance normalization, and deterministic split behavior. `test_svm.py` covers binary validation, updates/objective, scores, margin diagnostics, and nonlinear RFF learning. `test_tree_viz.py` checks readable/exportable graph output. `conftest.py` adds repository root and `src/` to the test import path so both `src.*` and direct model imports work.

## Run

Use `pytest -q` from the repository root. A failing assertion should be investigated against the active source modules and current artifact schema; do not change a test to accept a behavior that violates the documented data contract.
