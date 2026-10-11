# Dataset and generated data

The course project uses the Bina.az sale-listing dataset `bina_az_sale.csv`. The raw file is read only. Put it at `data/bina_az_sale.csv` (or pass `--input PATH` to `python -m src.data_prep`). The source columns are Azerbaijani and should remain UTF-8.

## Pipeline

Run `python -m src.explore --data data/bina_az_sale.csv --out reports/eda` for raw-data diagnostics, then `python -m src.data_prep --input data/bina_az_sale.csv --output-dir data/processed` for processed splits. `python -m src.run_all` runs both, trains/tunes models and generates all reports, figures, and presentation slides.

The modeling scope is Baku plus the three residential categories: new-build apartment, old-build apartment, and house/villa. Bare land is omitted because its `Sahə` unit is sot rather than residential square metres; commercial/office/garage listings are outside this task's population. Repeat scrapes are normalized to numeric advert IDs and the last scrape is retained. Similar measurement tuples are counted as candidate reposts but not deleted without address/identity evidence.

## `data/processed/` outputs

- `train.npz`, `validation.npz`, `test.npz`: compressed bundles with `X`, `y_log_price`, `tier`, `price_AZN`, and `area_m2`.
- `X_{train,valid,test}.npy`, `y_{train,valid,test}_log.npy`, `tier_{train,valid,test}.npy`: arrays for lightweight model loading.
- `{train,valid,test}_clean.csv`: readable rows and targets for audit.
- `feature_names.csv`: ordered names corresponding exactly to columns in `X`.
- `preprocessing_report.json`: counts, filters, split sizes, missingness and fitted numerical statistics.
- `preprocessing_rules.json`: train-only imputation values, location levels, tier boundary and scaling parameters.

The tier threshold, missing-value medians, one-hot location vocabulary, and continuous means/standard deviations are learned from train only. Fixed domain bounds were chosen after inspecting the full source distribution; this is disclosed in the project report. Feature construction excludes IDs, raw text, views, and target-derived fields such as `unit_price`/`total_price`. The raw CSV is never changed.
