# `src/web_demo.py` — real-model browser demo

This module is the small HTTP application behind the PegaSold estimator in `index.html`. It is intentionally connected to the existing coursework pipeline: it does not use a hand-written pricing formula, a mock prediction, or scikit-learn models.

## Startup and artifacts

Run `python -m src.web_demo` from the project root. `EstimateService` reads `X_train.npy`, `y_train_log.npy`, `tier_train.npy`, `preprocessing_rules.json`, and `train_clean.csv` from `data/processed/`. If any artifact is missing, startup fails with a message to run `python -m src.run_all`. This avoids silently displaying an estimate from a fallback model.

At process startup the service fits two instances of the project's NumPy `DecisionTree` using the selected settings in `reports/results.json`: MSE regression (`max_depth=None`, `min_samples_leaf=20`, `min_samples_split=40`) and Gini classification (`max_depth=12`, leaf 5, split 10). The selected hyperparameters were chosen using validation data in the experiment runner. Training uses only the saved training partition. Fitting once at startup keeps each form submission quick; no model fitting or tuning uses the visitor's input.

## Location features

The preprocessing contract expects both a location name and geographic coordinates. The user chooses a model-known Baku location. For latitude/longitude, the service calculates the median of valid coordinates for that location from `train_clean.csv`—training rows only. If the selected location has no usable coordinates, it uses the training medians saved by preprocessing, maps the location one-hot feature into `location=other` when appropriate, and flags that coordinate fallback in the response. It does not invent precise coordinates for an address.

## One-row transform

The request is checked against the project domain: supported property categories, area 10–2,000 m², 1–30 rooms, nonnegative plot area, and a valid floor/building-floor pair. A one-row DataFrame is constructed with the exact parsed-column names expected by `build_features()`. House records can omit apartment-floor data; their land area is accepted. Absent VIP, featured and owner attributes are set to zero because the demo form has no evidence for those listing-only flags.

Calling `build_features(row, rules)` applies the same log transforms, engineered features, location/category one-hot encoding, train-fitted imputation and train-fitted scaling used for the model matrices. A rare/unseen location naturally enters the `other` bucket. Feature names/order therefore remain consistent with `X_train.npy`.

## Two different predictions

The regression tree returns `ln(price)`. `exp()` converts that point prediction back to AZN. Because the model was fitted in log space, this is a central multiplicative-scale estimate; it is not a mathematically bias-corrected arithmetic expected price. The interface labels it a historical advertised-price estimate and makes no transaction-price or current-market guarantee.

The classification tree separately returns Standard/Premium. Its boundary is the training median stored in preprocessing rules (currently 209,000 AZN). The result card also displays the proportion of the predicted class among training rows in the reached leaf (`predict_proba`). This leaf proportion is a descriptive model output, not a calibrated probability or confidence interval; the page says so next to the bar.

## HTTP endpoints and safety

- `GET /api/health` says whether model initialization succeeded.
- `GET /api/metadata` returns supported categories/locations, training count and price-tier cutpoint.
- `POST /api/predict` accepts one JSON form request and returns the two actual model outputs. Invalid values receive HTTP 400 and an explanatory message.
- Other GET paths serve the website assets from the repository root. The handler resolves paths and rejects traversal outside that root.

The service binds to `127.0.0.1:8000`, so it is a local demonstration by default, not a publicly hosted valuation service.

## Browser files

`index.html` contains the accessible Azerbaijani form and result panels. `demo.js` loads real metadata from the API, validates/submits user choices, and renders the API result without hard-coded prices. `demo.css` implements the supplied minimal black/lime/light-surface visual system responsively. The PegaSold logo is copied to `assets/pegasold-logo.png`. `report.html` remains the project-results page.

## Tests

`tests/test_web_demo.py` checks artifact/model initialization, a real end-to-end tree prediction, house features without floor information, validation boundaries, and the unseen-location fallback. Run all project checks with `pytest -q`.
