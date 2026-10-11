# `src/data_prep.py` — full implementation walkthrough

Source: [`src/data_prep.py`](../../src/data_prep.py). This guide follows the file from its constants through `main()`. The module's own comments explain the code in place; this document explains the reasoning and tricky interactions. The numbered pipeline stages match the source headings.

## Module contract and imports

The module reads the raw Bina.az CSV, constructs a restricted modeling table, splits it, learns imputation/encoding/scaling state from training only, writes clean artifacts, and returns arrays to `run_all.py`. It never writes to the input CSV. `argparse` exposes input/output paths; `hashlib` fingerprints ID sets; `json` stores audits/rules; `Path` handles filesystem paths; NumPy handles arrays and numeric transforms; pandas handles table parsing and aligned columns.

`RANDOM_SEED` fixes row allocation. `SEED` is a compatibility alias consumed by the experiment runner. `TRAIN_SHARE` and `VALID_SHARE` are used inside each descriptive price stratum; test receives the remaining examples. This results in nearly, rather than mathematically exactly, 70/15/15 counts when strata have indivisible sizes.

## Constants: declared scope, limits, and features

- `ADVERT_URL` identifies a listing. `parse()` extracts the numeric path ID, so `/items/123` and `/items/123.html` become the same ID.
- `KEEP_CITY` and `KEEP_CATEGORIES` define this project's population: Baku apartments and houses. Other property categories remain in the raw data; excluding them is a scope decision, not a claim that those rows are corrupt.
- Price, price-per-square-metre, and area bounds are fixed constants. They were selected after inspecting this dataset's full distributions. They are not recomputed separately by split, but the report discloses that source-level selection used the whole dataset.
- `LAT_RANGE`/`LNG_RANGE` are a broad Azerbaijan sanity box. Invalid or incomplete coordinates are set missing; they do not by themselves delete otherwise usable rows.
- The Baku centre and kilometres-per-degree values are external constants. `distance_to_centre_km` is an approximate local planar distance, suitable for this city-scale region, not a geodesic distance for long journeys.
- `MIN_ROWS_PER_LOCATION=100` prevents very rare location columns. Frequencies are measured on train only; validation/test-only names map to `location=other`.
- `LEAKAGE_COLUMNS` is an audit list. More fields are excluded in practice because `parse()` builds an allow-listed table instead of copying all source columns.
- `CONTINUOUS_FEATURES` is the ordered set standardized with training mean and population standard deviation (`ddof=0`). `BINARY_FEATURES` documents 0/1 feature names. `READABLE_COLUMNS` is a separate human-audit schema; it intentionally includes raw values and IDs that never enter `X`.

## Helpers

### `number_from_text(series)`

Converts strings such as `145 m²`, `1.3 sot`, and `3 450 AZN/m²`: cast to pandas nullable strings; remove whitespace used as a thousands separator; change decimal commas to periods; extract the first numeric token; coerce failures to missing numeric values. This parser intentionally does not infer units. Its caller must know whether a field is m² or sot. A limitation is that punctuation conventions other than decimal commas/thousands spaces are not inferred.

### `fingerprint(ids)`

Sorts string representations of IDs, joins them with newline separators, computes SHA-256, and keeps the first 16 hexadecimal characters. Sorting makes the digest independent of row order. It fingerprints membership, not full rows or the raw file. Empty/missing IDs should be treated carefully; `deduplicate()` preserves missing-ID rows rather than collapsing all nulls into one group.

## Steps 1–2: `load()` and `parse()`

`load(path)` delegates CSV decoding/type inference to pandas and reports its dimensions. `parse(df)` creates a new frame indexed like the source and copies only fields needed for identity, target, features, filtering, and readable tables.

Identity: `estate_rel_url_x` is converted to pandas string and the first digit group becomes `item_id`; `day_x` is parsed with the known `%d.%m.%Y` format. Parse failures become `NaT`. Target `price` is coerced to numeric; bad values become `NaN` and are later removed by the fixed price window.

Measurements: `Sahə` is parsed, but rows in the bare-land `Torpaq` category get `area_m2=NaN`, since that source field is in sot for land. The selected scope subsequently removes bare land. `Torpaq sahəsi` is multiplied by 100 to convert sot to square metres; absent plot area is filled with 0 after scope because a flat has no plot. Rooms are parsed numerically. A floor string like `7 / 9` is split into current floor and total floors; invalid/unlisted floors stay missing.

Location: city/category/location remain categorical text. Coordinates are numeric-coerced and retained only when both fall within the declared national bounds; bad pairs become missing. This is an imputation case, not row deletion.

Flags: affirmative-only scraper flags become 0/1 based on presence/value. The mapping relies on this dataset's observed convention (`var` means yes; blank or its observed negative token means no). This should be rechecked if the source schema changes. `owner_title == 'mülkiyyətçi'` defines the owner flag. Views and long-form text are deliberately not copied, avoiding a post-publication signal and target/identity leakage.

## Step 3: `deduplicate()` — what it does and does not claim

First, rows with a valid normalized `item_id` are sorted by ID and scrape date, then the last dated observation is retained. Missing IDs are preserved separately; they are not all treated as the same property. Ties on date retain pandas sort/source ordering behavior, so exact tie selection is a reproducibility detail worth disclosing.

Second, the code counts rows that share coordinates, area, room count, and price. These are only *candidate reposts*: separate apartments can share those values. The current implementation deliberately does not delete them without independent identity evidence. `rows_removed_by_reposted_twins` is therefore zero and `ambiguous_repost_candidate_rows` records the candidates. This avoids false deletions. Residual limitation: candidate homes under distinct IDs are not guaranteed to stay in one split, so the score can retain some near-duplicate optimism; the report calls this out instead of claiming perfect property-level deduplication.

The returned counts distinguish repeat scrapes from candidate matches and reset the index for downstream positional array work.

## Step 4: `apply_scope()`

Builds a boolean mask requiring both the exact Baku label and one of the three declared residential categories. It records rows before/after and the category/city settings. The exact Azerbaijani text match is intentional for this known CSV. A future spelling/casing variant would need explicit normalization rather than silently changing scope.

## Step 5: `apply_filters()`

Missing land plot size is set to zero. It then computes `price_per_m2=price/area_m2` and builds separate boolean masks for price window, area window, price-per-m² window, rooms, floor, and building height. Missing price/area naturally fail `.between()` and are removed. Missing room/floor data are allowed; they are either structurally absent (house floors) or later imputed/flagged. Floor greater than building height is invalid. The masks are reported individually, but they overlap: summing their counts double-counts rows. `drop` is their row-wise OR, so `total_removed_by_domain_filters` is the true unique count.

One special case: the fixed price/m² denominator is safe after `area_m2` has already been range-checked by the resulting OR mask; zero and missing area cannot survive. Rules were set after exploring this particular source; fixed bounds avoid per-split quantile drift but do not erase the source-selection limitation.

## Step 6: `split()` — stratification without changing the target rule

The full cleaned frame's median is used only to form two allocation strata. Within each stratum, a seeded NumPy permutation allocates approximately 70% train, 15% validation, and the remainder test. This helps each partition contain both broad price bands. The target threshold is *not* taken from this global median: `fit_preprocessor()` recomputes it from training rows only. The two uses are distinct and the split allocation use is recorded in the report.

Because group sizes are discrete, the partitions can differ slightly from exact 70/15/15. The split is row-wise after normalized-advert deduplication; ambiguous duplicate-post candidates are retained and not group-split.

## Step 7: `remove_training_price_outliers()` — robust log-price MAD

The raw price distribution is strongly right-skewed, so symmetric fences on AZN prices would be dominated by a few very large values. This function takes `log(price)` for the training partition, finds its median, then finds the median absolute deviation (MAD), which is the median of `abs(log_price - median)`. It computes a two-sided modified z-score `0.67449 * abs(log_price - median) / MAD` and flags rows above 3.5. The `0.67449` normal-consistency factor makes the score comparable to a standard z-score when the data are normal. The audit saves the learned AZN fence, log median, MAD, cutoff, and row counts.

The bounds are learned from training and used only to remove training rows. Validation/test remain untouched, preserving the full evaluation population and its tails; filtering those partitions by actual target price would bias the reported score. `preprocessing_report.json` records the fence and counts, with zero removals from validation/test by design. This reduces the tail's effect on fitting but does not erase its effect on honest test errors.

## Step 7: `fit_preprocessor(train)` — fit state once

1. Count training locations and sort the names meeting the minimum count.
2. Compute train medians for rooms, latitude, and longitude. `median` is robust to skew and missingness.
3. Compute the training median price for Task B. Later `build_targets()` labels `price >= threshold` as premium. Repeated values equal to the threshold are deliberately placed in class 1.
4. Call `build_features(train, rules, scale=False)` to obtain a training-only unscaled matrix. Measure each continuous feature's mean and population standard deviation. Zero standard deviation is replaced with 1 so division cannot produce infinity.
5. Store ordered feature names. That order is the column contract for all models and `.npy` matrices.

No validation/test feature value contributes to imputation, category vocabulary, threshold, or scaling statistics.

## Step 8: `build_features()` and `build_targets()`

`build_features()` writes an output DataFrame in deliberate order. `log(area)` compresses size; `log1p(land)` maps zero plots safely to zero. Missing rooms/coordinates get train medians. Houses have no floor number, so their floor fields become zero and `has_floor_info` identifies that those zeros mean “not applicable” rather than an actual ground-floor home.

Engineered features reshape existing information: approximate Baku-centre distance combines both coordinates; floor ratio normalizes a floor by building height; ground/top indicators expose common edge floors; area-per-room expresses average room size; land ratio compares plot to built area. They do not add source facts. Derived divisions guard zero rooms with `.clip(lower=1)` and guard zero building heights; logs use `log1p` for values that may be zero.

The binary indicators remain 0/1. Category columns are one-hot indicators in a fixed category order. Each retained location has an indicator; rare/unseen values activate a single `other` column. This guarantees validation/test matrices match train columns.

Only `CONTINUOUS_FEATURES` are standardized. If `scale=False`, that step is skipped for estimating the training statistics. Returned matrices are float arrays. Trees are invariant to monotone feature scale in split ordering, while the SVM optimization is scale-sensitive; using one shared matrix keeps the experiment pipeline consistent.

`build_targets()` returns `ln(price)` for regression and a 0/1 class from the training-only median for classification. The log target makes multiplicative price differences more comparable under squared loss; exponentiating predictions later produces a geometric-scale price estimate and can be biased for the arithmetic conditional mean.

## Step 9: `save()` and artifacts

For each of `train`, `validation`, and `test`, save the dense numeric feature matrix, log-price target, tier target, and a compressed `.npz` bundle containing `X`, `y_log_price`, `tier`, `price_AZN`, and `area_m2`. The readable CSV retains the agreed columns and `log_price`; IDs are included there strictly for auditing, not as model inputs. `feature_names.csv` defines matrix column order. `preprocessing_report.json` contains cleaning counts and fitted statistics; `preprocessing_rules.json` contains the fitted location/category vocabulary and medians/scaling state. These files live only under the output directory, normally `data/processed/`; the raw source is untouched.

At the start of saving, the function removes six specifically named outputs from the repository's older preprocessing schema (old manifests and `*_metadata.csv` tables). This prevents stale artifacts from appearing to belong to the new pipeline. It does not recursively clear the output directory or remove unrelated user files.

## Orchestration: `_prepare()`, `prepare_data()`, `run()`, and CLI

`_prepare()` calls the nine stages once, constructs the report, saves artifacts, and builds a dictionary of arrays/metadata expected by `run_all.py`. Keys are predictable: `X_train`, `y_train_log`, `tier_train`, `price_train`, `metadata_train`, `area_train_m2` (and corresponding validation/test keys). It also reports the number of continuous-plus-binary columns, feature list, threshold, missingness, and legacy-friendly row audit keys.

`prepare_data()` is the model-runner API and returns the dictionary; `run()` is the command-line API and returns the audit report. The parser supplies default source/destination paths. The `if __name__ == '__main__'` guard prevents running the pipeline just by importing the module, while `SystemExit(main())` passes the program's success/error code to the shell.

## Output validation checklist

- Train/validation/test `X` shapes have the same number of columns and every element is finite.
- No target-derived input (`total_price`, `unit_price`, description) or raw ID is in `feature_names`.
- The recorded tier threshold equals the train price median, never the validation/test median.
- Split item-ID fingerprints do not overlap.
- Each clean CSV's `log_price` equals `ln(price)`.
- All written artifacts are under `data/processed/`; the raw CSV's bytes remain unchanged.
