# `src/explore.py` — full raw-data EDA walkthrough

Source: [`src/explore.py`](../../src/explore.py). The module reads the raw CSV, prints and saves a detailed report, and writes plots. It never modifies the raw CSV or the model-ready processed splits. It answers data questions and explains cleaning choices; it is not model training.

## Imports, palette, and configuration

`argparse` exposes `--data` and `--out`; `Path` creates folders; Matplotlib's `Agg` backend permits rendering without a display; NumPy/Pandas implement summaries and plots. Constants define a consistent color palette and raw field groups: identity/bookkeeping, possible leakage, free text, numeric candidates, categorical candidates, presence-only flags, and geographic bounds. Those groups express hypotheses that later sections measure. They do not automatically dictate feature selection: `data_prep.py` is the operational source of truth.

`TARGET`, `ADVERT_ID`, `APARTMENTS`, and `RESIDENTIAL` connect the diagnostics to the actual Azerbaijani headers and chosen project population. `AZ_LAT`/`AZ_LNG` are broad coordinate checks, not a precise map boundary.

## `Report`

`Report.__call__` prints each message and stores the same text in a list; this keeps terminal output and saved `eda_report.txt` synchronized. `header()` adds a divider. `table()` uses Pandas' readable plain-text rendering. `save()` writes UTF-8 with a final newline so Azerbaijani characters survive and the file remains shell-friendly.

## Parsing and plot helpers

### `parse_number_from_text()`

Cast to nullable string; strip spaces; normalize decimal comma; extract the first numeric token; coerce malformed cells to `NaN`. It is a lightweight EDA parser, not the production conversion logic. It cannot infer units by itself.

### `parse_floor()`

Extract two numeric groups from strings like `7 / 9`; the first is the unit's floor, second is building height. Invalid strings yield missing values.

### `build_exploration_frame()`

Copies the input, then adds parsed area, plot size in source sot units, unit-price diagnostic, floor pair, room count, parsed scrape date, and a normalized numeric `item_id`. Numeric `area_m2` is not corrected for bare land here; analyses that relate floor area to price must filter to residential categories before use. Section 6 and 9 do so. The normalized identity ID intentionally mirrors data-prep URL extraction so reports count `.html` URL aliases consistently.

### `save()`, `style()`, `money()`, `compact()`

`save()` tightens layout, writes the PNG, closes the figure to reclaim memory, and logs its location. `style()` standardizes titles, labels, grid order, and removes top/right frame lines. `money()` formats AZN-like values with separators. `compact()` shortens axes to k/M/B to avoid overlapping labels.

## Sections 1–5: file, missingness, repeats, leakage, target

### 1. `section_overview()`

Reports dimensions/memory and builds a table with dtype, missing percentage, distinct count and one example per raw column. Constant columns have no variation; near-row-unique columns are identifier candidates. These are diagnostic categories rather than an automatic keep/drop verdict.

### 2. `section_missing()`

Computes missing fractions by column, then inspects presence-only fields. When the scraper writes a field only for affirmative values, blankness means “no” in this dataset and should become 0/1 instead of causing the row to be dropped. The report prints observed nonmissing values so that assumption is inspectable. The chart colors those flags separately from genuinely unknown numeric/text fields.

### 3. `section_duplicates()`

Counts exact duplicate rows, then groups by normalized `item_id` rather than raw URL text. It reports extra scrape rows, scrape multiplicity, and how many ads show changing price. The line chart truncates very high repeat counts for legibility; aggregate counts still appear in the text. This is why the pipeline deduplicates by normalized URL ID before splitting. Candidate properties under distinct IDs are analyzed later and not asserted to be exact duplicates.

### 4. `section_leakage()`

Measures how often `total_price` equals `price`; checks whether parsed `unit_price × area` reconstructs the target; samples descriptions for price words. This is evidence that those fields can reveal the answer. Descriptions can also contain addresses and identifiers. The production pipeline excludes them using an allow-list.

### 5. `section_target()`

Prints price quantiles and a few extremes, then compares raw and logarithmic histograms. The raw plot caps its display range to make the ordinary mass visible; this does not trim data. `log10` is only for visualization here; the project regression target uses natural log (`np.log`) in `data_prep.py`.

## Sections 6–10: measurements and descriptive associations

### 6. `section_numeric()`

First restricts data to the three residential categories so bare-land area-in-sot is not mixed with residential area-in-m². It prints quantiles and counts of selected plausibility issues. Area/price scatter uses log axes and a seeded sample to limit overplotting; room boxplots hide fliers to expose central ranges; floor medians show descriptive patterns. Correlations in plots are pairwise, not causal.

### 7. `section_categorical()`

For major raw categories it reports row counts and median prices. Median is less sensitive than mean to the long right tail. It checks whether redundant scraper flags agree with Azerbaijani source columns, plots category counts/medians, and compares frequently seen locations. Only top locations are plotted, while the text explains that production one-hot encoding has a train-only frequency threshold and an `other` category.

### 8. `section_geography()`

Counts coordinates inside/outside the broad bounds. The Baku price map uses a hexagonal binning scheme, coloring a cell by median asking price and hiding cells with fewer than ten records. This reduces overplotting and makes the spatial gradient visible. A map is descriptive and does not imply location causes a particular price.

### 9. `section_correlation()`

Restricts to residential data and positive prices, derives log price/log area, and computes a Pearson matrix for numeric columns. Each heatmap cell is explicitly labeled. Pearson correlation measures linear association after the stated transforms; correlated predictors can repeat information, and this matrix is not feature importance.

### 10. `section_tiers()`

Displays a whole-source median split strictly as a descriptive balance chart. This number/label is not used by models or evaluation. The actual class boundary comes from the training set only; the training target uses `price >= threshold`, while this raw illustration uses a strict `>` grouping. The explicit distinction prevents confusing EDA figures with benchmark labels.

## Sections 11–13: decisions data preparation needs

### 11. `section_scope()`

Compares category field availability, price-per-area spreads, and row counts for candidate populations after normalized-ID deduplication. Bare-land area is excluded from m² comparisons. It reports exact dynamic ratios/counts rather than hardcoded assertions. Its recommendation is Baku plus the three residential categories: a coherent project scope with usable rooms/floor/area features. This is a study-population choice, not data validity evidence.

### 12. `section_dedup()`

Uses normalized IDs, scrape dates, and within-ad price changes to justify keeping the last sighting. It examines same-coordinate/area/room/price tuples as possible cross-ID duplicates. Such a key may match different homes; it identifies candidates but cannot prove identity. Current preprocessing retains them rather than deleting rows on insufficient evidence. The report states this remaining duplicate risk.

### 13. `section_window()`

Shows tail quantiles, concrete extreme listings, price-per-area tails, and how candidate fixed windows alter counts. The selected rule is price 10,000–5,000,000 AZN, area 10–2,000 m², and price-per-m² 200–20,000 AZN/m²; production code also rejects nonmissing rooms outside 1–30 and invalid floor values. The bounds are fixed, applied before partitioning, and chosen after inspecting the full source distribution. That dataset-selection decision is disclosed; it is not described as independent of all test-distribution knowledge. The raw examples in the report are for diagnosis only, not manually edited.

### 14. `section_conclusions()`

Summarizes the feature allow-list implications: exclude IDs/bookkeeping, target-derived columns, free text, and post-publication views; parse typed measurements; represent affirmative flags; constrain domains; fit imputation/vocabulary/scaling on train only; and define price tier from training median. It is a readable recap, not a second pipeline.

## `explore_raw()`, `parse_args()`, and `main()`

`explore_raw(data_path, output_dir)` creates `output_dir/figures`, reads the CSV once, builds the EDA frame, executes sections in report order, saves the text file, and returns row/column/figure counts. It writes only to its chosen reports directory. `parse_args()` sets command defaults. `main()` calls the same reusable function and prints concise completion information. The `__main__` guard lets both `python -m src.explore` and the experiment runner use the code without executing anything on import.

## Reproducible outputs

The canonical raw diagnostic transcript is `reports/eda/eda_report.txt`; its PNGs are in `reports/eda/figures/`. Data preparation independently writes cleaned arrays/CSV/rules in `data/processed/`. To rerun EDA only: `python -m src.explore --data data/bina_az_sale.csv --out reports/eda`.
