# Website and results payload: `index.html`, `report.html`, `styles.css`, `app.js`, `src/site_data.py`

## Purpose and files

The website is a static presentation of the actual experiment, not an online valuation service.

- `index.html` is the concise project overview: task statement, headline test metrics, pipeline steps, selected charts, run instructions and deliverable links.
- `report.html` is the longer browser-readable report: cleaning audit, split policy, complete regression/classification tables, CV, feature importances, price-slice errors, plots, limitations and balanced speaking plan.
- `styles.css` contains the shared responsive layout and both pages' components.
- `app.js` fills dynamic values and tables from the compact `site-data.js` object, connects navigation/tabs, copies the run command and opens chart images in a modal.
- `src/site_data.py` turns `reports/results.json` into that small browser payload whenever report artifacts are regenerated.
- `tests/test_site_data.py` verifies values, fallbacks, omissions and the generated JavaScript assignment.

## The important integrity choice

The supplied `code.html` is a design reference for a property estimator. Its example prices, currency conversion, feature bonuses, confidence percentage and comparable-property cards are hard-coded demo values; they were not produced by our trained model. The project does not persist a fitted estimator with a prediction endpoint, so presenting those controls as live model predictions would be false. The site keeps the reference's clean dashboard surfaces, badges, cards, charts and action panel but displays tested experiment results instead.

The page says “asking price” and dates the scrape to 2024. It does not claim verified transaction amounts or current market valuation. The lime/noir AI Academy tokens are paired with the Colombian flag's yellow/blue/red accent strip. The footer is part of ordinary document flow rather than fixed over content: this prevents the screenshot template's bottom collision and keeps the complete page scrollable. Cards, plots, tables and navigation reflow at tablet/mobile breakpoints.

## Results payload: `src/site_data.py`

`build_site_data(results, test_log)` is deliberately a projection of experiment outputs:

1. It selects model metrics needed by the website from `results["test_results"]`.
2. It retains preprocessing counts, split sizes, selected settings, fold statistics, leading feature importances and price error slices.
3. It leaves out full ROC curve arrays. Those coordinates are already visualized in a chart and are not needed in the HTML, keeping the JS smaller.
4. It reads the real pytest terminal summary with a narrow pattern. If no test output is present it writes a truthful “unavailable” fallback.
5. `write_site_data` JSON-serializes with Unicode preserved and emits `window.PEGASOLD_DATA = ...`. JSON serialization is important: it safely quotes location names and uses proper `null` values rather than hand-built JavaScript strings.

The helper is called by `src.reporting.write_reports`, which is itself called at the end of `python -m src.run_all`. Thus a fresh model run refreshes the website data together with the report and slides. The HTML/CSS/JS remains hand-maintained presentation code; model scores are not duplicated there.

## Browser behavior: `app.js`

The script is an immediately invoked function so its temporary variables do not leak into global scope. Its only global data dependency is `window.PEGASOLD_DATA`, loaded before it from `site-data.js`.

- `set(id, value)` writes metrics into explicit slots without building HTML from the full results object.
- `fmt`, `pct`, `int` and `dollars` centralize number formatting; metrics are rendered to sensible precision.
- The classification and regression tables are built from fixed allow-lists of model names and labels. The results JSON determines numeric cell values, not markup or labels. This avoids injecting arbitrary report text as HTML.
- The cleaning table, cross-validation table, price-quartile error table and feature-importance bars are also built from explicit fields.
- `[data-result-tab]` changes the visible task panel. The home page has no such controls, so the query returns an empty list and is harmless there.
- `data-lightbox` stores a known local chart path. Clicking opens the native `<dialog>`; clicking its close button, backdrop or Escape closes it.
- Clipboard access is optional; if browser permission is denied, a toast shows the command.
- The mobile navigation button toggles a class and ARIA state. CSS handles the breakpoint/layout; JavaScript does not set sizes.

## Design system: `styles.css`

CSS variables are the single source for surface, text, lime, tricolor, borders and typography. Cards use light borders and restrained shadows. The project summary has a dark surface; the lower run panel uses lime for its action grouping; the footer has the full-width tricolor strip and remains in normal flow. `@media` rules reduce margins, stack the hero, collapse links and convert chart/report grids for narrower viewports. Plot images use `object-fit: contain`, so labels are not cropped to fill a card.

Space Grotesk and Inter are loaded from Google Fonts when available; the CSS has system fallbacks for offline use. The site has no framework/CDN runtime dependency and no server-side API. `python -m http.server 8000` is only a static file server; open `http://localhost:8000`.

## Asset path contract

HTML links to generated assets using root-relative project paths such as `reports/figures/roc_curves.png`. Keep `index.html`, `report.html`, `site-data.js`, `reports/figures/`, `report/` and `presentation/` at their repository locations. `WEBSITE.md` lists the files and start command. The local data CSV is not needed for viewing the built site; it is needed to rerun the experiment.

## Verification

`tests/test_site_data.py` uses a small synthetic results object and a temporary directory. It checks the cleaned-row/split values, model metric preservation, absent-test fallback, omission of unused ROC arrays and parseable JSON syntax after the JS assignment. The full project test suite also covers data prep, metrics, tree, SVM and visualization behavior.
