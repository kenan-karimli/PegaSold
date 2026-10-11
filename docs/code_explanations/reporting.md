# `src/reporting.py` — reports and 15-slide deck walkthrough

Source: [`src/reporting.py`](../../src/reporting.py). This module turns the experiment result dictionary into a detailed Markdown report, a compact two-column LaTeX/PDF report, and the synchronized Beamer source/PDF presentation. All numeric values are read from `results.json` data passed in by `run_all`; the module must not invent or hardcode model scores.

## Formatting helpers and team constants

`_f(value,digits)` renders a numeric value or em dash; `_table(headers,rows)` builds a Markdown table from already formatted cells. `TEAM` stores the four names and `TEAM_LINE` joins them for the report. Names belong on the title slide only in the presentation.

## `_markdown(results,data,reports,figures)`

The function reads the data audit and test metric maps. It creates separate rows for the two regression trees and six classification models. Metric names preserve the target units: log-space error and AZN-space error are separate columns. It then collects fold mean/std summaries, training-only EDA diagnostics, chosen hyperparameters, confusion matrices, top tree importances, subgroup errors, full validation sweep tables, discussion, limitations, contribution plan, AI-assistance disclosure, and generated figure inventory.

The data section documents the actual normalized-ID deduplication, exact Baku/residential scope, fixed domain bounds, ambiguous duplicate candidate groups, and the training-only imputation/scaling/category vocabulary. It explicitly discloses that domain bounds were selected after examining the source distribution. The tier boundary is derived from training only; global median bins are used only to allocate balanced splits. Candidate duplicate tuples are not described as confirmed identity.

The result tables keep each model's measured test score and timing. Confusion matrices use true class on rows and prediction on columns. The report's CV prose notes that imputation/scaling/category vocabulary were learned before the folds, so it is not fully nested cross-validation. Its Markdown tables include every tree/SVM validation candidate, not only the best one.

Raw EDA figures are appended with relative Markdown image links, then the full text transcript from `reports/eda/eda_report.txt` is included in a fenced appendix. A code appendix links every Python module to its module guide, while the complete runnable source remains in `src/`. This keeps results and explanations together without copying thousands of source lines into the result narrative.

## `_latex(results,report_dir)`

Builds a typeset two-column summary using the same raw/clean counts, feature count, split sizes, training cut, test scores, and comparison baselines from the result object. It describes the actual 10k–5M AZN, 10–2,000 m², and 200–20k AZN/m² bounds; the source-level EDA selection limitation; training-only imputation/scaling; retained ambiguous matches; and the absence of transaction-price labels. Figures are loaded from `../reports/figures` relative to the report directory.

The output LaTeX source is saved as `report/report.tex`. When `pdflatex` exists, the function compiles twice in a temporary folder so references settle, then copies only the PDF into `report/`. On failure it records the build detail in `report/latex_build_error.txt`; the Markdown and `.tex` sources remain available.

## `_slides_madrid(results,slide_dir,figures)`

This function builds the current 15-frame, 16:9 Beamer presentation from live result values, following the supplied Madrid example. It follows the report order: task and data, cleaning, features, the NumPy tree, regression errors, linear/nonlinear Pegasos, cross-validation, held-out comparisons, interpretation, limitations, and reproducibility. The four member names appear on the title page only; speaker labels identify presentation parts on relevant slides.

The visual system uses the standard Madrid theme: white canvas, blue headings and structure, and the theme's built-in bottom navigation/footer panel with project identity and page number. Plots use the original `reports/figures` assets and are sized to fit beside the text or inside the result frame. The custom `presentation` class and XeLaTeX are not available in the project environment, so the generator uses standard Beamer and pdfLaTeX with the supplied template's 16:9 layout and Madrid theme.

Compilation runs twice in a temporary directory and retains the PDF beside `slides.tex`; errors are written to `presentation/latex_build_error.txt`. `write_reports()` calls Markdown, LaTeX, and the presentation generator after `run_all` has written plots and result JSON.

## Accuracy and maintenance checks

- Search final Markdown, LaTeX and slide source for old fixed metrics/counts after changing pipeline settings.
- Verify all generated image targets exist relative to the rendered artifact directory.
- Confirm 15 Beamer frames, names on the first frame only, and the footer/page number on every content frame.
- Compare reported cleaning stage counts and feature count to preprocessing JSON/results JSON.
- Preserve the raw EDA transcript and figure links; do not convert descriptive EDA tiers into the training threshold.
