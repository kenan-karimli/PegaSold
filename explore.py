"""
explore_data.py  --  one-file exploratory data analysis (EDA) for the bina.az
real-estate sale dataset (ML final project, Problem 1).

HOW TO RUN (from the repo root):
    python explore_data.py
    python explore_data.py --data data/bina_az_sale.csv --out reports/eda

WHAT IT DOES (each numbered section below is a function):
    1.  Load the CSV
    2.  Basic overview            (shape, dtypes, memory, sample rows)
    3.  Missing values            (table + bar chart)
    4.  Duplicates                (exact duplicate rows)
    5.  Column type audit         (numbers hidden in text, constants, IDs)
    6.  Numeric summary           (describe, skew, zeros, negatives)
    7.  Categorical summary       (cardinality, top values, rare values)
    8.  Target analysis           (price distribution, log-transform, tiers)
    9.  Outlier scan              (IQR rule on every numeric column)
    10. Correlations              (Pearson + Spearman heatmaps)
    11. Leakage detection         (columns that secretly contain the price)
    12. Feature vs target plots   (scatter / boxplots against price)
    13. Geographic check          (lat/lng validity + map coloured by price)

OUTPUT:
    * Everything printed is also saved to  <out>/eda_report.txt
    * All figures are saved as PNG in      <out>/figures/
    * Nothing is modified: this script only READS the data. Cleaning decisions
      (what to drop, how to impute) are for you and your team to make in
      src/data_prep.py after reading the report.

Only needs: numpy, pandas, matplotlib (all already in requirements.txt).
"""

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")  # draw to files only, so it works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION  -- edit these if your column names differ
# =============================================================================
DATA_PATH = "data/bina_az_sale.csv"   # where the CSV lives
OUT_DIR = "reports/eda"               # where the report + figures go
TARGET_CANDIDATES = ["price", "Price", "qiymet", "Qiymət"]  # tried in order
RANDOM_SEED = 42                      # for reproducible subsampling in plots
MAX_PLOT_POINTS = 20_000              # scatter plots get slow with 100k+ points

# Column-name fragments that suggest a column is derived from the price.
# These are only HINTS; section 11 also checks the numbers themselves.
LEAKAGE_NAME_HINTS = ["unit_price", "total_price", "price_per", "per_m", "per_sqm", "qiymet_m"]

# Rough bounding box of Azerbaijan, used to spot impossible coordinates.
AZ_LAT_RANGE = (38.3, 41.95)
AZ_LNG_RANGE = (44.7, 50.9)

REPORT_LINES = []  # every log() call is stored here and written to disk at the end


# =============================================================================
# SMALL HELPERS
# =============================================================================
def log(text=""):
    """Print to the console AND remember the line for the saved text report."""
    print(text)
    REPORT_LINES.append(str(text))


def section(title):
    """Print a clearly visible section header."""
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def save_fig(fig, name, out_dir):
    """Save a matplotlib figure into <out_dir>/figures/ and close it (frees memory)."""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"  [figure saved] {path}")


def to_number(series):
    """
    Convert a text column such as '120 m²', '85 000 AZN' or '3,5' into floats.

    Steps: keep only digits, dots, commas and minus signs -> treat comma as a
    decimal point -> pd.to_numeric with errors='coerce' (bad values become NaN).
    This is a *heuristic*: always eyeball the result before trusting it.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d,.\-]", "", regex=True)  # drop letters, spaces, symbols
        .str.replace(",", ".", regex=False)          # '3,5' -> '3.5'
    )
    return pd.to_numeric(cleaned, errors="coerce")


def subsample(df, n=MAX_PLOT_POINTS):
    """Random sample of at most n rows, for plots that get slow on huge data."""
    if len(df) <= n:
        return df
    return df.sample(n, random_state=RANDOM_SEED)


def find_target(df):
    """Return the name of the price column (tries TARGET_CANDIDATES, then any col containing 'price')."""
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    for c in df.columns:
        if "price" in str(c).lower() and "unit" not in str(c).lower() and "total" not in str(c).lower():
            return c
    return None


# =============================================================================
# 1. LOAD
# =============================================================================
def load_data(path):
    """
    Read the CSV. Azerbaijani letters (ə ı ş ç ğ ö ü) need UTF-8, so we try
    utf-8 first, then utf-8-sig (UTF-8 with a BOM, common from Excel), then latin-1.
    """
    section("1. LOADING DATA")
    if not os.path.exists(path):
        sys.exit(f"ERROR: file not found: {path}\n"
                 f"Pass the right path with:  python explore_data.py --data <file.csv>")
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            log(f"Loaded '{path}' with encoding={enc}")
            return df
        except UnicodeDecodeError:
            continue
    sys.exit("ERROR: could not decode the file with utf-8 / utf-8-sig / latin-1.")


# =============================================================================
# 2. OVERVIEW
# =============================================================================
def overview(df):
    """Shape, memory, column dtypes and a few sample rows: the 'first look'."""
    section("2. BASIC OVERVIEW")
    log(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")
    log(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    log("\nColumn names and dtypes:")
    for c in df.columns:
        log(f"  {str(c):35s} {str(df[c].dtype):10s} non-null={df[c].notna().sum():,}")
    log("\nFirst 5 rows (transposed so long column lists stay readable):")
    log(df.head(5).T.to_string())
    log("\nRandom 5 rows (the head can be unrepresentative if the file is sorted):")
    log(df.sample(min(5, len(df)), random_state=RANDOM_SEED).T.to_string())


# =============================================================================
# 3. MISSING VALUES
# =============================================================================
def missing_values(df, out_dir):
    """
    Count and plot missing values per column.
    WHY: decides your imputation strategy. A column that is 90% empty is
    usually dropped; one that is 3% empty is usually imputed (median / mode).
    Also watch for 'hidden' missing values like '-', 'N/A', '' stored as text.
    """
    section("3. MISSING VALUES")
    n_missing = df.isna().sum()
    pct = (n_missing / len(df) * 100).round(2)
    table = pd.DataFrame({"missing": n_missing, "percent": pct}).sort_values("percent", ascending=False)
    log(table[table["missing"] > 0].to_string() if (table["missing"] > 0).any() else "No NaN values found.")

    # Hidden placeholders: text that means "missing" but pandas did not parse as NaN.
    placeholders = {"", "-", "--", "n/a", "na", "none", "null", "nan", "?", "yoxdur"}
    hidden = {}
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        count = df[c].astype(str).str.strip().str.lower().isin(placeholders).sum()
        if count:
            hidden[c] = int(count)
    if hidden:
        log("\nColumns with hidden 'missing' placeholders ('-', 'n/a', empty text...):")
        for c, k in hidden.items():
            log(f"  {c}: {k:,} rows  -> convert to NaN during cleaning")

    shown = table[table["missing"] > 0]
    if len(shown):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(shown))))
        ax.barh(shown.index[::-1].astype(str), shown["percent"][::-1])
        ax.set_xlabel("% missing")
        ax.set_title("Missing values per column")
        save_fig(fig, "03_missing_values.png", out_dir)


# =============================================================================
# 4. DUPLICATES
# =============================================================================
def duplicates(df):
    """
    Count exact duplicate rows. Real-estate sites often have the same ad
    re-posted. Duplicates are dangerous: the same house can land in BOTH train
    and test and inflate your test score.
    """
    section("4. DUPLICATES")
    n = df.duplicated().sum()
    log(f"Exact duplicate rows: {n:,} ({n / len(df) * 100:.2f}%)")
    id_like = [c for c in df.columns if re.search(r"(^id$|_id$|^id_|elan|listing)", str(c).lower())]
    for c in id_like:
        d = df[c].duplicated().sum()
        log(f"Possible ID column '{c}': {d:,} repeated values")
    log("Tip: also consider near-duplicates (same price + area + coordinates).")


# =============================================================================
# 5. COLUMN TYPE AUDIT
# =============================================================================
def column_audit(df):
    """
    Find structural problems:
      * text columns that are really numbers ('120 m²')
      * constant columns (useless for learning)
      * ID-like columns (every value unique -> pure noise / leakage risk)
    Returns (numeric_cols, categorical_cols) used by later sections.
    """
    section("5. COLUMN TYPE AUDIT")
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    log("Text columns that look numeric (>=80% of non-null values contain a number):")
    found = False
    for c in cat_cols:
        non_null = df[c].dropna()
        if non_null.empty:
            continue
        parsed = to_number(non_null)
        if parsed.notna().mean() >= 0.80 and non_null.nunique() > 10:
            sample = non_null.astype(str).head(3).tolist()
            log(f"  {c}: e.g. {sample}  -> parse into a numeric column")
            found = True
    if not found:
        log("  none")

    log("\nConstant columns (1 unique value) -> drop:")
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    log("  " + (", ".join(map(str, const)) if const else "none"))

    log("\nID-like columns (almost every value unique) -> usually drop or ignore:")
    ids = [c for c in df.columns if df[c].nunique() > 0.95 * len(df) and df[c].dtype == object]
    log("  " + (", ".join(map(str, ids)) if ids else "none"))
    return numeric_cols, cat_cols


# =============================================================================
# 6. NUMERIC SUMMARY
# =============================================================================
def numeric_summary(df, numeric_cols):
    """
    Descriptive stats for numeric columns, plus skewness (|skew| > 1 means a
    long tail -> consider log transform), count of zeros and negatives
    (an area or price of 0 or below is usually a data error).
    """
    section("6. NUMERIC COLUMNS SUMMARY")
    if not numeric_cols:
        log("No numeric columns detected (they may be stored as text; see section 5).")
        return
    desc = df[numeric_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc["skew"] = df[numeric_cols].skew()
    desc["zeros"] = (df[numeric_cols] == 0).sum()
    desc["negatives"] = (df[numeric_cols] < 0).sum()
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.3f}".format):
        log(desc.to_string())
    skewed = desc.index[desc["skew"].abs() > 1].tolist()
    log(f"\nHighly skewed columns (|skew|>1): {skewed}")


# =============================================================================
# 7. CATEGORICAL SUMMARY
# =============================================================================
def categorical_summary(df, cat_cols, out_dir):
    """
    For each text column: number of distinct values and the most frequent ones.
    WHY: cardinality decides encoding. Few values -> one-hot. Many values
    (e.g. 100+ districts) -> frequency/target encoding or grouping rare values.
    """
    section("7. CATEGORICAL COLUMNS SUMMARY")
    plotted = 0
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False)
        n_unique = df[c].nunique()
        rare = (vc < 0.01 * len(df)).sum()
        log(f"\n--- {c}  (unique={n_unique:,}, values under 1% of rows={rare:,}) ---")
        log(vc.head(10).to_string())
        # Plot only reasonably small columns so figures stay readable.
        if 2 <= n_unique <= 40 and plotted < 12:
            fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * min(n_unique, 15))))
            vc.head(15)[::-1].plot.barh(ax=ax)
            ax.set_title(f"Top values: {c}")
            safe = re.sub(r"[^\w]+", "_", str(c))
            save_fig(fig, f"07_cat_{safe}.png", out_dir)
            plotted += 1


# =============================================================================
# 8. TARGET ANALYSIS
# =============================================================================
def target_analysis(df, target, out_dir):
    """
    Study the thing you predict (price).
    * Real-estate prices are right-skewed -> log(price) is much closer to
      bell-shaped, which is why the project suggests regressing on log(price).
    * Task B: 'premium vs standard' uses the MEDIAN price as threshold, so the
      classes are balanced by construction. IMPORTANT: in your real pipeline
      compute that median on the TRAINING split only (here we use all data
      purely to illustrate).
    Returns the numeric price Series (NaN where unparseable).
    """
    section("8. TARGET ANALYSIS")
    if target is None:
        log("Could not find a price column. Set TARGET_CANDIDATES at the top of the file.")
        return None
    y = df[target] if pd.api.types.is_numeric_dtype(df[target]) else to_number(df[target])
    log(f"Target column: '{target}'   valid values: {y.notna().sum():,} / {len(y):,}")
    log(f"Non-positive prices (<=0): {(y <= 0).sum():,}  -> impossible for log(); must be cleaned")

    qs = y.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).round(0)
    log("\nQuantiles:\n" + qs.to_string())
    log(f"\nMean={y.mean():,.0f}  Median={y.median():,.0f}  Skew={y.skew():.2f}  (mean >> median = right tail)")

    pos = y[y > 0].dropna()
    log(f"Skew after log(price): {np.log(pos).skew():.2f}  (closer to 0 = more symmetric)")

    # IQR outlier rule on the raw price
    q1, q3 = pos.quantile([.25, .75])
    hi = q3 + 1.5 * (q3 - q1)
    log(f"IQR upper fence = {hi:,.0f} -> {(pos > hi).sum():,} listings above it ({(pos > hi).mean() * 100:.1f}%)")

    median = pos.median()
    tier = (y > median).astype(int)[y.notna()]
    log(f"\nExample tier split at median={median:,.0f}: premium share = {tier.mean() * 100:.1f}% (should be ~50%)")

    # Figure: 4 panels = raw hist, log hist, boxplot raw, boxplot log
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(pos.clip(upper=pos.quantile(0.99)), bins=60)
    axes[0, 0].set_title("Price (clipped at 99th pct for readability)")
    axes[0, 1].hist(np.log(pos), bins=60)
    axes[0, 1].set_title("log(price)  <- closer to bell shape")
    axes[1, 0].boxplot(pos, vert=False, showfliers=True)
    axes[1, 0].set_title("Price boxplot (dots = outliers)")
    axes[1, 1].boxplot(np.log(pos), vert=False)
    axes[1, 1].set_title("log(price) boxplot")
    save_fig(fig, "08_target_distribution.png", out_dir)
    return y


# =============================================================================
# 9. OUTLIER SCAN
# =============================================================================
def outlier_scan(df, numeric_cols, out_dir):
    """
    IQR rule on every numeric column: a value is flagged if it lies more than
    1.5 * IQR beyond the quartiles. Flagged != wrong. A 900 m2 villa is a real
    outlier; a 9000 m2 flat is a typo. Decide per column, then DOCUMENT the
    rule you chose (the project statement requires this).
    """
    section("9. OUTLIER SCAN (IQR rule)")
    rows = []
    for c in numeric_cols:
        s = df[c].dropna()
        if s.nunique() < 5:
            continue  # skip flags/binary columns
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append((c, lo, hi, s.min(), s.max(), n_out, n_out / len(s) * 100))
    if not rows:
        log("No numeric columns with enough distinct values.")
        return
    t = pd.DataFrame(rows, columns=["column", "low_fence", "high_fence", "min", "max", "n_outliers", "pct"])
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 200):
        log(t.sort_values("pct", ascending=False).to_string(index=False))

    cols = [r[0] for r in rows][:12]
    fig, axes = plt.subplots(int(np.ceil(len(cols) / 4)), 4, figsize=(14, 3 * int(np.ceil(len(cols) / 4))))
    for ax, c in zip(np.atleast_1d(axes).ravel(), cols):
        ax.boxplot(df[c].dropna(), vert=True)
        ax.set_title(str(c), fontsize=9)
    for ax in np.atleast_1d(axes).ravel()[len(cols):]:
        ax.axis("off")
    save_fig(fig, "09_outlier_boxplots.png", out_dir)


# =============================================================================
# 10. CORRELATIONS
# =============================================================================
def correlations(df, numeric_cols, y, target, out_dir):
    """
    Pearson = linear relationship; Spearman = monotonic (rank) relationship,
    robust to outliers. If Spearman >> Pearson for a feature, the relation
    is non-linear or outlier-driven (hint: trees / kernels may help).
    Very high correlation between two FEATURES = redundancy (multicollinearity).
    """
    section("10. CORRELATIONS")
    cols = [c for c in numeric_cols if df[c].nunique() > 1]
    if len(cols) < 2:
        log("Not enough numeric columns.")
        return
    data = df[cols].copy()
    for method in ("pearson", "spearman"):
        corr = data.corr(method=method)
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(cols)), max(5, 0.5 * len(cols))))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=8)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{method.capitalize()} correlation")
        save_fig(fig, f"10_corr_{method}.png", out_dir)

    if y is not None:
        tmp = data.copy()
        tmp["__target__"] = y
        t = tmp.corr(method="spearman")["__target__"].drop("__target__", errors="ignore")
        p = tmp.corr(method="pearson")["__target__"].drop("__target__", errors="ignore")
        out = pd.DataFrame({"spearman_vs_price": t, "pearson_vs_price": p})
        out = out.reindex(out["spearman_vs_price"].abs().sort_values(ascending=False).index)
        log("Correlation of each numeric column with the target (sorted by |Spearman|):")
        log(out.round(3).to_string())

    # Redundant feature pairs
    c = data.corr().abs()
    pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if c.loc[a, b] > 0.9]
    log("\nHighly correlated feature pairs (|r|>0.9):")
    log("\n".join(f"  {a} <-> {b}: {v:.3f}" for a, b, v in pairs) if pairs else "  none")


# =============================================================================
# 11. LEAKAGE DETECTION
# =============================================================================
def leakage_detection(df, numeric_cols, y, target):
    """
    DATA LEAKAGE = a feature that contains (or is computed from) the target,
    so the model 'cheats' and the score is meaningless on new listings.
    The project statement says columns like unit_price / total_price must be
    removed. Three checks here:
      A. column NAME suggests it is derived from price
      B. correlation with price is suspiciously near 1
      C. column ~ price / (another column)   e.g. unit_price = price / area
    """
    section("11. LEAKAGE DETECTION")
    if y is None:
        log("No target; skipped.")
        return []
    suspects = {}

    # A) name-based hints
    for c in df.columns:
        if c == target:
            continue
        name = str(c).lower()
        if any(h in name for h in LEAKAGE_NAME_HINTS):
            suspects.setdefault(c, []).append("name suggests it is derived from price")

    # B) near-perfect rank correlation with price
    for c in numeric_cols:
        if c == target or df[c].nunique() < 3:
            continue
        r = pd.concat([df[c], y], axis=1).dropna().corr(method="spearman").iloc[0, 1]
        if abs(r) > 0.95:
            suspects.setdefault(c, []).append(f"Spearman correlation with price = {r:.3f}")

    # C) ratio test: is price / col_a  ~=  col_c  ?  (sub-sampled for speed)
    sample = subsample(pd.concat([df[numeric_cols], y.rename("__y__")], axis=1), 5000).dropna()
    for a in numeric_cols:
        if a == target or (sample[a] <= 0).all():
            continue
        ratio = sample["__y__"] / sample[a].replace(0, np.nan)
        for c in numeric_cols:
            if c in (a, target):
                continue
            denom = sample[c].replace(0, np.nan)
            rel_err = ((ratio - denom).abs() / denom.abs()).dropna()
            if len(rel_err) > 100 and rel_err.median() < 0.02:
                suspects.setdefault(c, []).append(f"~= price / {a} (median rel. error {rel_err.median():.3%})")

    if suspects:
        log("SUSPECTED LEAKAGE COLUMNS (review manually, then add to LEAKAGE_COLUMNS in src/data_prep.py):")
        for c, reasons in suspects.items():
            log(f"  * {c}")
            for r in reasons:
                log(f"      - {r}")
    else:
        log("No obvious leakage found by these checks. Still read the column list by eye!")
    log("\nAlso consider dropping identifiers / contact info (owner, shop, phone, raw address text, URLs):")
    log("they identify a seller or a single ad rather than describe the property.")
    return list(suspects)


# =============================================================================
# 12. FEATURE vs TARGET
# =============================================================================
def feature_vs_target(df, numeric_cols, cat_cols, y, target, out_dir):
    """
    Plot how price changes with the most informative features.
    * Numeric: scatter of feature vs log(price) for the 6 columns most
      correlated with price (log scale on y makes the pattern visible).
    * Categorical: boxplot of log(price) per category (top 10 categories) for
      low-cardinality columns. Big differences between boxes = useful feature.
    """
    section("12. FEATURES vs TARGET")
    if y is None:
        log("No target; skipped.")
        return
    ok = y > 0
    logy = np.log(y.where(ok))

    cands = []
    for c in numeric_cols:
        if c == target or df[c].nunique() < 5:
            continue
        r = pd.concat([df[c], logy], axis=1).dropna().corr(method="spearman").iloc[0, 1]
        cands.append((c, abs(r)))
    top = [c for c, _ in sorted(cands, key=lambda t: -t[1])[:6]]
    if top:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        data = subsample(pd.concat([df[top], logy.rename("__ly__")], axis=1).dropna())
        for ax, c in zip(axes.ravel(), top):
            ax.scatter(data[c], data["__ly__"], s=3, alpha=0.3)
            ax.set_xlabel(str(c))
            ax.set_ylabel("log(price)")
        for ax in axes.ravel()[len(top):]:
            ax.axis("off")
        fig.suptitle("Top numeric features vs log(price)")
        save_fig(fig, "12_numeric_vs_target.png", out_dir)

    shown = 0
    for c in cat_cols:
        n = df[c].nunique()
        if not 2 <= n <= 60 or shown >= 6:
            continue
        top_vals = df[c].value_counts().head(10).index
        groups = [logy[df[c] == v].dropna().values for v in top_vals]
        keep = [(str(v), g) for v, g in zip(top_vals, groups) if len(g) > 5]
        if len(keep) < 2:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.boxplot([g for _, g in keep], showfliers=False)
        ax.set_xticklabels([v for v, _ in keep])  # works on every matplotlib version
        ax.set_title(f"log(price) by {c}")
        ax.tick_params(axis="x", rotation=45)
        safe = re.sub(r"[^\w]+", "_", str(c))
        save_fig(fig, f"12_box_{safe}.png", out_dir)
        shown += 1


# =============================================================================
# 13. GEOGRAPHIC CHECK
# =============================================================================
def geo_check(df, y, out_dir):
    """
    If lat/lng columns exist: count missing / impossible coordinates (outside
    Azerbaijan, or 0,0) and draw a map coloured by log(price). Location is
    usually the strongest price driver, so bad coordinates matter.
    """
    section("13. GEOGRAPHIC CHECK")
    lat_c = next((c for c in df.columns if str(c).lower() in ("lat", "latitude")), None)
    lng_c = next((c for c in df.columns if str(c).lower() in ("lng", "lon", "long", "longitude")), None)
    if lat_c is None or lng_c is None:
        log("No lat/lng columns found; skipped.")
        return
    lat, lng = pd.to_numeric(df[lat_c], errors="coerce"), pd.to_numeric(df[lng_c], errors="coerce")
    log(f"Missing coordinates: {(lat.isna() | lng.isna()).sum():,}")
    log(f"Exactly (0, 0): {((lat == 0) & (lng == 0)).sum():,}")
    outside = ~lat.between(*AZ_LAT_RANGE) | ~lng.between(*AZ_LNG_RANGE)
    outside &= lat.notna() & lng.notna()
    log(f"Outside Azerbaijan bounding box: {outside.sum():,}  -> set to NaN / drop")

    valid = lat.between(*AZ_LAT_RANGE) & lng.between(*AZ_LNG_RANGE)
    if valid.sum() == 0:
        log("No valid coordinates to plot.")
        return
    d = pd.DataFrame({"lat": lat[valid], "lng": lng[valid]})
    if y is not None:
        d["c"] = np.log(y[valid].where(y[valid] > 0))
    d = subsample(d.dropna())
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(d["lng"], d["lat"], c=d["c"] if "c" in d else None, s=4, cmap="viridis", alpha=0.6)
    if "c" in d:
        fig.colorbar(sc, label="log(price)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Listings by location (colour = log price)")
    save_fig(fig, "13_geo_map.png", out_dir)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Full EDA for the bina.az sale dataset")
    parser.add_argument("--data", default=DATA_PATH, help="path to the CSV")
    parser.add_argument("--out", default=OUT_DIR, help="output folder for report + figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.data)
    overview(df)
    missing_values(df, args.out)
    duplicates(df)
    numeric_cols, cat_cols = column_audit(df)
    numeric_summary(df, numeric_cols)
    categorical_summary(df, cat_cols, args.out)

    target = find_target(df)
    y = target_analysis(df, target, args.out)
    outlier_scan(df, numeric_cols, args.out)
    correlations(df, numeric_cols, y, target, args.out)
    leakage = leakage_detection(df, numeric_cols, y, target)
    feature_vs_target(df, numeric_cols, cat_cols, y, target, args.out)
    geo_check(df, y, args.out)

    section("DONE")
    log(f"Suspected leakage columns: {leakage}")
    log("Next: use these findings to write clean() / LEAKAGE_COLUMNS in src/data_prep.py.")

    report_path = os.path.join(args.out, "eda_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_LINES))
    print(f"\nFull text report saved to {report_path}")


if __name__ == "__main__":
    main()