"""
Data preprocessing for the bina.az real-estate price prediction project.

Pipeline:
1. Load raw CSV.
2. Extract item_id from estate_rel_url.
3. Deduplicate repeated listings before splitting.
4. Parse text-based numeric features.
5. Keep the recommended residential property categories.
6. Apply conservative domain sanity filters.
7. Fix invalid coordinates.
8. Convert boolean/yes-no fields.
9. Split into train / validation / test.
10. Calculate the premium threshold from TRAIN ONLY.
11. Fit imputation, encoding and scaling on TRAIN ONLY.
12. Save cleaned splits and model-ready arrays.

Usage:
    python src/data_prep.py --input data/raw.csv --output-dir data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


RANDOM_STATE = 42

TEST_SIZE = 0.15
VALID_SIZE = 0.15

# Winsorization (train-fitted): clip legitimate-but-extreme values instead of
# deleting rows. Percentiles are computed on the TRAIN split only and applied
# to train / validation / test.
WINSORIZE_COLUMNS = ("area_m2", "land_m2")
WINSORIZE_LOWER_Q = 0.005
WINSORIZE_UPPER_Q = 0.995

# Categorical columns with more than this many distinct train values are
# frequency-encoded instead of one-hot encoded (avoids a sparse feature blow-up).
CARDINALITY_MAX_ONEHOT = 150

# Numeric feature pairs more correlated than this are redundant; we keep the
# one more correlated with the target.
CORRELATION_THRESHOLD = 0.90

# Implausible price density. The 99.99th percentile of price / area on the
# training data is about 20,000 AZN/m2, so anything above 50,000 is treated as
# a data-entry error (e.g. a 50 m2 flat listed at 60,000,000 AZN).
MAX_PRICE_PER_M2 = 50_000.0

# Columns that must never enter the model.
#   * location_text      : near-unique free text (2,284 train values).
#   * location_part_1..3 : derived from the same extra_info string and
#                          redundant with the canonical 'location' column.
FREE_TEXT_COLUMNS = {"location_text"}

REDUNDANT_LOCATION_COLUMNS = {
    "location_part_1",
    "location_part_2",
    "location_part_3",
}

AUTO_DROP_COLUMNS = (
    FREE_TEXT_COLUMNS | REDUNDANT_LOCATION_COLUMNS
)


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

# Recommended scope from the EDA report:
#   - Yeni tikili
#   - Köhnə tikili
#   - Həyət evi/Bağ evi
#
# Torpaq and Obyekt are left out of the main experiment.
# Ofis and Qaraj are also excluded because they have very few observations.
RESIDENTIAL_CATEGORIES = {
    "Yeni tikili",
    "Köhnə tikili",
    "Həyət evi/Bağ evi",
}


# ---------------------------------------------------------------------------
# Leakage / metadata columns
# ---------------------------------------------------------------------------

LEAKAGE_COLUMNS = {
    "unit_price",
    "total_price",
    "description",
}


METADATA_COLUMNS = {
    "rel_url",
    "updated",
    "city_when",
    "img_url",
    "views",
    "owner_name",
    "shop_name",
    "shop_title",
    "owner_title",
    "products_label",
    "vip",
    "featured",
    "address",
    "currency",
    "Binanın növü",
}


DUPLICATE_COLUMNS = {
    "id_x",
    "id_y",
    "estate_id",
    "estate_details_id_x",
    "estate_details_id_y",
    "estate_rel_url_x",
    "estate_rel_url_y",
    "datetime_scrape_x",
    "datetime_scrape_y",
    "day_x",
    "day_y",
    "hour_x",
    "hour_y",
    "currency_x",
    "currency_y",
    "repair",
    "bill_of_sale",
    "mortgage",
}


RAW_PARSED_COLUMNS = {
    "Sahə",
    "Mərtəbə",
    "Torpaq sahəsi",
    "Otaq sayı",
    "attributes",
    "extra_info",
    "Təmir",
    "Çıxarış",
    "İpoteka",
}


ALWAYS_DROP = {
    "estate_rel_url",
    "id",
    "item_id",
    "scrape_date",
    "datetime_scrape",
    "date_scrape",
}


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series

    tier_train: pd.Series
    tier_valid: pd.Series
    tier_test: pd.Series


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def num(series: pd.Series, pattern: str) -> pd.Series:
    """
    Extract the first captured number from a text series.

    Example:
        '145 m²' -> 145.0
        '1.3 sot' -> 1.3
    """

    extracted = series.astype("string").str.extract(
        pattern,
        expand=False,
    )

    if isinstance(extracted, pd.DataFrame):
        extracted = extracted.iloc[:, 0]

    extracted = (
        extracted
        .astype("string")
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(
        extracted,
        errors="coerce",
    )


def item_id_sha256(series: pd.Series) -> str:
    """
    Stable fingerprint of a set of listing ids.

    Used by the validator to confirm each split is reproducible.
    """

    joined = "\n".join(
        sorted(series.astype(str))
    )

    return hashlib.sha256(
        joined.encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert important text-based numerical columns into numeric features.
    """

    df = df.copy()

    # Area
    if "Sahə" in df.columns:
        df["area_m2"] = num(
            df["Sahə"],
            r"([\d.,]+)\s*m²",
        )

    # Floor
    if "Mərtəbə" in df.columns:

        floor_text = df["Mərtəbə"].astype("string")

        df["floor"] = pd.to_numeric(
            floor_text.str.extract(
                r"^\s*(\d+)",
                expand=False,
            ),
            errors="coerce",
        )

        df["n_floors"] = pd.to_numeric(
            floor_text.str.extract(
                r"/\s*(\d+)",
                expand=False,
            ),
            errors="coerce",
        )

    # Land area
    #
    # 1 sot = 100 m²
    if "Torpaq sahəsi" in df.columns:

        df["land_m2"] = (
            num(
                df["Torpaq sahəsi"],
                r"([\d.,]+)\s*sot",
            )
            * 100.0
        )

    # Rooms
    #
    # First use Otaq sayı.
    # If missing, try attributes.
    rooms_from_attributes = (
        num(
            df["attributes"],
            r"(\d+(?:[.,]\d+)?)\s*otaql",
        )
        if "attributes" in df.columns
        else pd.Series(
            np.nan,
            index=df.index,
        )
    )

    if "Otaq sayı" in df.columns:

        rooms = pd.to_numeric(
            df["Otaq sayı"],
            errors="coerce",
        )

        df["rooms"] = rooms.fillna(
            rooms_from_attributes
        )

    else:
        df["rooms"] = rooms_from_attributes

    return df


# ---------------------------------------------------------------------------
# Listing ID / deduplication
# ---------------------------------------------------------------------------

def extract_item_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the numeric listing ID from estate_rel_url.

    Example:
        /items/4645538 -> 4645538
    """

    df = df.copy()

    url_column = None

    for column in (
        "estate_rel_url",
        "estate_rel_url_x",
        "estate_rel_url_y",
    ):

        if column in df.columns:
            url_column = column
            break

    if url_column is None:
        raise ValueError(
            "No estate_rel_url column found. "
            "It is required for listing-level deduplication."
        )

    df["item_id"] = num(
        df[url_column],
        r"/items/(\d+)",
    )

    missing = int(
        df["item_id"].isna().sum()
    )

    if missing > 0:
        raise ValueError(
            f"{missing} rows have no extractable item_id. "
            "Inspect estate_rel_url before continuing."
        )

    return df


def get_scrape_datetime(
    df: pd.DataFrame,
) -> pd.Series | None:
    """
    Try to find a usable scrape timestamp.

    Used only to determine which repeated listing is the latest.
    """

    candidates = [
        "datetime_scrape",
        "datetime_scrape_x",
        "datetime_scrape_y",
    ]

    for column in candidates:

        if column in df.columns:

            parsed = pd.to_datetime(
                df[column],
                errors="coerce",
            )

            if parsed.notna().any():
                return parsed

    combinations = [
        ("date_scrape", "hour"),
        ("scrape_date", "hour"),
        ("day", "hour"),
        ("day_x", "hour_x"),
        ("day_y", "hour_y"),
    ]

    for date_column, hour_column in combinations:

        if date_column not in df.columns:
            continue

        date = pd.to_datetime(
            df[date_column],
            errors="coerce",
        )

        if hour_column in df.columns:

            hours = pd.to_numeric(
                df[hour_column],
                errors="coerce",
            ).fillna(0)

            return date + pd.to_timedelta(
                hours,
                unit="h",
            )

        return date

    return None


def deduplicate_listings(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Keep one row per item_id.

    If scrape time exists:
        keep the latest scrape.

    Otherwise:
        keep the last occurrence.
    """

    df = df.copy()

    before = len(df)

    scrape_datetime = get_scrape_datetime(df)

    if scrape_datetime is not None:

        df["_scrape_datetime_for_dedup"] = (
            scrape_datetime
        )

        df = (
            df
            .sort_values(
                [
                    "item_id",
                    "_scrape_datetime_for_dedup",
                ]
            )
            .drop_duplicates(
                "item_id",
                keep="last",
            )
            .drop(
                columns="_scrape_datetime_for_dedup"
            )
        )

    else:

        df = df.drop_duplicates(
            "item_id",
            keep="last",
        )

    removed = before - len(df)

    return (
        df.reset_index(drop=True),
        removed,
    )


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def clean_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert categorical yes/no and flag columns.

    Important:
        'yoxdur' = 0
        NaN = unknown
    """

    df = df.copy()

    if "Təmir" in df.columns:

        df["has_repair"] = df["Təmir"].map(
            {
                "var": 1,
                "yoxdur": 0,
            }
        )

    if "Çıxarış" in df.columns:

        df["has_extract"] = df["Çıxarış"].map(
            {
                "var": 1,
                "yoxdur": 0,
            }
        )

    if "İpoteka" in df.columns:

        df["has_mortgage"] = (
            df["İpoteka"]
            .notna()
            .astype(int)
        )

    # These are advertising flags rather than property attributes.
    # We create them here, but they are removed from the model below.
    if "vip" in df.columns:

        df["is_vip"] = (
            df["vip"]
            .notna()
            .astype(int)
        )

    if "featured" in df.columns:

        df["is_featured"] = (
            df["featured"]
            .notna()
            .astype(int)
        )

    return df


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

def split_location(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Split extra_info into separate location components.

    Example:
        Xəzər r.* Mərdəkan q.

    becomes approximately:
        location_part_1 = Xəzər r.
        location_part_2 = Mərdəkan q.
    """

    df = df.copy()

    if "extra_info" not in df.columns:
        return df

    parts = (
        df["extra_info"]
        .fillna("")
        .astype("string")
        .str.split(
            r"\s*\*\s*",
            expand=True,
        )
    )

    names = [
        "location_part_1",
        "location_part_2",
        "location_part_3",
    ]

    for i, name in enumerate(names):

        if i < parts.shape[1]:

            df[name] = (
                parts.iloc[:, i]
                .replace("", pd.NA)
            )

        else:

            df[name] = pd.NA

    df["location_text"] = (
        df["extra_info"]
        .astype("string")
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    return df


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def fix_coordinates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Set clearly invalid coordinates to NaN.

    Do NOT use IQR here.

    The EDA report found a small number of coordinates outside the
    Azerbaijan geographic range.
    """

    df = df.copy()

    if (
        "lat" not in df.columns
        or "lng" not in df.columns
    ):
        return df, 0

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce",
    )

    df["lng"] = pd.to_numeric(
        df["lng"],
        errors="coerce",
    )

    valid = (
        df["lat"].between(
            38.0,
            42.0,
        )
        &
        df["lng"].between(
            44.5,
            51.0,
        )
    )

    invalid = (
        df["lat"].notna()
        &
        df["lng"].notna()
        &
        ~valid
    )

    changed = int(
        invalid.sum()
    )

    df.loc[
        invalid,
        ["lat", "lng"],
    ] = np.nan

    return df, changed


# ---------------------------------------------------------------------------
# Property scope
# ---------------------------------------------------------------------------

def apply_scope(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Keep only the three recommended residential categories.
    """

    if "Kateqoriya" not in df.columns:

        raise ValueError(
            "Required column 'Kateqoriya' is missing."
        )

    before = len(df)

    counts_before = (
        df["Kateqoriya"]
        .value_counts(
            dropna=False,
        )
        .to_dict()
    )

    keep = df["Kateqoriya"].isin(
        RESIDENTIAL_CATEGORIES
    )

    df = df.loc[keep].copy()

    counts_after = (
        df["Kateqoriya"]
        .value_counts(
            dropna=False,
        )
        .to_dict()
    )

    report = {
        "rows_before_scope": before,
        "rows_after_scope": len(df),
        "rows_removed_by_scope": (
            before - len(df)
        ),
        "category_counts_before": {
            str(k): int(v)
            for k, v in counts_before.items()
        },
        "category_counts_after": {
            str(k): int(v)
            for k, v in counts_after.items()
        },
    }

    return df, report


# ---------------------------------------------------------------------------
# Domain sanity filters
# ---------------------------------------------------------------------------

def apply_domain_filters(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply conservative domain sanity rules.

    Important:
        We do NOT use an IQR rule for price.

    The report explicitly recommends:
        price >= 5,000 AZN

    Price is otherwise left on its natural scale (the heavy tail is handled by
    the log target). The only upper price rule is a price-density sanity check
    (MAX_PRICE_PER_M2) that removes obvious data-entry errors without trimming
    the legitimate luxury segment.
    """

    df = df.copy()

    removed = {}

    def remove(
        name: str,
        mask: pd.Series,
    ) -> None:

        nonlocal df

        mask = mask.fillna(False)

        removed[name] = int(
            mask.sum()
        )

        df = df.loc[
            ~mask
        ].copy()

    if "price" not in df.columns:

        raise ValueError(
            "Required target column 'price' is missing."
        )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce",
    )

    # Target must exist and be positive.
    remove(
        "price_missing_or_nonpositive",
        df["price"].isna()
        | (df["price"] <= 0),
    )

    # Explicit rule from the EDA report.
    remove(
        "price_below_5000_AZN",
        df["price"] < 5000,
    )

    # Area
    if "area_m2" in df.columns:

        remove(
            "area_nonpositive",
            df["area_m2"].notna()
            & (df["area_m2"] <= 0),
        )

        remove(
            "area_above_5000_m2",
            df["area_m2"].notna()
            & (df["area_m2"] > 5000),
        )

        # Implausible price density (see MAX_PRICE_PER_M2).
        price_per_m2 = (
            df["price"]
            / df["area_m2"].replace(0, np.nan)
        )

        remove(
            "price_per_m2_above_50000",
            price_per_m2 > MAX_PRICE_PER_M2,
        )

    # Land area
    if "land_m2" in df.columns:

        remove(
            "land_nonpositive",
            df["land_m2"].notna()
            & (df["land_m2"] <= 0),
        )

        remove(
            "land_above_100000_m2",
            df["land_m2"].notna()
            & (df["land_m2"] > 100000),
        )

    # Rooms
    if "rooms" in df.columns:

        remove(
            "rooms_nonpositive",
            df["rooms"].notna()
            & (df["rooms"] <= 0),
        )

        remove(
            "rooms_above_30",
            df["rooms"].notna()
            & (df["rooms"] > 30),
        )

    # Floor
    if "floor" in df.columns:

        remove(
            "floor_nonpositive",
            df["floor"].notna()
            & (df["floor"] <= 0),
        )

    # Number of floors
    if "n_floors" in df.columns:

        remove(
            "n_floors_nonpositive",
            df["n_floors"].notna()
            & (df["n_floors"] <= 0),
        )

    # Floor cannot be above total number of floors.
    if (
        "floor" in df.columns
        and "n_floors" in df.columns
    ):

        remove(
            "floor_above_n_floors",
            df["floor"].notna()
            & df["n_floors"].notna()
            & (
                df["floor"]
                > df["n_floors"]
            ),
        )

    removed[
        "total_removed_by_domain_filters"
    ] = int(
        sum(removed.values())
    )

    return (
        df.reset_index(drop=True),
        removed,
    )


# ---------------------------------------------------------------------------
# Complete cleaning
# ---------------------------------------------------------------------------

def build_clean_dataset(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Run all row-level cleaning before train/validation/test splitting.
    """

    df = raw.copy()

    report = {
        "raw_rows": len(df),
        "raw_columns": len(df.columns),
    }

    # ---------------------------------------------------------------
    # 1. Extract item ID
    # ---------------------------------------------------------------

    df = extract_item_id(df)

    # ---------------------------------------------------------------
    # 2. Deduplicate BEFORE splitting
    # ---------------------------------------------------------------

    df, duplicate_removed = (
        deduplicate_listings(df)
    )

    report[
        "rows_removed_by_item_deduplication"
    ] = duplicate_removed

    report[
        "rows_after_deduplication"
    ] = len(df)

    # ---------------------------------------------------------------
    # 3. Parse numerical values
    # ---------------------------------------------------------------

    df = parse_numeric_columns(df)

    # ---------------------------------------------------------------
    # 4. Convert flags
    # ---------------------------------------------------------------

    df = clean_flags(df)

    # ---------------------------------------------------------------
    # 5. Parse location
    # ---------------------------------------------------------------

    df = split_location(df)

    # ---------------------------------------------------------------
    # 6. Property scope
    # ---------------------------------------------------------------

    df, scope_report = apply_scope(df)

    report["scope"] = scope_report

    # ---------------------------------------------------------------
    # 7. Coordinates
    # ---------------------------------------------------------------

    df, coordinate_fixes = (
        fix_coordinates(df)
    )

    report[
        "invalid_coordinates_set_to_nan"
    ] = coordinate_fixes

    # ---------------------------------------------------------------
    # 8. Domain sanity filters
    # ---------------------------------------------------------------

    df, filter_report = (
        apply_domain_filters(df)
    )

    report[
        "domain_filters"
    ] = filter_report

    report[
        "rows_after_cleaning"
    ] = len(df)

    # ---------------------------------------------------------------
    # 9. Log target
    # ---------------------------------------------------------------

    df["log_price"] = np.log(
        df["price"]
    )

    return df, report


# ---------------------------------------------------------------------------
# Model feature selection
# ---------------------------------------------------------------------------

def choose_model_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Determine numeric and categorical model features.

    Target/leakage/metadata/raw columns are excluded.
    """

    excluded = (
        LEAKAGE_COLUMNS
        | METADATA_COLUMNS
        | DUPLICATE_COLUMNS
        | RAW_PARSED_COLUMNS
        | ALWAYS_DROP
        | AUTO_DROP_COLUMNS
        | {
            "price",
            "log_price",
        }
    )

    # Do not accidentally retain merge duplicates.
    excluded |= {
        column
        for column in df.columns
        if column.endswith("_x")
        or column.endswith("_y")
    }

    numeric_columns = []
    categorical_columns = []

    for column in df.columns:

        if column in excluded:
            continue

        # item_id is ONLY for grouping/splitting.
        if column == "item_id":
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            numeric_columns.append(
                column
            )

        else:

            categorical_columns.append(
                column
            )

    return (
        numeric_columns,
        categorical_columns,
    )


# ---------------------------------------------------------------------------
# Winsorization (train-fitted)
# ---------------------------------------------------------------------------

def fit_winsorization(
    X_train: pd.DataFrame,
    columns: Iterable[str] = WINSORIZE_COLUMNS,
    lower_q: float = WINSORIZE_LOWER_Q,
    upper_q: float = WINSORIZE_UPPER_Q,
) -> dict:
    """
    Compute winsorization fences from the TRAIN split only.

    Returns a mapping column -> {"low": float, "high": float}.
    """

    fences: dict = {}

    for column in columns:

        if column not in X_train.columns:
            continue

        values = X_train[column].dropna()

        if values.empty:
            continue

        fences[column] = {
            "low": float(values.quantile(lower_q)),
            "high": float(values.quantile(upper_q)),
        }

    return fences


def apply_winsorization(
    df: pd.DataFrame,
    fences: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Clip values to the train-derived fences. Rows are never removed.

    Returns the clipped frame and a per-column count of clipped values.
    """

    df = df.copy()

    report: dict = {}

    for column, bounds in fences.items():

        if column not in df.columns:
            continue

        low, high = bounds["low"], bounds["high"]

        clipped = int(
            ((df[column] < low) | (df[column] > high)).sum()
        )

        df[column] = df[column].clip(lower=low, upper=high)

        report[column] = {
            "low": low,
            "high": high,
            "values_clipped": clipped,
        }

    return df, report


# ---------------------------------------------------------------------------
# Feature selection (train-fitted)
# ---------------------------------------------------------------------------

def select_numeric_features(
    X_train: pd.DataFrame,
    numeric_columns: Iterable[str],
    y_train: pd.Series,
) -> tuple[list[str], dict]:
    """
    Drop numeric features that are constant or redundant on TRAIN.

    Redundancy = absolute Pearson correlation above CORRELATION_THRESHOLD.
    Of a redundant pair we keep the feature more correlated with the target.
    """

    kept = list(numeric_columns)

    decisions: dict = {}

    # Constant columns carry no information.
    for column in list(kept):

        if X_train[column].nunique(dropna=True) <= 1:

            kept.remove(column)

            decisions[column] = "dropped: constant on train"

    if len(kept) < 2:
        return kept, decisions

    correlation = X_train[kept].corr().abs()

    target_correlation = (
        X_train[kept].corrwith(y_train).abs()
    )

    dropped: set[str] = set()

    for i, a in enumerate(kept):

        for b in kept[i + 1:]:

            if a in dropped or b in dropped:
                continue

            r = correlation.loc[a, b]

            if pd.isna(r) or r <= CORRELATION_THRESHOLD:
                continue

            a_target = float(target_correlation.get(a, 0.0))
            b_target = float(target_correlation.get(b, 0.0))

            drop, keep = (
                (a, b) if a_target < b_target else (b, a)
            )

            dropped.add(drop)

            decisions[drop] = (
                f"dropped: |corr|={r:.3f} with '{keep}' "
                f"(target |corr| {target_correlation.get(drop, 0.0):.3f} "
                f"< {target_correlation.get(keep, 0.0):.3f})"
            )

    kept = [c for c in kept if c not in dropped]

    return kept, decisions


def split_categorical_features(
    X_train: pd.DataFrame,
    categorical_columns: Iterable[str],
) -> tuple[list[str], list[str], dict]:
    """
    Split categorical columns by train cardinality.

    Low cardinality -> one-hot encoding.
    High cardinality -> frequency encoding.
    """

    low_cardinality: list[str] = []
    high_cardinality: list[str] = []

    decisions: dict = {}

    for column in categorical_columns:

        n_unique = int(X_train[column].nunique(dropna=True))

        if n_unique > CARDINALITY_MAX_ONEHOT:
            high_cardinality.append(column)
            decisions[column] = (
                f"frequency-encoded ({n_unique} train values)"
            )
        else:
            low_cardinality.append(column)
            decisions[column] = (
                f"one-hot encoded ({n_unique} train values)"
            )

    return low_cardinality, high_cardinality, decisions


class FrequencyEncoder(
    BaseEstimator,
    TransformerMixin,
):
    """
    Replace each category with its TRAIN frequency.

    Unseen categories at transform time map to 0.
    """

    def fit(self, X, y=None):

        X = pd.DataFrame(X)

        self.columns_ = list(X.columns)

        self.frequencies_ = {
            column: X[column].value_counts(
                normalize=True,
                dropna=True,
            )
            for column in self.columns_
        }

        return self

    def transform(self, X):

        X = pd.DataFrame(X)

        encoded = np.zeros(
            (len(X), len(self.columns_)),
            dtype=float,
        )

        for j, column in enumerate(self.columns_):

            encoded[:, j] = (
                X[column]
                .map(self.frequencies_[column])
                .to_numpy(dtype=float)
            )

        encoded = np.nan_to_num(encoded, nan=0.0)

        return encoded

    def get_feature_names_out(self, input_features=None):

        return np.array(
            [f"freq_{c}" for c in self.columns_],
            dtype=object,
        )


# ---------------------------------------------------------------------------
# Train-only preprocessing
# ---------------------------------------------------------------------------

def make_preprocessor(
    numeric_columns: Iterable[str],
    low_cardinality_columns: Iterable[str],
    high_cardinality_columns: Iterable[str],
) -> ColumnTransformer:
    """
    Create the preprocessing transformer.

    It must be fitted on TRAIN ONLY.
    """

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                RobustScaler(),
            ),
        ]
    )

    low_cardinality_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                ),
            ),
        ]
    )

    high_cardinality_pipeline = Pipeline(
        [
            (
                "frequency",
                FrequencyEncoder(),
            ),
        ]
    )

    transformers = [
        (
            "numeric",
            numeric_pipeline,
            list(numeric_columns),
        ),
    ]

    if len(list(low_cardinality_columns)):
        transformers.append(
            (
                "lowcard",
                low_cardinality_pipeline,
                list(low_cardinality_columns),
            )
        )

    if len(list(high_cardinality_columns)):
        transformers.append(
            (
                "highcard",
                high_cardinality_pipeline,
                list(high_cardinality_columns),
            )
        )

    return ColumnTransformer(
        transformers,
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
) -> tuple[SplitData, float]:
    """
    Split the cleaned dataset.

    The tier threshold is calculated from TRAIN ONLY.

    Split proportions:
        train = 70%
        validation = 15%
        test = 15%
    """

    if "item_id" not in df.columns:

        raise ValueError(
            "item_id is required for leakage-safe splitting."
        )

    indices = np.arange(
        len(df)
    )

    # ---------------------------------------------------------------
    # First create the training set.
    # ---------------------------------------------------------------

    train_indices, temp_indices = (
        train_test_split(
            indices,
            test_size=(
                TEST_SIZE
                + VALID_SIZE
            ),
            random_state=RANDOM_STATE,
        )
    )

    # ---------------------------------------------------------------
    # Premium threshold comes from TRAIN ONLY.
    # ---------------------------------------------------------------

    tier_threshold = float(
        df.iloc[
            train_indices
        ]["price"].median()
    )

    tier = (
        df["price"]
        >= tier_threshold
    ).astype(int)

    # ---------------------------------------------------------------
    # Split temporary set into validation/test.
    # Stratification uses the train-derived threshold.
    # ---------------------------------------------------------------

    validation_fraction = (
        VALID_SIZE
        / (
            TEST_SIZE
            + VALID_SIZE
        )
    )

    valid_indices, test_indices = (
        train_test_split(
            temp_indices,
            test_size=(
                1
                - validation_fraction
            ),
            random_state=RANDOM_STATE,
            stratify=tier.iloc[
                temp_indices
            ],
        )
    )

    split = SplitData(
        X_train=df.iloc[
            train_indices
        ].copy(),

        X_valid=df.iloc[
            valid_indices
        ].copy(),

        X_test=df.iloc[
            test_indices
        ].copy(),

        y_train=df.iloc[
            train_indices
        ]["log_price"].copy(),

        y_valid=df.iloc[
            valid_indices
        ]["log_price"].copy(),

        y_test=df.iloc[
            test_indices
        ]["log_price"].copy(),

        tier_train=tier.iloc[
            train_indices
        ].copy(),

        tier_valid=tier.iloc[
            valid_indices
        ].copy(),

        tier_test=tier.iloc[
            test_indices
        ].copy(),
    )

    return (
        split,
        tier_threshold,
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_split_tables(
    split: SplitData,
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split.X_train.to_csv(
        output_dir / "train_clean.csv",
        index=False,
    )

    split.X_valid.to_csv(
        output_dir / "valid_clean.csv",
        index=False,
    )

    split.X_test.to_csv(
        output_dir / "test_clean.csv",
        index=False,
    )


def save_metadata(
    metadata: dict,
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir
        / "preprocessing_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    input_path: str | Path,
    output_dir: str | Path,
) -> None:

    input_path = Path(
        input_path
    )

    output_dir = Path(
        output_dir
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input CSV does not exist: "
            f"{input_path}"
        )

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    raw = pd.read_csv(
        input_path,
        encoding="utf-8",
    )

    print(
        f"Loaded {len(raw):,} rows "
        f"and {len(raw.columns):,} columns."
    )

    # ---------------------------------------------------------------
    # Clean
    # ---------------------------------------------------------------

    clean, clean_report = (
        build_clean_dataset(raw)
    )

    # ---------------------------------------------------------------
    # Split
    # ---------------------------------------------------------------

    split, tier_threshold = (
        split_data(clean)
    )

    # ---------------------------------------------------------------
    # Winsorize features (train-fitted, applied to all splits)
    # ---------------------------------------------------------------

    fences = fit_winsorization(split.X_train)

    winsor_report: dict = {}

    (
        split.X_train,
        winsor_report["train"],
    ) = apply_winsorization(split.X_train, fences)

    (
        split.X_valid,
        winsor_report["validation"],
    ) = apply_winsorization(split.X_valid, fences)

    (
        split.X_test,
        winsor_report["test"],
    ) = apply_winsorization(split.X_test, fences)

    # ---------------------------------------------------------------
    # Select features (train-fitted)
    # ---------------------------------------------------------------

    numeric_columns, categorical_columns = (
        choose_model_columns(clean)
    )

    (
        candidate_numeric_columns
    ) = list(numeric_columns)

    numeric_columns, selection_report = (
        select_numeric_features(
            split.X_train,
            numeric_columns,
            split.y_train,
        )
    )

    (
        low_cardinality_columns,
        high_cardinality_columns,
        encoding_report,
    ) = split_categorical_features(
        split.X_train,
        categorical_columns,
    )

    print(
        f"Numeric features kept: "
        f"{len(numeric_columns)} "
        f"(of {len(candidate_numeric_columns)})"
    )

    print(
        f"Low-cardinality categorical: "
        f"{len(low_cardinality_columns)}"
    )

    print(
        f"High-cardinality categorical "
        f"(frequency-encoded): "
        f"{len(high_cardinality_columns)}"
    )

    # ---------------------------------------------------------------
    # Fit preprocessing ONLY on TRAIN
    # ---------------------------------------------------------------

    preprocessor = make_preprocessor(
        numeric_columns,
        low_cardinality_columns,
        high_cardinality_columns,
    )

    # Normalize pandas missing values before sklearn preprocessing.
    # sklearn's SimpleImputer expects np.nan rather than pandas pd.NA.
    for df in (split.X_train, split.X_valid, split.X_test):
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    preprocessor.fit(
        split.X_train
    )

    # ---------------------------------------------------------------
    # Transform all three sets
    # ---------------------------------------------------------------

    X_train = preprocessor.transform(
        split.X_train
    )

    X_valid = preprocessor.transform(
        split.X_valid
    )

    X_test = preprocessor.transform(
        split.X_test
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # ---------------------------------------------------------------
    # Output directory (flat layout)
    # ---------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Save model-ready arrays
    # ---------------------------------------------------------------

    np.save(
        output_dir / "X_train.npy",
        X_train,
    )

    np.save(
        output_dir / "X_valid.npy",
        X_valid,
    )

    np.save(
        output_dir / "X_test.npy",
        X_test,
    )

    np.save(
        output_dir / "y_train_log.npy",
        split.y_train.to_numpy(),
    )

    np.save(
        output_dir / "y_valid_log.npy",
        split.y_valid.to_numpy(),
    )

    np.save(
        output_dir / "y_test_log.npy",
        split.y_test.to_numpy(),
    )

    np.save(
        output_dir / "tier_train.npy",
        split.tier_train.to_numpy(),
    )

    np.save(
        output_dir / "tier_valid.npy",
        split.tier_valid.to_numpy(),
    )

    np.save(
        output_dir / "tier_test.npy",
        split.tier_test.to_numpy(),
    )

    # ---------------------------------------------------------------
    # Save feature names
    # ---------------------------------------------------------------

    pd.DataFrame(
        {
            "feature": feature_names
        }
    ).to_csv(
        output_dir
        / "feature_names.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Save cleaned, human-readable tables
    # ---------------------------------------------------------------

    save_split_tables(
        split,
        output_dir,
    )

    # ---------------------------------------------------------------
    # Save preprocessing report
    # ---------------------------------------------------------------

    metadata = {
        **clean_report,

        "final_rows": len(clean),

        "split_rows": {
            "train": len(
                split.X_train
            ),
            "validation": len(
                split.X_valid
            ),
            "test": len(
                split.X_test
            ),
        },

        "split_item_id_sha256": {
            "train": item_id_sha256(
                split.X_train["item_id"]
            ),
            "validation": item_id_sha256(
                split.X_valid["item_id"]
            ),
            "test": item_id_sha256(
                split.X_test["item_id"]
            ),
        },

        "target": "log_price",

        "original_target": "price",

        "tier_threshold_AZN": (
            tier_threshold
        ),

        "winsorization": {
            "method": (
                "percentile clip, train-fitted, "
                "applied to all splits"
            ),
            "lower_quantile": (
                WINSORIZE_LOWER_Q
            ),
            "upper_quantile": (
                WINSORIZE_UPPER_Q
            ),
            "fences": fences,
            "clipped_values": winsor_report,
        },

        "feature_selection": {
            "correlation_threshold": (
                CORRELATION_THRESHOLD
            ),
            "decisions": selection_report,
        },

        "encoding": {
            "cardinality_max_onehot": (
                CARDINALITY_MAX_ONEHOT
            ),
            "auto_dropped_columns": (
                sorted(AUTO_DROP_COLUMNS)
            ),
            "free_text_columns_dropped": (
                sorted(FREE_TEXT_COLUMNS)
            ),
            "redundant_location_columns_dropped": (
                sorted(REDUNDANT_LOCATION_COLUMNS)
            ),
            "decisions": encoding_report,
        },

        "numeric_features": (
            numeric_columns
        ),

        "low_cardinality_features": (
            low_cardinality_columns
        ),

        "high_cardinality_features": (
            high_cardinality_columns
        ),

        "categorical_features": (
            categorical_columns
        ),

        "n_processed_features": (
            len(feature_names)
        ),

        "random_state": (
            RANDOM_STATE
        ),

        "test_size": TEST_SIZE,

        "validation_size": (
            VALID_SIZE
        ),

        "leakage_columns_removed": (
            sorted(
                LEAKAGE_COLUMNS
            )
        ),
    }

    save_metadata(
        metadata,
        output_dir,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print()
    print(
        "Preprocessing complete."
    )

    print(
        f"Raw rows: "
        f"{clean_report['raw_rows']:,}"
    )

    print(
        f"After item deduplication: "
        f"{clean_report['rows_after_deduplication']:,}"
    )

    print(
        f"After cleaning: "
        f"{clean_report['rows_after_cleaning']:,}"
    )

    print(
        "Split: "
        f"train={len(split.X_train):,}, "
        f"valid={len(split.X_valid):,}, "
        f"test={len(split.X_test):,}"
    )

    print(
        f"Train tier threshold: "
        f"{tier_threshold:,.2f} AZN"
    )

    for column, bounds in fences.items():

        print(
            f"Winsorized {column}: "
            f"[{bounds['low']:,.2f}, {bounds['high']:,.2f}] "
            f"clipped "
            f"{winsor_report['train'][column]['values_clipped']:,} "
            f"(train)"
        )

    print(
        f"Features dropped as redundant: "
        f"{len(selection_report):,}"
    )

    print(
        f"Processed features: "
        f"{len(feature_names):,}"
    )

    print(
        f"Saved to: "
        f"{output_dir.resolve()}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Clean and preprocess "
            "bina.az real-estate data."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw UTF-8 CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help=(
            "Directory for cleaned "
            "tables and arrays."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run(
        args.input,
        args.output_dir,
    )