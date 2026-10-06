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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

TEST_SIZE = 0.15
VALID_SIZE = 0.15


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

    It does not provide a trustworthy universal price/m²
    upper bound, so we do not invent one here.
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
# Train-only preprocessing
# ---------------------------------------------------------------------------

def make_preprocessor(
    numeric_columns: Iterable[str],
    categorical_columns: Iterable[str],
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
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
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

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                list(numeric_columns),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(categorical_columns),
            ),
        ],
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
    # Select features
    # ---------------------------------------------------------------

    numeric_columns, categorical_columns = (
        choose_model_columns(clean)
    )

    print(
        f"Numeric features: "
        f"{len(numeric_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    # ---------------------------------------------------------------
    # Fit preprocessing ONLY on TRAIN
    # ---------------------------------------------------------------

    preprocessor = make_preprocessor(
        numeric_columns,
        categorical_columns,
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
    # Output directories
    # ---------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_dir = (
        output_dir / "processed"
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Save model-ready arrays
    # ---------------------------------------------------------------

    np.save(
        processed_dir / "X_train.npy",
        X_train,
    )

    np.save(
        processed_dir / "X_valid.npy",
        X_valid,
    )

    np.save(
        processed_dir / "X_test.npy",
        X_test,
    )

    np.save(
        processed_dir / "y_train_log.npy",
        split.y_train.to_numpy(),
    )

    np.save(
        processed_dir / "y_valid_log.npy",
        split.y_valid.to_numpy(),
    )

    np.save(
        processed_dir / "y_test_log.npy",
        split.y_test.to_numpy(),
    )

    # ---------------------------------------------------------------
    # Save feature names
    # ---------------------------------------------------------------

    pd.DataFrame(
        {
            "feature": feature_names
        }
    ).to_csv(
        processed_dir
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

        "target": "log_price",

        "original_target": "price",

        "tier_threshold_AZN": (
            tier_threshold
        ),

        "numeric_features": (
            numeric_columns
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