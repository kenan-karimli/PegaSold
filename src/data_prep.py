"""
Turn the raw bina.az CSV into model-ready arrays.

Run it with:

    python -m src.data_prep --input data/bina_az_sale.csv --output-dir data/processed

The pipeline runs top to bottom, one step per function, in the order the
functions appear in this file:

    1. load              read the raw CSV
    2. parse             pull numbers out of text, turn flags into 0/1
    3. deduplicate       one row per property
    4. apply_scope       keep only the rows that belong to the task
    5. apply_filters     drop impossible values
    6. split             70 / 15 / 15 into train / validation / test
    7. fit_preprocessor  learn encoding rules FROM THE TRAINING SET ONLY
    8. transform         apply those rules to each split
    9. save              write arrays, tables and a report

Every decision below comes from a measurement in reports/eda. The three
big ones:

    SCOPE   Baku + the three residential categories. Bare land is measured in
            a different unit (sot, not m²), and land / offices / garages /
            commercial units have no rooms and no floor at all, so they are a
            different prediction problem. See reports/eda section 11.

    DEDUP   The file is 45 days of repeated scrapes; 36% of rows are re-sightings
            of an advert that was still online. We keep the LAST sighting of each
            advert, because when a price moved it went down 2,606 times against
            714 up — the last sighting is the price the market saw last. Then we
            drop properties posted twice under two different advert addresses.
            Both passes happen BEFORE the split, so one property can never be in
            both train and test. See reports/eda section 12.

    WINDOW  Keep 10,000-5,000,000 AZN and 200-20,000 AZN per m². These are fixed
            numbers written into this file, not percentiles recomputed from the
            data, so the test set never influences the training set. Together
            they remove about 0.1% of rows. See reports/eda section 13.

Anything learned from the data — which locations get their own column, the
median used to fill a gap, the mean and standard deviation used to scale, the
median that splits premium from standard — is computed on the TRAINING SET ONLY
and then applied unchanged to validation and test. That is the whole point of
step 7 being separate from step 8.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings. Everything tunable lives here so nothing is buried in the code.
# ---------------------------------------------------------------------------
RANDOM_SEED = 42          # fixed so two runs produce byte-identical output
TRAIN_SHARE = 0.70
VALID_SHARE = 0.15        # test gets whatever is left (0.15)

# The column holding the advert's page address, e.g. "/items/4521724".
ADVERT_URL = "estate_rel_url_x"

# SCOPE — see the module docstring.
KEEP_CITY = "bakı"
KEEP_CATEGORIES = ("Yeni tikili", "Köhnə tikili", "Həyət evi/Bağ evi")

# WINDOW — fixed, declared thresholds.
MIN_PRICE, MAX_PRICE = 10_000, 5_000_000            # AZN
MIN_PRICE_PER_M2, MAX_PRICE_PER_M2 = 200, 20_000    # AZN per m²
MIN_AREA, MAX_AREA = 10, 2_000                      # m²

# Azerbaijan's bounding box. Anything outside is a broken coordinate.
LAT_RANGE, LNG_RANGE = (38.3, 41.95), (44.7, 50.6)

# The middle of Baku, used to measure how far out a property is. This is
# Icherisheher, the old walled city on the bay, which the price map in
# reports/eda/figures/fig09_map_of_baku.png shows as the expensive core.
# It is a fixed landmark, not a value measured from the data, so it cannot
# shift between runs or leak anything from the test set.
BAKU_CENTRE_LAT, BAKU_CENTRE_LNG = 40.3667, 49.8352

# How many kilometres one degree is worth at Baku's latitude. A degree of
# latitude is about 111 km everywhere; a degree of longitude shrinks as you
# move away from the equator, to about 85 km here.
KM_PER_DEGREE_LAT = 111.0
KM_PER_DEGREE_LNG = 111.0 * float(np.cos(np.radians(BAKU_CENTRE_LAT)))

# A location gets its own column only if it appears at least this often in the
# training set. Rarer ones share a single "other" column, so we do not add 140
# columns that are almost always zero.
MIN_ROWS_PER_LOCATION = 100

# Columns that are the target in disguise. Measured in reports/eda section 4:
# total_price equals price in 99.99% of rows, and unit_price x area rebuilds
# price exactly on 100% of rows. They never reach the model.
LEAKAGE_COLUMNS = ["total_price", "unit_price", "description"]

# The continuous features. These get scaled in step 8.
CONTINUOUS_FEATURES = [
    "log_area_m2",      # log of floor area — EDA showed log(area) vs log(price)
                        # is close to a straight line, so the log is the shape a
                        # linear model can actually use
    "rooms",
    "floor",
    "n_floors",
    "log_land_m2",      # log(1 + land). Flats have no land, so 0 is common and
                        # log1p keeps 0 at 0 instead of producing -infinity
    "lat",
    "lng",

    # --- engineered: four things the raw columns imply but do not state ---
    "distance_to_centre_km",  # how far from the middle of Baku
    "floor_ratio",            # how high up the building, as 0..1
    "area_per_room",          # m² per room: a cramped 3-room vs a spacious one
    "log_land_ratio",         # log(1 + land / floor area): how much garden
]

# The 0/1 features. Already on a sensible scale, so they are left alone.
BINARY_FEATURES = [
    "has_floor_info",       # 0 for houses: they genuinely have no floor number
    "has_repair",
    "has_bill_of_sale",
    "has_mortgage",
    "is_vip",
    "is_featured",
    "sold_by_owner",

    # --- engineered ---
    "is_ground_floor",      # the floor buyers like least
    "is_top_floor",         # the floor with the roof above it
]

# Columns kept in the *_clean.csv tables so a human can read a row back.
# Columns kept in the *_clean.csv tables so a human can read a row back.
# Listed explicitly rather than derived from BINARY_FEATURES, because that
# list also holds engineered columns (has_floor_info, is_ground_floor,
# is_top_floor) which exist only in the feature matrix. Here a missing
# `floor` already tells you the property has no floor number.
READABLE_COLUMNS = [
    "item_id", "price", "log_price", "area_m2", "rooms", "floor",
    "n_floors", "land_m2", "lat", "lng", "category", "location", "scraped",
    "has_repair", "has_bill_of_sale", "has_mortgage",
    "is_vip", "is_featured", "sold_by_owner",
]


# ===========================================================================
# Small helpers
# ===========================================================================
def number_from_text(series: pd.Series) -> pd.Series:
    """Pull the first number out of a text cell.

    The raw file stores numbers inside words: "145 m²", "1.3 sot",
    "3 450 AZN/m²". We remove the spaces that act as thousand separators, turn
    a decimal comma into a dot, then take the first number we find. Anything
    unreadable becomes NaN rather than crashing.
    """
    cleaned = (
        series.astype("string")
        .str.replace(r"\s", "", regex=True)
        .str.replace(",", ".", regex=False)
        .str.extract(r"(\d+\.?\d*)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def fingerprint(ids: pd.Series) -> str:
    """A short, stable hash of a set of item ids.

    src/validate.py re-runs this pipeline and compares these hashes, which is
    how we prove the split is reproducible rather than merely claiming it.
    """
    joined = "\n".join(sorted(ids.astype(str)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# ===========================================================================
# STEP 1 — load
# ===========================================================================
def load(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV exactly as it is. No cleaning happens here."""
    df = pd.read_csv(path)
    print(f"  [1] loaded {len(df):,} rows x {df.shape[1]} columns from {path}")
    return df


# ===========================================================================
# STEP 2 — parse
# ===========================================================================
def parse(df: pd.DataFrame) -> pd.DataFrame:
    """Build tidy, correctly typed columns from the raw text ones.

    The raw file is a join of two scraped tables, so it also carries a lot of
    bookkeeping (ids, scrape timestamps, image urls). We simply do not copy
    those across: `out` starts empty and only gains what we intend to use.
    """
    out = pd.DataFrame(index=df.index)

    # --- identity and time -------------------------------------------------
    # "/items/4521724" -> "4521724". One id per advert, used for deduplication
    # and for the split fingerprints.
    # We key on the NUMBER, not the url text, because the same advert is
    # sometimes stored both as "/items/2467670" and "/items/2467670.html".
    # Treating those as two adverts would leave 669 duplicate properties in
    # the file, free to straddle the train/test boundary.
    out["item_id"] = df[ADVERT_URL].astype("string").str.extract(r"(\d+)", expand=False)
    out["scraped"] = pd.to_datetime(df["day_x"], format="%d.%m.%Y", errors="coerce")

    # --- the target --------------------------------------------------------
    out["price"] = pd.to_numeric(df["price"], errors="coerce")

    # --- size --------------------------------------------------------------
    # CAREFUL: `Sahə` is in m² for flats and houses but in SOT for bare land
    # (1 sot = 100 m²). We only keep flats and houses in step 4, so by the time
    # this column is used it is always m². The guard below makes that explicit
    # rather than relying on the reader to remember it.
    area = number_from_text(df["Sahə"])
    is_land = df["Kateqoriya"].eq("Torpaq")
    out["area_m2"] = area.where(~is_land)   # land's area is in the wrong unit -> NaN

    # `Torpaq sahəsi` is the plot a house stands on, always in sot. Convert to
    # m² so every area in the frame is in one unit. Flats have no plot, so NaN
    # here means "no land", which step 5 turns into 0.
    out["land_m2"] = number_from_text(df["Torpaq sahəsi"]) * 100

    out["rooms"] = pd.to_numeric(df["Otaq sayı"], errors="coerce")

    # "7 / 9" means the 7th floor of a 9-storey building.
    floors = df["Mərtəbə"].astype("string").str.extract(r"(\d+)\s*/\s*(\d+)")
    out["floor"] = pd.to_numeric(floors[0], errors="coerce")
    out["n_floors"] = pd.to_numeric(floors[1], errors="coerce")

    # --- where -------------------------------------------------------------
    out["city"] = df["city"]
    out["category"] = df["Kateqoriya"]
    out["location"] = df["location"]

    # A coordinate outside Azerbaijan is broken. We blank the coordinate rather
    # than delete the row: the rest of that advert is still perfectly usable.
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lng = pd.to_numeric(df["lng"], errors="coerce")
    inside = lat.between(*LAT_RANGE) & lng.between(*LNG_RANGE)
    out["lat"] = lat.where(inside)
    out["lng"] = lng.where(inside)

    # --- the flag columns --------------------------------------------------
    # These five are only ever written when the answer is yes; the scraper
    # leaves them empty for no. Each has exactly one distinct non-empty value,
    # which reports/eda section 2 verifies. So "is it present?" IS the answer,
    # and filling them with a mean or dropping their rows would be wrong.
    out["has_repair"] = df["Təmir"].eq("var").astype(int)
    out["has_bill_of_sale"] = df["Çıxarış"].eq("var").astype(int)
    out["has_mortgage"] = df["İpoteka"].notna().astype(int)
    out["is_vip"] = df["vip"].notna().astype(int)
    out["is_featured"] = df["featured"].notna().astype(int)

    # Who is selling. Agents list 90% of adverts and at a higher median price.
    out["sold_by_owner"] = df["owner_title"].eq("mülkiyyətçi").astype(int)

    # NOTE: `repair`, `bill_of_sale` and `mortgage` are not read at all. They
    # agree with Təmir / Çıxarış / İpoteka on 100.0% of rows (reports/eda
    # section 7) — the same column twice. We keep one of each pair.
    # `views` is not read either: it counts people who opened the advert AFTER
    # it went live, so it cannot be known about a flat you are about to price.

    print(f"  [2] parsed {out.shape[1]} working columns")
    return out


# ===========================================================================
# STEP 3 — deduplicate
# ===========================================================================
def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reduce the file to one row per property.

    Two passes, because there are two different kinds of duplicate, and both
    must happen before the split — otherwise the same flat lands in train and
    test and the test score becomes a memory test.
    """
    start = len(df)

    # Pass 1: the same advert scraped on several days. Sorting by scrape date
    # and keeping the last row keeps the most recent asking price.
    df = (df.sort_values(["item_id", "scraped"])
            .drop_duplicates(subset=["item_id"], keep="last"))
    after_advert = len(df)

    # Pass 2: one property posted twice under two advert addresses. If two
    # adverts agree on position, size, rooms AND price, it is the same flat.
    twin_key = ["lat", "lng", "area_m2", "rooms", "price"]
    complete = df[twin_key].notna().all(axis=1)
    is_twin = complete & df.duplicated(subset=twin_key, keep="first")
    df = df[~is_twin]
    after_twins = len(df)

    stats = {
        "rows_before": start,
        "rows_removed_by_repeat_scrapes": start - after_advert,
        "rows_removed_by_reposted_twins": after_advert - after_twins,
        "rows_after": after_twins,
    }
    print(f"  [3] deduplicated {start:,} -> {after_twins:,} rows "
          f"(-{start - after_advert:,} repeat scrapes, "
          f"-{after_advert - after_twins:,} reposts)")
    return df.reset_index(drop=True), stats


# ===========================================================================
# STEP 4 — scope
# ===========================================================================
def apply_scope(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep only Baku residential property.

    Two reasons, both measured in reports/eda section 11:
      - 99.6% of adverts are in Baku. The 260 elsewhere are a different market
        sampled 260 times: too few to learn from, enough to add noise.
      - land / commercial units / offices / garages record no rooms and no
        floor, and land's area is in sot rather than m². They are not the same
        prediction problem.
    `category` stays as a feature, so the model can still tell a house from a
    new-build flat.
    """
    start = len(df)
    keep = df["city"].eq(KEEP_CITY) & df["category"].isin(KEEP_CATEGORIES)
    df = df[keep]

    stats = {
        "rows_before": start,
        "rows_removed_by_scope": start - len(df),
        "rows_after": len(df),
        "kept_city": KEEP_CITY,
        "kept_categories": list(KEEP_CATEGORIES),
    }
    print(f"  [4] scope: {start:,} -> {len(df):,} rows "
          f"(Baku + {len(KEEP_CATEGORIES)} residential categories)")
    return df.reset_index(drop=True), stats


# ===========================================================================
# STEP 5 — filters
# ===========================================================================
def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop records that cannot be true, and count every one we drop.

    The thresholds are the fixed numbers declared at the top of this file.
    They are NOT percentiles of this data: a rule like "drop the top 1%"
    computed over the whole file would let the test set decide what the
    training set contains.
    """
    start = len(df)
    df = df.copy()

    # A flat has no plot of land, so a missing land size means 0, not unknown.
    df["land_m2"] = df["land_m2"].fillna(0.0)

    # Each rule, kept separately so the report can say what each one cost.
    price_per_m2 = df["price"] / df["area_m2"]
    rules = {
        "price_outside_window": ~df["price"].between(MIN_PRICE, MAX_PRICE),
        "area_missing_or_outside_window": ~df["area_m2"].between(MIN_AREA, MAX_AREA),
        "price_per_m2_outside_window": ~price_per_m2.between(MIN_PRICE_PER_M2, MAX_PRICE_PER_M2),
        "floor_above_building_height": df["floor"] > df["n_floors"],
    }
    removed = {name: int(mask.sum()) for name, mask in rules.items()}

    drop = pd.concat(rules.values(), axis=1).any(axis=1)
    df = df[~drop]

    stats = {
        "rows_before": start,
        "removed_by_rule": removed,          # rules overlap, so these overlap too
        "total_removed_by_domain_filters": int(drop.sum()),
        "rows_after": len(df),
        "price_window_AZN": [MIN_PRICE, MAX_PRICE],
        "price_per_m2_window_AZN": [MIN_PRICE_PER_M2, MAX_PRICE_PER_M2],
        "area_window_m2": [MIN_AREA, MAX_AREA],
    }
    print(f"  [5] filters: {start:,} -> {len(df):,} rows (-{int(drop.sum()):,})")
    for name, n in removed.items():
        print(f"        {name}: {n:,}")
    return df.reset_index(drop=True), stats


# ===========================================================================
# STEP 6 — split
# ===========================================================================
def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Shuffle once with a fixed seed, then cut 70 / 15 / 15.

    A plain random split is enough here because step 3 already guaranteed that
    each property appears exactly once, so nothing can leak across the cut.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(df))

    n_train = int(len(df) * TRAIN_SHARE)
    n_valid = int(len(df) * VALID_SHARE)

    parts = {
        "train": df.iloc[order[:n_train]],
        "validation": df.iloc[order[n_train:n_train + n_valid]],
        "test": df.iloc[order[n_train + n_valid:]],
    }
    parts = {k: v.reset_index(drop=True) for k, v in parts.items()}
    print("  [6] split " + ", ".join(f"{k}={len(v):,}" for k, v in parts.items()))
    return parts


# ===========================================================================
# STEP 7 — fit the preprocessor ON THE TRAINING SET ONLY
# ===========================================================================
def fit_preprocessor(train: pd.DataFrame) -> dict:
    """Learn every rule we will later apply to all three splits.

    Returns a plain dictionary so the rules can be printed, saved and checked.
    Nothing here ever looks at validation or test — that is what makes the
    final score honest.
    """
    # Which locations deserve their own column. Rare ones share "other".
    counts = train["location"].value_counts()
    locations = sorted(counts[counts >= MIN_ROWS_PER_LOCATION].index.tolist())

    # Values used to fill gaps. The median is used rather than the mean because
    # these columns are skewed.
    fill_values = {col: float(train[col].median())
                   for col in ["rooms", "lat", "lng"]}

    # The tier boundary for Task B: the median price of the TRAINING set.
    # Using the median of the whole file would let test prices decide the
    # class boundary, which is leakage.
    tier_threshold = float(train["price"].median())

    rules: dict = {
        "locations": locations,
        "categories": list(KEEP_CATEGORIES),
        "fill_values": fill_values,
        "tier_threshold_AZN": tier_threshold,
    }

    # The mean and standard deviation used to scale. They must be measured on
    # the training set AFTER the gaps are filled, so we build the training
    # features once with scaling switched off and read the statistics off that.
    unscaled = build_features(train, rules, scale=False)
    rules["scaler_mean"] = {c: float(unscaled[c].mean()) for c in CONTINUOUS_FEATURES}
    rules["scaler_std"] = {c: float(unscaled[c].std(ddof=0)) or 1.0
                           for c in CONTINUOUS_FEATURES}
    rules["feature_names"] = unscaled.columns.tolist()

    print(f"  [7] fitted on train: {len(locations)} named locations, "
          f"tier threshold {tier_threshold:,.0f} AZN")
    return rules


# ===========================================================================
# STEP 8 — build the feature matrix
# ===========================================================================
def build_features(df: pd.DataFrame, rules: dict, scale: bool = True) -> pd.DataFrame:
    """Turn a cleaned table into the numeric matrix the models receive.

    `rules` comes from fit_preprocessor and is identical for all three splits.
    """
    out = pd.DataFrame(index=df.index)

    # --- continuous --------------------------------------------------------
    # Logs for the two heavily skewed sizes. log1p(x) = log(1 + x) keeps a
    # flat's land area of 0 at 0 instead of producing -infinity.
    out["log_area_m2"] = np.log(df["area_m2"])
    out["log_land_m2"] = np.log1p(df["land_m2"])
    out["rooms"] = df["rooms"].fillna(rules["fill_values"]["rooms"])
    out["lat"] = df["lat"].fillna(rules["fill_values"]["lat"])
    out["lng"] = df["lng"].fillna(rules["fill_values"]["lng"])

    # Houses have no floor number at all — it is not missing, it does not
    # apply. Filling it with the median would invent a 5th floor for a house.
    # Instead we put 0 and add a flag saying whether the number is real, so the
    # model can learn "when has_floor_info is 0, ignore the floor columns".
    has_floor = df["floor"].notna() & df["n_floors"].notna()
    out["floor"] = df["floor"].where(has_floor, 0.0)
    out["n_floors"] = df["n_floors"].where(has_floor, 0.0)

    # --- engineered features ----------------------------------------------
    # These four add no new information: every one is computed from columns we
    # already have. What they add is SHAPE. A straight-line model like the SVM
    # can only add its inputs up with weights; it cannot divide one by another
    # or measure a distance. A decision tree can approximate these by stacking
    # many splits, but it has to spend depth doing it. Writing them out as
    # columns hands both models the relationship directly.

    # 1. Distance from the middle of Baku, in kilometres. The price map shows a
    #    smooth fade from the centre outwards, but a model given only latitude
    #    and longitude has to rediscover that from two numbers that each mean
    #    nothing alone. We convert degrees to kilometres and use the ordinary
    #    flat-surface distance, which is accurate enough across one city.
    km_north = (out["lat"] - BAKU_CENTRE_LAT) * KM_PER_DEGREE_LAT
    km_east = (out["lng"] - BAKU_CENTRE_LNG) * KM_PER_DEGREE_LNG
    out["distance_to_centre_km"] = np.sqrt(km_north ** 2 + km_east ** 2)

    # 2. How high up the building the flat sits, as a fraction: 0.1 is near the
    #    bottom, 1.0 is the top floor. Floor 3 means something different in a
    #    4-storey building than in a 20-storey one, and the raw floor number
    #    cannot say which. Houses have no floors, so they get 0.
    out["floor_ratio"] = np.where(
        has_floor & (df["n_floors"] > 0),
        df["floor"] / df["n_floors"].replace(0, np.nan),
        0.0,
    )
    # The two floors buyers react to: the ground floor (noise, security, no
    # view) and the top floor (the roof, and the lift when it breaks).
    out["is_ground_floor"] = (has_floor & df["floor"].eq(1)).astype(int)
    out["is_top_floor"] = (has_floor & df["floor"].eq(df["n_floors"])).astype(int)

    # 3. Average room size. Two 90 m² flats, one with 2 rooms and one with 4,
    #    are different products at the same size and the same room count is
    #    only half the story. Division is exactly what a linear model cannot do.
    out["area_per_room"] = df["area_m2"] / out["rooms"].clip(lower=1)

    # 4. How much land comes with the building, relative to the building
    #    itself. A house on a large plot is worth more than the same house on a
    #    small one. Flats have no land, so this is 0 for them.
    out["log_land_ratio"] = np.log1p(df["land_m2"] / df["area_m2"])

    # --- binary ------------------------------------------------------------
    out["has_floor_info"] = has_floor.astype(int)
    for col in BINARY_FEATURES:
        if col not in ("has_floor_info", "is_ground_floor", "is_top_floor"):
            out[col] = df[col].astype(int)

    # --- one-hot: property type -------------------------------------------
    # One column per category, 1 where the row belongs to it. With three
    # categories this adds three columns, which is cheap and easy to read.
    for name in rules["categories"]:
        out[f"category={name}"] = df["category"].eq(name).astype(int)

    # --- one-hot: location -------------------------------------------------
    # Only locations frequent enough in the training set get a column. A row
    # whose location is rare, unknown, or appeared only after training lands in
    # "location=other", so an unseen value can never break the transform.
    known = set(rules["locations"])
    for name in rules["locations"]:
        out[f"location={name}"] = df["location"].eq(name).astype(int)
    out["location=other"] = (~df["location"].isin(known)).astype(int)

    # --- scaling -----------------------------------------------------------
    # Only the continuous columns are scaled; the 0/1 columns are already on a
    # sensible scale. The SVM needs this (area in the hundreds would otherwise
    # drown a 0/1 flag); the decision tree is unaffected either way, and it is
    # simpler to feed both models the same matrix.
    if scale:
        for col in CONTINUOUS_FEATURES:
            out[col] = (out[col] - rules["scaler_mean"][col]) / rules["scaler_std"][col]

    return out.astype(float)


def build_targets(df: pd.DataFrame, rules: dict) -> tuple[np.ndarray, np.ndarray]:
    """The two targets.

    Task A: the natural log of the price. Prices span six orders of magnitude,
    so in plain AZN one mistake on a 5,000,000 flat outweighs thousands of
    ordinary ones and squared error chases the outliers. The log turns "twice
    as expensive" into a constant step.

    Task B: 1 ("premium") if the price is at or above the TRAINING median,
    else 0 ("standard"). The comparison is >= rather than > because 117
    training rows sit exactly on the threshold — prices cluster on round
    numbers — and putting them on the high side splits the training set
    50.01 / 49.99 instead of 49.71 / 50.29.
    """
    y_log = np.log(df["price"].to_numpy(dtype=float))
    tier = (df["price"] >= rules["tier_threshold_AZN"]).to_numpy().astype(int)
    return y_log, tier


# ===========================================================================
# STEP 9 — save
# ===========================================================================
def save(parts: dict[str, pd.DataFrame], rules: dict, report: dict,
         output_dir: str | Path) -> None:
    """Write the arrays, the readable tables and the report.

    The file names are the ones src/validate.py and src/evaluate.py expect.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # The .npy suffix each split is saved under, matching the downstream scripts.
    suffix = {"train": "train", "validation": "valid", "test": "test"}

    for name, frame in parts.items():
        X = build_features(frame, rules).to_numpy()
        y_log, tier = build_targets(frame, rules)
        s = suffix[name]
        np.save(out / f"X_{s}.npy", X)
        np.save(out / f"y_{s}_log.npy", y_log)
        np.save(out / f"tier_{s}.npy", tier)
        # A readable copy of the same rows, for the validator and for eyeballing.
        # `log_price` is the Task A target spelled out next to the raw price,
        # so a human can check the two agree without running anything.
        readable = frame.assign(log_price=np.log(frame["price"]))
        readable[READABLE_COLUMNS].to_csv(out / f"{s}_clean.csv", index=False)

    pd.DataFrame({"feature": rules["feature_names"]}).to_csv(
        out / "feature_names.csv", index=False)
    (out / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  [9] wrote arrays, tables and preprocessing_report.json to {out}/")


# ===========================================================================
# The whole pipeline
# ===========================================================================
def run(input_path: str | Path, output_dir: str | Path) -> dict:
    """Run every step in order and return the report dictionary."""
    print(f"\ndata_prep: {input_path} -> {output_dir}")

    raw = load(input_path)
    raw_rows = len(raw)

    df = parse(raw)
    df, dedup_stats = deduplicate(df)
    df, scope_stats = apply_scope(df)
    df, filter_stats = apply_filters(df)

    parts = split(df)
    rules = fit_preprocessor(parts["train"])

    # Build the training matrix once here purely to count the columns for the
    # report; save() rebuilds it per split.
    n_features = build_features(parts["train"], rules).shape[1]
    print(f"  [8] built {n_features} features per row")

    report = {
        "random_seed": RANDOM_SEED,
        "raw_rows": raw_rows,

        "rows_after_deduplication": dedup_stats["rows_after"],
        "rows_removed_by_item_deduplication":
            dedup_stats["rows_removed_by_repeat_scrapes"]
            + dedup_stats["rows_removed_by_reposted_twins"],
        "deduplication": dedup_stats,

        "scope": {
            "rows_after_scope": scope_stats["rows_after"],
            "rows_removed_by_scope": scope_stats["rows_removed_by_scope"],
            "kept_city": scope_stats["kept_city"],
            "kept_categories": scope_stats["kept_categories"],
        },

        "rows_after_cleaning": filter_stats["rows_after"],
        "domain_filters": filter_stats,

        # The ranges every surviving row is guaranteed to lie inside. We do not
        # winsorize (clip an extreme value back to a fence): a row outside a
        # window is a broken record, so step 5 removes it outright. The bounds
        # are recorded here so src/validate.py can confirm they hold.
        "value_ranges": {
            "method": "rows outside the window are removed, not clipped",
            "fences": {
                "price": {"low": MIN_PRICE, "high": MAX_PRICE},
                "area_m2": {"low": MIN_AREA, "high": MAX_AREA},
            },
        },

        "final_rows": len(df),
        "split_rows": {k: len(v) for k, v in parts.items()},
        "split_item_id_sha256": {k: fingerprint(v["item_id"]) for k, v in parts.items()},

        "leakage_columns_removed": LEAKAGE_COLUMNS,
        "tier_threshold_AZN": rules["tier_threshold_AZN"],

        "n_processed_features": n_features,
        "numeric_features": CONTINUOUS_FEATURES,
        "binary_features": BINARY_FEATURES,
        "n_location_columns": len(rules["locations"]) + 1,   # + "other"
        "feature_names": rules["feature_names"],
        "scaler_mean": rules["scaler_mean"],
        "scaler_std": rules["scaler_std"],
    }

    save(parts, rules, report, output_dir)
    print(f"done: {len(df):,} rows, {n_features} features, "
          f"tier threshold {rules['tier_threshold_AZN']:,.0f} AZN\n")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare the bina.az data for modelling.")
    p.add_argument("--input", default="data/bina_az_sale.csv")
    p.add_argument("--output-dir", default="data/processed")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run(args.input, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
