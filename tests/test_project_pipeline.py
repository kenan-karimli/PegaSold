import numpy as np
import pandas as pd

from src.data_prep import number_from_text, prepare_data, remove_training_price_outliers
from src.evaluate import classification_metrics, regression_metrics
from decision_tree import DecisionTree
from svm import RandomFourierPegasosSVM


def test_measurement_parser_handles_sot_and_missing_values():
    values = number_from_text(pd.Series(["145 m²", "1.3 sot", None]))
    assert values.iloc[0] == 145
    assert values.iloc[1] == 1.3
    assert pd.isna(values.iloc[2])


def test_log_price_mad_filters_training_only_and_keeps_evaluation_rows():
    train = pd.DataFrame({"price": [100_000, 110_000, 120_000, 130_000, 140_000, 5_000_000],
                          "item_id": list("abcdef")})
    validation = pd.DataFrame({"price": [5_000_000], "item_id": ["v"]})
    test = pd.DataFrame({"price": [5_000_000], "item_id": ["t"]})
    parts, audit = remove_training_price_outliers(
        {"train": train, "validation": validation, "test": test})
    assert "f" not in set(parts["train"]["item_id"])
    assert parts["validation"].equals(validation)
    assert parts["test"].equals(test)
    assert audit["train_outliers_removed"] == 1
    assert audit["rows_by_split"]["test"]["outliers_removed"] == 0


def test_prepare_data_deduplicates_filters_and_returns_finite_splits(tmp_path):
    rows = []
    for i in range(80):
        rows.append({
            "estate_rel_url_x": f"/items/{i // 2 if i < 2 else i}",
            "day_x": f"{5 + (i % 2):02d}.10.2024",
            "city": "bakı",
            "Kateqoriya": ("Yeni tikili", "Köhnə tikili", "Həyət evi/Bağ evi")[i % 3],
            "price": 100000 + i * 5000, "total_price": 100000 + i * 5000,
            "unit_price": "1 500 AZN/m²", "currency_x": "AZN",
            "Sahə": f"{100 + i} m²", "Otaq sayı": 2 + i % 3,
            "Mərtəbə": "3 / 9", "Torpaq sahəsi": "1 sot",
            "lat": 40.4, "lng": 49.8,
            "Təmir": "var" if i % 2 else "yoxdur", "Çıxarış": "var", "İpoteka": None,
            "location": f"district-{i % 4}", "description": "not a feature",
            "vip": None, "featured": None, "owner_title": "mülkiyyətçi",
        })
    path = tmp_path / "listings.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    processed = tmp_path / "processed"
    d = prepare_data(path, output_dir=processed)
    assert d["audit"]["raw_rows"] == 80
    assert d["audit"]["after_dedup"] == 79
    assert d["audit"]["after_scope"] == 79
    assert "description" not in d["feature_names"]
    assert "unit_price" not in d["feature_names"]
    assert np.isfinite(d["X_train"]).all()
    assert d["X_train"].shape[1] == d["X_test"].shape[1]
    assert d["tier_threshold"] == np.median(d["price_train"])
    for split in ("train", "validation", "test"):
        artifact = np.load(processed / f"{split}.npz")
        assert artifact["X"].shape[1] == d["X_train"].shape[1]
        suffix = "valid" if split == "validation" else split
        assert (processed / f"{suffix}_clean.csv").exists()
    id_sets = [set(pd.read_csv(processed / f"{name}_clean.csv")["item_id"].astype(str))
               for name in ("train", "valid", "test")]
    assert not (id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2])
    assert (processed / "preprocessing_report.json").exists()
    assert (processed / "preprocessing_rules.json").exists()


def test_exploration_parser_does_not_modify_input_frame():
    from src.explore import build_exploration_frame
    source = pd.DataFrame({"Sahə": ["145 m²"], "Torpaq sahəsi": ["1.3 sot"],
                           "unit_price": ["3 450 AZN/m²"], "Mərtəbə": ["7 / 9"],
                           "Otaq sayı": [3], "day_x": ["05.10.2024"],
                           "estate_rel_url_x": ["/items/123.html"]})
    original = source.copy(deep=True)
    parsed = build_exploration_frame(source)
    pd.testing.assert_frame_equal(source, original)
    assert parsed["item_id"].iloc[0] == "123"
    assert parsed["area_m2"].iloc[0] == 145


def test_exploration_uses_latest_normalized_ad_snapshot_like_preprocessing():
    from src.explore import build_exploration_frame, latest_listing_per_id
    source = pd.DataFrame({
        "estate_rel_url_x": ["/items/12", "/items/12.html", None],
        "day_x": ["05.10.2024", "06.10.2024", "06.10.2024"],
        "Sahə": ["80 m²", "82 m²", "90 m²"],
        "Torpaq sahəsi": [None, None, None], "unit_price": [None, None, None],
        "Mərtəbə": ["2 / 9", "3 / 9", "1 / 2"], "Otaq sayı": [2, 3, 2],
    })
    one = latest_listing_per_id(build_exploration_frame(source))
    identified = one[one["item_id"].notna()]
    assert len(identified) == 1
    assert identified.iloc[0]["area_m2"] == 82
    assert one["item_id"].isna().sum() == 1


def test_numpy_metrics_match_known_binary_and_regression_values():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    metrics = classification_metrics(y, pred, scores=np.array([.1, .8, .7, .9]))
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == .75
    assert np.isclose(metrics["precision"], 2 / 3)
    assert metrics["recall"] == 1
    assert np.isclose(metrics["f1"], .8)
    assert np.isclose(metrics["roc_auc"], .75)
    reg = regression_metrics(np.array([1., 2., 3.]), np.array([1., 3., 2.]))
    assert np.isclose(reg["mae"], 2 / 3)
    assert np.isclose(reg["rmse"], np.sqrt(2 / 3))
    assert np.isclose(reg["r2"], .0)


def test_tree_feature_importances_are_normalized():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    model = DecisionTree(max_depth=2).fit(X, y)
    assert model.feature_importances_.shape == (2,)
    np.testing.assert_allclose(model.feature_importances_.sum(), 1.0, atol=1e-12)


def test_random_fourier_pegasos_learns_nonlinear_pattern():
    rng = np.random.default_rng(10)
    angles = rng.uniform(0, 2*np.pi, 700)
    radius = np.where(np.arange(len(angles)) % 2, 3.0, 1.0) + rng.normal(0, .12, len(angles))
    X = np.c_[radius*np.cos(angles), radius*np.sin(angles)]
    y = (radius > 2).astype(int)
    model = RandomFourierPegasosSVM(lambda_param=.001, epochs=25,
                                    n_components=256, gamma=.5, random_state=4).fit(X, y)
    assert (model.predict(X) == y).mean() > .94
    assert model.margin_width_ > 0
