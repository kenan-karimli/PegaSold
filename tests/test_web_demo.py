import json
from pathlib import Path

import numpy as np
import pytest

from src.web_demo import EstimateService


@pytest.fixture(scope="module")
def service():
    return EstimateService()


def sample(**updates):
    payload = {
        "category": "Yeni tikili", "location": "Nəsimi r.",
        "area_m2": 90, "rooms": 3, "floor": 5, "n_floors": 12,
        "land_m2": 0, "has_repair": True,
        "has_bill_of_sale": True, "has_mortgage": False,
    }
    payload.update(updates)
    return payload


def test_models_load_selected_training_configuration(service):
    metadata = service.metadata()
    assert metadata["training_rows"] == len(np.load("data/processed/X_train.npy"))
    rules = json.loads(Path("data/processed/preprocessing_rules.json").read_text())
    assert metadata["threshold_azn"] == rules["tier_threshold_AZN"]
    assert metadata["location_count"] > 0
    assert service.regressor.max_depth == service.model_selection["tree_regression"]["max_depth"]
    assert service.regressor.min_samples_leaf == service.model_selection["tree_regression"]["min_samples_leaf"]
    assert service.classifier.max_depth == service.model_selection["tree_classification"]["max_depth"]


def test_predict_returns_finite_price_and_separate_tree_class(service):
    result = service.predict(sample())
    assert result["price_azn"] > 0
    assert result["price_per_m2_azn"] > 0
    assert result["tier"] in {"Premium", "Standard"}
    assert 0 <= result["tier_probability"] <= 1
    assert result["model"] == "PegaSold NumPy Decision Tree"


def test_house_uses_plot_area_and_no_floor_metadata(service):
    result = service.predict(sample(category="Həyət evi/Bağ evi", land_m2=400,
                                    floor=None, n_floors=None))
    assert result["price_azn"] > 0


@pytest.mark.parametrize("updates", [
    {"area_m2": 2}, {"rooms": 0}, {"land_m2": -1},
    {"category": "Torpaq"}, {"floor": 13, "n_floors": 12},
])
def test_rejects_inputs_outside_model_domain(service, updates):
    with pytest.raises(ValueError):
        service.predict(sample(**updates))


@pytest.mark.parametrize("updates", [
    {"rooms": 2.5}, {"floor": 2.5}, {"n_floors": 12.5},
    {"has_repair": "nope"}, {"location": "   "},
])
def test_rejects_malformed_property_values(service, updates):
    with pytest.raises(ValueError):
        service.predict(sample(**updates))


def test_string_boolean_values_are_parsed_without_python_truthiness(service):
    yes = service.predict(sample(has_repair="true"))
    no = service.predict(sample(has_repair="false"))
    assert yes["price_azn"] > 0 and no["price_azn"] > 0


def test_unseen_location_is_mapped_to_other_and_uses_global_coordinate_median(service):
    result = service.predict(sample(location="Nümunə, təlimdə yoxdur"))
    assert result["price_azn"] > 0
    assert result["location_coordinate_approximate"] is True
