"""Serve the interactive PegaSold estimate demo using the project's NumPy trees."""
from __future__ import annotations

import json
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from .data_prep import build_features
from .decision_tree import DecisionTree

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
WEB_ROOT = ROOT


class EstimateService:
    """Load the saved training split, fit selected project trees, and predict."""

    def __init__(self, processed_dir: Path = PROCESSED) -> None:
        self.processed_dir = Path(processed_dir)
        required = ["X_train.npy", "y_train_log.npy", "tier_train.npy",
                    "preprocessing_rules.json", "train_clean.csv"]
        missing = [name for name in required if not (self.processed_dir / name).is_file()]
        results_path = self.processed_dir.parent.parent / "reports" / "results.json"
        if not results_path.is_file():
            missing.append(str(results_path))
        if missing:
            raise FileNotFoundError(
                "Prepared project data is missing: " + ", ".join(missing)
                + ". Run `python -m src.run_all` first."
            )
        self.rules = json.loads((self.processed_dir / "preprocessing_rules.json").read_text(encoding="utf-8"))
        results = json.loads(results_path.read_text(encoding="utf-8"))
        selected = results.get("selected", {})
        reg_settings = selected.get("tree_regression")
        cls_settings = selected.get("tree_classification")
        if not reg_settings or not cls_settings:
            raise ValueError(
                f"Selected tree configurations are missing from {results_path}; "
                "run `python -m src.run_all` to regenerate experiment results."
            )
        X = np.load(self.processed_dir / "X_train.npy")
        y_reg = np.load(self.processed_dir / "y_train_log.npy")
        y_cls = np.load(self.processed_dir / "tier_train.npy")
        if not (len(X) == len(y_reg) == len(y_cls)):
            raise ValueError("Training features and targets have inconsistent row counts")

        # Read the selected configuration from the experiment output so the demo
        # cannot silently drift away from the evaluated models.
        reg_leaf = int(reg_settings["min_samples_leaf"])
        cls_leaf = int(cls_settings["min_samples_leaf"])
        self.regressor = DecisionTree(
            task="regression", criterion=reg_settings["criterion"],
            max_depth=reg_settings["max_depth"], min_samples_leaf=reg_leaf,
            min_samples_split=max(2, 2 * reg_leaf),
        ).fit(X, y_reg)
        self.classifier = DecisionTree(
            task="classification", criterion=cls_settings["criterion"],
            max_depth=cls_settings["max_depth"], min_samples_leaf=cls_leaf,
            min_samples_split=max(2, 2 * cls_leaf),
        ).fit(X, y_cls)
        self.model_selection = {
            "tree_regression": reg_settings,
            "tree_classification": cls_settings,
        }
        self.train_rows = int(len(X))

        clean = pd.read_csv(self.processed_dir / "train_clean.csv")
        clean["lat"] = pd.to_numeric(clean["lat"], errors="coerce")
        clean["lng"] = pd.to_numeric(clean["lng"], errors="coerce")
        coordinates = clean.dropna(subset=["lat", "lng"]).groupby("location")[["lat", "lng"]].median()
        self.coordinates = {str(k): [float(v.lat), float(v.lng)] for k, v in coordinates.iterrows()}

    def metadata(self) -> dict:
        locations = sorted(self.rules["locations"])
        return {
            "categories": self.rules["categories"],
            "locations": locations,
            "threshold_azn": float(self.rules["tier_threshold_AZN"]),
            "training_rows": self.train_rows,
            "location_count": len(locations),
        }

    def predict(self, request: dict) -> dict:
        try:
            category = str(request["category"])
            location = str(request["location"])
            area = float(request["area_m2"])
            rooms_value = float(request["rooms"])
            if not rooms_value.is_integer():
                raise ValueError("Room count must be a whole number.")
            rooms = int(rooms_value)
            floor = request.get("floor")
            n_floors = request.get("n_floors")
            land = float(request.get("land_m2", 0) or 0)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Please check the property details.") from exc
        if category not in self.rules["categories"]:
            raise ValueError("Select a supported residential property type.")
        if not (10 <= area <= 2000):
            raise ValueError("Floor area must be between 10 and 2,000 m².")
        if not (1 <= rooms <= 30):
            raise ValueError("Room count must be between 1 and 30.")
        if not (0 <= land <= 200000):
            raise ValueError("Land area must be between 0 and 200,000 m².")
        # Rare and unseen locations correctly become the preprocessing 'other' bucket.
        location_for_features = location.strip()
        if not location_for_features:
            raise ValueError("Select a location.")
        has_floor = floor not in (None, "") and n_floors not in (None, "")
        try:
            floor_value = float(floor) if has_floor else np.nan
            floors_value = float(n_floors) if has_floor else np.nan
        except (TypeError, ValueError) as exc:
            raise ValueError("Floor values must be whole numbers.") from exc
        if has_floor and (not floor_value.is_integer() or not floors_value.is_integer()):
            raise ValueError("Floor values must be whole numbers.")
        floor = int(floor_value) if has_floor else np.nan
        n_floors = int(floors_value) if has_floor else np.nan
        if has_floor and (floor < 1 or n_floors < floor or n_floors > 100):
            raise ValueError("Floor number must fit within the building's total floors.")

        coords = self.coordinates.get(location)
        approximate_location = coords is None
        if coords is None:
            coords = [self.rules["fill_values"]["lat"], self.rules["fill_values"]["lng"]]
        def flag(name: str) -> int:
            value = request.get(name, False)
            if isinstance(value, bool):
                return int(value)
            if value in (0, 1):
                return int(value)
            if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                return int(value.strip().lower() == "true")
            raise ValueError(f"{name} must be true or false.")

        row = pd.DataFrame([{
            "area_m2": area, "land_m2": land, "rooms": rooms,
            "floor": floor, "n_floors": n_floors,
            "lat": coords[0], "lng": coords[1], "category": category,
            "location": location_for_features,
            "has_repair": flag("has_repair"),
            "has_bill_of_sale": flag("has_bill_of_sale"),
            "has_mortgage": flag("has_mortgage"),
            "is_vip": 0, "is_featured": 0, "sold_by_owner": 0,
        }])
        vector = build_features(row, self.rules).to_numpy(dtype=float)
        log_price = float(self.regressor.predict(vector)[0])
        price = float(np.exp(np.clip(log_price, -50, 50)))
        predicted_class = int(self.classifier.predict(vector)[0])
        probabilities = self.classifier.predict_proba(vector)[0]
        class_index = int(np.flatnonzero(self.classifier.classes_ == predicted_class)[0])
        return {
            "price_azn": round(price),
            "price_per_m2_azn": round(price / area),
            "tier": "Premium" if predicted_class == 1 else "Standard",
            "tier_probability": float(probabilities[class_index]),
            "tier_threshold_azn": float(self.rules["tier_threshold_AZN"]),
            "location_coordinate_approximate": approximate_location,
            "model": "PegaSold NumPy Decision Tree",
        }


SERVICE = None


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print("[web] " + fmt % args)

    def send_json(self, status: int, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json(200, {"ready": SERVICE is not None})
            return
        if path == "/api/metadata":
            self.send_json(200, SERVICE.metadata())
            return
        relative = "index.html" if path in ("", "/") else unquote(path.lstrip("/"))
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            self.send_error(403)
            return
        if not target.is_file():
            self.send_error(404)
            return
        mime = "text/html; charset=utf-8" if target.suffix == ".html" else {
            ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
            ".png": "image/png", ".svg": "image/svg+xml", ".jpg": "image/jpeg",
        }.get(target.suffix, "application/octet-stream")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16_384:
                raise ValueError("Request size is invalid.")
            payload = json.loads(self.rfile.read(length))
            self.send_json(200, SERVICE.predict(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})


def main() -> None:
    global SERVICE
    parser = argparse.ArgumentParser(description="Run the local PegaSold estimate demo.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    SERVICE = EstimateService()
    port = args.port
    print(f"PegaSold price demo is ready: http://127.0.0.1:{port}")
    print("Models fitted from the saved training split; Ctrl+C to stop.")
    ThreadingHTTPServer(("127.0.0.1", port), DemoHandler).serve_forever()


if __name__ == "__main__":
    main()
