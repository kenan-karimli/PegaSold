# PegaSold interactive estimator

The root `index.html` is an English, responsive estimator for a Baku residential property's category, location, area, rooms, optional floor/plot size and listing attributes. It returns an asking-price estimate and a separate Standard/Premium classification from the project's NumPy decision trees.

## Run

From the repository root:

```bash
python -m src.web_demo
```

Open `http://127.0.0.1:8765`. To choose another local port, pass `--port 8000`. The server needs `data/processed/` artifacts created by `python -m src.run_all`. It refits the selected trees from the saved training arrays when it starts. It does not provide a static or mocked prediction when the model server is unavailable.

## Files

- `index.html`, `demo.css`, `demo.js` — estimator interface and responsive supplied black/lime/light-card design.
- `assets/pegasold-logo.png` — provided team logo.
- `src/web_demo.py` — local API, preprocessing integration and model predictions.
- `report.html` — experiment report and project results.

Endpoint details, feature construction, output interpretation and limits are explained in [`docs/code_explanations/web_demo.md`](docs/code_explanations/web_demo.md).
