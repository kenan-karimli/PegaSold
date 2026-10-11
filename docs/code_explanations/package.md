# `src/__init__.py` — package entry

This file marks `src` as the project's Python package and documents the standard entry point, `python -m src.run_all`. It intentionally does not import all modules at package load time: importing NumPy, pandas, plotting, and scikit-learn eagerly would make even a simple package import slow and could introduce circular imports.
