# `tests/conftest.py` — pytest import setup

The fixture/setup module resolves the repository's `src/` directory and inserts it into `sys.path` for older tests that import `data_prep`, `decision_tree`, `svm`, or `tree_viz` as direct modules. The project itself uses package imports (`from .module import ...`) when invoked with `python -m src.run_all`.

This path setup is test-only: pytest loads `conftest.py` before test modules, so the direct imports can resolve predictably regardless of the launch directory. It defines no application data, does not import the dataset, and does not train a model. The tradeoff is that a few unit tests test direct imports rather than package-qualified imports; the actual runner separately exercises the package entry point.
