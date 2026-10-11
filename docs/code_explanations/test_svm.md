# `tests/test_svm.py` — Pegasos and SVM diagnostics tests

The tests validate the linear primal solver and its public estimator behavior using synthetic data. They deliberately avoid Bina.az so edge cases run fast and deterministically.

## Data/objective helpers

- `blobs(n,sep,d,seed)` builds two-class clusters with controllable separation/dimensionality. It lets tests change task difficulty without changing model code.
- `reference_objective(X,y01,lam,w,b)` directly evaluates regularized hinge loss independently of the model's private `_objective`; objective parity detects formula/sign mistakes.
- `clf_data()` supplies a stable train/holdout case through pytest fixtures.

## Test classes

### `TestInit`

Parameterized invalid lambda, epochs, numeric learning rates, unsupported schedule strings, and constant-rate `eta*lambda>=1` must raise. Defaults/types and NumPy scalar compatibility check normalization of settings.

### `TestFitValidation`

Checks X/y dimensionality, row count, empty rows/features, NaN/infinity, single/three-class labels, predict-before-fit, wrong feature count, one-dimensional prediction, `fit` returning self, and input non-mutation. Each prevents a common silent failure from NumPy broadcasting or an estimator whose state is unusable.

### `TestLabels`

Proves arbitrary binary class labels map to signs internally and map back externally. Cases include integer/string labels, sorted positive-class orientation, flipping labels, agreement between score sign and prediction, Python list inputs, and pandas-like objects. This ensures a decision-function sign is interpretable.

### `TestLearning`

Measures exact/separable learning and holdout behavior as cluster separation varies. It also covers one feature, one epoch, tiny and imbalanced data, duplicate rows, constant features, a nonlinear pattern, and refitting. These distinguish reasonable learning from shape-specific overfitting.

Regularization tests compare weight norm under increasing lambda, ensure huge lambda suppresses weights, exercise constant learning-rate mode, and check the Pegasos projection/radius behavior loosely rather than imposing a brittle exact coefficient.

### `TestDeterminism`

Same seed should reproduce the same update path and outputs. Different seeds may take different stochastic paths but should reach similar quality. Row reorder can alter sample sequence, so tests emphasize prediction quality rather than exact weights.

### `TestObjective`

Checks one history entry per epoch, finite/nonnegative values, broad improvement, equality to an independently computed objective, final history alignment, and the known zero-weight hinge baseline of 1. These tests catch a common bug where history stores a different objective from the one optimized.

### `TestAttributes`

Verifies `classes_`, feature count, positive margin width, support-vector mask as `signed_margin<=1`, and fewer support points for a well-separated example. It does not claim these primal margin points are dual coefficients from an exact quadratic-program SVC.

### `TestAgainstSklearn`

Compares objective-matched Pegasos with a linear SVC on controlled scaled data, checks that more epochs approach the reference solution, and checks decision-boundary agreement on random points. Reference SVC is a test oracle only, not the custom implementation.

### `TestBiasRegression`

Confirms the bounded unregularized bias remains controlled on centered examples over selected lambdas; it documents why the practical solver's bias behavior is tested separately from weight shrinkage.

## Interpreting failures

- Score/objective discrepancy: inspect class sign convention, shrink order, violation condition, and intercept update.
- Different-seed exact coefficient mismatch alone is expected; quality or same-seed reproducibility failure is meaningful.
- Sklearn disagreement can indicate scaling/C convention mismatch. Keep test feature scaling and objective parameters matched before changing solver math.
- Synthetic tests establish mechanics, not superiority on the real housing dataset.
