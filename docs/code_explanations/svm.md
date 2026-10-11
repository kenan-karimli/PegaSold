# `src/svm.py` — detailed walkthrough

The module implements a linear binary soft-margin classifier and a nonlinear wrapper that maps examples into seeded random Fourier features. Both optimizers and decision scores are NumPy code. The runner is responsible for feature standardization; the estimator warns in its module docstring that scaling is important.

## Shared binary-class convention

The public API accepts any two distinct labels, including strings or `{0,1}`. `np.unique(y)` produces deterministic sorted `classes_`. `classes_[1]` is the positive class, mapped to `+1`; all rows from `classes_[0]` map to `-1`. This makes the mathematical SVM objective simple while `predict` can return the user's original labels. It also means positive decision scores favor the second sorted label.

## `PegasosSVM.__init__`

Parameters:

- `lambda_param`: positive regularization coefficient. It controls the trade-off between small weight norm and hinge violations. Larger lambda shrinks weights more strongly.
- `epochs`: number of full shuffled passes through the training examples.
- `learning_rate`: either the canonical `"pegasos"` schedule or a constant positive value. For a constant rate, the code requires `eta*lambda < 1`, so the shrink factor remains positive.
- `random_state`: seed used to make the per-epoch row order reproducible.

Invalid values raise immediately. `w=None` marks the not-yet-fitted state; `b=0` is the initial intercept. `float`/`int` normalization prevents NumPy scalar settings from causing subtle downstream type differences.

## `fit(X,y)` — validate, encode, initialize

X is converted to a float matrix; y remains in its original dtype to preserve labels. The code requires 2-D X, 1-D y, equal nonzero row counts, at least one feature, finite numeric X, and exactly two unique classes. Invalid input fails before allocating a model.

`classes_` stores sorted original labels. `signs` is a float vector of `-1/+1`; the same row index selects X, y, and signs. `n_features_in_` enables shape validation later. `w` starts as a zero vector of length `n_features`; `b` starts at zero. A local `default_rng` isolates randomness from global NumPy random state.

## The Pegasos update, line by line

For each epoch, `rng.permutation(X.shape[0])` returns a fresh row ordering. A global `step` counter increments over all epochs, not once per epoch; this is important because the scheduled learning rate should decrease throughout training.

When `learning_rate == "pegasos"`, `eta=1/(lambda*step)`. Otherwise the fixed user rate is used. The signed margin is

`margin = y_i * (w dot x_i + b)`.

If margin is at least one, the point is outside/on the correct margin boundary and causes no hinge-loss gradient contribution. If it is below one, the example violates the margin or is misclassified.

Every step first applies `w *= 1 - eta*lambda`. This is the L2 regularization part of the subgradient. If the margin is below one, the code adds `eta*y_i*x_i` to w and adjusts the intercept in the class direction.

### Why the intercept has a clipped update

The SVM objective regularizes `w` but leaves `b` unregularized. An unregularized bias under a large early Pegasos rate can jump sharply; this implementation uses `min(eta,1.0)` for the bias step. It is a bounded-bias variant and is explicitly documented as a limitation. The code does not pretend it is an exact update for a model that regularizes an augmented bias coefficient.

At the end of each epoch `_objective(X,signs)` records the regularized hinge objective. This produces one scalar per epoch for convergence visualization. It need not decrease monotonically each epoch because the underlying updates are stochastic and the objective is measured on all training rows after each pass.

## After optimization: margins and diagnostics

The signed margin for each training point is recomputed. `support_vector_mask_ = signed_margin <= 1` labels points on or inside the margin; in soft-margin learning some can be misclassified. `support_vectors_` stores their feature rows for inspection. For a linear boundary, geometric margin width is `2/||w||`; a zero vector has infinite width by convention because it defines no finite separating direction. This implementation does not perform the optional canonical Pegasos projection of `w` onto a norm ball; its actual update is the shrink-plus-hinge subgradient shown above.

The fit returns `self`, allowing estimator-style chaining. Re-fitting resets weights, bias, random generator, objective history, and support-vector state rather than continuing old optimization.

## `decision_function` and `predict`

`decision_function` checks fitted state, 2-D input, feature count, and finiteness. It returns matrix multiplication `X @ w + b`, one real score per row. The sign is the class decision; magnitude is distance in unnormalized score units, not a calibrated probability.

`predict` uses `scores >= 0` to select class index 1, otherwise class index 0. The returned dtype/values are original target labels. No probability API is claimed for the SVM.

## `_objective(X, signs)`

The vector expression `max(0,1 - y*(Xw+b))` calculates hinge loss for all rows. `mean()` makes it average loss rather than total loss, and `0.5*lambda*dot(w,w)` is the L2 penalty. This objective is diagnostic and corresponds to the weight regularizer plus hinge term; the bounded unregularized intercept update is the stated practical optimization variant.

## `RandomFourierPegasosSVM`

### Parameters and random map

`n_components=D` is the number of mapped features; `gamma` controls RBF bandwidth and must be positive. A `default_rng(random_state)` creates a matrix `W` of shape `(original_features,D)` with independent normal entries of standard deviation `sqrt(2*gamma)` and a phase vector of D uniform values on `[0,2π)`. The mapping is

`sqrt(2/D) * cos(X @ W + phase)`.

The scale `sqrt(2/D)` is part of the random Fourier approximation and keeps mapped feature magnitude from growing with the number of components. For fixed seed, parameters and input, the same X always maps identically.

### Fit and reuse

`fit` validates non-empty finite 2-D X, records original feature count, samples W and phase once, then fits a contained `PegasosSVM` on mapped rows. It exposes the inner model's classes, objective history, margin width, and support-vector mask for convenient diagnostics. The RFF support vectors refer to mapped training examples, not a kernel solver's dual coefficients.

At prediction, `_map` uses the already-fitted W and phase; resampling the random frequencies would make train and test live in different feature spaces and invalidate the model. The original feature count and finite values are checked before mapping. `predict` routes mapped decision signs back through the original classes.

## `__main__` demonstration

The bottom block runs only when `python src/svm.py` is executed directly. It creates a synthetic classification dataset, standardizes it, fits Pegasos and a scikit-learn linear SVC, then prints their training accuracy and final objective. The actual project runner does not rely on this demo; it uses real prepared data and objective-matched baselines.

## Common mistakes this design avoids

- Never feed unscaled raw features to an SVM when feature units differ by orders of magnitude.
- Do not fit a scaler on validation/test data; runner and CV use train-only scaling.
- Do not compare raw `w` values across differently scaled feature spaces.
- Do not pass predicted class labels into ROC-AUC; use continuous `decision_function` scores.
- Do not confuse support-vector-like margin points with a dual optimization's stored support vectors; this implementation uses a primal subgradient solver.
- RFFs approximate an RBF kernel with finite D; increasing D can help approximation quality but increases memory and compute.
