"""Linear soft-margin SVM trained with the Pegasos sub-gradient method.

Objective: lambda/2 * ||w||^2 + mean(max(0, 1 - y * (Xw + b))). Scale input
features before fitting. Labels may be any two distinct values and are mapped
internally to -1/+1.
"""

import numpy as np


class PegasosSVM:
    """Binary linear SVM using stochastic sub-gradient updates."""

    def __init__(self, lambda_param=0.01, epochs=100, learning_rate="pegasos", random_state=42):
        if lambda_param <= 0:
            raise ValueError("lambda_param must be positive")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if isinstance(learning_rate, str):
            if learning_rate != "pegasos":
                raise ValueError("learning_rate must be 'pegasos' or a positive number")
        else:
            if not np.isscalar(learning_rate) or learning_rate <= 0:
                raise ValueError("learning_rate must be 'pegasos' or a positive number")
            if learning_rate * lambda_param >= 1:
                raise ValueError("learning_rate * lambda_param must be < 1")
        self.lambda_param = float(lambda_param)
        self.epochs = int(epochs)
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")
        if X.shape[0] != y.size or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X and y must contain matching, non-empty samples and features")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("PegasosSVM requires exactly two target classes")

        self.classes_ = classes
        signs = np.where(y == classes[1], 1.0, -1.0)
        self.n_features_in_ = X.shape[1]
        self.w = np.zeros(self.n_features_in_, dtype=float)
        self.b = 0.0
        rng = np.random.default_rng(self.random_state)
        self.objective_history_ = []
        step = 0

        for _ in range(self.epochs):
            for i in rng.permutation(X.shape[0]):
                step += 1
                eta = (1.0 / (self.lambda_param * step)
                       if self.learning_rate == "pegasos"
                       else float(self.learning_rate))
                margin = signs[i] * (np.dot(self.w, X[i]) + self.b)
                # Regularization applies to w; the intercept is not penalized.
                self.w *= 1.0 - eta * self.lambda_param
                if margin < 1.0:
                    self.w += eta * signs[i] * X[i]
                    # Bound the unregularized intercept update. The first
                    # Pegasos steps can be very large for small lambda; without
                    # this cap the bias can become oversized on centered data,
                    # destabilizing the objective and decision boundary.
                    self.b += min(eta, 1.0) * signs[i]
                # Pegasos includes a projection onto ||w|| <= 1/sqrt(lambda).
                # Without it, the first 1/(lambda*t) updates can leave the
                # feasible ball (especially at t=1), making the optimizer
                # depend heavily on the first few shuffled rows.
                norm = np.linalg.norm(self.w)
                radius = 1.0 / np.sqrt(self.lambda_param)
                if norm > radius:
                    self.w *= radius / norm
            self.objective_history_.append(self._objective(X, signs))

        self.support_vector_mask_ = signs * self.decision_function(X) <= 1.0
        self.support_vectors_ = X[self.support_vector_mask_]
        norm = np.linalg.norm(self.w)
        self.margin_width_ = float(2.0 / norm) if norm > 0 else np.inf
        return self

    def decision_function(self, X):
        if self.w is None:
            raise ValueError("fit must be called before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must be 2D with the fitted feature count")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return X @ self.w + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]

    def _objective(self, X, signs):
        hinge = np.maximum(0.0, 1.0 - signs * (X @ self.w + self.b))
        return float(0.5 * self.lambda_param * np.dot(self.w, self.w) + hinge.mean())


class RandomFourierPegasosSVM:
    """Nonlinear RBF approximation followed by the same NumPy Pegasos solver.

    Random Fourier features approximate the RBF kernel using
    ``sqrt(2/D) cos(X @ W + phase)``. This meets the nonlinear-model option
    without calling a library kernel/model implementation.
    """

    def __init__(self, lambda_param=0.01, epochs=30, n_components=256,
                 gamma=0.1, random_state=42):
        if n_components < 1 or gamma <= 0:
            raise ValueError("n_components and gamma must be positive")
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.random_state = random_state

    def _map(self, X):
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.random_weights_ + self.random_phase_)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a non-empty finite 2D array")
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.random_weights_ = rng.normal(0, np.sqrt(2 * self.gamma),
                                          size=(self.n_features_in_, self.n_components))
        self.random_phase_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        self.model_ = PegasosSVM(lambda_param=self.lambda_param, epochs=self.epochs,
                                 random_state=self.random_state).fit(self._map(X), y)
        self.classes_ = self.model_.classes_
        self.objective_history_ = self.model_.objective_history_
        self.margin_width_ = self.model_.margin_width_
        self.support_vector_mask_ = self.model_.support_vector_mask_
        self.n_support_vectors_ = int(self.support_vector_mask_.sum())
        return self

    def decision_function(self, X):
        if not hasattr(self, "model_"):
            raise ValueError("fit must be called before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("X must be finite, 2D, with the fitted feature count")
        return self.model_.decision_function(self._map(X))

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y = make_classification(n_samples=1000, n_features=10, random_state=0)
    X = StandardScaler().fit_transform(X)
    lam = 0.01
    peg = PegasosSVM(lambda_param=lam, epochs=200).fit(X, y)
    ref = SVC(kernel="linear", C=1 / (lam * len(y))).fit(X, y)
    print("Pegasos accuracy:", (peg.predict(X) == y).mean())
    print("sklearn accuracy:", ref.score(X, y))
    print("Pegasos objective:", peg.objective_history_[-1])
