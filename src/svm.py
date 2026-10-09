"""Linear SVM trained with mini-batch Pegasos, with one-vs-rest multiclass.
 
Objective (per binary problem):
    lambda/2 * ||w||^2 + mean_i( c_i * max(0, 1 - y_i * (w.x_i + b)) )
where c_i are optional class weights. Scale features before fitting.
 
Compatible with evaluate.py: fit / predict / decision_function, classes_,
objective_history_, support_vector_mask_, margin_width_, and the constructor
arguments lambda_param, epochs, random_state used by svm_lambda_study.
 
Binary problem  -> decision_function returns shape (n,)    (ROC-AUC works)
Multiclass      -> decision_function returns shape (n, K)  (argmax prediction)
"""
 
import numpy as np
 
 
class PegasosSVMOvR:
    """Linear SVM (Pegasos) for binary and multiclass (one-vs-rest) targets."""
 
    def __init__(self, lambda_param=0.01, epochs=50, batch_size=32,
                 class_weight=None, tol=None, project=True, random_state=42):
        """
        Args:
            lambda_param: L2 regularization strength (> 0).
            epochs: passes over the training data.
            batch_size: mini-batch size; 1 gives classic Pegasos.
            class_weight: None or "balanced" (weights hinge loss per class).
            tol: stop early when the objective improves by less than tol
                between epochs (None disables early stopping).
            project: project w onto the ball ||w|| <= 1/sqrt(lambda), as in
                the original Pegasos paper.
            random_state: seed for shuffling.
        """
        if lambda_param <= 0:
            raise ValueError("lambda_param must be positive")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")
        self.lambda_param = float(lambda_param)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.class_weight = class_weight
        self.tol = tol
        self.project = project
        self.random_state = random_state
        self.W_ = None
        self.b_ = None
 
    # ------------------------------------------------------------------ fit
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
        if classes.size < 2:
            raise ValueError("PegasosSVMOvR requires at least two target classes")
 
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
 
        # One binary problem if 2 classes, otherwise one per class (OvR).
        targets = [classes[1]] if classes.size == 2 else list(classes)
        W, B, histories, epochs_run = [], [], [], []
        for cls in targets:
            signs = np.where(y == cls, 1.0, -1.0)
            weights = self._sample_weights(signs)
            w, b, hist = self._fit_binary(X, signs, weights, rng)
            W.append(w)
            B.append(b)
            histories.append(hist)
            epochs_run.append(len(hist))
 
        self.W_ = np.vstack(W)
        self.b_ = np.asarray(B)
        self.objective_history_per_class_ = histories
        self.n_epochs_run_ = epochs_run
        # Mean objective across OvR problems, aligned to the shortest run.
        n = min(len(h) for h in histories)
        self.objective_history_ = list(np.mean([h[:n] for h in histories], axis=0))
 
        # Support vectors: points on/inside the margin of at least one problem.
        viol = np.zeros(X.shape[0], dtype=bool)
        widths = []
        for k, cls in enumerate(targets):
            signs = np.where(y == cls, 1.0, -1.0)
            viol |= signs * (X @ self.W_[k] + self.b_[k]) <= 1.0
            norm = np.linalg.norm(self.W_[k])
            widths.append(2.0 / norm if norm > 0 else np.inf)
        self.support_vector_mask_ = viol
        self.support_vectors_ = X[viol]
        self.margin_widths_ = np.asarray(widths)
        self.margin_width_ = float(np.mean(widths))
        return self
 
    def _sample_weights(self, signs):
        """Per-sample hinge weights; 'balanced' gives each side equal total."""
        if self.class_weight != "balanced":
            return np.ones_like(signs)
        n = signs.size
        n_pos = (signs > 0).sum()
        n_neg = n - n_pos
        return np.where(signs > 0, n / (2.0 * n_pos), n / (2.0 * n_neg))
 
    def _fit_binary(self, X, signs, weights, rng):
        n, d = X.shape
        lam = self.lambda_param
        w = np.zeros(d)
        b = 0.0
        history = []
        t = 0
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                t += 1
                eta = 1.0 / (lam * t)
                Xb, sb, cb = X[idx], signs[idx], weights[idx]
                violated = sb * (Xb @ w + b) < 1.0
                coef = (cb * sb)[violated]
                w *= 1.0 - eta * lam
                if coef.size:
                    w += (eta / idx.size) * (coef @ Xb[violated])
                    b += (eta / idx.size) * coef.sum()
                if self.project:
                    norm = np.linalg.norm(w)
                    limit = 1.0 / np.sqrt(lam)
                    if norm > limit:
                        w *= limit / norm
            obj = self._objective(X, signs, weights, w, b)
            history.append(obj)
            if self.tol is not None and len(history) > 1 \
                    and abs(history[-2] - obj) < self.tol:
                break
        return w, b, history
 
    def _objective(self, X, signs, weights, w, b):
        hinge = np.maximum(0.0, 1.0 - signs * (X @ w + b))
        return float(0.5 * self.lambda_param * (w @ w) + np.mean(weights * hinge))
 
    # ------------------------------------------------------------ inference
    def decision_function(self, X):
        if self.W_ is None:
            raise ValueError("fit must be called before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must be 2D with the fitted feature count")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        scores = X @ self.W_.T + self.b_
        return scores[:, 0] if self.classes_.size == 2 else scores
 
    def predict(self, X):
        scores = self.decision_function(X)
        if self.classes_.size == 2:
            return self.classes_[(scores >= 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]
 
    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))