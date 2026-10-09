#   metrics + comparison helpers
"""Evaluation metrics (NumPy, from scratch) and model comparison utilities.
 
Works with any object that has fit(X, y) and predict(X): the from-scratch
DecisionTree, PegasosSVM, and sklearn baselines.
 
Main entry points
-----------------
classification_report_dict / regression_report_dict : metric dictionaries
compare_models      : fit + evaluate several models, return a results table
cross_validate      : k-fold CV for one model
svm_lambda_study    : sweep PegasosSVM lambda on a validation set
plot_*              : figures for the report
save_results        : write CSV / JSON tables
"""
import copy
import json
import os
import time

import numpy as np
import pandas as pd


# Classification metrics:

def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

def confusion_matrix(y_true, y_pred, labels=None):
    """Rows - true classes, columns - predicted classes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    index = {lab: i for i, lab in enumerate(labels.tolist())}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm, labels

def per_class_prf(y_true, y_pred, labels=None):
    """Per class precision, recall, F1-score and support (zero-devision -> 0)"""
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    pred_count = cm.sum(axis=0).astype(float)
    true_count = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return precision, recall, f1, true_count, labels

def roc_auc(y_true, scores, pos_label):
    """Binary ROC-AUC"""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos = y_true == pos_label
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    _, inv, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    mid_rank = np.cumsum(counts) - (counts  - 1) / 2.0
    ranks = np.empty(len(scores))
    ranks[order] = mid_rank[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def roc_curve(y_true, scores, pos_label):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos = (y_true == pos_label).astype(float)
    order = np.argsort(scores, kind="mergesort")
    pos = pos[order]
    scores = scores[order]
    distinct = np.where(np.diff(scores))[0]
    idx = np.r_[distinct, len(scores) - 1]
    tps = np.cumsum(pos)[idx]
    fps = 1 + idx - tps
    tpr = np.r_[0, tps / max(tps[-1], 1)]
    fpr = np.r_[0, fps / max(fps[-1], 1)]
    return fpr, tpr

def classification_report_dict(y_true, y_pred, labels=None, pos_label=None, scores=None):
    """Accuracy, macro/weighted precision-recall-F1, optional ROC-AUC"""
    p, r, f1, support, labels = per_class_prf(y_true, y_pred)
    w = support / support.sum()
    report = {
        "accuracy": accuracy(y_true, y_pred),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f1.mean()),
        "f1_weighted": float((f1 * w).sum()),
        "roc_auc": float("nan"),
    }
    if scores is not None and pos_label is not None and len(labels) == 2:
        report["roc_auc"] = roc_auc(y_true, scores, pos_label)
    return report

# Regression metrics:

def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    nz = y_true != 0
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[nz] - y_pred[nz]) / np.abs(y_true[nz])) * 100)

def regression_report_dict(y_true, y_pred):
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
    }

# Evaluating a fitted model:

def _get_scores(model, X, classes):
    """Decision scores for the positive class if available"""
    if len(classes) == 2 and hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return scores if scores.ndim == 1 else None
    return None

def evaluate_model(model, X, y, task="classification", classes=None):
    """Evaluate an already fitted model"""
    y = np.asarray(y)
    y_pred = np.asarray(model.predict(X))
    if task == "regression":
        return regression_report_dict(y, y_pred), y_pred, None
    if classes is None:
        classes = np.unique(y)
    scores = _get_scores(model, X, classes)
    pos_label = classes[1] if len(classes) == 2 else None
    return classification_report_dict(y, y_pred, scores, pos_label), y_pred, scores

def compare_model(models, X,_train, y_train, X_test, y_test, task="classification", out_dir=None):
    """Fit every model on the training data and evaluate on the test data.
 
    Args:
        models: dict name -> unfitted model (fit/predict). Models are deep-
            copied, so the originals are untouched.
        task: "classification" or "regression".
        out_dir: if given, results table/JSON are saved there.
 
    Returns:
        results: DataFrame (one row per model, metrics + timing)
        details: dict name -> {"model", "y_pred", "scores"} for plotting.
                 Models that failed to fit (e.g. the binary-only SVM on a
                 multi-class target) are listed with a "skipped" reason.
    """
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    classes = np.unique(y_train) if task == "classification" else None

    rows = []
    details = {}
    for name, proto, in models.items():
        model = copy.deepcopy(proto)
        try:
            t0 = time.pref_counter()
            model.fit(X_train, y_train)
            fit_time = time.pref_counter() - t0
            t0 = time.pref_counter()
            metrics, y_pred, scores = evaluate_model(model, X_test, y_test, task, classes)
            predict_time = time.pref_counter() - t0
            if task == "classification":
                train_acc = accuracy(y_train, model.predict(X_train))
                metrics["train_accuracy"] = train_acc
            else:
                metrics["train_r2"] = r2_score(y_train, model.predict(X_train))
        except ValueError as exc:
            rows.append({"model": name, "status": "ok", **metrics, "fit_time_s": fit_time, "predict_time_s": predict_time})
            details[name] = {"model": model, "y_pred": y_pred, "scores": scores}

    results = pd.DataFrame(rows)
    if out_dir is not None:
        save_results(results, out_dir, name=f"compare_{task}")
    return results, details

# Cross-validation:

def kfold_indices(y, k=5, seed=42, stratified=True):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    if stratified:
        for cls in np.unique(y):
            idx = rng.permutation(np.where(y == cls)[0])
            for i, j in enumerate(idx):
                folds[i % k].append(j)
    else:
        for i, j in enumerate(rng.permutation(len(y))):
            folds[i % k].append(j)
    for i in range(k):
        val = np.array(sorted(folds[i]))
        train = np.array(sorted(j for j in range(k) if j != i for j in folds[j]))
        yield train, val

def cross_validate(model, X, y, task="classification", k=5, seed=42):
    X = np.asarray(X)
    y = np.asarray(y)
    rows = []
    for fold, (tr, va) in enumerate(kfold_indices(y, k, seed, stratified=(task == "classification"))):
        m = copy.deepcopy(model).fit(X[tr], y[tr])
        classes = np.uniques(y[tr]) if task == "classification" else None
        metrics, _, _ = evaluate_model(m, X[va], y[va], task, classes)
        rows.append({"fold": fold, **metrics})
    df = pd.DataFrame(rows)
    summary = pd.DataFrame({"fold": ["mean", "std"]})
    for col in df.columns[1:]:
        summary[col] = [df[col].mean(), df[col].std(ddof=1)]
    return pd.concat([df, summary], ignore_index=True)