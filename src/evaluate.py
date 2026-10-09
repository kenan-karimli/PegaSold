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
    "Per class precision, recall, F1-score and support (zero-devision -> 0)"
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    pred_count = cm.sum(axis=0).astype(float)
    true_count = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return precision, recall, f1, true_count, labels