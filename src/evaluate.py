"""
evaluate.py  --  diagnostic plots and model comparison for the bina.az project.

Tries to load real processed data from data/processed/.
Falls back to realistic synthetic data (matching the preprocessing report
statistics) if that directory is missing — so the plots always render.

Produces five figure files in reports/figures/:
  fig1_bias_variance.png      depth sweep + learning curves
  fig2_predictions.png        actual vs predicted + residuals (Task A)
  fig3_classification.png     ROC, confusion matrix, PR curve (Task B)
  fig4_model_comparison.png   cross-model metric bar chart
  fig5_error_analysis.png     error sliced by price range / category

Usage:
    python evaluate.py
    python evaluate.py --dir data/processed --out reports/figures
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix,
    mean_absolute_error, mean_squared_error,
    precision_recall_curve, r2_score,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# DATA  (real or synthetic)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic(
    n_train: int = 40_446,
    n_valid: int = 8_667,
    n_test: int = 8_667,
    n_features: int = 138,
    n_numeric: int = 12,
    tier_threshold_log: float = 12.25,   # ln(209 000)
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Generate synthetic data whose statistics match the preprocessing report.

    The ground-truth log-price is a linear combination of a few strong features
    plus Gaussian noise — realistic enough to show genuine bias-variance curves.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    N = n_train + n_valid + n_test

    # ── numeric block (RobustScaled, centered ≈ 0) ──────────────────────────
    # Feature layout matches preprocessing_report.json numeric_features order:
    #   0:lat  1:lng  2:area_m2  3:floor  4:n_floors  5:land_m2
    #   6:rooms  7:has_repair  8:has_extract  9:has_mortgage
    #   10:is_vip  11:is_featured
    X_num = rng.standard_normal((N, n_numeric))
    # Binary flag columns: snap to 0/1 (threshold at 0)
    for col in (7, 8, 9, 10, 11):
        X_num[:, col] = (X_num[:, col] > 0).astype(float)
    # Rooms and area are integers, positive, and correlated
    X_num[:, 6] = np.clip(rng.normal(0, 1, N), -2, 3)   # rooms (scaled)
    X_num[:, 2] = X_num[:, 6] * 0.7 + rng.normal(0, 0.5, N)  # area corr with rooms

    # ── one-hot block ────────────────────────────────────────────────────────
    n_onehot = n_features - n_numeric
    # location (≈126 columns) + city (8) + Kateqoriya (3) minus rare values
    # We use a Dirichlet-Multinomial approach to get sparse one-hot rows
    n_loc = 120   # location OHE columns (after min_freq=5 pruning)
    n_city = 8
    n_kat = 3     # Kateqoriya (remaining = n_onehot - n_loc - n_city)

    loc_probs = rng.dirichlet(np.ones(n_loc) * 0.3)
    city_probs = np.array([0.94, 0.015, 0.013, 0.01, 0.008, 0.007, 0.004, 0.003])
    kat_probs  = np.array([0.625, 0.209, 0.166])   # Yeni / Köhnə / Həyət

    loc_idx  = rng.choice(n_loc,  N, p=loc_probs)
    city_idx = rng.choice(n_city, N, p=city_probs)
    kat_idx  = rng.choice(n_kat,  N, p=kat_probs)

    X_ohe = np.zeros((N, n_loc + n_city + n_kat))
    X_ohe[np.arange(N), loc_idx] = 1.0
    X_ohe[np.arange(N), n_loc + city_idx] = 1.0
    X_ohe[np.arange(N), n_loc + n_city + kat_idx] = 1.0

    X = np.hstack([X_num, X_ohe])

    # ── ground-truth log-price ───────────────────────────────────────────────
    # Known strong effects
    beta_num = np.array([
        -0.08,   # lat  (south Baku neighborhoods cheaper)
         0.03,   # lng
         0.28,   # area_m2   ← strong
         0.04,   # floor
         0.01,   # n_floors
         0.03,   # land_m2
         0.32,   # rooms     ← strongest (Spearman 0.528 in EDA)
         0.10,   # has_repair
         0.07,   # has_extract
         0.04,   # has_mortgage
         0.07,   # is_vip
         0.09,   # is_featured
    ])
    # Location premium: each district shifts price by a draw from N(0, 0.4)
    loc_effects  = rng.normal(0, 0.40, n_loc)
    city_effects = np.array([0.80, 0.05, -0.10, -0.15, -0.18, -0.20, -0.25, -0.30])
    kat_effects  = np.array([0.12, 0.00, -0.20])   # Yeni/Köhnə/Həyət

    y = (
        tier_threshold_log
        + X_num @ beta_num
        + loc_effects[loc_idx]
        + city_effects[city_idx]
        + kat_effects[kat_idx]
        + rng.normal(0, 0.45, N)           # irreducible noise
    )

    tier = (y >= tier_threshold_log).astype(int)

    def split(arr):
        return arr[:n_train], arr[n_train:n_train+n_valid], arr[n_train+n_valid:]

    X_tr, X_va, X_te   = split(X)
    y_tr, y_va, y_te   = split(y)
    t_tr, t_va, t_te   = split(tier)

    return dict(
        X_train=X_tr, X_valid=X_va, X_test=X_te,
        y_train=y_tr, y_valid=y_va, y_test=y_te,
        tier_train=t_tr, tier_valid=t_va, tier_test=t_te,
        synthetic=True,
    )


def load_real(data_dir: Path) -> dict | None:
    required = [
        "X_train.npy", "X_valid.npy", "X_test.npy",
        "y_train_log.npy", "y_valid_log.npy", "y_test_log.npy",
        "tier_train.npy", "tier_valid.npy", "tier_test.npy",
    ]
    if not all((data_dir / f).exists() for f in required):
        return None

    return dict(
        X_train=np.load(data_dir / "X_train.npy"),
        X_valid=np.load(data_dir / "X_valid.npy"),
        X_test=np.load(data_dir / "X_test.npy"),
        y_train=np.load(data_dir / "y_train_log.npy"),
        y_valid=np.load(data_dir / "y_valid_log.npy"),
        y_test=np.load(data_dir / "y_test_log.npy"),
        tier_train=np.load(data_dir / "tier_train.npy"),
        tier_valid=np.load(data_dir / "tier_valid.npy"),
        tier_test=np.load(data_dir / "tier_test.npy"),
        synthetic=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save(fig: plt.Figure, path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    out = path / name
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1  bias-variance (depth sweep + learning curves)
# ─────────────────────────────────────────────────────────────────────────────

def fig_bias_variance(data: dict, out: Path) -> None:
    print("\n[1/5] bias-variance / depth sweep …")

    X_tr, y_tr   = data["X_train"], data["y_train"]
    X_va, y_va   = data["X_valid"], data["y_valid"]
    t_tr          = data["tier_train"]
    t_va          = data["tier_valid"]

    depths = list(range(1, 31))

    # ── regression ───────────────────────────────────────────────────────────
    rmse_train_r, rmse_val_r = [], []
    for d in depths:
        m = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE)
        m.fit(X_tr, y_tr)
        rmse_train_r.append(rmse(y_tr, m.predict(X_tr)))
        rmse_val_r.append(rmse(y_va, m.predict(X_va)))

    # ── classification ───────────────────────────────────────────────────────
    acc_train_c, acc_val_c = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
        m.fit(X_tr, t_tr)
        acc_train_c.append(accuracy_score(t_tr, m.predict(X_tr)))
        acc_val_c.append(accuracy_score(t_va, m.predict(X_va)))

    # ── learning curves (fixed depth) ────────────────────────────────────────
    best_depth_r = depths[int(np.argmin(rmse_val_r))]
    best_depth_c = depths[int(np.argmax(acc_val_c))]
    sizes = np.linspace(0.05, 1.0, 15)
    n = len(X_tr)
    lc_rmse_tr, lc_rmse_va = [], []
    lc_acc_tr,  lc_acc_va  = [], []
    for frac in sizes:
        k = max(50, int(n * frac))
        idx = np.random.choice(n, k, replace=False)
        Xs, ys, ts = X_tr[idx], y_tr[idx], t_tr[idx]
        mr = DecisionTreeRegressor(max_depth=best_depth_r, random_state=RANDOM_STATE).fit(Xs, ys)
        mc = DecisionTreeClassifier(max_depth=best_depth_c, random_state=RANDOM_STATE).fit(Xs, ts)
        lc_rmse_tr.append(rmse(ys, mr.predict(Xs)))
        lc_rmse_va.append(rmse(y_va, mr.predict(X_va)))
        lc_acc_tr.append(accuracy_score(ts, mc.predict(Xs)))
        lc_acc_va.append(accuracy_score(t_va, mc.predict(X_va)))

    train_sizes_k = (sizes * n / 1000).round(1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bias–Variance Analysis  (Decision Tree baseline)", fontsize=14, fontweight="bold")

    # top-left: depth vs RMSE
    ax = axes[0, 0]
    ax.plot(depths, rmse_train_r, "o-", label="Train RMSE", color="steelblue")
    ax.plot(depths, rmse_val_r,   "s-", label="Val RMSE",   color="tomato")
    ax.axvline(best_depth_r, ls="--", color="gray", alpha=0.7,
               label=f"Best depth = {best_depth_r}")
    ax.fill_between(depths, rmse_train_r, rmse_val_r, alpha=0.10, color="tomato",
                    label="Variance gap")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("RMSE  (log-price)")
    ax.set_title("Task A — Regression: depth vs error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # annotate regions
    ax.text(2, max(rmse_val_r) * 0.98, "← High bias\n  (underfitting)", fontsize=8,
            color="steelblue", va="top")
    ax.text(depths[-4], min(rmse_val_r) + (max(rmse_val_r)-min(rmse_val_r))*0.5,
            "Overfitting →\n(high variance)", fontsize=8, color="tomato", ha="right")

    # top-right: depth vs accuracy
    ax = axes[0, 1]
    ax.plot(depths, acc_train_c, "o-", label="Train Acc", color="steelblue")
    ax.plot(depths, acc_val_c,   "s-", label="Val Acc",   color="tomato")
    ax.axvline(best_depth_c, ls="--", color="gray", alpha=0.7,
               label=f"Best depth = {best_depth_c}")
    ax.fill_between(depths, acc_val_c, acc_train_c, alpha=0.10, color="tomato",
                    label="Variance gap")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Task B — Classification: depth vs accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.4, 1.02)

    # bottom-left: learning curve regression
    ax = axes[1, 0]
    ax.plot(train_sizes_k, lc_rmse_tr, "o-", label="Train RMSE", color="steelblue")
    ax.plot(train_sizes_k, lc_rmse_va, "s-", label="Val RMSE",   color="tomato")
    ax.set_xlabel("Training set size (thousands)")
    ax.set_ylabel("RMSE  (log-price)")
    ax.set_title(f"Learning curve — Regression (depth={best_depth_r})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # bottom-right: learning curve classification
    ax = axes[1, 1]
    ax.plot(train_sizes_k, lc_acc_tr, "o-", label="Train Acc", color="steelblue")
    ax.plot(train_sizes_k, lc_acc_va, "s-", label="Val Acc",   color="tomato")
    ax.set_xlabel("Training set size (thousands)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning curve — Classification (depth={best_depth_c})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.4, 1.02)

    save(fig, out, "fig1_bias_variance.png")
    return best_depth_r, best_depth_c


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2  prediction analysis (Task A)
# ─────────────────────────────────────────────────────────────────────────────

def fig_predictions(data: dict, best_depth_r: int, out: Path) -> None:
    print("[2/5] prediction analysis (Task A) …")

    m = DecisionTreeRegressor(max_depth=best_depth_r, random_state=RANDOM_STATE)
    m.fit(data["X_train"], data["y_train"])

    y_val  = data["y_valid"]
    y_pred = m.predict(data["X_valid"])
    resid  = y_val - y_pred

    val_rmse = rmse(y_val, y_pred)
    val_mae  = float(mean_absolute_error(y_val, y_pred))
    val_r2   = float(r2_score(y_val, y_pred))

    print(f"     Val RMSE={val_rmse:.4f}  MAE={val_mae:.4f}  R²={val_r2:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Task A — Prediction Analysis  (DT depth={best_depth_r})\n"
                 f"Val  RMSE={val_rmse:.4f}   MAE={val_mae:.4f}   R²={val_r2:.4f}",
                 fontsize=12, fontweight="bold")

    # top-left: actual vs predicted
    ax = axes[0, 0]
    ax.scatter(y_val, y_pred, s=4, alpha=0.3, color="steelblue", rasterized=True)
    lo, hi = min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual  log(price)")
    ax.set_ylabel("Predicted  log(price)")
    ax.set_title("Actual vs Predicted")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # top-right: residuals vs predicted
    ax = axes[0, 1]
    ax.scatter(y_pred, resid, s=4, alpha=0.3, color="steelblue", rasterized=True)
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.axhline( resid.std(), color="gray", lw=1, ls=":")
    ax.axhline(-resid.std(), color="gray", lw=1, ls=":", label="±1 σ")
    ax.set_xlabel("Predicted  log(price)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # bottom-left: residual histogram
    ax = axes[1, 0]
    ax.hist(resid, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0,          color="red",  lw=2, ls="--", label="Zero")
    ax.axvline(resid.mean(), color="orange", lw=1.5, ls="--",
               label=f"Mean={resid.mean():.3f}")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution  (σ={resid.std():.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # bottom-right: RMSE by price bin (in AZN, back-transformed)
    ax = axes[1, 1]
    bins = np.percentile(y_val, np.linspace(0, 100, 7))  # 6 equal-frequency bins
    bin_rmse, bin_centers, bin_counts = [], [], []
    for lo_b, hi_b in zip(bins[:-1], bins[1:]):
        mask = (y_val >= lo_b) & (y_val < hi_b)
        if mask.sum() < 10:
            continue
        bin_rmse.append(rmse(y_val[mask], y_pred[mask]))
        center_azn = int(np.exp((lo_b + hi_b) / 2))
        bin_centers.append(f"{center_azn:,}")
        bin_counts.append(mask.sum())

    bars = ax.bar(range(len(bin_rmse)), bin_rmse, color="steelblue",
                  edgecolor="white", alpha=0.85)
    for bar, cnt in zip(bars, bin_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"n={cnt:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(bin_rmse)))
    ax.set_xticklabels(bin_centers, rotation=30, fontsize=8)
    ax.set_xlabel("Price range mid-point (AZN)")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by Price Range")
    ax.grid(axis="y", alpha=0.3)

    save(fig, out, "fig2_predictions.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3  classification analysis (Task B)
# ─────────────────────────────────────────────────────────────────────────────

def fig_classification(data: dict, best_depth_c: int, out: Path) -> None:
    print("[3/5] classification analysis (Task B) …")

    m = DecisionTreeClassifier(max_depth=best_depth_c, random_state=RANDOM_STATE)
    m.fit(data["X_train"], data["tier_train"])

    t_val   = data["tier_valid"]
    t_pred  = m.predict(data["X_valid"])
    t_prob  = m.predict_proba(data["X_valid"])[:, 1]

    acc  = accuracy_score(t_val, t_pred)
    auc_ = roc_auc_score(t_val, t_prob)
    print(f"     Val Acc={acc:.4f}  ROC-AUC={auc_:.4f}")

    fpr, tpr, _  = roc_curve(t_val, t_prob)
    prec, rec, _ = precision_recall_curve(t_val, t_prob)
    cm = confusion_matrix(t_val, t_pred)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Task B — Classification Analysis  (DT depth={best_depth_c})\n"
                 f"Val  Acc={acc:.4f}   ROC-AUC={auc_:.4f}",
                 fontsize=12, fontweight="bold")

    # ROC
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"DT (AUC = {auc_:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Confusion matrix
    ax = axes[0, 1]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, fraction=0.046, label="%")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm_pct[i, j] > 60 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred: Standard", "Pred: Premium"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True: Standard", "True: Premium"])
    ax.set_title("Confusion Matrix")

    # Precision-Recall
    ax = axes[1, 0]
    ax.plot(rec, prec, color="steelblue", lw=2,
            label=f"DT (AP = {auc(rec, prec):.3f})")
    baseline = t_val.mean()
    ax.axhline(baseline, ls="--", color="gray", lw=1,
               label=f"Random ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # Probability calibration histogram
    ax = axes[1, 1]
    ax.hist(t_prob[t_val == 0], bins=40, alpha=0.6,
            label="Standard (true=0)", color="steelblue")
    ax.hist(t_prob[t_val == 1], bins=40, alpha=0.6,
            label="Premium (true=1)",  color="tomato")
    ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Threshold 0.5")
    ax.set_xlabel("Predicted probability  P(premium)")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by True Class")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    save(fig, out, "fig3_classification.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4  model comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_model_comparison(data: dict, best_depth_r: int, best_depth_c: int, out: Path) -> None:
    print("[4/5] model comparison …")

    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_valid"], data["y_valid"]
    t_tr, t_va = data["tier_train"], data["tier_valid"]
    X_te, y_te = data["X_test"],  data["y_test"]
    t_te        = data["tier_test"]

    models_r = {
        "Dummy\n(mean)":         DummyRegressor(strategy="mean"),
        f"DT\n(depth={best_depth_r})": DecisionTreeRegressor(max_depth=best_depth_r, random_state=RANDOM_STATE),
        "DT\n(no limit)":        DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Ridge\n(α=1)":          Ridge(alpha=1.0),
        "LinearSVR\n(C=0.1)":   LinearSVR(C=0.1, max_iter=2000, random_state=RANDOM_STATE),
    }
    models_c = {
        "Dummy\n(stratified)":   DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        f"DT\n(depth={best_depth_c})": DecisionTreeClassifier(max_depth=best_depth_c, random_state=RANDOM_STATE),
        "DT\n(no limit)":        DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SGD\n(SVM hinge)":      SGDClassifier(loss="hinge", max_iter=1000, random_state=RANDOM_STATE),
        "LinearSVC\n(C=0.1)":   LinearSVC(C=0.1, max_iter=2000, random_state=RANDOM_STATE),
    }

    def score_r(model):
        model.fit(X_tr, y_tr)
        return {
            "train_rmse": rmse(y_tr, model.predict(X_tr)),
            "val_rmse":   rmse(y_va, model.predict(X_va)),
            "test_rmse":  rmse(y_te, model.predict(X_te)),
        }

    def score_c(model):
        model.fit(X_tr, t_tr)
        return {
            "train_acc": accuracy_score(t_tr, model.predict(X_tr)),
            "val_acc":   accuracy_score(t_va, model.predict(X_va)),
            "test_acc":  accuracy_score(t_te, model.predict(X_te)),
        }

    results_r = {name: score_r(m) for name, m in models_r.items()}
    results_c = {name: score_c(m) for name, m in models_c.items()}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Model Comparison  (sklearn baselines — replace with your from-scratch DT & Pegasos SVM)",
                 fontsize=11, fontweight="bold")

    # ── regression bar chart ─────────────────────────────────────────────────
    ax = axes[0]
    names = list(results_r.keys())
    x = np.arange(len(names))
    w = 0.28
    r_train = [results_r[n]["train_rmse"] for n in names]
    r_val   = [results_r[n]["val_rmse"]   for n in names]
    r_test  = [results_r[n]["test_rmse"]  for n in names]
    ax.bar(x - w, r_train, w, label="Train", color="steelblue",   alpha=0.85)
    ax.bar(x,     r_val,   w, label="Val",   color="tomato",      alpha=0.85)
    ax.bar(x + w, r_test,  w, label="Test",  color="forestgreen", alpha=0.85)
    for xi, (tr, va, te) in zip(x, zip(r_train, r_val, r_test)):
        ax.text(xi - w, tr + 0.003, f"{tr:.3f}", ha="center", fontsize=7, rotation=90)
        ax.text(xi,     va + 0.003, f"{va:.3f}", ha="center", fontsize=7, rotation=90)
        ax.text(xi + w, te + 0.003, f"{te:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("RMSE  (log-price)")
    ax.set_title("Task A — Regression RMSE")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── classification bar chart ─────────────────────────────────────────────
    ax = axes[1]
    names_c = list(results_c.keys())
    x = np.arange(len(names_c))
    c_train = [results_c[n]["train_acc"] for n in names_c]
    c_val   = [results_c[n]["val_acc"]   for n in names_c]
    c_test  = [results_c[n]["test_acc"]  for n in names_c]
    ax.bar(x - w, c_train, w, label="Train", color="steelblue",   alpha=0.85)
    ax.bar(x,     c_val,   w, label="Val",   color="tomato",      alpha=0.85)
    ax.bar(x + w, c_test,  w, label="Test",  color="forestgreen", alpha=0.85)
    for xi, (tr, va, te) in zip(x, zip(c_train, c_val, c_test)):
        ax.text(xi - w, tr + 0.003, f"{tr:.3f}", ha="center", fontsize=7, rotation=90)
        ax.text(xi,     va + 0.003, f"{va:.3f}", ha="center", fontsize=7, rotation=90)
        ax.text(xi + w, te + 0.003, f"{te:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(names_c, fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Task B — Classification Accuracy")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.12)

    save(fig, out, "fig4_model_comparison.png")

    # Print table
    print("\n  ── Task A (RMSE log-price) ──────────────────────────────")
    print(f"  {'Model':22s}  {'Train':>8}  {'Val':>8}  {'Test':>8}")
    for n in names:
        r = results_r[n]
        print(f"  {n.replace(chr(10),' '):22s}  {r['train_rmse']:8.4f}  {r['val_rmse']:8.4f}  {r['test_rmse']:8.4f}")
    print("\n  ── Task B (Accuracy) ────────────────────────────────────")
    print(f"  {'Model':22s}  {'Train':>8}  {'Val':>8}  {'Test':>8}")
    for n in names_c:
        c = results_c[n]
        print(f"  {n.replace(chr(10),' '):22s}  {c['train_acc']:8.4f}  {c['val_acc']:8.4f}  {c['test_acc']:8.4f}")

    return results_r, results_c


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5  error analysis (price range + error distribution)
# ─────────────────────────────────────────────────────────────────────────────

def fig_error_analysis(data: dict, best_depth_r: int, best_depth_c: int, out: Path) -> None:
    print("[5/5] error analysis …")

    mr = DecisionTreeRegressor(max_depth=best_depth_r, random_state=RANDOM_STATE)
    mc = DecisionTreeClassifier(max_depth=best_depth_c, random_state=RANDOM_STATE)
    mr.fit(data["X_train"], data["y_train"])
    mc.fit(data["X_train"], data["tier_train"])

    y_va   = data["y_valid"]
    t_va   = data["tier_valid"]
    y_pred = mr.predict(data["X_valid"])
    t_pred = mc.predict(data["X_valid"])
    resid  = y_va - y_pred
    err_r  = np.abs(resid)         # absolute error in log-price
    err_r_azn = np.abs(np.exp(y_va) - np.exp(y_pred))   # error in AZN

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Error Analysis", fontsize=13, fontweight="bold")

    # top-left: abs error vs predicted
    ax = axes[0, 0]
    ax.scatter(y_pred, err_r, s=3, alpha=0.25, color="steelblue", rasterized=True)
    # running median
    order  = np.argsort(y_pred)
    window = max(1, len(y_pred) // 50)
    running_med = pd.Series(err_r[order]).rolling(window, center=True).median()
    ax.plot(y_pred[order], running_med, "r-", lw=2, label="Median MAE (rolling)")
    ax.set_xlabel("Predicted log(price)")
    ax.set_ylabel("|Residual|")
    ax.set_title("Absolute Error vs Predicted")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # top-middle: error in AZN by price bin
    ax = axes[0, 1]
    price_azn = np.exp(y_va)
    price_bins = [0, 50e3, 100e3, 200e3, 300e3, 500e3, np.inf]
    labels_bin  = ["<50k", "50–100k", "100–200k", "200–300k", "300–500k", ">500k"]
    mae_azn, counts = [], []
    for lo_b, hi_b in zip(price_bins[:-1], price_bins[1:]):
        mask = (price_azn >= lo_b) & (price_azn < hi_b)
        if mask.sum() < 5:
            mae_azn.append(0); counts.append(0)
            continue
        mae_azn.append(float(np.median(err_r_azn[mask])))
        counts.append(mask.sum())
    bars = ax.bar(range(len(labels_bin)), [v/1000 for v in mae_azn],
                  color="steelblue", alpha=0.85, edgecolor="white")
    for bar, cnt in zip(bars, counts):
        if cnt:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"n={cnt:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(labels_bin)))
    ax.set_xticklabels(labels_bin, rotation=25, fontsize=8)
    ax.set_ylabel("Median |error|  (×1,000 AZN)")
    ax.set_title("Median AZN Error by Price Bin")
    ax.grid(axis="y", alpha=0.3)

    # top-right: % error by price bin
    ax = axes[0, 2]
    pct_err = []
    for lo_b, hi_b in zip(price_bins[:-1], price_bins[1:]):
        mask = (price_azn >= lo_b) & (price_azn < hi_b)
        if mask.sum() < 5:
            pct_err.append(0); continue
        pct = float(np.median(err_r_azn[mask] / price_azn[mask] * 100))
        pct_err.append(pct)
    ax.bar(range(len(labels_bin)), pct_err, color="tomato", alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(labels_bin)))
    ax.set_xticklabels(labels_bin, rotation=25, fontsize=8)
    ax.set_ylabel("Median |error| / price  (%)")
    ax.set_title("Relative Error by Price Bin")
    ax.grid(axis="y", alpha=0.3)

    # bottom-left: Q-Q plot of residuals
    ax = axes[1, 0]
    from scipy import stats as sp_stats
    (osm, osr), (slope, intercept, r) = sp_stats.probplot(resid, dist="norm", fit=True)
    ax.plot(osm, osr, ".", ms=2, color="steelblue", alpha=0.5)
    ax.plot(osm, slope * np.array(osm) + intercept, "r--", lw=2, label=f"Normal fit  r={r:.3f}")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title("Q–Q Plot of Residuals")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # bottom-middle: classification error by tier
    ax = axes[1, 1]
    for tier_val, label, colour in [(0, "Standard", "steelblue"), (1, "Premium", "tomato")]:
        mask = t_va == tier_val
        probs = mc.predict_proba(data["X_valid"])[mask, 1]
        correct = (t_pred[mask] == t_va[mask])
        ax.hist(probs[correct],  bins=25, alpha=0.6, color=colour, label=f"{label} correct")
        ax.hist(probs[~correct], bins=25, alpha=0.6, color=colour,
                label=f"{label} wrong", histtype="step", lw=2, ls="--")
    ax.axvline(0.5, color="black", ls="--", lw=1.5)
    ax.set_xlabel("P(premium)")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution: Correct vs Wrong")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # bottom-right: cumulative error (what % of predictions are within x% error)
    ax = axes[1, 2]
    thresholds = np.linspace(0, 1.0, 200)
    pct_within = np.array([100 * (err_r <= t).mean() for t in thresholds])
    ax.plot(thresholds, pct_within, "steelblue", lw=2)
    for thresh, color in [(0.1, "green"), (0.2, "orange"), (0.5, "red")]:
        pct = 100 * (err_r <= thresh).mean()
        ax.axvline(thresh, color=color, ls="--", lw=1,
                   label=f"|err|<{thresh:.1f}: {pct:.1f}% of val")
    ax.set_xlabel("|Residual|  (log-price units)")
    ax.set_ylabel("% of validation set")
    ax.set_title("Cumulative Error Distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 101)

    save(fig, out, "fig5_error_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# METRICS dump (metrics.json + metrics.md for the report)
# ─────────────────────────────────────────────────────────────────────────────

def write_metrics(data: dict, best_depth_r: int, best_depth_c: int,
                  results_r: dict, results_c: dict, out: Path) -> dict:
    """Headline numbers for report/report.tex + reports/summary.md."""
    mr = DecisionTreeRegressor(max_depth=best_depth_r, random_state=RANDOM_STATE)
    mc = DecisionTreeClassifier(max_depth=best_depth_c, random_state=RANDOM_STATE)
    mr.fit(data["X_train"], data["y_train"])
    mc.fit(data["X_train"], data["tier_train"])

    y_va, y_te = data["y_valid"], data["y_test"]
    t_va, t_te = data["tier_valid"], data["tier_test"]
    pv_va, pv_te = mr.predict(data["X_valid"]), mr.predict(data["X_test"])
    cv_va, cv_te = mc.predict(data["X_valid"]), mc.predict(data["X_test"])
    prob_va = mc.predict_proba(data["X_valid"])[:, 1]

    metrics = {
        "synthetic": bool(data.get("synthetic", False)),
        "best_depth_regression": best_depth_r,
        "best_depth_classification": best_depth_c,
        "task_a_val": {
            "rmse": rmse(y_va, pv_va),
            "mae": float(mean_absolute_error(y_va, pv_va)),
            "r2": float(r2_score(y_va, pv_va)),
        },
        "task_a_test": {
            "rmse": rmse(y_te, pv_te),
            "mae": float(mean_absolute_error(y_te, pv_te)),
            "r2": float(r2_score(y_te, pv_te)),
        },
        "task_b_val": {
            "acc": float(accuracy_score(t_va, cv_va)),
            "roc_auc": float(roc_auc_score(t_va, prob_va)),
        },
        "task_b_test": {
            "acc": float(accuracy_score(t_te, cv_te)),
            "roc_auc": float(roc_auc_score(t_te, mc.predict_proba(data["X_test"])[:, 1])),
        },
        "baselines_val_rmse": {k: v["val_rmse"] for k, v in results_r.items()},
        "baselines_val_acc": {k: v["val_acc"] for k, v in results_c.items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    a, b = metrics["task_a_val"], metrics["task_b_val"]
    at, bt = metrics["task_a_test"], metrics["task_b_test"]
    (out / "metrics.md").write_text(
        "# Model metrics (sklearn DecisionTree baselines)\n\n"
        f"- synthetic={metrics['synthetic']}\n"
        f"- Task A valid: RMSE={a['rmse']:.4f} MAE={a['mae']:.4f} R²={a['r2']:.4f}\n"
        f"- Task A test: RMSE={at['rmse']:.4f} MAE={at['mae']:.4f} R²={at['r2']:.4f}\n"
        f"- Task B valid: Acc={b['acc']:.4f} ROC-AUC={b['roc_auc']:.4f}\n"
        f"- Task B test: Acc={bt['acc']:.4f} ROC-AUC={bt['roc_auc']:.4f}\n"
        f"- best_depth regression={best_depth_r}, classification={best_depth_c}\n",
        encoding="utf-8",
    )
    print(f"  saved → {out / 'metrics.json'}")
    print(f"  saved → {out / 'metrics.md'}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/processed",
                        help="directory with .npy arrays from data_prep.py")
    parser.add_argument("--out", default="reports/figures",
                        help="output directory for PNG figures")
    parser.add_argument("--synthetic", action="store_true",
                        help="allow synthetic fallback when processed arrays are missing")
    args = parser.parse_args()

    out = Path(args.out)

    # load real data; synthetic only on explicit opt-in
    data = load_real(Path(args.dir))
    if data is None:
        if not args.synthetic:
            print(f"ERROR: no processed arrays in '{args.dir}'. "
                  "Run data_prep.py first (or pass --synthetic for a smoke test).")
            return 2
        print(f"NOTE: using synthetic data (matches preprocessing report stats).")
        data = make_synthetic()
    else:
        print(f"Loaded real data from '{args.dir}'")

    synthetic_tag = " [SYNTHETIC DATA — run data_prep.py for real results]" \
                    if data["synthetic"] else ""
    print(f"Train: {data['X_train'].shape}  "
          f"Valid: {data['X_valid'].shape}  "
          f"Test: {data['X_test'].shape}{synthetic_tag}\n")

    best_depth_r, best_depth_c = fig_bias_variance(data, out)
    fig_predictions(data, best_depth_r, out)
    fig_classification(data, best_depth_c, out)
    results_r, results_c = fig_model_comparison(data, best_depth_r, best_depth_c, out)
    fig_error_analysis(data, best_depth_r, best_depth_c, out)
    write_metrics(data, best_depth_r, best_depth_c, results_r, results_c, out)

    print(f"\nAll figures saved to {out.resolve()}")
    print("Replace the sklearn DT and LinearSVR/LinearSVC with your "
          "from-scratch DecisionTree and PegasosSVM to get the real project plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())