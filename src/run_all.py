"""Run cleaning, model selection, cross-validation, final test and reporting."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             roc_auc_score)
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as SKTreeC
from sklearn.tree import DecisionTreeRegressor as SKTreeR

from .data_prep import SEED, prepare_data
from .explore import explore_raw
from .decision_tree import DecisionTree
from .evaluate import classification_metrics, regression_metrics
from .svm import PegasosSVM, RandomFourierPegasosSVM
from .tree_viz import plot_tree
from .reporting import write_reports

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def _score_regression(y, pred):
    return regression_metrics(y, pred)


def _tree(task, criterion, depth, leaf):
    return DecisionTree(task=task, criterion=criterion, max_depth=depth,
                        min_samples_leaf=leaf, min_samples_split=max(2, 2 * leaf))


def _sk_tree(task, criterion, depth, leaf):
    if task == "regression":
        return SKTreeR(criterion="squared_error", max_depth=depth,
                       min_samples_leaf=leaf, min_samples_split=max(2, 2*leaf), random_state=SEED)
    return SKTreeC(criterion=criterion, max_depth=depth,
                   min_samples_leaf=leaf, min_samples_split=max(2, 2*leaf), random_state=SEED)


def _fit_time(model, X, y):
    start = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - start


def _cv_tree(X, y, task, criterion, depth, leaf, seed=SEED):
    # Use every available training row for the uncertainty estimate. The
    # separately held validation set still selects hyperparameters; test is
    # never included in CV.
    n = len(y)
    xx, yy = np.asarray(X), np.asarray(y)
    splitter = (StratifiedKFold(5, shuffle=True, random_state=seed) if task == "classification"
                else KFold(5, shuffle=True, random_state=seed))
    ours, reference = [], []
    ours_mae_azn, ref_mae_azn = [], []
    ours_rmse_azn, ref_rmse_azn = [], []
    ours_r2_azn, ref_r2_azn = [], []
    for tr, va in splitter.split(xx, yy):
        m = _tree(task, criterion, depth, leaf).fit(xx[tr], yy[tr])
        sk = _sk_tree(task, criterion, depth, leaf).fit(xx[tr], yy[tr])
        po, ps = m.predict(xx[va]), sk.predict(xx[va])
        if task == "classification":
            ours.append(float(accuracy_score(yy[va], po)))
            reference.append(float(accuracy_score(yy[va], ps)))
        else:
            ours.append(float(np.sqrt(mean_squared_error(yy[va], po))))
            reference.append(float(np.sqrt(mean_squared_error(yy[va], ps))))
            actual_azn = np.exp(yy[va])
            ours_azn = np.exp(np.clip(po, 0, 25))
            ref_azn = np.exp(np.clip(ps, 0, 25))
            ours_metrics = regression_metrics(actual_azn, ours_azn)
            ref_metrics = regression_metrics(actual_azn, ref_azn)
            ours_mae_azn.append(ours_metrics["mae"])
            ref_mae_azn.append(ref_metrics["mae"])
            ours_rmse_azn.append(ours_metrics["rmse"])
            ref_rmse_azn.append(ref_metrics["rmse"])
            ours_r2_azn.append(ours_metrics["r2"])
            ref_r2_azn.append(ref_metrics["r2"])
    metric = "accuracy" if task == "classification" else "rmse_log"
    result = {"metric": metric, "sample_n": n, "fold_count": 5,
              "scratch": {"fold_scores": ours, "mean": float(np.mean(ours)), "std": float(np.std(ours))},
              "sklearn": {"fold_scores": reference, "mean": float(np.mean(reference)), "std": float(np.std(reference))}}
    if task == "regression":
        result["azn_metrics"] = {
            "scratch_mae": {"fold_scores": ours_mae_azn, "mean": float(np.mean(ours_mae_azn)), "std": float(np.std(ours_mae_azn))},
            "sklearn_mae": {"fold_scores": ref_mae_azn, "mean": float(np.mean(ref_mae_azn)), "std": float(np.std(ref_mae_azn))},
            "scratch_rmse": {"fold_scores": ours_rmse_azn, "mean": float(np.mean(ours_rmse_azn)), "std": float(np.std(ours_rmse_azn))},
            "sklearn_rmse": {"fold_scores": ref_rmse_azn, "mean": float(np.mean(ref_rmse_azn)), "std": float(np.std(ref_rmse_azn))},
            "scratch_r2": {"fold_scores": ours_r2_azn, "mean": float(np.mean(ours_r2_azn)), "std": float(np.std(ours_r2_azn))},
            "sklearn_r2": {"fold_scores": ref_r2_azn, "mean": float(np.mean(ref_r2_azn)), "std": float(np.std(ref_r2_azn))},
        }
    return result


def _cv_svm(X, y, lam, seed=SEED):
    # Use the full training partition; validation/test remain separate.
    xx, yy = np.asarray(X), np.asarray(y)
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    ours, refs = [], []
    for tr, va in folds.split(xx, yy):
        scaler = StandardScaler().fit(xx[tr])
        xt, xv = scaler.transform(xx[tr]), scaler.transform(xx[va])
        model = PegasosSVM(lambda_param=lam, epochs=12, random_state=seed).fit(xt, yy[tr])
        svc = Pipeline([("scale", StandardScaler()),
                        ("svc", LinearSVC(loss="hinge", C=1/(lam*len(tr)), dual="auto",
                                           max_iter=10000, tol=1e-3, random_state=seed))]).fit(xx[tr], yy[tr])
        ours.append(float(accuracy_score(yy[va], model.predict(xv))))
        # Pipeline owns its scaler; feed raw fold rows so scaling happens once.
        refs.append(float(accuracy_score(yy[va], svc.predict(xx[va]))))
    return {"metric": "accuracy", "sample_n": len(yy), "fold_count": 5, "epochs": 12,
            "scratch": {"fold_scores": ours, "mean": float(np.mean(ours)), "std": float(np.std(ours))},
            "sklearn": {"fold_scores": refs, "mean": float(np.mean(refs)), "std": float(np.std(refs))}}


def _plot_tuning(curves):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [f"d={x['depth_label']}, leaf={x['min_samples_leaf']}" for x in curves["tree_regression"]]
    vals = [x["validation_rmse_log"] for x in curves["tree_regression"]]
    axes[0].clear(); axes[0].bar(np.arange(len(vals)), vals, color="#3977a8"); axes[0].set_xticks(np.arange(len(vals)), labels, rotation=55, ha="right", fontsize=7)
    axes[0].set_title("Tree regression: validation RMSE (log price)"); axes[0].set_ylabel("Lower is better")
    labels = [f"{x['criterion']}\nd={x['depth_label']}\nleaf={x['min_samples_leaf']}" for x in curves["tree_classification"]]
    vals = [x["validation_f1"] for x in curves["tree_classification"]]
    axes[1].bar(np.arange(len(vals)), vals, color="#bd4f3a"); axes[1].set_xticks(np.arange(len(vals)), labels, fontsize=6)
    axes[1].set_title("Tree classification: validation F1"); axes[1].set_ylabel("Higher is better")
    fig.tight_layout(); fig.savefig(FIGURES / "tree_tuning.png", dpi=160); plt.close(fig)


def _learning_curves(d, best_reg, best_cls):
    rng = np.random.default_rng(SEED + 7)
    sizes = sorted(set(min(n, len(d["X_train"])) for n in (1000, 4000, 12000, 24000, len(d["X_train"]))))
    curves = {"regression": [], "classification": []}
    for size in sizes:
        ix = rng.choice(len(d["X_train"]), size, replace=False)
        for task, best, target_key, valid_key, metric in (
            ("regression", best_reg, "y_train_log", "y_validation_log", "rmse"),
            ("classification", best_cls, "tier_train", "tier_validation", "accuracy")):
            m = _tree(task, best["criterion"], best["max_depth"], best["min_samples_leaf"]).fit(d["X_train"][ix], d[target_key][ix])
            pt, pv = m.predict(d["X_train"][ix]), m.predict(d["X_validation"])
            if task == "regression":
                train_score = float(np.sqrt(mean_squared_error(d[target_key][ix], pt)))
                val_score = float(np.sqrt(mean_squared_error(d[valid_key], pv)))
            else:
                train_score = float(accuracy_score(d[target_key][ix], pt))
                val_score = float(accuracy_score(d[valid_key], pv))
            curves[task].append({"n": size, "train": train_score, "validation": val_score, "metric": metric})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task, title, ylabel in ((axes[0], "regression", "Tree learning curve — regression", "RMSE in log-price space"),
                                    (axes[1], "classification", "Tree learning curve — classification", "Accuracy")):
        c = curves[task]; ax.plot([x["n"] for x in c], [x["train"] for x in c], "o-", label="training")
        ax.plot([x["n"] for x in c], [x["validation"] for x in c], "o-", label="validation")
        ax.set(title=title, xlabel="Training rows", ylabel=ylabel); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "tree_learning_curves.png", dpi=160); plt.close(fig)
    return curves


def _make_figures(d, results, final_models):
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Price distribution and cleaning attrition.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(d["price_train"], bins=60, color="#3977a8"); ax[0].set(xscale="log", title="Sale prices, AZN (log axis)", xlabel="Price (AZN)", ylabel="Listings")
    stages = results["data_audit"]; labels = ["Raw", "Unique listing", "Residential", "Valid"]
    vals = [stages["raw_rows"], stages["after_dedup"], stages["after_scope"], stages["final_rows"]]
    ax[1].bar(labels, vals, color=["#8fa4b5", "#3977a8", "#7c9c59", "#bd4f3a"]); ax[1].set(title="Data cleaning attrition", ylabel="Rows")
    fig.tight_layout(); fig.savefig(FIGURES / "data_and_prices.png", dpi=160); plt.close(fig)
    # Measured values vs target with property category colors.
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(d["price_train"]), min(3500, len(d["price_train"])), replace=False)
    categories = d["metadata_train"].iloc[take]["category"].astype("category").cat.codes.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5)); scat = ax.scatter(d["area_train_m2"][take], d["price_train"][take], c=categories, s=10, alpha=.45, cmap="viridis")
    ax.set(xscale="log", yscale="log", xlabel="Area (m², log scale)", ylabel="Sale price (AZN, log scale)", title="Price vs area, colored by property category"); fig.colorbar(scat, ax=ax, label="Category code"); fig.tight_layout(); fig.savefig(FIGURES / "price_vs_area.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    balance_groups = [d["tier_train"], d["tier_validation"], d["tier_test"]]
    standard = [float(np.mean(y == 0)) for y in balance_groups]
    premium = [float(np.mean(y == 1)) for y in balance_groups]
    ax.bar(["Train", "Validation", "Test"], standard, label="Standard", color="#3977a8")
    ax.bar(["Train", "Validation", "Test"], premium, bottom=standard, label="Premium", color="#bd4f3a")
    ax.set(ylim=(0, 1), ylabel="Share", title="Price-tier balance by split"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "class_balance.png", dpi=160); plt.close(fig)
    missing = d["missing_feature_percent"]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(list(missing), list(missing.values()), color="#7c9c59")
    ax.set(xlabel="Missing before imputation (%)", title="Numeric feature missingness"); fig.tight_layout(); fig.savefig(FIGURES / "missingness.png", dpi=160); plt.close(fig)
    room_values = d["rooms_train"]
    room_numbers = sorted(set(int(x) for x in room_values if 1 <= x <= 8))
    keep = [(n, d["price_train"][room_values == n]) for n in room_numbers]
    keep = [(n, values) for n, values in keep if len(values)]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.boxplot([x[1] for x in keep], tick_labels=[str(x[0]) for x in keep], showfliers=False)
    ax.set(yscale="log", xlabel="Number of rooms", ylabel="Sale price (AZN, log scale)", title="Training price by room count"); fig.tight_layout(); fig.savefig(FIGURES / "price_vs_rooms.png", dpi=160); plt.close(fig)
    location_values = d["metadata_train"]["location"].astype(str)
    locations = []
    for name, positions in location_values.groupby(location_values).groups.items():
        positions = np.asarray(list(positions), dtype=int)
        if len(positions) >= 100:
            locations.append((name, len(positions), float(np.median(d["price_train"][positions]))))
    locations = sorted(locations, key=lambda x: x[2], reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh([x[0] for x in locations][::-1], [x[2] for x in locations][::-1], color="#bd4f3a")
    ax.set(xlabel="Median advertised price (AZN)", title="Highest-median training locations (n ≥ 100)"); fig.tight_layout(); fig.savefig(FIGURES / "location_prices.png", dpi=160); plt.close(fig)
    # Best tree/SVM confusion matrices.
    for key, title, filename in (("decision_tree_classification", "Scratch decision tree", "tree_confusion_matrix.png"),
                                 ("pegasos_svm", "Scratch Pegasos SVM", "pegasos_confusion_matrix.png"),
                                 ("rff_pegasos_svm", "Random Fourier Pegasos", "rff_confusion_matrix.png")):
        cm = results["test_results"][key]["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4.4, 3.8)); ax.imshow(cm, cmap="Blues"); ax.set(title=title + " — test", xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1], xticklabels=["Standard", "Premium"], yticklabels=["Standard", "Premium"])
        for (i, j), value in np.ndenumerate(cm): ax.text(j, i, f"{value:,}", ha="center", va="center")
        fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=160); plt.close(fig)
    # Test ROC comparisons.
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ("decision_tree_classification", "sklearn_tree_classification", "pegasos_svm", "linear_svc", "rff_pegasos_svm", "rbf_svc"):
        roc = results["test_results"][name].get("roc_curve")
        if roc:
            ax.plot(roc["fpr"], roc["tpr"], label=f"{name} AUC={results['test_results'][name]['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Price-tier ROC curves on the held-out test set"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "roc_curves.png", dpi=160); plt.close(fig)
    # Test accuracy and F1, matched two tasks where applicable.
    names = ["Scratch tree", "sklearn tree", "Pegasos", "Linear SVC", "RFF Pegasos", "RBF SVC"]
    keys = ["decision_tree_classification", "sklearn_tree_classification", "pegasos_svm", "linear_svc", "rff_pegasos_svm", "rbf_svc"]
    xx = np.arange(len(keys)); width = .36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xx-width/2, [results["test_results"][k]["accuracy"] for k in keys], width, label="Accuracy")
    ax.bar(xx+width/2, [results["test_results"][k]["f1"] for k in keys], width, label="F1")
    ax.set(ylim=(0, 1), ylabel="Score", title="Classification test scores"); ax.set_xticks(xx, names, rotation=20, ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "classification_comparison.png", dpi=160); plt.close(fig)
    # Pegasos objective history and margin score distribution.
    svm = final_models["pegasos_svm"]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(np.arange(1, len(svm.objective_history_)+1), svm.objective_history_, "o-", ms=3)
    ax.set(xlabel="Epoch", ylabel="Regularized hinge objective", title="Pegasos objective by epoch"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "pegasos_objective.png", dpi=160); plt.close(fig)
    scores = svm.decision_function(final_models["X_test_scaled"]); labels = d["tier_test"]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(scores[labels == 0], bins=45, alpha=.65, label="Standard"); ax.hist(scores[labels == 1], bins=45, alpha=.65, label="Premium"); ax.axvline(0, color="black", ls="--"); ax.set(title="Pegasos decision scores and margin boundary", xlabel="w·x + b", ylabel="Listings"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "svm_margin_scores.png", dpi=160); plt.close(fig)
    # Error diagnosis for the final regression model.
    pred_log = final_models["tree_regression"].predict(d["X_test"])
    true_price = d["price_test"]; pred_price = np.exp(np.clip(pred_log, 0, 25))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    take = rng.choice(len(true_price), min(3000, len(true_price)), replace=False)
    ax[0].scatter(true_price[take], pred_price[take], s=9, alpha=.35); lo, hi = np.quantile(true_price[take], [.01, .99]); ax[0].plot([lo, hi], [lo, hi], "r--"); ax[0].set(xscale="log", yscale="log", xlabel="Actual price (AZN)", ylabel="Predicted price (AZN)", title="Tree price predictions")
    err = np.log(true_price) - pred_log; ax[1].scatter(pred_log[take], err[take], s=9, alpha=.35); ax[1].axhline(0, color="red", ls="--"); ax[1].set(xlabel="Predicted ln(price)", ylabel="Actual − predicted ln(price)", title="Regression residuals")
    fig.tight_layout(); fig.savefig(FIGURES / "regression_error_analysis.png", dpi=160); plt.close(fig)
    # Feature importance and readable top-three-level tree.
    for key, model in (("regression", final_models["tree_regression"]), ("classification", final_models["tree_classification"])):
        imp = model.feature_importances_; ix = np.argsort(imp)[-15:]
        fig, ax = plt.subplots(figsize=(7.5, 5)); ax.barh([d["feature_names"][i] for i in ix], imp[ix], color="#3977a8"); ax.set(title=f"Decision-tree {key} feature importance", xlabel="Normalized weighted impurity decrease"); fig.tight_layout(); fig.savefig(FIGURES / f"feature_importance_{key}.png", dpi=160); plt.close(fig)
    small = _tree("classification", "gini", 3, 30).fit(d["X_train"], d["tier_train"])
    plot_tree(small, feature_names=d["feature_names"], X=d["X_train"], y=d["tier_train"], save_path=str(FIGURES / "decision_tree.png"), fontsize=7)


def _with_roc(y, pred, scores):
    from sklearn.metrics import roc_curve
    out = classification_metrics(y, pred, scores)
    fpr, tpr, _ = roc_curve(y, scores)
    step = max(1, len(fpr) // 200)
    indices = list(range(0, len(fpr), step))
    if indices[-1] != len(fpr) - 1:
        indices.append(len(fpr) - 1)
    out["roc_curve"] = {"fpr": fpr[indices].tolist(), "tpr": tpr[indices].tolist()}
    return out


def run():
    REPORTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    explore_raw(output_dir=REPORTS / "eda")
    d = prepare_data()
    print(f"Prepared {d['audit']['final_rows']:,} listings; tuning trees...", flush=True)
    Xtr, Xv, Xtest = d["X_train"], d["X_validation"], d["X_test"]
    curves = {"tree_regression": [], "tree_classification": [], "svm_linear": [], "svm_rff": []}

    # Task A: tune tree depth and minimum leaf count using validation RMSE.
    regression_candidates = []
    for depth in (4, 8, 12, None):
        for leaf in (5, 20):
            model, fit_s = _fit_time(_tree("regression", "mse", depth, leaf), Xtr, d["y_train_log"])
            pred = model.predict(Xv); score = float(np.sqrt(mean_squared_error(d["y_validation_log"], pred)))
            rec = {"criterion": "mse", "max_depth": depth, "depth_label": depth or "None", "min_samples_leaf": leaf,
                   "validation_rmse_log": score, "validation_mae_log": regression_metrics(d["y_validation_log"], pred)["mae"], "fit_seconds": fit_s}
            curves["tree_regression"].append(rec); regression_candidates.append((score, rec))
    best_reg = min(regression_candidates, key=lambda item: item[0])[1]

    # Task B: tune Gini vs entropy, depth and leaf size by validation F1.
    classification_candidates = []
    for criterion in ("gini", "entropy"):
        for depth in (4, 8, 12):
            for leaf in (5, 20):
                model, fit_s = _fit_time(_tree("classification", criterion, depth, leaf), Xtr, d["tier_train"])
                pred = model.predict(Xv); scores = model.predict_proba(Xv)[:, 1]
                met = classification_metrics(d["tier_validation"], pred, scores)
                rec = {"criterion": criterion, "max_depth": depth, "depth_label": depth,
                       "min_samples_leaf": leaf, "validation_accuracy": met["accuracy"],
                       "validation_f1": met["f1"], "validation_roc_auc": met["roc_auc"], "fit_seconds": fit_s}
                curves["tree_classification"].append(rec); classification_candidates.append((met["f1"], rec))
    best_cls = max(classification_candidates, key=lambda item: item[0])[1]
    print("Tree validation sweep complete; tuning linear Pegasos...", flush=True)

    # SVM scaling is learned from train only. Sweep lambda on a fixed train subset,
    # select on validation, then refit the chosen model on every training row.
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xv_s, Xtest_s = (scaler.transform(x) for x in (Xtr, Xv, Xtest))
    rng = np.random.default_rng(SEED)
    svm_tune_idx = rng.choice(len(Xtr_s), min(12000, len(Xtr_s)), replace=False)
    linear_candidates = []
    for lam in (.001, .01, .1):
        model, fit_s = _fit_time(PegasosSVM(lambda_param=lam, epochs=15, random_state=SEED), Xtr_s[svm_tune_idx], d["tier_train"][svm_tune_idx])
        score = model.decision_function(Xv_s); pred = model.predict(Xv_s)
        met = classification_metrics(d["tier_validation"], pred, score)
        rec = {"lambda": lam, "epochs": 15, "validation_accuracy": met["accuracy"], "validation_f1": met["f1"], "validation_roc_auc": met["roc_auc"], "fit_seconds": fit_s, "tuning_train_n": len(svm_tune_idx)}
        curves["svm_linear"].append(rec); linear_candidates.append((met["f1"], rec))
    best_linear = max(linear_candidates, key=lambda item: item[0])[1]
    print("Linear Pegasos sweep complete; tuning nonlinear RFF Pegasos...", flush=True)

    # Statement's nonlinear choice: RBF random Fourier features + our same Pegasos.
    rff_idx = rng.choice(len(Xtr_s), min(8000, len(Xtr_s)), replace=False)
    rff_scaler = StandardScaler().fit(Xtr[rff_idx])
    Xrff_train, Xrff_valid, Xrff_test = (rff_scaler.transform(x) for x in (Xtr[rff_idx], Xv, Xtest))
    rff_candidates = []
    for gamma in (.01, .1, 1.0):
        model, fit_s = _fit_time(RandomFourierPegasosSVM(lambda_param=best_linear["lambda"], epochs=10,
                                    n_components=128, gamma=gamma, random_state=SEED), Xrff_train, d["tier_train"][rff_idx])
        pred, scores = model.predict(Xrff_valid), model.decision_function(Xrff_valid)
        met = classification_metrics(d["tier_validation"], pred, scores)
        rec = {"gamma": gamma, "n_components": 128, "epochs": 10, "lambda": best_linear["lambda"],
               "validation_accuracy": met["accuracy"], "validation_f1": met["f1"], "validation_roc_auc": met["roc_auc"], "fit_seconds": fit_s, "tuning_train_n": len(rff_idx)}
        curves["svm_rff"].append(rec); rff_candidates.append((met["f1"], rec))
    best_rff = max(rff_candidates, key=lambda item: item[0])[1]
    print("RFF sweep complete; fitting final models and matched baselines...", flush=True)

    # Final model refits on train only; test data is used once from this point on.
    reg, reg_fit = _fit_time(_tree("regression", "mse", best_reg["max_depth"], best_reg["min_samples_leaf"]), Xtr, d["y_train_log"])
    cls, cls_fit = _fit_time(_tree("classification", best_cls["criterion"], best_cls["max_depth"], best_cls["min_samples_leaf"]), Xtr, d["tier_train"])
    sk_reg, sk_reg_fit = _fit_time(_sk_tree("regression", "mse", best_reg["max_depth"], best_reg["min_samples_leaf"]), Xtr, d["y_train_log"])
    sk_cls, sk_cls_fit = _fit_time(_sk_tree("classification", best_cls["criterion"], best_cls["max_depth"], best_cls["min_samples_leaf"]), Xtr, d["tier_train"])
    svm, svm_fit = _fit_time(PegasosSVM(lambda_param=best_linear["lambda"], epochs=15, random_state=SEED), Xtr_s, d["tier_train"])
    lam = best_linear["lambda"]
    svc, svc_fit = _fit_time(Pipeline([("scale", StandardScaler()),
                                      ("svc", LinearSVC(loss="hinge", C=1/(lam*len(d["tier_train"])),
                                                        dual="auto", max_iter=10000, tol=1e-3, random_state=SEED))]), Xtr, d["tier_train"])
    rff, rff_fit = _fit_time(RandomFourierPegasosSVM(lambda_param=best_rff["lambda"], epochs=12,
                           n_components=128, gamma=best_rff["gamma"], random_state=SEED), Xrff_train, d["tier_train"][rff_idx])
    rbf, rbf_fit = _fit_time(Pipeline([("scale", StandardScaler()),
                                      ("svc", SVC(kernel="rbf", C=1/(lam*len(rff_idx)), gamma=best_rff["gamma"]))]), Xtr[rff_idx], d["tier_train"][rff_idx])
    print("Final models fitted; computing held-out scores and CV...", flush=True)

    # Collect only final held-out test results here.
    test_metrics = {}
    regression_models = {"decision_tree_regression": reg, "sklearn_tree_regression": sk_reg}
    regression_fit = {"decision_tree_regression": reg_fit, "sklearn_tree_regression": sk_reg_fit}
    regression_preds = {}
    for name, model in regression_models.items():
        start = time.perf_counter(); pred_log = model.predict(Xtest); predict_s = time.perf_counter() - start
        regression_preds[name] = pred_log
        met = regression_metrics(d["y_test_log"], pred_log)
        pred_price = np.exp(np.clip(pred_log, 0, 25))
        raw = regression_metrics(d["price_test"], pred_price)
        absolute_error = np.abs(d["price_test"] - pred_price)
        worst_index = int(np.argmax(absolute_error))
        worst_row = d["metadata_test"].iloc[worst_index]
        met.update({"rmse_AZN": raw["rmse"], "mae_AZN": raw["mae"], "r2_AZN": raw["r2"],
                    "median_absolute_error_AZN": float(np.median(absolute_error)),
                    "max_absolute_error_AZN": float(absolute_error[worst_index]),
                    "max_error_case": {
                        "actual_AZN": float(d["price_test"][worst_index]),
                        "predicted_AZN": float(pred_price[worst_index]),
                        "category": str(worst_row["category"]),
                        "area_m2": float(worst_row["area_m2"]),
                        "rooms": float(worst_row["rooms"]) if pd.notna(worst_row["rooms"]) else None,
                        "location": str(worst_row["location"]),
                    },
                    "fit_seconds": regression_fit[name], "predict_seconds": predict_s})
        test_metrics[name] = met
    classification_models = {"decision_tree_classification": (cls, Xtest),
                              "sklearn_tree_classification": (sk_cls, Xtest),
                              "pegasos_svm": (svm, Xtest_s), "linear_svc": (svc, Xtest),
                              "rff_pegasos_svm": (rff, Xrff_test), "rbf_svc": (rbf, Xtest)}
    fit_times = {"decision_tree_classification": cls_fit, "sklearn_tree_classification": sk_cls_fit,
                 "pegasos_svm": svm_fit, "linear_svc": svc_fit, "rff_pegasos_svm": rff_fit, "rbf_svc": rbf_fit}
    for name, (model, matrix) in classification_models.items():
        start = time.perf_counter()
        if name.startswith("decision_tree") or name == "pegasos_svm" or name == "rff_pegasos_svm":
            scores = model.decision_function(matrix) if name != "decision_tree_classification" else model.predict_proba(matrix)[:, 1]
            pred = model.predict(matrix)
        elif name.startswith("sklearn_tree"):
            scores = model.predict_proba(matrix)[:, 1]; pred = model.predict(matrix)
        else:
            scores = model.decision_function(matrix); pred = model.predict(matrix)
        pred_s = time.perf_counter() - start
        test_metrics[name] = _with_roc(d["tier_test"], pred, scores)
        test_metrics[name].update({"fit_seconds": fit_times[name], "predict_seconds": pred_s})
    test_metrics["pegasos_svm"].update({"weight_norm": float(np.linalg.norm(svm.w)), "margin_width": svm.margin_width_,
                                         "support_vector_count": int(svm.support_vector_mask_.sum()), "train_n": len(d["tier_train"])})
    test_metrics["rff_pegasos_svm"].update({"weight_norm": float(np.linalg.norm(rff.model_.w)), "margin_width": rff.margin_width_,
                                             "support_vector_count": rff.n_support_vectors_, "train_n": len(rff_idx),
                                             "rff_components": rff.n_components, "gamma": rff.gamma})
    linear_coef = svc.named_steps["svc"].coef_.ravel()
    linear_margin = svc.decision_function(Xtr)
    linear_sign = np.where(d["tier_train"] == svc.classes_[1], 1.0, -1.0)
    test_metrics["linear_svc"].update({"train_n": len(d["tier_train"]),
                                       "weight_norm": float(np.linalg.norm(linear_coef)),
                                       "margin_width": float(2/np.linalg.norm(linear_coef)),
                                       "support_vector_count": int(np.sum(linear_sign*linear_margin <= 1.0))})
    test_metrics["rbf_svc"].update({"support_vector_count": int(rbf.named_steps["svc"].n_support_.sum()), "train_n": len(rff_idx)})

    # CV runs on a seeded 6k-row sample of training data to provide a repeatable
    # uncertainty estimate without making the two-week student project unusably slow.
    best_reg_cv = _cv_tree(Xtr, d["y_train_log"], "regression", "mse", best_reg["max_depth"], best_reg["min_samples_leaf"])
    best_cls_cv = _cv_tree(Xtr, d["tier_train"], "classification", best_cls["criterion"], best_cls["max_depth"], best_cls["min_samples_leaf"])
    svm_cv = _cv_svm(Xtr, d["tier_train"], lam)
    print("Cross-validation complete; generating figures and reports...", flush=True)
    cv_results = {"decision_tree_regression": best_reg_cv, "decision_tree_classification": best_cls_cv,
                  "pegasos_svm": svm_cv}

    # Explicit validation comparisons with matched baseline hyperparameters.
    val_results = {
        "decision_tree_regression": {"scratch": regression_metrics(d["y_validation_log"], reg.predict(Xv)), "sklearn": regression_metrics(d["y_validation_log"], sk_reg.predict(Xv))},
        "decision_tree_classification": {"scratch": classification_metrics(d["tier_validation"], cls.predict(Xv), cls.predict_proba(Xv)[:, 1]), "sklearn": classification_metrics(d["tier_validation"], sk_cls.predict(Xv), sk_cls.predict_proba(Xv)[:, 1])},
        "pegasos_svm": {"scratch": classification_metrics(d["tier_validation"], svm.predict(Xv_s), svm.decision_function(Xv_s)), "sklearn_linear_svc": classification_metrics(d["tier_validation"], svc.predict(Xv), svc.decision_function(Xv))},
        "rff_pegasos_svm": {"scratch": classification_metrics(d["tier_validation"], rff.predict(Xrff_valid), rff.decision_function(Xrff_valid)), "sklearn_rbf": classification_metrics(d["tier_validation"], rbf.predict(Xv), rbf.decision_function(Xv))}}

    # Error slices, learned feature rankings, and CV/learning diagnostics.
    err_log = np.log(d["price_test"]) - regression_preds["decision_tree_regression"]
    quartile = np.digitize(d["price_test"], np.quantile(d["price_train"], [.25, .5, .75]))
    subgroup = []
    for q in range(4):
        mask = quartile == q
        subgroup.append({"slice": f"training-price-quartile-{q+1}", "n": int(mask.sum()),
                         "median_actual_AZN": float(np.median(d["price_test"][mask])),
                         "mae_AZN": float(np.mean(np.abs(d["price_test"][mask] - np.exp(np.clip(regression_preds["decision_tree_regression"][mask], 0, 25))))),
                         "mae_log": float(np.mean(np.abs(err_log[mask])))})
    for category in sorted(d["metadata_test"]["category"].dropna().unique()):
        mask = d["metadata_test"]["category"].to_numpy() == category
        subgroup.append({"slice": f"property:{category}", "n": int(mask.sum()),
                         "mae_AZN": float(np.mean(np.abs(d["price_test"][mask] - np.exp(np.clip(regression_preds["decision_tree_regression"][mask], 0, 25))))),
                         "mae_log": float(np.mean(np.abs(err_log[mask])))})
    importance = {}
    for key, model in (("regression", reg), ("classification", cls)):
        indices = np.argsort(model.feature_importances_)[::-1][:20]
        importance[key] = [{"feature": d["feature_names"][i], "importance": float(model.feature_importances_[i])} for i in indices]
    numeric_corr = []
    for i, name in enumerate(d["feature_names"][:d["numeric_feature_count"]]):
        corr = float(np.corrcoef(Xtr[:, i], d["y_train_log"])[0, 1])
        numeric_corr.append({"feature": name, "pearson_r_with_log_price": corr if np.isfinite(corr) else 0.0})
    numeric_corr.sort(key=lambda x: abs(x["pearson_r_with_log_price"]), reverse=True)
    location_summary = []
    loc_values = d["metadata_train"]["location"].astype(str)
    for name, positions in loc_values.groupby(loc_values).groups.items():
        positions = np.asarray(list(positions), dtype=int)
        if len(positions) >= 100:
            location_summary.append({"location": name, "n": int(len(positions)),
                                     "median_price_AZN": float(np.median(d["price_train"][positions]))})
    location_summary.sort(key=lambda x: x["median_price_AZN"], reverse=True)

    results = {"seed": SEED, "data_audit": d["audit"],
               "split": {"train": len(Xtr), "validation": len(Xv), "test": len(Xtest)},
               "tier_threshold_AZN": d["tier_threshold"],
               "feature_count": len(d["feature_names"]), "numeric_feature_count": d["numeric_feature_count"],
               "feature_names": d["feature_names"], "leakage_columns_excluded": d["audit"]["leakage_columns_excluded"],
               "tuning": curves,
               "selected": {"tree_regression": best_reg, "tree_classification": best_cls,
                            "linear_svm": best_linear, "rff_svm": best_rff},
               "validation_comparison": val_results, "cross_validation": cv_results,
               "test_results": test_metrics, "feature_importance": importance,
               "regression_error_slices": subgroup,
               "exploratory_analysis": {"missing_numeric_percent": d["missing_feature_percent"],
                                        "tier_prevalence": {"train": float(d["tier_train"].mean()), "validation": float(d["tier_validation"].mean()), "test": float(d["tier_test"].mean())},
                                        "numeric_correlations_with_log_price": numeric_corr,
                                        "locations_min_100_train_rows_top_median": location_summary[:10]},
               "svm_diagnostics": {"pegasos_objective_by_epoch": [float(x) for x in svm.objective_history_],
                                   "rff_objective_by_epoch": [float(x) for x in rff.objective_history_],
                                   "pegasos_train_sample_count": len(d["tier_train"]),
                                   "rff_train_sample_count": len(rff_idx)}}
    learning = _learning_curves(d, best_reg, best_cls)
    results["learning_curves"] = learning
    _plot_tuning(curves)
    _make_figures(d, results, {"pegasos_svm": svm, "X_test_scaled": Xtest_s,
                                "tree_regression": reg, "tree_classification": cls})
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_reports(results, d, REPORTS, FIGURES)
    print(f"Cleaned {d['audit']['final_rows']:,} listings; reports: reports/final_report.md and report/report.pdf")
    for name, met in test_metrics.items():
        if "accuracy" in met: print(f"{name:32s} test accuracy={met['accuracy']:.4f} F1={met['f1']:.4f} AUC={met['roc_auc']:.4f}")
        else: print(f"{name:32s} test RMSE(log)={met['rmse']:.4f}; RMSE(AZN)={met['rmse_AZN']:,.0f}")
    return results


if __name__ == "__main__":
    run()
