"""
Tuned Gradient Boosting & HistGradientBoosting for LOB graph classification
with optional outer cross-validation.

- Loads node/edge CSVs and builds one feature vector per LOB (line of business).
- Creates binary labels based on how many "high" nodes each LOB has.
- Tries two tree-based models:
    * Model A: GradientBoostingClassifier
    * Model B: HistGradientBoostingClassifier
- Uses RandomizedSearchCV to tune hyperparameters for both models.
- Optionally runs full outer cross-validation, or a single train/val/test split.
- Tunes a probability threshold for classification (not just 0.5).
- Saves metrics, plots, and feature importances (single-split mode).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


# Configuration handling


@dataclass
class Config:
    """
    Simple container for all runtime options.

    Most values can be controlled via environment variables so one can
    re-run without touching the code.
    """
    artifacts_dir: str
    use_noisy: bool
    outdir: str
    use_cv: bool
    cv_folds: int
    inner_val: float
    seed: int

    @classmethod
    def from_env(cls) -> "Config":
        """
        Read settings from environment variables,
        create output directory if needed, and return a Config instance.
        """
        # Where to find the raw node/edge CSVs
        art = os.environ.get("ARTIFACTS", os.path.join("..", "Data Generation"))

        # Whether to use *_noisy.csv files instead of the clean ones
        use_noisy = os.environ.get("USE_NOISY", "0") == "1"

        # Where to save metrics and plots
        outdir = os.environ.get("OUTDIR", os.path.join(art, "baseline_metrics"))

        # Whether to run full outer cross-validation or just a single split
        use_cv = os.environ.get("USE_CV", "0") == "1"
        cv_folds = int(os.environ.get("CV_FOLDS", 5))

        # Fraction of training data held out as a small inner validation set
        inner_val = float(os.environ.get("INNER_VAL", 0.2))

        # Global random seed used for splits and model tuning
        seed = int(os.environ.get("SEED", 42))

        os.makedirs(outdir, exist_ok=True)

        return cls(
            artifacts_dir=art,
            use_noisy=use_noisy,
            outdir=outdir,
            use_cv=use_cv,
            cv_folds=cv_folds,
            inner_val=inner_val,
            seed=seed,
        )


# Small numeric helper functions

def safe_mean(x) -> float:
    """Compute a mean that tolerates empty input and NaNs."""
    s = pd.Series(x, dtype=float)
    return float(s.mean(skipna=True)) if len(s) else np.nan


def safe_std(x) -> float:
    """Compute a standard deviation that tolerates empty input and NaNs."""
    s = pd.Series(x, dtype=float)
    return float(s.std(skipna=True)) if len(s) else np.nan


def label_entropy(sub_labels: pd.Series) -> float:
    """
    Compute entropy over labels {'high','medium','low'}.

    Intuition:
    - Entropy is low if a LOB is almost all one label.
    - Entropy is high if labels are more mixed.
    """
    counts = pd.Series(sub_labels).value_counts()
    total = counts.sum()
    if total == 0:
        return 0.0
    p = (
        counts[["high", "medium", "low"]]
        .reindex(["high", "medium", "low"])
        .fillna(0)
        .values.astype(float)
        / total
    )
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Pick a probability threshold that maximizes F1 on a validation set.

    Instead of always using 0.5, scan possible thresholds and pick the one
    that gives the best precision/recall tradeoff (F1).
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
    idx = int(np.nanargmax(f1s))
    # precision_recall_curve returns thresholds of length len(prec)-1
    best_thr = thr[idx - 1] if 0 < idx < len(thr) else 0.5
    return float(best_thr), float(f1s[idx])


def eval_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    """
    Evaluate standard metrics at a given probability threshold.

    Returns a small dict of:
      - ROC AUC, Average Precision
      - Accuracy, F1, Precision, Recall
    """
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        roc_auc=roc_auc_score(y_true, y_prob),
        avg_prec=average_precision_score(y_true, y_prob),
        accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
    )


# Data loading & LOB-level feature engineering

def load_graph_data(cfg: Config):
    """
    Load node/edge tables and build one feature vector per LOB.

    High-level steps:
      1. Read nodes.csv and edges.csv (or noisy versions).
      2. Clean types / enforce basic schema.
      3. Build a parent->children map from 'parent_of' edges.
      4. For each LOB node:
           - Find all descendants.
           - Compute summary stats and label fractions.
      5. Turn 'fraction of high nodes' into a binary label via an adaptive rule.

    Returns
    -------
    X : np.ndarray
        Feature matrix (one row per LOB).
    y : np.ndarray
        Binary labels for each LOB.
    high_fracs : np.ndarray
        Fraction of 'high' nodes for each LOB.
    lob_ids : np.ndarray
        IDs of LOB nodes used.
    used_thresh : float
        Threshold used to define y from high_fracs.
    thresh_method : str
        How that threshold was chosen ("fixed>0.10", "median", or "q75").
    nodes_file, edges_file : str
        Names of the CSV files actually used (for logging/metrics).
    """
    nodes_file = "nodes_noisy.csv" if cfg.use_noisy else "nodes.csv"
    edges_file = "edges_noisy.csv" if cfg.use_noisy else "edges.csv"

    print(f"ART={cfg.artifacts_dir}")
    print(f"Using files: nodes={nodes_file}, edges={edges_file}")
    print(
        f"OUTDIR={cfg.outdir} | USE_CV={cfg.use_cv} | "
        f"CV_FOLDS={cfg.cv_folds} | INNER_VAL={cfg.inner_val}"
    )

    # Read raw node and edge tables
    nodes = pd.read_csv(os.path.join(cfg.artifacts_dir, nodes_file), low_memory=False)
    edges = pd.read_csv(os.path.join(cfg.artifacts_dir, edges_file), low_memory=False)

    # Basic type cleanup: make sure IDs and numeric columns are numeric
    for col in ("id", "deg", "level"):
        if col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce")
    for col in ("source", "target"):
        if col in edges.columns:
            edges[col] = pd.to_numeric(edges[col], errors="coerce")

    nodes = nodes.dropna(subset=["id"])
    nodes["id"] = nodes["id"].astype(int)

    # Check that required columns are present
    required_nodes = {"type_name", "label"}
    missing_nodes = required_nodes - set(nodes.columns)
    if missing_nodes:
        raise SystemExit(f"Missing columns in nodes: {missing_nodes}")

    required_edges = {"source", "target", "note"}
    missing_edges = required_edges - set(edges.columns)
    if missing_edges:
        raise SystemExit(f"Missing columns in edges: {missing_edges}")

    edges = edges.dropna(subset=["source", "target"])
    edges["source"] = edges["source"].astype(int)
    edges["target"] = edges["target"].astype(int)

    # Ensure degree and level exist, even if missing from the input schema
    if "deg" not in nodes.columns:
        nodes["deg"] = np.nan
    if "level" not in nodes.columns:
        nodes["level"] = np.nan

    # Build a parent -> list_of_children lookup from parent_of edges
    child_by_parent: Dict[int, List[int]] = {}
    po = edges[edges["note"] == "parent_of"][["source", "target"]].values
    for s, t in po:
        child_by_parent.setdefault(int(s), []).append(int(t))

    def descendants(root: int) -> set[int]:
        """
        Get all descendants (children, grandchildren, etc.) for one root node.

        This is a simple DFS over the parent_of edges.
        """
        stack = [root]
        seen = {root}
        out: set[int] = set()
        while stack:
            u = stack.pop()
            for v in child_by_parent.get(u, []):
                if v not in seen:
                    seen.add(v)
                    out.add(v)
                    stack.append(v)
        return out

    # List of LOB-level features we will compute
    feature_names = [
        "num_desc",
        "label_entropy",
        "deg_mean",
        "deg_std",
        "level_mean",
        "level_std",
        "high_frac",
        "med_frac",
        "low_frac",
    ]

    X, high_fracs = [], []

    # Only treat nodes tagged as LOB as "graph roots" for this model
    lob_ids = nodes.loc[nodes["type_name"] == "LOB", "id"].astype(int).values

    # Build feature vector per LOB by summarizing its descendants
    for lob in lob_ids:
        desc = descendants(lob)
        if not desc:
            # Skip LOBs with no descendants
            continue
        sub = nodes[nodes["id"].isin(desc)]

        num_desc = len(sub)
        deg_mean = safe_mean(sub["deg"])
        deg_std_ = safe_std(sub["deg"])
        level_mean = safe_mean(sub["level"].replace(-1, np.nan))
        level_std_ = safe_std(sub["level"].replace(-1, np.nan))

        high_frac = (sub["label"] == "high").mean()
        med_frac = (sub["label"] == "medium").mean()
        low_frac = (sub["label"] == "low").mean()

        ent = label_entropy(sub["label"])

        feats = [
            num_desc,
            ent,
            deg_mean,
            deg_std_,
            level_mean,
            level_std_,
            high_frac,
            med_frac,
            low_frac,
        ]
        # Replace NaNs with 0.0 so the models don't choke on missing values
        X.append([0.0 if pd.isna(v) else float(v) for v in feats])
        high_fracs.append(float(high_frac))

    if len(X) < 4:
        # We need at least a few LOBs for a meaningful classification problem
        raise SystemExit(
            "Not enough LOBs found. Try using the clean dataset or increase generator scale."
        )

    X_arr = np.array(X, dtype=float)
    high_fracs_arr = np.array(high_fracs, dtype=float)

    # Turn high_fracs into binary labels using a flexible rule
    def make_labels(hf: np.ndarray) -> Tuple[np.ndarray, float, str]:
        """
        Create binary labels from high_frac with a few fallbacks.

        1. Try fixed threshold > 0.10.
        2. If that gives only one class, use the median.
        3. If still one class, use the 75th percentile.

        This helps avoid degenerate cases where all LOBs look similar.
        """
        y_bin = (hf > 0.10).astype(int)
        if np.unique(y_bin).size >= 2:
            return y_bin, 0.10, "fixed>0.10"

        med_ = float(np.median(hf))
        y_bin = (hf > med_).astype(int)
        if np.unique(y_bin).size >= 2:
            return y_bin, med_, "median"

        q75 = float(np.quantile(hf, 0.75))
        y_bin = (hf > q75).astype(int)
        return y_bin, q75, "q75"

    y, used_thresh, thresh_method = make_labels(high_fracs_arr)
    classes, counts = np.unique(y, return_counts=True)
    print(
        "Label distribution:",
        dict(zip(classes.tolist(), counts.tolist())),
        f"(method={thresh_method}, thresh={used_thresh:.4f})",
    )
    if np.unique(y).size < 2:
        raise SystemExit("Labels are still single-class after adaptive thresholds.")

    return (
        X_arr,
        y,
        high_fracs_arr,
        lob_ids,
        used_thresh,
        thresh_method,
        feature_names,
        nodes_file,
        edges_file,
    )


# Model definitions and hyperparameter search spaces

def get_models_and_spaces():
    """
    Define both models and their hyperparameter search spaces.

    Note:
    - We always fix random_state=42 for the base models and inner CV so
      runs are reproducible.
    """
    # Classic Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    param_distributions_gb = {
        "n_estimators": [100, 200, 300, 400, 600, 800],
        "learning_rate": [0.03, 0.05, 0.07, 0.1, 0.15],
        "max_depth": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 4, 8, 16],
        "subsample": [0.6, 0.75, 0.9, 1.0],
        "max_features": [None, 1.0, 0.75, 0.5],
    }
    cv_inner_default = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # HistGradientBoosting (faster, works well with tabular data)
    hgb = HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        scoring="average_precision",
    )
    param_distributions_hgb = {
        "learning_rate": [0.03, 0.05, 0.07, 0.1, 0.15],
        "max_depth": [None, 3, 5, 7, 9],
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [1, 5, 10, 20],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0],
        "max_bins": [255, 128, 64],
    }
    return gb, param_distributions_gb, hgb, param_distributions_hgb, cv_inner_default

# Outer cross-validation (multiple train/test splits)

def run_outer_cv(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    gb,
    param_distributions_gb,
    hgb,
    param_distributions_hgb,
    cv_inner_default,
) -> None:
    """
    Run a full nested CV:
      - Outer loop creates train/test splits.
      - Inner RandomizedSearchCV tunes hyperparameters on train/val.

    We report test metrics for each outer fold and print a summary at the end.
    """
    skf = StratifiedKFold(
        n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed
    )
    metrics_gb, metrics_hgb = [], []
    print("\n>>> Running outer CV...")

    for fold, (idx_tr, idx_te) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[idx_tr], X[idx_te]
        y_train, y_test = y[idx_tr], y[idx_te]

        # Split off a small inner validation set from the outer train split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=cfg.inner_val,
            stratify=y_train,
            random_state=cfg.seed + fold,
        )

        # Model A: Gradient Boosting (GB)
        rs_gb = RandomizedSearchCV(
            estimator=gb,
            param_distributions=param_distributions_gb,
            n_iter=40,
            scoring="average_precision",
            n_jobs=-1,
            cv=cv_inner_default,
            verbose=0,
            random_state=cfg.seed + fold,
            refit=True,
        )
        rs_gb.fit(X_tr, y_tr)
        gb_best = rs_gb.best_estimator_
        gb_val_prob = gb_best.predict_proba(X_val)[:, 1]
        gb_thr, _ = tune_threshold(y_val, gb_val_prob)

        # Refit GB on train+val using the tuned hyperparameters
        gb_final = GradientBoostingClassifier(random_state=42, **rs_gb.best_params_)
        gb_final.fit(
            np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val])
        )
        gb_test_prob = gb_final.predict_proba(X_test)[:, 1]
        m_gb = eval_at_threshold(y_test, gb_test_prob, gb_thr)
        metrics_gb.append(m_gb)

        # Model B: HistGradientBoosting (HistGB)
        rs_hgb = RandomizedSearchCV(
            estimator=hgb,
            param_distributions=param_distributions_hgb,
            n_iter=40,
            scoring="average_precision",
            n_jobs=-1,
            cv=cv_inner_default,
            verbose=0,
            random_state=cfg.seed + fold,
            refit=True,
        )
        rs_hgb.fit(X_tr, y_tr)
        hgb_best = rs_hgb.best_estimator_
        hgb_val_prob = hgb_best.predict_proba(X_val)[:, 1]
        hgb_thr, _ = tune_threshold(y_val, hgb_val_prob)

        hgb_final = HistGradientBoostingClassifier(
            random_state=42, **rs_hgb.best_params_
        )
        hgb_final.fit(
            np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val])
        )
        hgb_test_prob = hgb_final.predict_proba(X_test)[:, 1]
        m_hgb = eval_at_threshold(y_test, hgb_test_prob, hgb_thr)
        metrics_hgb.append(m_hgb)

        print(
            f"[Fold {fold}/{cfg.cv_folds}]  "
            f"GB  AUC {m_gb['roc_auc']:.4f} | AP {m_gb['avg_prec']:.4f} | F1 {m_gb['f1']:.4f}    "
            f"HGB AUC {m_hgb['roc_auc']:.4f} | AP {m_hgb['avg_prec']:.4f} | F1 {m_hgb['f1']:.4f}"
        )

    # Helper to summarize metrics over folds
    def summarize(ms, name: str):
        if not ms:
            return
        agg = {k: np.array([m[k] for m in ms], dtype=float) for k in ms[0].keys()}
        print(f"\n===== Outer CV Summary — {name} =====")
        for k, v in agg.items():
            mu, sd = np.mean(v), np.std(v)
            print(f"{k:>10s}: {mu:.4f} ± {sd:.4f}")

    summarize(metrics_gb, "GB")
    summarize(metrics_hgb, "HistGB")

# Single train/val/test run with plots + CSV logging

def run_single_holdout(
    X: np.ndarray,
    y: np.ndarray,
    high_fracs: np.ndarray,
    used_thresh: float,
    thresh_method: str,
    feature_names: List[str],
    cfg: Config,
    gb,
    param_distributions_gb,
    hgb,
    param_distributions_hgb,
    cv_inner_default,
    nodes_file: str,
    edges_file: str,
) -> None:
    """
    Run one classic train/validation/test experiment and save a full set
    of metrics and plots for both models.
    """
    # First split: top-level train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Second split: train -> train/val (so overall ~70/17.5/12.5 split)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    # Gradient Boosting (GB)
    rs_gb = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_distributions_gb,
        n_iter=40,
        scoring="average_precision",
        n_jobs=-1,
        cv=cv_inner_default,
        verbose=0,
        random_state=42,
        refit=True,
    )
    t0 = time.time()
    rs_gb.fit(X_tr, y_tr)
    gb_tune_sec = time.time() - t0
    gb_best = rs_gb.best_estimator_
    print("\n[GB] Best params:", rs_gb.best_params_)
    print(f"[GB] Tuning seconds: {gb_tune_sec:.3f}s")

    # Tune threshold on validation set
    gb_val_prob = gb_best.predict_proba(X_val)[:, 1]
    gb_thr, gb_val_f1 = tune_threshold(y_val, gb_val_prob)
    print(f"[GB] Chosen threshold from val: {gb_thr:.3f} (val F1={gb_val_f1:.4f})")

    # Refit GB on train+val using best hyperparameters
    gb_final = GradientBoostingClassifier(random_state=42, **rs_gb.best_params_)
    t1 = time.time()
    gb_final.fit(np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))
    gb_train_sec = time.time() - t1

    # Evaluate on held-out test data
    gb_test_prob = gb_final.predict_proba(X_test)[:, 1]
    gb_test_pred = (gb_test_prob >= gb_thr).astype(int)

    gb_roc = roc_auc_score(y_test, gb_test_prob)
    gb_ap = average_precision_score(y_test, gb_test_prob)
    gb_acc = accuracy_score(y_test, gb_test_pred)
    gb_f1 = f1_score(y_test, gb_test_pred, zero_division=0)
    gb_prc = precision_score(y_test, gb_test_pred, zero_division=0)
    gb_rec = recall_score(y_test, gb_test_pred, zero_division=0)

    print("\n=== Test metrics (GB tuned) ===")
    print(f"ROC AUC:   {gb_roc:.4f}  |  PR AUC:   {gb_ap:.4f}")
    print(f"Accuracy:  {gb_acc:.4f}  |  F1:       {gb_f1:.4f}")
    print(f"Precision: {gb_prc:.4f}  |  Recall:   {gb_rec:.4f}")
    print(f"Train sec: {gb_train_sec:.3f}")

    # HistGradientBoosting (HistGB)
    rs_hgb = RandomizedSearchCV(
        estimator=hgb,
        param_distributions=param_distributions_hgb,
        n_iter=40,
        scoring="average_precision",
        n_jobs=-1,
        cv=cv_inner_default,
        verbose=0,
        random_state=42,
        refit=True,
    )
    t2 = time.time()
    rs_hgb.fit(X_tr, y_tr)
    hgb_tune_sec = time.time() - t2
    hgb_best = rs_hgb.best_estimator_
    print("\n[HGB] Best params:", rs_hgb.best_params_)
    print(f"[HGB] Tuning seconds: {hgb_tune_sec:.3f}s")

    hgb_val_prob = hgb_best.predict_proba(X_val)[:, 1]
    hgb_thr, hgb_val_f1 = tune_threshold(y_val, hgb_val_prob)
    print(f"[HGB] Chosen threshold from val: {hgb_thr:.3f} (val F1={hgb_val_f1:.4f})")

    hgb_final = HistGradientBoostingClassifier(random_state=42, **rs_hgb.best_params_)
    t3 = time.time()
    hgb_final.fit(np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))
    hgb_train_sec = time.time() - t3

    hgb_test_prob = hgb_final.predict_proba(X_test)[:, 1]
    hgb_test_pred = (hgb_test_prob >= hgb_thr).astype(int)

    hgb_roc = roc_auc_score(y_test, hgb_test_prob)
    hgb_ap = average_precision_score(y_test, hgb_test_prob)
    hgb_acc = accuracy_score(y_test, hgb_test_pred)
    hgb_f1 = f1_score(y_test, hgb_test_pred, zero_division=0)
    hgb_prc = precision_score(y_test, hgb_test_pred, zero_division=0)
    hgb_rec = recall_score(y_test, hgb_test_pred, zero_division=0)

    print("\n=== Test metrics (HistGB tuned) ===")
    print(f"ROC AUC:   {hgb_roc:.4f}  |  PR AUC:   {hgb_ap:.4f}")
    print(f"Accuracy:  {hgb_acc:.4f}  |  F1:       {hgb_f1:.4f}")
    print(f"Precision: {hgb_prc:.4f}  |  Recall:   {hgb_rec:.4f}")
    print(f"Train sec: {hgb_train_sec:.3f}")

    # Plot helpers
    def plot_and_save_cm(y_true, y_pred, title, fname):
        """Plot confusion matrix and save it to disk."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure()
        disp.plot(values_format="d")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.outdir, fname), dpi=150)
        plt.show()

    def plot_and_save_roc(y_true, y_prob, auc_val, title, fname):
        """Plot ROC curve if both classes appear in y_true."""
        if np.unique(y_true).size < 2:
            print("[WARN] Single-class y_true; ROC undefined.")
            return
        plt.figure()
        RocCurveDisplay.from_predictions(y_true, y_prob, name=f"AUC={auc_val:.3f}")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.outdir, fname), dpi=150)
        plt.show()

    def plot_and_save_pr(y_true, y_prob, ap_val, title, fname):
        """Plot Precision Recall curve if both classes appear in y_true."""
        if np.unique(y_true).size < 2:
            print("[WARN] Single-class y_true; PR curve degenerate.")
            return
        plt.figure()
        PrecisionRecallDisplay.from_predictions(
            y_true, y_prob, name=f"AP={ap_val:.3f}"
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.outdir, fname), dpi=150)
        plt.show()

    # GB plots
    plot_and_save_cm(y_test, gb_test_pred, "Confusion Matrix (GB tuned)", "gb_tuned_cm.png")
    plot_and_save_roc(y_test, gb_test_prob, gb_roc, "ROC Curve (GB tuned)", "gb_tuned_roc.png")
    plot_and_save_pr(y_test, gb_test_prob, gb_ap, "Precision–Recall (GB tuned)", "gb_tuned_pr.png")

    # HGB plots
    plot_and_save_cm(y_test, hgb_test_pred, "Confusion Matrix (HistGB tuned)", "hgb_tuned_cm.png")
    plot_and_save_roc(
        y_test,
        hgb_test_prob,
        hgb_roc,
        "ROC Curve (HistGB tuned)",
        "hgb_tuned_roc.png",
    )
    plot_and_save_pr(
        y_test,
        hgb_test_prob,
        hgb_ap,
        "Precision–Recall (HistGB tuned)",
        "hgb_tuned_pr.png",
    )

    # Feature importances (GB only; HistGB doesn't show them in the same way)
    importances = getattr(gb_final, "feature_importances_", None)
    if importances is not None:
        order = np.argsort(importances)
        plt.figure(figsize=(7, 5))
        plt.barh(range(len(importances)), importances[order])
        plt.yticks(range(len(importances)), [feature_names[i] for i in order])
        plt.xlabel("Importance")
        plt.title("Feature Importances (GB tuned)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(cfg.outdir, "gb_tuned_feature_importances.png"), dpi=150
        )
        plt.show()

    # Show how the high_frac values are distributed across LOBs, along with the threshold used to turn them into binary labels.
    plt.figure(figsize=(6, 4))
    plt.hist(high_fracs, bins=20)
    plt.axvline(used_thresh, linestyle="--")
    plt.xlabel("high_frac among descendants")
    plt.ylabel("Count of LOBs")
    plt.title(
        f"Distribution of high_frac (label method={thresh_method}, "
        f"thresh={used_thresh:.3f})"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, "gb_high_frac_hist.png"), dpi=150)
    plt.show()

    # Metric logging
    # Store the most important settings and test metrics for both models
    row_gb = {
        "model": "GB",
        "use_noisy": int(cfg.use_noisy),
        "nodes_file": nodes_file,
        "edges_file": edges_file,
        "label_method": thresh_method,
        "label_threshold": round(float(used_thresh), 6),
        "cv_scoring": "average_precision",
        "chosen_prob_threshold": round(float(gb_thr), 6),
        "tune_seconds": round(gb_tune_sec, 3),
        "train_seconds": round(gb_train_sec, 3),
        "roc_auc": round(gb_roc, 4),
        "avg_precision": round(gb_ap, 4),
        "accuracy": round(gb_acc, 4),
        "f1": round(gb_f1, 4),
        "precision": round(gb_prc, 4),
        "recall": round(gb_rec, 4),
    }

    row_hgb = {
        "model": "HistGB",
        "use_noisy": int(cfg.use_noisy),
        "nodes_file": nodes_file,
        "edges_file": edges_file,
        "label_method": thresh_method,
        "label_threshold": round(float(used_thresh), 6),
        "cv_scoring": "average_precision",
        "chosen_prob_threshold": round(float(hgb_thr), 6),
        "tune_seconds": round(hgb_tune_sec, 3),
        "train_seconds": round(hgb_train_sec, 3),
        "roc_auc": round(hgb_roc, 4),
        "avg_precision": round(hgb_ap, 4),
        "accuracy": round(hgb_acc, 4),
        "f1": round(hgb_f1, 4),
        "precision": round(hgb_prc, 4),
        "recall": round(hgb_rec, 4),
    }

    csv_path = os.path.join(cfg.outdir, "graph_classification_tuned_models_metrics.csv")
    pd.DataFrame([row_gb, row_hgb]).to_csv(
        csv_path, mode="a", index=False, header=not os.path.exists(csv_path)
    )
    print(f"[✓] Metrics appended to {csv_path}")
    print(f"[✓] Plots saved to {cfg.outdir}")

# Script entry point

def main() -> None:
    """
    Main driver:
      1) Read config from environment.
      2) Build LOB-level dataset from node/edge CSVs.
      3) Create both models + hyperparameter spaces.
      4) Run either outer CV or a single hold-out experiment.
    """
    cfg = Config.from_env()
    (
        X,
        y,
        high_fracs,
        lob_ids,
        used_thresh,
        thresh_method,
        feature_names,
        nodes_file,
        edges_file,
    ) = load_graph_data(cfg)

    gb, param_distributions_gb, hgb, param_distributions_hgb, cv_inner_default = (
        get_models_and_spaces()
    )

    if cfg.use_cv:
        # Full nested CV for more reliable estimates
        run_outer_cv(
            X,
            y,
            cfg,
            gb,
            param_distributions_gb,
            hgb,
            param_distributions_hgb,
            cv_inner_default,
        )
    else:
        # Single train/val/test run with plots and CSV logging
        run_single_holdout(
            X,
            y,
            high_fracs,
            used_thresh,
            thresh_method,
            feature_names,
            cfg,
            gb,
            param_distributions_gb,
            hgb,
            param_distributions_hgb,
            cv_inner_default,
            nodes_file,
            edges_file,
        )


if __name__ == "__main__":
    main()
