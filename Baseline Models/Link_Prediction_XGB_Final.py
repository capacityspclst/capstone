"""

Link Prediction Pipeline (XGBoost GPU + Graph Heuristics)

This script is structured as three notebook-style cells:
   1) Experiment configuration:
        - Choose artifacts directory, relation name, noisy vs. clean data
        - Configure cross-validation and random seed
        - Export all settings as environment variables and sanity-check files

   2) Main training pipeline:
        - Load graph nodes/edges and edge-split CSVs
        - Build an undirected NetworkX graph
        - Compute classic link-prediction heuristics for each node pair
        - Transform & scale features
        - Run GPU-accelerated XGBoost with simple hyperparam search + early stopping
        - Refit on train+val with best settings
        - Evaluate on test and save metrics/plots

   3) Extended metrics and cross-validation:
        - Compute richer test metrics (accuracy, F1, confusion matrix, etc.)
        - Optionally run outer cross-validation with inner validation for tuning
        - Generate per-fold confusion matrices and metric bar charts
        - Save summary CSVs and confusion-matrix arrays

"""

# Imports
import os
import pathlib
import json
import time
import math
from collections import Counter
from statistics import mean, pstdev

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
    train_test_split,
)

# USER Switches

# Root directory containing nodes*.csv, edges*.csv, and edge_splits/*
ART = "../Data Generation"

# Subfolder under ART where metrics, plots, etc. will be written
OUTDIR_NAME = "baseline_metrics"

# Relation name: subfolder under edge_splits/ that this run is targeting
REL_NAME = "depends_on"

# Whether to load "*_noisy.csv" instead of the clean versions
USE_NOISY = True

# Whether to run the extended cross-validation routine
USE_CV = True

# Cross-validation controls
CV_FOLDS = 5          # number of CV folds
INNER_VAL = 0.20      # fraction of per-fold train reserved as inner validation

# Reproducibility and device configuration
SEED = 42
GPU_ID = 0            # CUDA_VISIBLE_DEVICES index (int) for the selected GPU

# If noisy artifacts live somewhere else, point to that directory when USE_NOISY=True
NOISY_ART_OVERRIDE = None

def _abspath(p: str) -> str:
    """Return absolute, expanded path for filesystem consistency."""
    return str(pathlib.Path(p).expanduser().resolve())

# Effective artifacts directory (clean or noisy override)
ART_EFF = (
    _abspath(NOISY_ART_OVERRIDE)
    if (USE_NOISY and (NOISY_ART_OVERRIDE is not None))
    else _abspath(ART)
)

# Output directory: ART_EFF / OUTDIR_NAME
OUTDIR = _abspath(os.path.join(ART_EFF, OUTDIR_NAME))
os.makedirs(OUTDIR, exist_ok=True)

# Export all relevant configuration switches to the environment so that downstream scripts can read them.
os.environ.update(
    {
        # GPU selection
        "CUDA_VISIBLE_DEVICES": str(GPU_ID),
        "GPU_ID": str(int(GPU_ID)),

        # Artifact locations
        "ARTIFACTS": ART_EFF,
        "OUTDIR": OUTDIR,

        # Relation and noisy data toggle
        "REL": REL_NAME,
        "USE_NOISY": "1" if USE_NOISY else "0",

        # Cross-validation controls
        "USE_CV": "1" if USE_CV else "0",
        "CV_FOLDS": str(int(CV_FOLDS)),
        "INNER_VAL": str(float(INNER_VAL)),

        # Reproducibility
        "SEED": str(int(SEED)),
    }
)

nodes_file = "nodes_noisy.csv" if USE_NOISY else "nodes.csv"
edges_file = "edges_noisy.csv" if USE_NOISY else "edges.csv"


def _exists(p: str) -> bool:
    """Robust 'exists' check; returns False on errors instead of raising."""
    try:
        return os.path.exists(p)
    except Exception:
        return False

expected = {
    "ARTIFACTS": ART_EFF,
    "nodes": os.path.join(ART_EFF, nodes_file),
    "edges": os.path.join(ART_EFF, edges_file),
    "edge_splits/train_pos.csv": os.path.join(
        ART_EFF, "edge_splits", REL_NAME, "train_pos.csv"
    ),
    "edge_splits/train_neg.csv": os.path.join(
        ART_EFF, "edge_splits", REL_NAME, "train_neg.csv"
    ),
    "edge_splits/test_pos.csv": os.path.join(
        ART_EFF, "edge_splits", REL_NAME, "test_pos.csv"
    ),
    "edge_splits/test_neg.csv": os.path.join(
        ART_EFF, "edge_splits", REL_NAME, "test_neg.csv"
    ),
    "edge_splits/val_pos.csv (optional)": os.path.join(
        ART_EFF, "edge_splits", REL_NAME, "val_pos.csv"
    ),
    "edge_splits/val_neg.csv (optional)": os.path.join(
        ART_EFF, "edge_splits", REL_NAME, "val_neg.csv"
    ),
    "OUTDIR (will be created)": OUTDIR,
}

print("=== Effective Run Config ===")
print(
    json.dumps(
        {
            "ARTIFACTS": ART_EFF,
            "OUTDIR": OUTDIR,
            "REL": REL_NAME,
            "USE_NOISY": USE_NOISY,
            "USE_CV": os.environ.get("USE_CV") == "1",
            "CV_FOLDS": int(os.environ["CV_FOLDS"]),
            "INNER_VAL": float(os.environ["INNER_VAL"]),
            "SEED": int(os.environ["SEED"]),
            "GPU_ID": GPU_ID,
            "nodes_file": nodes_file,
            "edges_file": edges_file,
        },
        indent=2,
    )
)

print("\n=== Expected Files / Dirs ===")
for k, p in expected.items():
    flag = "✓" if _exists(p) else ("(create)" if k.startswith("OUTDIR") else "✗")
    print(f"{k:>28s}: {p} {flag}")



# Paths / configuration

# Read required settings back from environment
REL = os.environ.get("REL", "depends_on")
ART = os.environ.get("ARTIFACTS", os.path.join("..", "Data Generation"))
USE_NOISY = os.environ.get("USE_NOISY", "0") == "1"
OUTDIR = os.environ.get("OUTDIR", os.path.join(ART, "baseline_metrics"))
USE_CV = os.environ.get("USE_CV", "0") == "1"       # enable cross-validation
CV_FOLDS = int(os.environ.get("CV_FOLDS", 5))       # number of CV folds
INNER_VAL = float(os.environ.get("INNER_VAL", 0.2)) # inner val size per fold
SEED = int(os.environ.get("SEED", 42))
GPU_ID = int(os.environ.get("GPU_ID", 0))           # which GPU (if multiple)

os.makedirs(OUTDIR, exist_ok=True)
print(
    "ART:", ART,
    "| OUTDIR:", OUTDIR,
    "| REL:", REL,
    "| USE_CV:", USE_CV,
    "| CV_FOLDS:", CV_FOLDS,
)

# RNG for any NumPy-based randomness
rng = np.random.default_rng(SEED)

# Load edge splits

nodes_file = "nodes_noisy.csv" if USE_NOISY else "nodes.csv"
edges_file = "edges_noisy.csv" if USE_NOISY else "edges.csv"

# Root directory for edge split CSVs for this relation
root = os.path.join(ART, "edge_splits", REL)


def read_pairs(fname: str):
    """Load a pair-list CSV (u,v) or return None if it does not exist."""
    p = os.path.join(root, fname)
    return pd.read_csv(p).values if os.path.exists(p) else None

# Positive / negative splits for train, val, and test
train_pos = read_pairs("train_pos.csv")
train_neg = read_pairs("train_neg.csv")
val_pos = read_pairs("val_pos.csv")
val_neg = read_pairs("val_neg.csv")
test_pos = read_pairs("test_pos.csv")
test_neg = read_pairs("test_neg.csv")

# Require at minimum train and test splits to exist
if train_pos is None or train_neg is None or test_pos is None or test_neg is None:
    raise SystemExit("Missing required splits. Need train_pos/neg.csv and test_pos/neg.csv")

# Presence of both val_pos and val_neg determines whether we use a dedicated val set
use_val = (val_pos is not None) and (val_neg is not None)

# Build graph from nodes/edges

# Load raw node and edge tables
nodes = pd.read_csv(os.path.join(ART, nodes_file), low_memory=False)
edges = pd.read_csv(os.path.join(ART, edges_file), low_memory=False)

# Ensure integer IDs and drop rows with missing endpoints
nodes["id"] = pd.to_numeric(nodes["id"], errors="coerce")
edges["source"] = pd.to_numeric(edges["source"], errors="coerce")
edges["target"] = pd.to_numeric(edges["target"], errors="coerce")
nodes = nodes.dropna(subset=["id"]).astype({"id": int})
edges = edges.dropna(subset=["source", "target"]).astype({"source": int, "target": int})

# Build an undirected NetworkX graph for computing neighborhood-based heuristics
G = nx.Graph()
G.add_nodes_from(nodes["id"].tolist())
G.add_edges_from(edges[["source", "target"]].values.tolist())

# Precompute degrees and cache neighbor sets for speed
deg = dict(G.degree())
neighbors_cache = {}


def neigh(u: int):
    """Return cached neighbor set for node u."""
    if u not in neighbors_cache:
        neighbors_cache[u] = set(G.neighbors(u))
    return neighbors_cache[u]


# Heuristic link-prediction features

def CN(u, v):
    """Common neighbors count."""
    return len(neigh(u) & neigh(v))


def AA(u, v):
    """Adamic–Adar score: sum(1 / log(deg(w))) over common neighbors."""
    s = 0.0
    for w in neigh(u) & neigh(v):
        dw = deg.get(w, 0)
        if dw > 1:
            s += 1.0 / math.log(dw)
    return s


def JACC(u, v):
    """Jaccard coefficient based on neighbor sets."""
    Nu, Nv = neigh(u), neigh(v)
    denom = len(Nu | Nv)
    return len(Nu & Nv) / denom if denom else 0.0


def RA(u, v):
    """Resource allocation index: sum(1 / deg(w)) over common neighbors."""
    s = 0.0
    for w in neigh(u) & neigh(v):
        dw = deg.get(w, 0)
        if dw > 0:
            s += 1.0 / dw
    return s


def PA(u, v):
    """Preferential attachment: deg(u) * deg(v)."""
    return deg.get(u, 0) * deg.get(v, 0)


def LHN(u, v):
    """Leicht–Holme–Newman index: CN / (deg(u)*deg(v))."""
    du, dv = deg.get(u, 0), deg.get(v, 0)
    if du == 0 or dv == 0:
        return 0.0
    return CN(u, v) / (du * dv)


def HPI(u, v):
    """Hub promoted index: CN / max(deg(u), deg(v))."""
    du, dv = deg.get(u, 0), deg.get(v, 0)
    m = max(du, dv)
    return CN(u, v) / m if m > 0 else 0.0


def HDI(u, v):
    """Hub depressed index: CN / min(deg(u), deg(v))."""
    du, dv = deg.get(u, 0), deg.get(v, 0)
    m = min(du, dv)
    return CN(u, v) / m if m > 0 else 0.0


def SALTON(u, v):
    """Salton index: CN / sqrt(deg(u)*deg(v))."""
    du, dv = deg.get(u, 0), deg.get(v, 0)
    d = math.sqrt(du * dv)
    return CN(u, v) / d if d > 0 else 0.0


def SORENSEN(u, v):
    """Sørensen index: 2*CN / (deg(u)+deg(v))."""
    du, dv = deg.get(u, 0), deg.get(v, 0)
    s = du + dv
    return (2.0 * CN(u, v)) / s if s > 0 else 0.0


def SP_INV(u, v, cap=3):
    """
    Inverse shortest path distance (capped):
        1 / (1 + d(u,v)) if a path of length <= cap exists, else 0.
    """
    try:
        d = nx.shortest_path_length(G, u, v, cutoff=cap)
        # If d is a dict (multi-source) or None, treat as no path
        if isinstance(d, dict) or d is None:
            return 0.0
        return 1.0 / (1.0 + float(d))
    except nx.NetworkXNoPath:
        return 0.0
    except Exception:
        return 0.0


# Degree-based features
def DEG_U(u, v):
    return float(deg.get(u, 0))


def DEG_V(u, v):
    return float(deg.get(v, 0))


def DEG_SUM(u, v):
    return float(deg.get(u, 0) + deg.get(v, 0))


def DEG_PROD(u, v):
    return float(deg.get(u, 0) * deg.get(v, 0))


def DEG_ADF(u, v):
    return float(abs(deg.get(u, 0) - deg.get(v, 0)))


# List of all features (name, function) for consistent ordering
FEATURES = [
    ("CN", CN),
    ("AA", AA),
    ("Jaccard", JACC),
    ("RA", RA),
    ("PA", PA),
    ("LHN", LHN),
    ("HPI", HPI),
    ("HDI", HDI),
    ("Salton", SALTON),
    ("Sorensen", SORENSEN),
    ("SP_inv", SP_INV),
    ("deg_u", DEG_U),
    ("deg_v", DEG_V),
    ("deg_sum", DEG_SUM),
    ("deg_prod", DEG_PROD),
    ("deg_absdiff", DEG_ADF),
]

# Column names (ordered) and indices that are constrained to be non-negative
colnames = [name for name, _ in FEATURES]
nonneg_cols = [
    i
    for i, n in enumerate(colnames)
    if n
    in {
        "CN",
        "AA",
        "RA",
        "PA",
        "LHN",
        "HPI",
        "HDI",
        "Salton",
        "Sorensen",
        "SP_inv",
        "deg_u",
        "deg_v",
        "deg_sum",
        "deg_prod",
        "deg_absdiff",
    }
]
# Any columns not listed as non-negative are treated as signed
signed_cols = [i for i in range(len(colnames)) if i not in nonneg_cols]

# Build X / y from edge pairs

def build_xy(pos_pairs, neg_pairs):
    """
    Given positive and negative (u,v) pairs, compute feature matrix X and label vector y.
    Positive labels are 1; negative labels are 0.
    """
    X = []
    # Positive examples
    for (u, v) in pos_pairs:
        u, v = int(u), int(v)
        X.append([fn(u, v) for _, fn in FEATURES])

    # Negative examples
    for (u, v) in neg_pairs:
        u, v = int(u), int(v)
        X.append([fn(u, v) for _, fn in FEATURES])

    X = np.array(X, dtype=float)
    y = np.concatenate(
        [np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))]
    ).astype(int)
    return X, y


# Build training and test matrices
X_tr, y_tr = build_xy(train_pos, train_neg)
X_te, y_te = build_xy(test_pos, test_neg)

# Use provided validation split if available, else create one using StratifiedShuffleSplit
if use_val:
    X_va, y_va = build_xy(val_pos, val_neg)
else:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    (tr_idx, va_idx), = sss.split(X_tr, y_tr)
    X_tr, X_va, y_tr, y_va = (
        X_tr[tr_idx],
        X_tr[va_idx],
        y_tr[tr_idx],
        y_tr[va_idx],
    )

# Feature transforms & scaling

def transform_fit(X_train: np.ndarray):
    """
    Fit transformation on training data:
      - log1p on non-negative features
      - arcsinh on signed features
      - standard scaling (zero-mean, unit-variance)
    Returns the fitted scaler and transformed training matrix.
    """
    X = X_train.copy()

    # Log-transform heavy-tailed non-negative features
    if nonneg_cols:
        X[:, nonneg_cols] = np.log1p(
            np.clip(X[:, nonneg_cols], a_min=0, a_max=None)
        )

    # Arcsinh-transform signed features (robust to outliers & keeps sign)
    if signed_cols:
        X[:, signed_cols] = np.arcsinh(X[:, signed_cols])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return scaler, X


def transform_apply(X: np.ndarray, scaler: StandardScaler):
    """
    Apply the previously-fitted transform to new data.
    """
    Xc = X.copy()
    if nonneg_cols:
        Xc[:, nonneg_cols] = np.log1p(
            np.clip(Xc[:, nonneg_cols], a_min=0, a_max=None)
        )
    if signed_cols:
        Xc[:, signed_cols] = np.arcsinh(Xc[:, signed_cols])
    return scaler.transform(Xc)


# Class weights / imbalance handling

def scale_pos_weight(y: np.ndarray) -> float:
    """Return negative/positive ratio used by XGBoost's scale_pos_weight."""
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    return float(neg) / float(pos)



# XGBoost (GPU) — native Booster API for early stopping

# Ensure XGBoost is aware of the chosen GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# Base parameters GPU training + binary logistic + AUC metric
XGB_TRAIN_PARAMS_BASE = dict(
    device="cuda",
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="auc",
)

# Modest hyperparameter grid; rely on early stopping for number of rounds
xgb_grid = [
    dict(
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        num_boost_round=2000,
    ),
    dict(
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        num_boost_round=2500,
    ),
    dict(
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        num_boost_round=3000,
    ),
]


def _params_from_pg(pg: dict, spw: float):
    """
    Merge grid parameters into base params and attach scale_pos_weight.
    Returns (params, num_boost_round).
    """
    p = dict(XGB_TRAIN_PARAMS_BASE)
    # Alias 'learning_rate' to 'eta' if provided
    if "learning_rate" in pg:
        p["eta"] = pg["learning_rate"]
    # Copy other XGBoost params from grid dict if present
    for k in [
        "max_depth",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "min_child_weight",
    ]:
        if k in pg:
            p[k] = pg[k]
    p["scale_pos_weight"] = float(spw)
    nrounds = int(pg.get("num_boost_round", 2000))
    return p, nrounds


def _train_native_es(Xtr, ytr, Xva, yva, params, num_boost_round):
    """
    Train a native XGBoost Booster with early stopping on validation data.
    Returns (best_booster, best_iteration).
    """
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    watch = [(dtr, "train"), (dva, "valid")]

    bst = xgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        evals=watch,
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    best_round = getattr(bst, "best_iteration", None)
    if best_round is None:
        best_round = int(bst.best_ntree_limit) - 1
    return bst, int(best_round)


def _predict_proba_booster(bst, X):
    """
    Convenience wrapper: predict probabilities P(y=1) using a Booster.
    """
    return bst.predict(xgb.DMatrix(X))


class BoosterProbaWrapper:
    """
    Adapter to make a trained XGBoost Booster compatible with sklearn tools
    that expect an estimator with predict_proba (e.g., permutation_importance).

    This object does NOT train; it only wraps an existing Booster.
    """

    def __init__(self, booster):
        self.booster = booster
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"

    def fit(self, X, y=None):
        """
        fit that just sets n_features_in_ for sklearn compatibility.
        """
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        """Return [P(y=0), P(y=1)] for each sample."""
        p1 = _predict_proba_booster(self.booster, X).reshape(-1)
        p0 = 1.0 - p1
        return np.c_[p0, p1]

    def predict(self, X):
        """Return hard labels with threshold 0.5 on P(y=1)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=False):
        return {}

    def set_params(self, **params):
        return self


def fit_with_search_gpu(X_tr_tf, y_tr, X_va_tf, y_va, seed=SEED):
    """
    Run a small hyperparameter search over xgb_grid with early stopping,
    selecting the configuration with highest validation ROC AUC.

    Returns:
        best_bst, best_params, best_val_auc, best_round, tune_seconds
    """
    spw = scale_pos_weight(y_tr)
    best_bst, best_params, best_va_auc, best_round = None, None, -np.inf, None
    t0 = time.time()

    for i, pg in enumerate(xgb_grid, 1):
        print(f"[XGB-SEARCH] Config {i}/{len(xgb_grid)}: {pg}")
        params, num_boost_round = _params_from_pg(pg, spw)

        t1 = time.time()
        bst, br = _train_native_es(X_tr_tf, y_tr, X_va_tf, y_va, params, num_boost_round)
        fit_sec = time.time() - t1

        proba_va = _predict_proba_booster(bst, X_va_tf)
        auc_va = roc_auc_score(y_va, proba_va)
        print(
            f"[XGB-SEARCH]   fit {fit_sec:.1f}s | best_round={br} | val AUC={auc_va:.4f}"
        )

        if auc_va > best_va_auc:
            best_va_auc = auc_va
            best_bst = bst
            best_params = dict(params)
            best_round = int(br)

    tune_sec = time.time() - t0
    print(
        f"[XGB-SEARCH] Best val AUC={best_va_auc:.4f} | "
        f"best_round={best_round} | tune={tune_sec:.1f}s"
    )
    return best_bst, best_params, best_va_auc, best_round, tune_sec


def refit_final_gpu(best_params, best_round, X_trva_tf, y_trva, seed=SEED):
    """
    Refit a final XGBoost model on (train + validation) data using the
    best hyperparameters and number of boosting rounds from the search.

    Early stopping is NOT used here; we train for exactly best_round+1 trees.
    """
    dtrva = xgb.DMatrix(X_trva_tf, label=y_trva)
    nrounds = int(best_round) + 1 if best_round is not None else 2000

    t1 = time.time()
    bst = xgb.train(best_params, dtrva, num_boost_round=nrounds, verbose_eval=False)
    train_sec = time.time() - t1
    print(f"[XGB-REFIT] Trained on train+val in {train_sec:.1f}s (rounds={nrounds})")
    return bst, train_sec



# (B) Main pipeline: tune on val, refit on train+val, test

# Fit transforms on training data and apply to val/test
scaler_main, X_tr_tf = transform_fit(X_tr)
X_va_tf = transform_apply(X_va, scaler_main)
X_te_tf = transform_apply(X_te, scaler_main)

print("\n>>> Hyperparameter search with early stopping on validation …")
best_bst, best_params, best_va_auc, best_round, tune_sec = fit_with_search_gpu(
    X_tr_tf, y_tr, X_va_tf, y_va, seed=SEED
)

print("\n>>> Refit on train+val with best_round")
X_trva_tf = np.vstack([X_tr_tf, X_va_tf])
y_trva = np.concatenate([y_tr, y_va])
xgb_final, train_sec = refit_final_gpu(
    best_params, best_round, X_trva_tf, y_trva, seed=SEED
)

# Evaluate on test set
proba_te = _predict_proba_booster(xgb_final, X_te_tf)
pred_te = (proba_te >= 0.5).astype(int)

roc = roc_auc_score(y_te, proba_te)
ap = average_precision_score(y_te, proba_te)

print("\n=== Test metrics (XGBoost GPU) ===")
print(f"ROC AUC: {roc:.4f}   |   PR AUC: {ap:.4f}")
print(
    f"Train seconds: {train_sec:.3f} "
    f"(best config from val; tune {tune_sec:.3f}s)"
)

# Plots: ROC and Precision–Recall curves

plt.figure()
RocCurveDisplay.from_predictions(y_te, proba_te)
plt.title(f"ROC — XGBoost GPU ({REL})")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, f"xgb_gpu_roc_{REL}.png"), dpi=150)
plt.show()

plt.figure()
PrecisionRecallDisplay.from_predictions(y_te, proba_te)
plt.title(f"PR — XGBoost GPU ({REL})")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, f"xgb_gpu_pr_{REL}.png"), dpi=150)
plt.show()

# Permutation importance (with BoosterProbaWrapper)

try:
    # Wrapper fit sets n_features_in_ for sklearn, but does not retrain
    wrapper = BoosterProbaWrapper(xgb_final).fit(X_trva_tf, y_trva)
    pi = permutation_importance(
        wrapper,
        X_te_tf,
        y_te,
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
        scoring="roc_auc",
    )
    order = np.argsort(pi.importances_mean)[::-1]
    top_k = min(20, len(order))
    order = order[:top_k]

    names = np.array(colnames)[order]
    means = pi.importances_mean[order]
    stds = pi.importances_std[order]

    plt.figure(figsize=(8, 5))
    plt.barh(range(top_k), means[::-1], xerr=stds[::-1])
    plt.yticks(range(top_k), names[::-1])
    plt.xlabel("Permutation importance (Δ ROC AUC)")
    plt.title("Top feature importances — XGBoost GPU")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"xgb_gpu_perm_importance_{REL}.png"), dpi=150)
    plt.show()
except Exception as e:
    print("[info] Permutation importance skipped:", e)

# Save summary metrics to CSV

row = {
    "model": "XGBoost_GPU_blend",
    "use_noisy": int(USE_NOISY),
    "nodes_file": nodes_file,
    "edges_file": edges_file,
    "val_roc_auc": round(best_va_auc, 4),
    "val_best_round": int(best_round) if best_round is not None else "",
    "test_roc_auc": round(roc, 4),
    "test_pr_auc": round(ap, 4),
    "tune_seconds": round(tune_sec, 3),
    "train_seconds": round(train_sec, 3),
    "n_features": len(FEATURES),
}
csv_path = os.path.join(OUTDIR, f"link_prediction_xgb_gpu_metrics_{REL}.csv")
pd.DataFrame([row]).to_csv(
    csv_path, mode="a", index=False, header=not os.path.exists(csv_path)
)
print(f"[✓] Metrics appended to {csv_path}")
print(f"[✓] Plots saved to {OUTDIR}")

if os.environ.get("USE_CV") == "1":
    print("\n>>> (Note) Extended CV with confusion matrices is available in Cell 3.")


plt.rcParams["figure.dpi"] = 120

OUTDIR = os.environ.get("OUTDIR", ".")
REL = os.environ.get("REL", "depends_on")

def evaluate_full(y_true, y_prob, threshold=0.5):
    """
    Compute a comprehensive set of metrics for binary classification:
        - accuracy, macro F1, macro precision/recall
        - ROC AUC, Average Precision (PR AUC)
        - confusion matrix
        - chosen threshold and positive prevalence
    """
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "macro_roc_auc": roc_auc_score(y_true, y_prob),
        "macro_avg_precision": average_precision_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "threshold": float(threshold),
        "pos_prevalence": float(np.mean(y_true)),
    }
    return metrics


def _print_metrics_block(title, m: dict):
    """Pretty-print metric dicts with a standard layout."""
    print(f"\n===== {title} =====")
    print(f"accuracy:            {m['accuracy']:.4f}")
    print(f"f1_macro:            {m['f1_macro']:.4f}")
    print(f"precision_macro:     {m['precision_macro']:.4f}")
    print(f"recall_macro:        {m['recall_macro']:.4f}")
    print(f"macro_roc_auc:       {m['macro_roc_auc']:.4f}")
    print(f"macro_avg_precision: {m['macro_avg_precision']:.4f}")
    print(f"threshold:           {m['threshold']:.3f}")
    print(f"pos_prevalence:      {m['pos_prevalence']:.4f}")
    print("Confusion matrix:\n", m["confusion_matrix"])


# TEST METRICS
# Requires y_te and proba_te

test_metrics = evaluate_full(y_te, proba_te, threshold=0.5)
_print_metrics_block(
    "Test Metrics (XGBoost GPU, SAFE, no embeddings)",
    test_metrics,
)


def _savefig(name: str):
    """
    Helper: save current matplotlib figure into OUTDIR and log its path.
    """
    try:
        path = os.path.join(OUTDIR, name)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print(f"[✓] Saved {path}")
    except Exception as e:
        print("[info] Could not save figure:", e)


# CROSS-VALIDATION (outer) with extended metrics & graphics
def run_cross_validation_extended_with_graphics():
    """
    Perform outer cross-validation (StratifiedKFold) on train ∪ val, and for each fold:
      - split into inner train/val for hyperparameter search (using existing XGB search)
      - refit on (inner train + inner val) with best params
      - evaluate on held-out fold
      - compute metrics + confusion matrix
      - generate and save per-fold confusion-matrix heatmaps
      - generate per-metric bar charts across folds
      - save CV summary CSV and confusion matrices NPZ
    """
    # Use train ∪ val as CV pool (same data used in main pipeline tuning)
    X_pool = np.vstack([X_tr, X_va])
    y_pool = np.concatenate([y_tr, y_va])

    n_folds = int(os.environ.get("CV_FOLDS", 5))
    seed = int(os.environ.get("SEED", 42))
    inner_val = float(os.environ.get("INNER_VAL", 0.2))

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed
    )

    # Store per-fold metrics
    metrics_store = {
        "accuracy": [],
        "f1_macro": [],
        "precision_macro": [],
        "recall_macro": [],
        "macro_roc_auc": [],
        "macro_avg_precision": [],
    }
    fold_cms = []
    start = time.time()

    for k, (idx_tr_fold, idx_te_fold) in enumerate(
        skf.split(X_pool, y_pool), 1
    ):
        print(f"\n[CV] Fold {k}/{skf.n_splits}")
        X_tr_fold, X_te_fold = X_pool[idx_tr_fold], X_pool[idx_te_fold]
        y_tr_fold, y_te_fold = y_pool[idx_tr_fold], y_pool[idx_te_fold]

        # Inner validation split within the current training fold
        tr_idx, va_idx = train_test_split(
            np.arange(len(y_tr_fold)),
            test_size=inner_val,
            stratify=y_tr_fold,
            random_state=seed + k,
        )
        Xtr, Xva = X_tr_fold[tr_idx], X_tr_fold[va_idx]
        ytr, yva = y_tr_fold[tr_idx], y_tr_fold[va_idx]

        # Transform per fold (no leakage from outer test)
        scaler, Xtr_tf = transform_fit(Xtr)
        Xva_tf = transform_apply(Xva, scaler)
        Xte_tf = transform_apply(X_te_fold, scaler)

        # Hyperparameter search + refit on (train+val) of current fold
        best_bst, best_params, best_va_auc, best_round, _ = fit_with_search_gpu(
            Xtr_tf, ytr, Xva_tf, yva, seed=seed + k
        )
        Xtrva_tf = np.vstack([Xtr_tf, Xva_tf])
        ytrva = np.concatenate([ytr, yva])
        bst_fold, _ = refit_final_gpu(
            best_params, best_round, Xtrva_tf, ytrva, seed=seed + k
        )

        # Evaluate on held-out outer fold
        proba_fold = _predict_proba_booster(bst_fold, Xte_tf)
        m = evaluate_full(y_te_fold, proba_fold, threshold=0.5)

        # Record metrics
        for key in metrics_store.keys():
            metrics_store[key].append(m[key])
        fold_cms.append(m["confusion_matrix"])

        _print_metrics_block(f"Fold {k} Metrics", m)

        # Confusion matrix heatmap per fold
        plt.figure()
        cm = m["confusion_matrix"]
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix — Fold {k}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        # Add numeric values into the heatmap cells
        for (i_row, i_col), val in np.ndenumerate(cm):
            plt.text(i_col, i_row, int(val), ha="center", va="center")

        _savefig(f"cm_fold_{k}_{REL}.png")
        plt.show()

    elapsed = time.time() - start

    # Summary printout across folds
    print("\n===== Outer CV Summary (XGB GPU, SAFE, no embeddings) =====")
    for metric in [
        "accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "macro_roc_auc",
        "macro_avg_precision",
    ]:
        mu = mean(metrics_store[metric])
        sd = pstdev(metrics_store[metric])
        print(f"{metric:20s}{mu:.4f} ± {sd:.4f}")
    print(
        f"[✓] CV done in {elapsed:.2f}s | "
        f"confusion matrices collected for {len(fold_cms)} folds"
    )

    # Per-metric bar charts across folds
    fold_ids = np.arange(1, len(fold_cms) + 1)

    def plot_metric_across_folds(name: str, values):
        """Bar chart for a metric across CV folds with a dashed mean line."""
        plt.figure()
        plt.bar(fold_ids, values)
        plt.axhline(mean(values), linestyle="--")
        plt.title(f"{name} across folds")
        plt.xlabel("Fold")
        plt.ylabel(name)
        _savefig(f"cv_{name.replace(' ', '_')}_{REL}.png")
        plt.show()

    plot_metric_across_folds("accuracy", metrics_store["accuracy"])
    plot_metric_across_folds("f1_macro", metrics_store["f1_macro"])
    plot_metric_across_folds("precision_macro", metrics_store["precision_macro"])
    plot_metric_across_folds("recall_macro", metrics_store["recall_macro"])
    plot_metric_across_folds("macro_roc_auc", metrics_store["macro_roc_auc"])
    plot_metric_across_folds(
        "macro_avg_precision",
        metrics_store["macro_avg_precision"],
    )

    # Save summary CSV and confusion matrices NPZ
    try:
        cv_summary = pd.DataFrame(
            [
                {
                    "accuracy_mean": mean(metrics_store["accuracy"]),
                    "accuracy_std": pstdev(metrics_store["accuracy"]),
                    "f1_macro_mean": mean(metrics_store["f1_macro"]),
                    "f1_macro_std": pstdev(metrics_store["f1_macro"]),
                    "precision_macro_mean": mean(
                        metrics_store["precision_macro"]
                    ),
                    "precision_macro_std": pstdev(
                        metrics_store["precision_macro"]
                    ),
                    "recall_macro_mean": mean(metrics_store["recall_macro"]),
                    "recall_macro_std": pstdev(
                        metrics_store["recall_macro"]
                    ),
                    "macro_roc_auc_mean": mean(
                        metrics_store["macro_roc_auc"]
                    ),
                    "macro_roc_auc_std": pstdev(
                        metrics_store["macro_roc_auc"]
                    ),
                    "macro_avg_precision_mean": mean(
                        metrics_store["macro_avg_precision"]
                    ),
                    "macro_avg_precision_std": pstdev(
                        metrics_store["macro_avg_precision"]
                    ),
                    "folds": len(fold_cms),
                }
            ]
        )
        csv_path = os.path.join(OUTDIR, f"cv_summary_{REL}.csv")
        cv_summary.to_csv(csv_path, index=False)

        cms_path = os.path.join(
            OUTDIR, f"cv_confusion_matrices_{REL}.npz"
        )
        np.savez_compressed(cms_path, *fold_cms)

        print(f"[✓] Saved CV summary to {csv_path}")
        print(f"[✓] Saved per-fold confusion matrices to {cms_path}")
    except Exception as e:
        print("[info] Skipped saving CV artifacts:", e)


if os.environ.get("USE_CV") == "1":
    run_cross_validation_extended_with_graphics()
