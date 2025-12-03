"""
Node-level XGBoost CV pipeline for tabular + graph data.

This script:
  - Configures paths, labels, imbalance handling, and feature options via
    user-editable "toggles" at the top.
  - Writes these settings into environment variables so downstream blocks
    can read a consistent configuration.
  - Loads node and edge CSV files, prepares labels (including optional
    blank-label handling), and builds an automatic feature matrix from
    numeric, categorical, and datetime columns.
  - Optionally adds graph-based PageRank features using the edges table.
  - Runs GPU-accelerated XGBoost with stratified K-fold cross-validation,
    including a lightweight hyperparameter search per fold.
  - Saves per-fold metrics, plots, feature importances, and (optionally)
    predictions for nodes with blank labels.
"""

import os
import pathlib
import json
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_string_dtype,
    CategoricalDtype,
)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.exceptions import UndefinedMetricWarning

import xgboost as xgb
import re

# USER Config
# These are the main switches for controlling data paths, labels, features, resampling strategy, model export, and prediction behavior.

ART = "../Data Generation"          # Folder containing nodes*.csv, edges*.csv
USE_NOISY = True                    # True -> use nodes_noisy.csv / edges_noisy.csv
OUTDIR_NAME = "baseline_metrics_xgb_cv"

# CV controls
CV_FOLDS = 5
INNER_VAL = 0.20
SEED = 42
GPU_ID = 0

# Labels (in nodes.csv)
LABEL_COLUMN = "label"
CLASS_ORDER = ["low", "medium", "high"]
BLANK_STRATEGY = "drop"        # "drop" = ignore blanks in training; "unknown" = add 4th class

# Imbalance handling
RESAMPLE = "ros"               # "none" | "ros" | "smote"
WEIGHT_MODE = "inv_freq"       # "inv_freq" or "none"
RESAMPLE_SEED = SEED

# Threshold tuning
USE_THRESHOLD_TUNING = True
THRESH_OBJECTIVE = "f1"        # "f1" | "recall" | "precision"
THRESH_FALLBACK = "argmax"

# Graph features (cheap + scalable)
ADD_ROLE_FEATS = True          # add PageRank if edges file is present
PAGERANK_ALPHA = 0.85

# Auto-feature settings
EXCLUDE_NODE_COLS = {
    "label", "y", "target", "split", "train_idx", "val_idx", "test_idx",
    "index", "id", "node_id"
}
CAT_TOP_K = 50
CAT_MIN_FREQ = 20
CAT_HASHING = False
HASH_N_FEATURES = 128
PARSE_DATETIME = True

# KEEP-LIST after feature build (supports prefix "*" wildcard)
USE_KEEP_FEATURES = True
KEEP_FEATURES = [
    "log_deg_in",
    "deg_in",
    "log_deg_out",
    "level_bin",
    "level_clip0",
    "level",
    "type_id",
    "level_nonneg",
    "deg",
    "type_name_AppInstance*", 
    "deg_out",
    "log_deg",
    "type_name_Service*", 
]

# Save per-fold probability & model export
SAVE_FOLD_PROBS = True
FOLD_PROBS_DIR = "node_cv_fold_preds"
EXPORT_MODEL = True
MODEL_DIR = "node_cv_models"
IMPORTANCE_TOPK = 30
IMPORTANCE_TYPE = "gain"       # "weight"|"gain"|"cover"|"total_gain"|"total_cover"

# Predict blanks after CV
PREDICT_BLANKS = True
BLANK_PRED_OUT = "node_blank_predictions.csv"

# GLOBAL SETUP


def _abspath(p):
    """Expand a path (including '~') and return its absolute string form."""
    return str(pathlib.Path(p).expanduser().resolve())


# Effective artifacts directory and output directory
ART_EFF = _abspath(ART)
OUTDIR = _abspath(os.path.join(ART_EFF, OUTDIR_NAME))
os.makedirs(OUTDIR, exist_ok=True)

# Select GPU via environment
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# Core env variables shared across blocks and potential subprocesses
os.environ.update({
    "ARTIFACTS": ART_EFF, "OUTDIR": OUTDIR, "USE_NOISY": "1" if USE_NOISY else "0",
    "CV_FOLDS": str(int(CV_FOLDS)), "INNER_VAL": str(float(INNER_VAL)),
    "SEED": str(int(SEED)), "GPU_ID": str(int(GPU_ID)),
    "LABEL_COLUMN": LABEL_COLUMN, "BLANK_STRATEGY": BLANK_STRATEGY,
    "RESAMPLE": RESAMPLE.lower(), "WEIGHT_MODE": WEIGHT_MODE.lower(), "RESAMPLE_SEED": str(int(RESAMPLE_SEED)),
    "USE_THRESHOLD_TUNING": "1" if USE_THRESHOLD_TUNING else "0", "THRESH_OBJECTIVE": THRESH_OBJECTIVE, "THRESH_FALLBACK": THRESH_FALLBACK,
    "ADD_ROLE_FEATS": "1" if ADD_ROLE_FEATS else "0", "PAGERANK_ALPHA": str(float(PAGERANK_ALPHA)),
    "SAVE_FOLD_PROBS": "1" if SAVE_FOLD_PROBS else "0", "FOLD_PROBS_DIR": FOLD_PROBS_DIR,
    "EXPORT_MODEL": "1" if EXPORT_MODEL else "0", "MODEL_DIR": MODEL_DIR,
    "IMPORTANCE_TOPK": str(int(IMPORTANCE_TOPK)), "IMPORTANCE_TYPE": IMPORTANCE_TYPE,
    "PREDICT_BLANKS": "1" if PREDICT_BLANKS else "0", "BLANK_PRED_OUT": BLANK_PRED_OUT,
    "USE_KEEP_FEATURES": "1" if USE_KEEP_FEATURES else "0",
    "KEEP_FEATURES": json.dumps(KEEP_FEATURES),
})

# Compute which files will actually be used
nodes_file = "nodes_noisy.csv" if USE_NOISY else "nodes.csv"
edges_file = "edges_noisy.csv" if USE_NOISY else "edges.csv"

# Print a JSON config summary for logging / reproducibility
print("=== Node Config ===")
print(json.dumps({
    "ARTIFACTS": ART_EFF, "OUTDIR": OUTDIR, "USE_NOISY": USE_NOISY,
    "CV_FOLDS": CV_FOLDS, "INNER_VAL": INNER_VAL, "SEED": SEED, "GPU_ID": GPU_ID,
    "nodes_file": nodes_file, "edges_file": edges_file,
    "LABEL_COLUMN": LABEL_COLUMN, "CLASS_ORDER": CLASS_ORDER, "BLANK_STRATEGY": BLANK_STRATEGY,
    "RESAMPLE": RESAMPLE, "WEIGHT_MODE": WEIGHT_MODE,
    "USE_THRESHOLD_TUNING": USE_THRESHOLD_TUNING, "THRESH_OBJECTIVE": THRESH_OBJECTIVE,
    "ADD_ROLE_FEATS": bool(int(os.environ["ADD_ROLE_FEATS"])), "PAGERANK_ALPHA": float(os.environ["PAGERANK_ALPHA"]),
    "AUTO_FEATURES": {
        "EXCLUDE_NODE_COLS": sorted(list(EXCLUDE_NODE_COLS)),
        "CAT_TOP_K": CAT_TOP_K, "CAT_MIN_FREQ": CAT_MIN_FREQ, "CAT_HASHING": CAT_HASHING,
        "HASH_N_FEATURES": HASH_N_FEATURES, "PARSE_DATETIME": PARSE_DATETIME
    },
    "KEEP_FEATURES": KEEP_FEATURES,
    "SAVE_FOLD_PROBS": SAVE_FOLD_PROBS, "EXPORT_MODEL": EXPORT_MODEL, "MODEL_DIR": MODEL_DIR,
    "IMPORTANCE_TOPK": IMPORTANCE_TOPK, "IMPORTANCE_TYPE": IMPORTANCE_TYPE,
    "PREDICT_BLANKS": PREDICT_BLANKS, "BLANK_PRED_OUT": BLANK_PRED_OUT
}, indent=2))

# RUNTIME SETUP

# Filter out noisy UndefinedMetric warnings from sklearn metrics
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="No positive class found in y_true, recall is set to one")

# Ensure OUTDIR exists (again, in case of separate execution context)
OUTDIR = os.environ["OUTDIR"]; os.makedirs(OUTDIR, exist_ok=True)

# Read core configuration from environment (set above)
ART = os.environ["ARTIFACTS"]; USE_NOISY = os.environ["USE_NOISY"] == "1"
LABEL_COLUMN = os.environ["LABEL_COLUMN"]; BLANK_STRATEGY = os.environ["BLANK_STRATEGY"].lower()

# Decide which CSV filenames to use
nodes_file = "nodes_noisy.csv" if USE_NOISY else "nodes.csv"
edges_file = "edges_noisy.csv" if USE_NOISY else "edges.csv"
nodes_path = os.path.join(ART, nodes_file); edges_path = os.path.join(ART, edges_file)

# Basic file existence checks
if not os.path.exists(nodes_path): raise FileNotFoundError(nodes_path)
if not os.path.exists(edges_path): print("[warn] edges file missing; graph features that need edges will be skipped.")

# Load node and edge tables
nodes = pd.read_csv(nodes_path, low_memory=False)
edges = pd.read_csv(edges_path, low_memory=False) if os.path.exists(edges_path) else None

# Fixed class order and mapping
class_order = ["low","medium","high"]; label_map = {c:i for i,c in enumerate(class_order)}

# Normalize label column: lowercase strings, stripped of whitespace
raw = nodes.get(LABEL_COLUMN, pd.Series([None]*len(nodes))).astype("string").str.strip().str.lower()

# Identify blanks or label values that are not in the allowed class set
is_blank = raw.isna() | (raw == "") | (~raw.isin(class_order))

# Handle blank labels according to configured BLANK_STRATEGY
if BLANK_STRATEGY == "unknown":
    # Turn blanks and out-of-vocabulary labels into a 4th "unknown" class
    nodes["y"] = raw.fillna("unknown"); nodes.loc[~nodes["y"].isin(class_order + ["unknown"]), "y"] = "unknown"
    class_names = class_order + ["unknown"]; label_map_ext = {c:i for i,c in enumerate(class_names)}
    y_all = nodes["y"].map(label_map_ext)   # no NaNs
    classes = np.arange(len(label_map_ext), dtype=int)
else:
    # Default: leave blanks as NaN (ignored later during training)
    nodes["y"] = raw.where(~is_blank)
    class_names = class_order; y_all = nodes["y"].map(label_map)  # NaNs for blanks
    classes = np.arange(len(class_order), dtype=int)

print(f"[labels] classes={class_names}")
print(f"[labels] labeled nodes: {nodes['y'].notna().sum()} / {len(nodes)} | blanks: {nodes['y'].isna().sum()}")

# AUTO FEATURES
# Build numeric / categorical / datetime feature matrix from node attributes.

# Settings for categorical handling
EXCLUDE_NODE_COLS = set(json.loads(json.dumps(list({
    "label","y","target","split","train_idx","val_idx","test_idx","index","id","node_id"
}))))
CAT_TOP_K = int(os.environ.get("CAT_TOP_K","50"))
CAT_MIN_FREQ = int(os.environ.get("CAT_MIN_FREQ","20"))
CAT_HASHING = os.environ.get("CAT_HASHING","0") == "1"
HASH_N_FEATURES = int(os.environ.get("HASH_N_FEATURES","128"))
PARSE_DATETIME = os.environ.get("PARSE_DATETIME","1") == "1"

df_nodes = nodes.copy()

# Ensure core degree fields exist (if absent, create as NaNs)
for col in ["deg_in", "deg_out", "deg", "level"]:
    if col not in df_nodes.columns: df_nodes[col] = np.nan

# Convert basic degree columns to numeric
df_nodes["deg_in"]  = pd.to_numeric(df_nodes["deg_in"], errors="coerce")
df_nodes["deg_out"] = pd.to_numeric(df_nodes["deg_out"], errors="coerce")
df_nodes["deg"]     = pd.to_numeric(df_nodes["deg"], errors="coerce")

# Engineered SAFE-like features using degrees and level
df_nodes["deg_ratio"]   = df_nodes["deg_in"] / (df_nodes["deg_out"].abs() + 1.0)
df_nodes["deg_balance"] = df_nodes["deg_in"] - df_nodes["deg_out"]
df_nodes["log_deg_in"]  = np.log1p(df_nodes["deg_in"].clip(lower=0))
df_nodes["log_deg_out"] = np.log1p(df_nodes["deg_out"].clip(lower=0))
df_nodes["log_deg"]     = np.log1p(df_nodes["deg"].clip(lower=0))

lvl = pd.to_numeric(df_nodes["level"], errors="coerce")
df_nodes["level_missing"] = lvl.isna().astype(np.int32)
df_nodes["level_nonneg"]  = (lvl >= 0).astype(np.int32)
df_nodes["level_clip0"]   = lvl.fillna(0).clip(lower=0)
try:
    q = pd.qcut(lvl.fillna(lvl.median()), q=4, labels=False, duplicates="drop")
    df_nodes["level_bin"] = q.astype(float).fillna(-1)
except Exception:
    df_nodes["level_bin"] = -1.0

# Columns that belong to engineered numeric block
engineered_cols = [
    "deg_in","deg_out","deg","level",
    "deg_ratio","deg_balance","log_deg_in","log_deg_out","log_deg",
    "level_missing","level_nonneg","level_clip0","level_bin"
]

# Candidate input columns (excluding label/index columns)
all_cols = [c for c in df_nodes.columns if c not in EXCLUDE_NODE_COLS]

# Containers for feature blocks and their names
feat_arrays, feat_names = [], []

def _add_matrix(names, mat):
    """Append a new 2D feature block and its column names to the global lists."""
    feat_arrays.append(np.asarray(mat, dtype=np.float32)); feat_names.extend(names)

# Engineered numeric block first
num_engineered = df_nodes[engineered_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32).values
_add_matrix(engineered_cols, num_engineered)

# Categorize remaining columns into numeric/bool/datetime/categorical
num_cols, bool_cols, dt_cols, cat_cols = [], [], [], []
for c in all_cols:
    if c in engineered_cols: continue
    s = df_nodes[c]
    if s.isna().all(): continue
    if is_bool_dtype(s):
        bool_cols.append(c)
    elif is_numeric_dtype(s):
        num_cols.append(c)
    elif is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif is_string_dtype(s) or isinstance(s.dtype, CategoricalDtype) or s.dtype == "object":
        cat_cols.append(c)
    else:
        # Try numeric fallback if dtype is ambiguous
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() > 0:
            df_nodes[c] = s_num; num_cols.append(c)
        else:
            cat_cols.append(c)

# Add extra numeric columns
extra_num = [c for c in num_cols if c not in engineered_cols]
if extra_num:
    _add_matrix(extra_num, df_nodes[extra_num].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32).values)

# Add boolean columns as 0/1 features
if bool_cols:
    _add_matrix(bool_cols, df_nodes[bool_cols].astype("Int64").fillna(0).astype(np.float32).values)

# Datetime expansion helper
def _dt_expand(series, prefix):
    """Expand a datetime-like column into useful numeric parts (year, month, etc.)."""
    s = pd.to_datetime(series, errors="coerce", utc=True)
    yr = s.dt.year.fillna(0).astype(np.float32).values
    mo = s.dt.month.fillna(0).astype(np.float32).values
    da = s.dt.day.fillna(0).astype(np.float32).values
    dow= s.dt.dayofweek.fillna(0).astype(np.float32).values
    is_wkend = ((dow >= 5).astype(np.float32))
    epoch = (s.view("int64") // 10**9).fillna(0).astype(np.float32).values
    mat = np.vstack([yr,mo,da,dow,is_wkend,epoch]).T
    names = [f"{prefix}_year", f"{prefix}_month", f"{prefix}_day", f"{prefix}_dow", f"{prefix}_isweekend", f"{prefix}_epoch_s"]
    return names, mat

# Expand all datetime columns
if PARSE_DATETIME and dt_cols:
    all_dt_names, all_dt_blocks = [], []
    for c in dt_cols:
        names, mat = _dt_expand(df_nodes[c], c)
        all_dt_names.extend(names); all_dt_blocks.append(mat)
    if all_dt_blocks:
        dt_block = np.hstack(all_dt_blocks) if len(all_dt_blocks)>1 else all_dt_blocks[0]
        _add_matrix(all_dt_names, dt_block)

# Categorical handling: top-k one-hot or hashing
def _categorical_topk_onehot(series, name, top_k=50, min_freq=20):
    """One-hot encode top_k frequent categories, mapping rare ones to '__OTHER__'."""
    ser = series.fillna("__NA__").astype(str)
    vc = ser.value_counts(dropna=False)
    keep = list(vc[vc >= max(1, min_freq)].index)[:top_k]
    ser2 = ser.where(ser.isin(keep), "__OTHER__")
    dummies = pd.get_dummies(ser2, prefix=name, dtype=np.float32)
    dummies = dummies.reindex(columns=sorted(dummies.columns), fill_value=0.0)
    return list(dummies.columns), dummies.values

def _categorical_hash(series, name, n_features=128):
    """Hash high-cardinality categoricals, using signed feature bins."""
    ser = series.fillna("__NA__").astype(str)
    N = len(ser); Xh = np.zeros((N, n_features), dtype=np.float32)
    for i, val in enumerate(ser):
        h = hash((name, val)); j = abs(h) % n_features; sign = -1.0 if (h & 1) else 1.0
        Xh[i, j] += sign
    names = [f"{name}_hash_{k}" for k in range(n_features)]
    return names, Xh

# Add categorical feature blocks
if cat_cols:
    use_hash = os.environ.get("CAT_HASHING","0") == "1"
    for c in cat_cols:
        s = df_nodes[c]; nunique = s.nunique(dropna=True)
        if use_hash and nunique > (CAT_TOP_K * 5):
            names, mat = _categorical_hash(s, c, HASH_N_FEATURES)
        else:
            names, mat = _categorical_topk_onehot(s, c, top_k=CAT_TOP_K, min_freq=CAT_MIN_FREQ)
        _add_matrix(names, mat)

# Final dense feature matrix
if feat_arrays:
    X_all = np.hstack(feat_arrays).astype(np.float32); node_feat_names = feat_names
else:
    # Fallback: if no blocks were created, use all numeric columns directly
    X_all = df_nodes.select_dtypes(include=[np.number]).fillna(0.0).astype(np.float32).values
    node_feat_names = list(df_nodes.select_dtypes(include=[np.number]).columns)

print(f"[auto-feats] X_all shape: {X_all.shape} | n_features={len(node_feat_names)}")

# OPTIONAL FEATURE SELECTION

USE_KEEP = os.environ.get("USE_KEEP_FEATURES","0") == "1"
KEEP_FEATURES = json.loads(os.environ.get("KEEP_FEATURES","[]"))

if USE_KEEP and KEEP_FEATURES:
    def _indices_from_patterns(all_names, patterns):
        """
        Convert a list of exact / prefix patterns into feature indices.

        Uses '*' suffix as a wildcard to match any feature name starting with
        the given prefix.
        """
        idx = []
        for i, n in enumerate(all_names):
            for p in patterns:
                if p.endswith("*"):
                    if n.startswith(p[:-1]): idx.append(i); break
                else:
                    if n == p: idx.append(i); break
        # stable unique
        seen=set(); out=[]
        for i in idx:
            if i not in seen:
                out.append(i); seen.add(i)
        return sorted(out)

    _keep_idx = _indices_from_patterns(node_feat_names, KEEP_FEATURES)
    if not _keep_idx:
        raise ValueError("No features matched KEEP_FEATURES. Inspect names in node_feat_names.")
    X_all = X_all[:, _keep_idx]
    node_feat_names = [node_feat_names[i] for i in _keep_idx]
    print(f"[select] Kept {len(node_feat_names)} features (from keep-list).")
else:
    print("[select] KEEP_FEATURES disabled or empty; using all auto-features.")

# GRAPH ROLE FEATURES

ADD_ROLE_FEATS = os.environ["ADD_ROLE_FEATS"] == "1"
if ADD_ROLE_FEATS and isinstance(edges, pd.DataFrame):
    try:
        # Import networkx so it remains optional
        import networkx as nx
        # Use first two columns for src/dst if "src"/"dst" not present
        if not {'src','dst'}.issubset(set(edges.columns)):
            c0, c1 = edges.columns[:2]; edges = edges.rename(columns={c0:"src", c1:"dst"})
        g = nx.DiGraph()
        g.add_edges_from(edges[["src","dst"]].dropna().astype(int).itertuples(index=False, name=None))
        pr = nx.pagerank(g, alpha=float(os.environ["PAGERANK_ALPHA"]))
        pr_vec = np.zeros(len(nodes), dtype=np.float32)
        for nidx, score in pr.items():
            if 0 <= nidx < len(nodes): pr_vec[nidx] = float(score)
        X_all = np.hstack([X_all, pr_vec.reshape(-1,1)]).astype(np.float32)
        node_feat_names = node_feat_names + ["pagerank"]
        print("[graph] added PageRank feature")
    except Exception as e:
        print("[graph] pagerank skipped:", e)
else:
    print("[graph] skipped extra graph features")

# RESAMPLING / WEIGHT HELPERS

def apply_resample(X, y):
    """
    Apply class imbalance handling based on RESAMPLE mode.

    Modes:
      - "none": no resampling.
      - "ros":  Random OverSampling: upsample minority classes to max count.
      - "smote": SMOTE oversampling (requires imblearn).
    """
    mode = os.environ["RESAMPLE"]
    if mode == "none":
        return X, y
    if mode == "ros":
        rng = np.random.default_rng(int(os.environ["RESAMPLE_SEED"]))
        counts = Counter(y); maxc = max(counts.values()); idxs = []
        for c, cnt in counts.items():
            idx_c = np.where(y == c)[0]
            if cnt < maxc:
                extra = rng.choice(idx_c, size=maxc - cnt, replace=True)
                idxs.append(np.concatenate([idx_c, extra]))
            else:
                idxs.append(idx_c)
        idx_all = np.concatenate(idxs)
        return X[idx_all], y[idx_all]
    if mode == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=int(os.environ["RESAMPLE_SEED"]))
            return sm.fit_resample(X, y)
        except Exception:
            print("[resample] SMOTE requested but imblearn not available; using ROS.")
            os.environ["RESAMPLE"]="ros"; return apply_resample(X, y)
    return X, y

def make_sample_weights(y_):
    """
    Compute inverse-frequency sample weights if WEIGHT_MODE == 'inv_freq',
    otherwise return None.
    """
    if os.environ["WEIGHT_MODE"] != "inv_freq":
        return None
    cnt = Counter(y_); total = len(y_); classes_ = list(cnt.keys())
    weight = {c: total/(len(classes_)*cnt[c]) for c in classes_}
    return np.array([weight[int(t)] for t in y_], dtype=np.float32)

# XGBOOST UTILITIES

def dmatrix(X_, y_=None, weight=None):
    """Create an xgboost DMatrix, optionally with labels and weights."""
    dm = xgb.DMatrix(X_, label=y_) if y_ is not None else xgb.DMatrix(X_)
    if weight is not None:
        if len(weight) != X_.shape[0]: raise ValueError(f"Weight length {len(weight)} != n_samples {X_.shape[0]}")
        dm.set_weight(weight)
    return dm

def train_with_es(Xtr, ytr, wtr, Xva, yva, params, num_boost_round):
    """
    Train XGBoost model with early stopping on validation set.

    Returns:
        bst        - trained booster
        best_round - index of best iteration
    """
    dtr = dmatrix(Xtr, ytr, wtr); dva = dmatrix(Xva, yva)
    watch = [(dtr,"train"),(dva,"valid")]
    bst = xgb.train(params, dtr, num_boost_round=num_boost_round, evals=watch, early_stopping_rounds=100, verbose_eval=False)
    best_round = getattr(bst, "best_iteration", None)
    if best_round is None: best_round = int(bst.best_ntree_limit) - 1
    return bst, int(best_round)

def predict_proba_booster(bst, X_): return bst.predict(dmatrix(X_))

def macro_scores_safe(y_true, proba, classes_):
    """
    Compute macro ROC AUC and macro Average Precision, skipping degenerate classes.

    Classes with all positives or all negatives are excluded from the macro average
    to avoid undefined metrics.
    """
    y_bin = label_binarize(y_true, classes=classes_); rocs, aps = [], []
    for k in range(len(classes_)):
        yk = y_bin[:, k]; s = int(yk.sum())
        if not (0 < s < len(yk)): continue
        try: rocs.append(roc_auc_score(yk, proba[:, k]))
        except Exception: pass
        try: aps.append(average_precision_score(yk, proba[:, k]))
        except Exception: pass
    roc_macro = float(np.mean(rocs)) if rocs else float("nan")
    ap_macro  = float(np.mean(aps))  if aps  else float("nan")
    return roc_macro, ap_macro

def _best_threshold_by_objective(y_true_bin, scores, objective="f1"):
    """
    Among points on the PR curve, pick the threshold that maximizes the
    requested objective ('f1', 'recall', or 'precision').
    """
    p, r, t = precision_recall_curve(y_true_bin, scores)
    if len(t) == 0: return 1.0
    if objective == "recall": vals = r[1:]
    elif objective == "precision": vals = p[1:]
    else: vals = 2 * p[1:] * r[1:] / np.clip(p[1:] + r[1:], 1e-12, None)
    idx = int(np.nanargmax(vals)); return float(t[idx])

def compute_pr_thresholds(y_true, proba, classes, objective="f1"):
    """
    Compute per-class thresholds using precision-recall curves.

    Returns:
        thresholds: array of length K (number of classes).
    """
    y_bin = label_binarize(y_true, classes=classes); K = len(classes)
    th = np.full(K, 1.0, dtype=np.float32)
    for k in range(K):
        yk = y_bin[:, k]; s = int(yk.sum())
        if not (0 < s < len(yk)): continue
        try: th[k] = _best_threshold_by_objective(yk, proba[:, k], objective=objective)
        except Exception: pass
    return th

def predict_with_thresholds(proba, thresholds, fallback="argmax"):
    """
    Apply a per-class threshold to probabilities and return predicted labels.

    Currently uses a simple ratio proba/threshold and picks the argmax class.
    """
    eps = 1e-9; scores = proba / np.maximum(thresholds.reshape(1,-1), eps)
    pred = scores.argmax(axis=1)
    return pred

# Small grid of XGBoost hyperparameters to search over per fold
XGB_GRID = [
    dict(max_depth=4,  eta=0.10, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,  num_boost_round=1200),
    dict(max_depth=6,  eta=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,  num_boost_round=1600),
    dict(max_depth=8,  eta=0.06, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,  num_boost_round=2000),
    dict(max_depth=10, eta=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,  num_boost_round=2500),
]

# MAIN CV

def run_cv_and_predict_blanks(X_all, y_series, feat_names, classes):
    """
    Run stratified K-fold CV with XGBoost, generate plots, and optionally
    train a final model to predict labels for blank nodes.

    Parameters
    ----------
    X_all : np.ndarray
        Feature matrix for all nodes.
    y_series : pd.Series
        Integer-encoded labels with NaNs for blanks.
    feat_names : list[str]
        Names of features corresponding to X_all columns.
    classes : np.ndarray
        Array of class indices.
    """
    # Indices of nodes that actually have labels
    labeled_idx = np.where(~y_series.isna())[0]
    X_lab = X_all[labeled_idx]
    y_lab = y_series.iloc[labeled_idx].astype(int).to_numpy()

    print("\n>>> Stratified K-fold CV (XGB, GPU)")
    skf = StratifiedKFold(n_splits=int(os.environ["CV_FOLDS"]), shuffle=True, random_state=int(os.environ["SEED"]))

    fold_rows = []
    cumulative_cm = np.zeros((len(classes), len(classes)), dtype=np.int64)

    # Optional per-fold probability saving
    save_fold_probs = os.environ["SAVE_FOLD_PROBS"] == "1"
    preds_dir = os.path.join(OUTDIR, os.environ["FOLD_PROBS_DIR"])
    if save_fold_probs: os.makedirs(preds_dir, exist_ok=True)

    # Optional model export
    export_model = os.environ["EXPORT_MODEL"] == "1"
    model_dir = os.path.join(OUTDIR, os.environ["MODEL_DIR"])
    if export_model: os.makedirs(model_dir, exist_ok=True)

    all_importance_maps = []

    # Base XGB params used for grid search
    params_base = dict(device="cuda", tree_method="hist", objective="multi:softprob",
                       num_class=len(classes), eval_metric="mlogloss", seed=int(os.environ["SEED"]))

    def search_xgb(X_tr, y_tr, X_va, y_va):
        """
        Run a small hyperparameter search (XGB_GRID) on the inner train/val split,
        returning the best config and its validation predictions.
        """
        X_tr_rs, y_tr_rs = apply_resample(X_tr, y_tr)
        w_tr_rs = make_sample_weights(y_tr_rs)
        best = (-np.inf, None, None, None, None)
        t0 = time.time()
        for i, pg in enumerate(XGB_GRID, 1):
            params = dict(params_base); params.update({k:v for k,v in pg.items() if k!="num_boost_round"})
            nrounds = int(pg.get("num_boost_round", 2000))
            bst, br = train_with_es(X_tr_rs, y_tr_rs, w_tr_rs, X_va, y_va, params, nrounds)
            p_val = predict_proba_booster(bst, X_va)
            ap_macro = macro_scores_safe(y_va, p_val, classes)[1]
            if np.isfinite(ap_macro) and ap_macro > best[0]:
                best = (ap_macro, params, bst, br, p_val)
            print(f"[SEARCH] {i}/{len(XGB_GRID)} macroAP={ap_macro:.4f} br={br}")
        tune_sec = time.time() - t0
        return best, tune_sec

    start = time.time()

    # Outer CV loop

    for fold, (idx_tr, idx_te) in enumerate(skf.split(X_lab, y_lab), 1):
        X_train, X_test = X_lab[idx_tr], X_lab[idx_te]
        y_train, y_test = y_lab[idx_tr], y_lab[idx_te]

        # Inner split (train/val) from the fold's training portion
        tr_idx, va_idx = train_test_split(np.arange(len(y_train)), test_size=float(os.environ["INNER_VAL"]), stratify=y_train, random_state=int(os.environ["SEED"])+fold)
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]; y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        # Hyperparameter search on inner train/val
        (best_ap, best_params, best_bst, best_round, val_proba), tune_sec = search_xgb(X_tr, y_tr, X_va, y_va)
        X_trva = np.vstack([X_tr, X_va]); y_trva = np.concatenate([y_tr, y_va])

        # Refit best model on full train+val data for this fold
        X_trva_rs, y_trva_rs = apply_resample(X_trva, y_trva); w_trva = make_sample_weights(y_trva_rs)
        dtrva = xgb.DMatrix(X_trva_rs, label=y_trva_rs, weight=w_trva)
        nrounds = int(best_round)+1 if best_round is not None else 2000
        bst_fold = xgb.train(best_params, dtrva, num_boost_round=nrounds, verbose_eval=False)

        # Threshold tuning using inner-val predictions
        if os.environ["USE_THRESHOLD_TUNING"] == "1":
            thresholds = compute_pr_thresholds(y_va, val_proba, classes, objective=os.environ["THRESH_OBJECTIVE"])
        else:
            thresholds = np.full(len(classes), 1.0, dtype=np.float32)

        # Evaluate on held-out test fold
        proba_fold = predict_proba_booster(bst_fold, X_test)
        pred_fold_thr = predict_with_thresholds(proba_fold, thresholds, fallback=os.environ["THRESH_FALLBACK"])

        if save_fold_probs:
            np.save(os.path.join(preds_dir, f"fold_{fold:02d}_y.npy"), y_test)
            np.save(os.path.join(preds_dir, f"fold_{fold:02d}_proba.npy"), proba_fold)

        macro_roc, macro_ap = macro_scores_safe(y_test, proba_fold, classes)
        acc = accuracy_score(y_test, pred_fold_thr); f1m = f1_score(y_test, pred_fold_thr, average="macro")
        prm = precision_score(y_test, pred_fold_thr, average="macro"); rem = recall_score(y_test, pred_fold_thr, average="macro")
        print(f"[CV {fold}/{os.environ['CV_FOLDS']}] f1={f1m:.4f} acc={acc:.4f} AUC={macro_roc if np.isfinite(macro_roc) else float('nan'):.4f} AP={macro_ap if np.isfinite(macro_ap) else float('nan'):.4f}")

        # Confusion matrix across folds
        cm_fold = confusion_matrix(y_test, pred_fold_thr, labels=classes); cumulative_cm += cm_fold

        # Save model per fold if requested
        if export_model:
            model_path = os.path.join(model_dir, f"node_xgb_fold_{fold:02d}.json"); bst_fold.save_model(model_path); print(f"[model] saved {model_path}")

        # Collect per-fold feature importance maps for later aggregation
        try:
            importance_type = os.environ.get("IMPORTANCE_TYPE","gain"); fmap = bst_fold.get_score(importance_type=importance_type)
            all_importance_maps.append(fmap)
        except Exception as e:
            print("[warn] importance extraction failed:", e)

        # Record metrics for this fold
        fold_rows.append(dict(
            mode="cv", fold=fold, tune_seconds=round(tune_sec,3),
            accuracy=acc, f1_macro=f1m, precision_macro=prm, recall_macro=rem,
            macro_roc_auc=(macro_roc if np.isfinite(macro_roc) else np.nan),
            macro_avg_precision=(macro_ap if np.isfinite(macro_ap) else np.nan)
        ))

    elapsed = time.time() - start
    print(f"\n[✓] CV done in {elapsed:.2f}s")

    # Save CV metrics + text summary
    df_fold = pd.DataFrame(fold_rows).sort_values("fold")
    csv_path = os.path.join(OUTDIR, "node_xgb_cv_metrics.csv"); df_fold.to_csv(csv_path, index=False)
    print(f"CV fold metrics saved to {csv_path}")

    # Overall mean ± std for key metrics
    agg = df_fold[["accuracy","f1_macro","precision_macro","recall_macro","macro_roc_auc","macro_avg_precision"]].agg(['mean','std'])
    print("\n===== Outer CV Summary (Node XGB GPU, SAFE) =====")
    for col in agg.columns:
        mu = agg.loc['mean', col]; sd = agg.loc['std', col]
        if np.isnan(mu): print(f"{col:>18s}: n/a")
        else:            print(f"{col:>18s}: {mu:.4f} ± {sd:.4f}")

    # Visualize metrics across folds
    folds = df_fold["fold"].to_numpy(); unique_folds = np.array(sorted(df_fold["fold"].unique()))
    cmap = plt.get_cmap("tab20"); fold_to_color = {f: cmap((i % 20) / 20.0) for i, f in enumerate(unique_folds)}
    metrics = [("accuracy","Accuracy"),("f1_macro","F1 (macro)"),("precision_macro","Precision (macro)"),("recall_macro","Recall (macro)"),("macro_roc_auc","Macro ROC AUC"),("macro_avg_precision","Macro PR AUC")]
    for key, nice in metrics:
        if key not in df_fold.columns: continue
        x = df_fold["fold"].to_numpy(dtype=int); yv = df_fold[key].to_numpy(dtype=float)
        plt.figure(figsize=(7,4))
        for xf, yf in zip(x, yv): plt.scatter([xf], [yf], s=60, color=fold_to_color[xf], label=f"Fold {int(xf)}")
        plt.plot(x, yv, linewidth=1, alpha=0.3, color="black")
        mu, sd = np.nanmean(yv), np.nanstd(yv); plt.axhline(mu, linestyle="--", linewidth=1.5, alpha=0.6, label=f"Mean = {mu:.4f}")
        plt.fill_between([min(x)-0.2, max(x)+0.2], mu - sd, mu + sd, alpha=0.12, label=f"±1 SD ({sd:.4f})")
        plt.title(f"{nice} across folds"); plt.xlabel("Fold"); plt.ylabel(nice); plt.grid(True, linestyle="--", alpha=0.4)
        handles, labels = plt.gca().get_legend_handles_labels(); seen=set(); h_dedup=[]; l_dedup=[]
        for h, lb in zip(handles, labels):
            if lb not in seen: h_dedup.append(h); l_dedup.append(lb); seen.add(lb)
        plt.legend(h_dedup, l_dedup, bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0.); plt.tight_layout()
        fn = os.path.join(OUTDIR, f"node_cv_metric_{key}_allfolds.png"); plt.savefig(fn, dpi=150); plt.show(); print(f"[plot] saved {fn}")

    # Confusion matrices
    plt.figure(); ConfusionMatrixDisplay(cumulative_cm, display_labels=[str(c) for c in classes]).plot(values_format="d")
    plt.title("Node CV Confusion Matrix — Overall (counts)"); plt.tight_layout()
    fn_cm_counts = os.path.join(OUTDIR, "node_cv_cm_overall_counts.png"); plt.savefig(fn_cm_counts, dpi=150); plt.show(); print(f"[plot] saved {fn_cm_counts}")
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cumulative_cm.sum(axis=1, keepdims=True); norm_cm = np.divide(cumulative_cm, row_sums, out=np.zeros_like(cumulative_cm, dtype=float), where=row_sums!=0)
    plt.figure(); ConfusionMatrixDisplay(norm_cm, display_labels=[str(c) for c in classes]).plot(values_format=".2f")
    plt.title("Node CV Confusion Matrix — Overall (row-normalized)"); plt.tight_layout()
    fn_cm_norm = os.path.join(OUTDIR, "node_cv_cm_overall_row_normalized.png"); plt.savefig(fn_cm_norm, dpi=150); plt.show(); print(f"[plot] saved {fn_cm_norm}")

    # Aggregated importances
    try:
        # Aggregate using mean across folds; we stored per-fold fmap dicts.
        agg_scores = {name: 0.0 for name in feat_names}
        for fmap in all_importance_maps:
            for fx, score in fmap.items():
                try:
                    idx = int(fx[1:])
                    if 0 <= idx < len(feat_names): agg_scores[feat_names[idx]] += float(score)
                except Exception: pass
        topk = int(os.environ.get("IMPORTANCE_TOPK","30")); items = sorted(agg_scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        if items:
            names, scores = zip(*items); plt.figure(figsize=(8, max(4, 0.25*len(names)+1)))
            y_pos = np.arange(len(names)); plt.barh(y_pos, scores); plt.yticks(y_pos, names); plt.gca().invert_yaxis()
            plt.xlabel(f"Importance ({os.environ.get('IMPORTANCE_TYPE','gain')})"); plt.title("Top Node Feature Importances (aggregated across folds)")
            plt.tight_layout(); fn_imp = os.path.join(OUTDIR, "node_cv_top_importances.png"); plt.savefig(fn_imp, dpi=150); plt.show(); print(f"[plot] saved {fn_imp}")
        else:
            print("[info] No non-zero importances collected.")
    except Exception as e:
        print("[warn] importance plotting failed:", e)

    # Predict blanks (optional)
    if os.environ["PREDICT_BLANKS"] == "1":
        blank_idx = np.where(y_series.isna())[0]
        if blank_idx.size == 0:
            print("[predict] No blank nodes detected; skipping.")
        else:
            # Use a fresh train/val split on all labeled points for threshold tuning
            tr_idx, va_idx = train_test_split(np.arange(len(y_lab)), test_size=float(os.environ["INNER_VAL"]), stratify=y_lab, random_state=int(os.environ["SEED"])+777)
            X_tr, X_va = X_lab[tr_idx], X_lab[va_idx]; y_tr, y_va = y_lab[tr_idx], y_lab[va_idx]

            params = dict(device="cuda", tree_method="hist", objective="multi:softprob", num_class=len(classes), eval_metric="mlogloss", seed=int(os.environ["SEED"]))
            bst_tmp, best_round = train_with_es(X_tr, y_tr, make_sample_weights(y_tr), X_va, y_va, params, 1500)
            val_proba = predict_proba_booster(bst_tmp, X_va)
            thresholds = compute_pr_thresholds(y_va, val_proba, classes, objective=os.environ["THRESH_OBJECTIVE"])

            # Fit final model on all labeled nodes with resampling/weights
            X_trva_rs, y_trva_rs = apply_resample(X_lab, y_lab); w_trva = make_sample_weights(y_trva_rs)
            dtrva = xgb.DMatrix(X_trva_rs, label=y_trva_rs, weight=w_trva)
            nrounds = int(best_round)+1 if best_round is not None else 1500
            bst_final = xgb.train(params, dtrva, num_boost_round=nrounds, verbose_eval=False)

            # Predict for all nodes, then filter down to blanks
            proba_all = predict_proba_booster(bst_final, X_all)
            preds_all = predict_with_thresholds(proba_all, thresholds, fallback=os.environ["THRESH_FALLBACK"])

            name_map = {i:n for i,n in enumerate(["low","medium","high"][:len(classes)])}
            out = pd.DataFrame({"node_id": np.arange(len(nodes)), "pred_label": preds_all})
            out["pred_name"] = out["pred_label"].map(name_map)
            for k in range(len(classes)): out[f"proba_{name_map[k]}"] = proba_all[:,k]
            out = out.loc[blank_idx].reset_index(drop=True)

            out_path = os.path.join(OUTDIR, os.environ["BLANK_PRED_OUT"])
            out.to_csv(out_path, index=False); print(f"[predict] wrote {len(out)} blank-node predictions -> {out_path}")

    return pd.DataFrame(fold_rows).sort_values("fold")

# Prepare classes array (3 fixed classes)
classes = np.arange(3, dtype=int)
