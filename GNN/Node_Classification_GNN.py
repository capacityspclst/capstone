"""
This script configures and runs a full Graph Neural Network (GNN) for
node classification on a graph dataset.

Script Summary:

1. USER CONFIG
   - Set all options here: where the data lives, whether to use
     noisy data, batch size, learning rate, number of layers, etc.
   - These settings are turned into environment variables so that the rest of
     the script read a single, consistent
     configuration.

2. PATHS & SANITY CHECKS
   - Resolves all file paths (nodes.csv, edges.csv, splits, output directory).
   - Prints a summary of what files it expects and whether they exist.

3. DATA LOADING & FEATURE ENGINEERING
   - Reads the node and edge CSV files.
   - Converts the human-readable labels (low/medium/high) to integer labels.
   - Automatically finds numeric columns to use as features.
   - Optionally adds "graph structure" features such as PageRank, core number,
     and clustering coefficient.

4. OPTIONAL NODE2VEC FEATURES
   - Runs Node2Vec to learn extra embedding features for each node using the
     graph structure.
   - Caches these embeddings to disk so the next run can reload them
     instead of retraining.

5. MODEL & TRAINING LOOP (GraphSAGE)
   - Builds a GraphSAGE model using PyTorch Geometric.
   - Trains with:
       * Label smoothing
       * Class weights (to handle imbalanced labels)
       * DropEdge (randomly dropping edges as regularization)
       * Consistency regularization (helps with stable predictions)
       * EMA (Exponential Moving Average) of weights for more stable test-time
         predictions
       * Label Propagation and Test-Time Augmentation (TTA) during evaluation.

6. EVALUATION & OUTPUT
   - Either:
       * Trains once with a fixed train/val/test split, OR
       * Runs K-fold cross-validation.
   - Computes many metrics: accuracy, F1, AUC, precision/recall, etc.
   - Saves:
       * JSON files with detailed metrics.
       * PNG plots for confusion matrices, ROC curves, PR curves, and training
         curves.
   - Prints warnings if any label classes are missing in validation or test
     splits so you can interpret the metrics correctly.

"""

import os, pathlib, json
from dataclasses import dataclass
from typing import Optional


# USER TOGGLES

@dataclass
class NodeClsToggles:
    # Paths
    art: str = "../Data Generation"          # Base folder with nodes.csv / *_idx.npy
    use_noisy: bool = True                   # Use nodes_noisy.csv / edges_noisy.csv when available
    outdir_name: str = "baseline_metrics"    # Subfolder inside artifacts for outputs
    noisy_art_override: Optional[str] = None # Optional different folder for noisy data

    # Experiment controls
    use_cv: bool = True          # True -> run K-fold CV; False -> single run
    cv_folds: int = 3
    fixed_test: bool = True       # Only matters when use_cv=True
    val_ratio: float = 0.10       # Validation fraction when rotating val/test
    seed: int = 42

    # Hardware
    gpu_id: int = 0
    amp: bool = True              # Automatic mixed precision

    # Model / sampler
    batch_size: int = 4096
    epochs: int = 200
    hidden: int = 512
    num_layers: int = 4
    dropout: float = 0.35
    fanouts: str = "75,50,25,10"  # Comma-separated fanouts per hop

    # Optimizer / schedule
    lr: float = 3e-3
    wd: float = 1e-4
    adamw: bool = True
    warmup_epochs: int = 10
    early_patience: int = 80

    # Regularization & extras
    edge_drop_p: float = 0.05
    label_smooth: float = 0.05
    use_class_weights: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    grad_clip_norm: float = 2.0

    # Consistency regularization
    consistency: bool = True
    cons_lambda: float = 0.08

    # Label Propagation + TTA
    use_lp: bool = True
    use_lp_val: bool = True
    lp_alpha: float = 0.5
    lp_steps: int = 10
    tta_k: int = 5

    # Node2Vec features (+ cache)
    use_n2v: bool = True
    n2v_dim: int = 64
    n2v_walk_len: int = 20
    n2v_context: int = 10
    n2v_walks_per_node: int = 4
    n2v_epochs: int = 5
    n2v_lr: float = 0.01
    n2v_sparse: bool = True
    n2v_batch_size: int = 1024
    n2v_cache: bool = True

    # Structural features
    enable_pagerank: bool = True
    enable_corenum: bool = False
    enable_clustering: bool = False

    # Viz
    vis: bool = True


TOG = NodeClsToggles()

# Keep original global names for compatibility with the rest of the script.

ART = TOG.art
USE_NOISY = TOG.use_noisy
OUTDIR_NAME = TOG.outdir_name
NOISY_ART_OVERRIDE = TOG.noisy_art_override

USE_CV = TOG.use_cv
CV_FOLDS = TOG.cv_folds
FIXED_TEST = TOG.fixed_test
VAL_RATIO = TOG.val_ratio
SEED = TOG.seed

GPU_ID = TOG.gpu_id
AMP = TOG.amp

BATCH_SIZE = TOG.batch_size
EPOCHS = TOG.epochs
HIDDEN = TOG.hidden
NUM_LAYERS = TOG.num_layers
DROPOUT = TOG.dropout
FANOUTS = [int(x) for x in TOG.fanouts.split(",") if x]

LR = TOG.lr
WD = TOG.wd
ADAMW = TOG.adamw
WARMUP_EPOCHS = TOG.warmup_epochs
EARLY_PATIENCE = TOG.early_patience

EDGE_DROP_P = TOG.edge_drop_p
LABEL_SMOOTH = TOG.label_smooth
USE_CLASS_WEIGHTS = TOG.use_class_weights
USE_EMA = TOG.use_ema
EMA_DECAY = TOG.ema_decay
GRAD_CLIP_NORM = TOG.grad_clip_norm

CONSISTENCY = TOG.consistency
CONS_LAMBDA = TOG.cons_lambda

USE_LP = TOG.use_lp
USE_LP_VAL = TOG.use_lp_val
LP_ALPHA = TOG.lp_alpha
LP_STEPS = TOG.lp_steps
TTA_K = TOG.tta_k

USE_N2V = TOG.use_n2v
N2V_DIM = TOG.n2v_dim
N2V_WALK_LEN = TOG.n2v_walk_len
N2V_CONTEXT = TOG.n2v_context
N2V_WALKS_PER_NODE = TOG.n2v_walks_per_node
N2V_EPOCHS = TOG.n2v_epochs
N2V_LR = TOG.n2v_lr
N2V_SPARSE = TOG.n2v_sparse
N2V_BATCH_SIZE = TOG.n2v_batch_size
N2V_CACHE = TOG.n2v_cache

ENABLE_PAGERANK = TOG.enable_pagerank
ENABLE_CORENUM = TOG.enable_corenum
ENABLE_CLUSTERING = TOG.enable_clustering

VIS = TOG.vis

def _abspath(p: str) -> str:
    """Expand ~ and make an absolute path."""
    return str(pathlib.Path(p).expanduser().resolve())


# Resolve artifact directory
if USE_NOISY and NOISY_ART_OVERRIDE:
    ART_EFF = _abspath(NOISY_ART_OVERRIDE)
else:
    ART_EFF = _abspath(ART)

# Output directory where all metrics, plots, and cached artifacts will go.
OUTDIR = _abspath(os.path.join(ART_EFF, OUTDIR_NAME))
os.makedirs(OUTDIR, exist_ok=True)

# Turn toggles into environment variables (rest of the script reads these).
env_pairs = {
    "ARTIFACTS": ART_EFF,
    "OUTDIR": OUTDIR,
    "USE_NOISY": "1" if USE_NOISY else "0",
    "USE_CV": "1" if USE_CV else "0",
    "CV_FOLDS": str(int(CV_FOLDS)),
    "FIXED_TEST": "1" if FIXED_TEST else "0",
    "VAL_RATIO": str(float(VAL_RATIO)),
    "SEED": str(int(SEED)),
    "GPU_ID": str(int(GPU_ID)),
    "BATCH_SIZE": str(int(BATCH_SIZE)),
    "EPOCHS": str(int(EPOCHS)),
    "HIDDEN": str(int(HIDDEN)),
    "NUM_LAYERS": str(int(NUM_LAYERS)),
    "DROPOUT": str(float(DROPOUT)),
    "FANOUTS": ",".join(str(int(x)) for x in FANOUTS),
    "LR": str(float(LR)),
    "WD": str(float(WD)),
    "ADAMW": "1" if ADAMW else "0",
    "WARMUP_EPOCHS": str(int(WARMUP_EPOCHS)),
    "EARLY_PATIENCE": str(int(EARLY_PATIENCE)),
    "EDGE_DROP_P": str(float(EDGE_DROP_P)),
    "LABEL_SMOOTH": str(float(LABEL_SMOOTH)),
    "USE_CLASS_WEIGHTS": "1" if USE_CLASS_WEIGHTS else "0",
    "USE_EMA": "1" if USE_EMA else "0",
    "EMA_DECAY": str(float(EMA_DECAY)),
    "GRAD_CLIP_NORM": str(float(GRAD_CLIP_NORM)),
    "CONSISTENCY": "1" if CONSISTENCY else "0",
    "CONS_LAMBDA": str(float(CONS_LAMBDA)),
    "USE_LP": "1" if USE_LP else "0",
    "USE_LP_VAL": "1" if USE_LP_VAL else "0",
    "LP_ALPHA": str(float(LP_ALPHA)),
    "LP_STEPS": str(int(LP_STEPS)),
    "TTA_K": str(int(TTA_K)),
    "USE_N2V": "1" if USE_N2V else "0",
    "N2V_DIM": str(int(N2V_DIM)),
    "N2V_WALK_LEN": str(int(N2V_WALK_LEN)),
    "N2V_CONTEXT": str(int(N2V_CONTEXT)),
    "N2V_WALKS_PER_NODE": str(int(N2V_WALKS_PER_NODE)),
    "N2V_EPOCHS": str(int(N2V_EPOCHS)),
    "N2V_LR": str(float(N2V_LR)),
    "N2V_SPARSE": "1" if N2V_SPARSE else "0",
    "N2V_BATCH_SIZE": str(int(N2V_BATCH_SIZE)),
    "N2V_CACHE": "1" if N2V_CACHE else "0",
    "AMP": "1" if AMP else "0",
    "VIS": "1" if VIS else "0",
    "ENABLE_PAGERANK": "1" if ENABLE_PAGERANK else "0",
    "ENABLE_CORENUM": "1" if ENABLE_CORENUM else "0",
    "ENABLE_CLUSTERING": "1" if ENABLE_CLUSTERING else "0",
}
for k, v in env_pairs.items():
    os.environ[k] = v

# Decide which node/edge CSV files to use based on USE_NOISY.
nodes_file = "nodes_noisy.csv" if USE_NOISY else "nodes.csv"
edges_file = "edges_noisy.csv" if USE_NOISY else "edges.csv"

def _exists(p: str) -> bool:
    """Safe existence check."""
    try:
        return os.path.exists(p)
    except Exception:
        return False

expected = {
    "ARTIFACTS": ART_EFF,
    "nodes": os.path.join(ART_EFF, nodes_file),
    "edges (optional)": os.path.join(ART_EFF, edges_file),
    "label_int.npy (optional)": os.path.join(ART_EFF, "label_int.npy"),
    "train_idx.npy": os.path.join(ART_EFF, "train_idx.npy"),
    "val_idx.npy":   os.path.join(ART_EFF, "val_idx.npy"),
    "test_idx.npy":  os.path.join(ART_EFF, "test_idx.npy"),
    "OUTDIR (will be created)": OUTDIR,
}

print("=== Effective Run Config ===")
print(json.dumps({**env_pairs, "nodes_file": nodes_file, "edges_file": edges_file}, indent=2))

print("\n=== Expected Files / Dirs ===")
for k, p in expected.items():
    flag = "✓" if _exists(p) else ("(create)" if k.startswith("OUTDIR") else "✗")
    print(f"{k:>24s}: {p} {flag}")

print("\nNotes:")
print("- Labels will be read from the `label` column in nodes_*.csv (low/medium/high). "
      "Blanks become -1 (unlabeled).")
print("- Optional heavy structural features (PageRank/core/clustering) are off by default for large graphs.")



# GraphSAGE

# From this point on, we consume the configuration above and actually run the GNN pipeline

import os, numpy as np, pandas as pd, torch, torch.nn.functional as F, random, inspect, hashlib, json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support, roc_curve, auc, precision_recall_curve,
    average_precision_score, balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef, log_loss
)
import matplotlib.pyplot as plt
from torch import amp

# Config
# Read configuration from environment variables
ART = os.environ.get("ARTIFACTS", os.path.join("..", "Data Generation"))
OUTDIR = os.environ.get("OUTDIR", ART)
os.makedirs(OUTDIR, exist_ok=True)
USE_NOISY = os.environ.get("USE_NOISY", "0") == "1"

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096))
EPOCHS = int(os.environ.get("EPOCHS", 200))
HIDDEN = int(os.environ.get("HIDDEN", 512))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 4))
DROPOUT = float(os.environ.get("DROPOUT", 0.35))

# Fanouts
# FANOUTS may be passed as a comma-separated string (e.g. "75,50,25,10")
FANOUTS_ENV = os.environ.get("FANOUTS", "").strip()

def default_fanouts(k): 
    # If fanouts were not explicitly set, fall back to default up to 4 layers
    return [75, 50, 25, 10][:k] if k >= 1 else []

NEI_FANOUTS = [int(x) for x in FANOUTS_ENV.split(",")] if FANOUTS_ENV else default_fanouts(NUM_LAYERS)

# Optimizer / schedule
LR = float(os.environ.get("LR", 3e-3))
WD = float(os.environ.get("WD", 1e-4))
USE_ADAMW = os.environ.get("ADAMW", "1") == "1"
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", 10))
EARLY_PATIENCE = int(os.environ.get("EARLY_PATIENCE", 80))

# Regularization/extras
EDGE_DROP_P = float(os.environ.get("EDGE_DROP_P", 0.05))
LABEL_SMOOTH = float(os.environ.get("LABEL_SMOOTH", 0.05))
USE_CLASS_WEIGHTS = os.environ.get("USE_CLASS_WEIGHTS", "1") == "1"
USE_EMA = os.environ.get("USE_EMA", "1") == "1"
EMA_DECAY = float(os.environ.get("EMA_DECAY", 0.999))
USE_LP = os.environ.get("USE_LP", "1") == "1"
USE_LP_VAL = os.environ.get("USE_LP_VAL", "1") == "1"
LP_ALPHA = float(os.environ.get("LP_ALPHA", 0.5))
LP_STEPS = int(os.environ.get("LP_STEPS", 10))

# Test-Time Augmentation
TTA_K = int(os.environ.get("TTA_K", 5))

# Consistency regularization
USE_CONSISTENCY = os.environ.get("CONSISTENCY", "1") == "1"
CONS_LAMBDA = float(os.environ.get("CONS_LAMBDA", 0.08))

# Grad clipping
GRAD_CLIP_NORM = float(os.environ.get("GRAD_CLIP_NORM", 2.0))

# Node2Vec features (+ cache)
USE_N2V = os.environ.get("USE_N2V", "1") == "1"
N2V_DIM = int(os.environ.get("N2V_DIM", 64))
N2V_WALK_LEN = int(os.environ.get("N2V_WALK_LEN", 20))
N2V_CONTEXT = int(os.environ.get("N2V_CONTEXT", 10))
N2V_WALKS_PER_NODE = int(os.environ.get("N2V_WALKS_PER_NODE", 4))
N2V_EPOCHS = int(os.environ.get("N2V_EPOCHS", 5))
N2V_LR = float(os.environ.get("N2V_LR", 0.01))
N2V_SPARSE = os.environ.get("N2V_SPARSE", "1") == "1"
N2V_BATCH_SIZE = int(os.environ.get("N2V_BATCH_SIZE", 1024))
N2V_CACHE = os.environ.get("N2V_CACHE", "1") == "1"

# Optional heavy structural features
ENABLE_PAGERANK = os.environ.get("ENABLE_PAGERANK", "0") == "1"
ENABLE_CORENUM = os.environ.get("ENABLE_CORENUM", "0") == "1"
ENABLE_CLUSTERING = os.environ.get("ENABLE_CLUSTERING", "0") == "1"

AMP = os.environ.get("AMP", "1") == "1"
VIS = os.environ.get("VIS", "1") == "1"
USE_CV = os.environ.get("USE_CV", "0") == "1"
CV_FOLDS = int(os.environ.get("CV_FOLDS", 3))
FIXED_TEST = os.environ.get("FIXED_TEST", "1") == "1"
VAL_RATIO = float(os.environ.get("VAL_RATIO", 0.1))
SEED = int(os.environ.get("SEED", 42))

# Repro

def seed_all(seed=0):
    """
    Set random seeds for Python, NumPy, and PyTorch so that runs are
    reproducible (given the same data and configuration).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

seed_all(SEED)

# Data IO

# Pick the node/edge CSVs based on the USE_NOISY flag.
nodes_file = "nodes_noisy.csv" if USE_NOISY else "nodes.csv"
edges_file = "edges_noisy.csv" if USE_NOISY else "edges.csv"

# Read nodes and edges into pandas DataFrames.
nodes = pd.read_csv(os.path.join(ART, nodes_file), low_memory=False)
edges = pd.read_csv(os.path.join(ART, edges_file), low_memory=False)

# Label mapping from nodes['label']
label_map = {"low": 0, "medium": 1, "high": 2}
if "label" not in nodes.columns:
    raise SystemExit("Expected a 'label' column in nodes CSV with values in {low, medium, high}.")

# Convert string labels to integers (0, 1, 2). Unlabeled/unknown becomes -1.
nodes["label_int"] = nodes["label"].map(label_map)
nodes["label_int"] = nodes["label_int"].fillna(-1).astype(int)

# Save the derived integer labels as a NumPy array
derived_label_path = os.path.join(OUTDIR, "node_gnn_derived_label_int.npy")
np.save(derived_label_path, nodes["label_int"].values)

labels = nodes["label_int"].values
# Keep track of which label values actually appear in the data (excluding -1).
LABELS_PRESENT = sorted([c for c in np.unique(labels) if c >= 0])

# Splits
train_i = np.load(os.path.join(ART, "train_idx.npy"))
val_i   = np.load(os.path.join(ART, "val_idx.npy"))
test_i  = np.load(os.path.join(ART, "test_idx.npy"))

# Features
# Automatically choose numeric columns to use as features.
exclude = {"label", "label_int", "name", "type_name"}
num_cols = [c for c in nodes.columns if c not in exclude and pd.api.types.is_numeric_dtype(nodes[c])]

# Log-transform degrees
for col in ["deg_in", "deg_out", "deg"]:
    if col in nodes.columns:
        nodes[col] = np.log1p(nodes[col])

# Create an additional degree ratio feature if both in/out degrees exist.
if "deg_in" in nodes.columns and "deg_out" in nodes.columns:
    # ratio features with safe division to avoid division by zero.
    nodes["deg_in_out_ratio"] = nodes["deg_in"] / (nodes["deg_out"] + 1e-6)
    num_cols += ["deg_in_out_ratio"]

feat_cols = [c for c in num_cols if c in nodes.columns]
# Convert features to a dense NumPy array and standardize them.
X = nodes[feat_cols].fillna(0).values
X = StandardScaler().fit_transform(X).astype(np.float32)

# Optional structural features (may be slow on very large graphs)
if ENABLE_PAGERANK or ENABLE_CORENUM or ENABLE_CLUSTERING:
    try:
        import networkx as nx
        # Build an undirected graph from the edges (for these structural metrics).
        G = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.Graph)
        if ENABLE_PAGERANK:
            pr = nx.pagerank(G, alpha=0.85)
            nodes["pagerank"] = nodes.index.to_series().map(pr).fillna(0.0)
            X = np.concatenate([X, nodes[["pagerank"]].values.astype(np.float32)], axis=1)
        if ENABLE_CORENUM:
            core = nx.core_number(G)
            nodes["core_num"] = nodes.index.to_series().map(core).fillna(0.0)
            X = np.concatenate([X, nodes[["core_num"]].values.astype(np.float32)], axis=1)
        if ENABLE_CLUSTERING:
            clus = nx.clustering(G)
            nodes["clustering"] = nodes.index.to_series().map(clus).fillna(0.0)
            X = np.concatenate([X, nodes[["clustering"]].values.astype(np.float32)], axis=1)
    except Exception as e:
        print("[struct] Skipping heavy structural features due to error or scale:", e)

# PyG

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.models import Node2Vec

def clip_indices(idx_arr, n):
    """
    Ensure that node indices are within [0, n).
    If any split indices point out of bounds, they are dropped.
    """
    idx_arr = np.asarray(idx_arr)
    keep = idx_arr[(idx_arr >= 0) & (idx_arr < n)]
    return keep, len(idx_arr) - len(keep)

n_nodes = len(nodes)
if n_nodes != len(labels):
    # If nodes and labels have different lengths, trim to the smaller size.
    n = min(n_nodes, len(labels))
    print(f"[align] len(nodes)={n_nodes}, len(labels)={len(labels)} -> using n={n} (clipping)")
    X = X[:n]
    labels = labels[:n]
    train_i, d_tr = clip_indices(train_i, n)
    val_i,   d_va = clip_indices(val_i, n)
    test_i,  d_te = clip_indices(test_i, n)
    if d_tr or d_va or d_te:
        print(f"[align] dropped out-of-range split indices: train {d_tr}, val {d_va}, test {d_te}")
else:
    n = n_nodes

# Edge sanitization
# Convert edge list into a NumPy array of integer indices.
e_np = edges[["source", "target"]].to_numpy(dtype=np.int64, copy=True)
# Keep only edges whose endpoints are valid node indices.
valid_edges = (e_np[:, 0] >= 0) & (e_np[:, 0] < n) & (e_np[:, 1] >= 0) & (e_np[:, 1] < n)
bad_edges = int((~valid_edges).sum())
if bad_edges > 0:
    print(f"[edge] Dropping {bad_edges}/{len(e_np)} edges with out-of-bounds endpoints (keeping {int(valid_edges.sum())}).")
e_np = e_np[valid_edges]
edge_index = torch.from_numpy(e_np.T).long()

# Node2Vec (DISK CACHE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _edge_hash(eidx_cpu: np.ndarray) -> str:
    """
    Compute a short hash representing the current graph and Node2Vec settings.
    Used to name the cache file for Node2Vec embeddings.
    """
    import hashlib, numpy as _np
    h = hashlib.md5()
    h.update(eidx_cpu.tobytes())
    h.update(_np.array([n, N2V_DIM, N2V_WALK_LEN, N2V_CONTEXT,
                       N2V_WALKS_PER_NODE, int(N2V_SPARSE)], dtype=_np.int64).tobytes())
    return h.hexdigest()

if USE_N2V:
    cache_dir = os.path.join(OUTDIR, "n2v_cache")
    os.makedirs(cache_dir, exist_ok=True)
    eh = _edge_hash(edge_index.cpu().numpy())
    cache_path = os.path.join(cache_dir, f"n2v_{eh}.npy")
    emb = None
    # Try to load cached embeddings if present and compatible.
    if N2V_CACHE and os.path.isfile(cache_path):
        tmp = np.load(cache_path)
        if tmp.shape == (n, N2V_DIM):
            emb = tmp
            print(f"[n2v] loaded cached embeddings: {cache_path} -> {emb.shape}")
        else:
            print("[n2v] cache shape mismatch; retraining and overwriting...")

    # If no valid cache, train Node2Vec from scratch.
    if emb is None:
        print(f"[n2v] training Node2Vec: dim={N2V_DIM} epochs={N2V_EPOCHS} walk_len={N2V_WALK_LEN} context={N2V_CONTEXT} "
              f"walks_per_node={N2V_WALKS_PER_NODE} sparse={N2V_SPARSE} batch_size={N2V_BATCH_SIZE}")
        eidx_dev = edge_index.to(device)
        n2v = Node2Vec(
            eidx_dev, embedding_dim=N2V_DIM, walk_length=N2V_WALK_LEN, context_size=N2V_CONTEXT,
            walks_per_node=N2V_WALKS_PER_NODE, num_nodes=n, sparse=N2V_SPARSE
        ).to(device)
        opt_n2v = (torch.optim.SparseAdam(list(n2v.parameters()), lr=N2V_LR)
                   if N2V_SPARSE else torch.optim.Adam(n2v.parameters(), lr=N2V_LR))

        # Older vs newer versions of Node2Vec may have different loss signatures.
        loss_params = list(inspect.signature(n2v.loss).parameters.keys())
        uses_loader = ('pos_rw' in loss_params) or ('neg_rw' in loss_params)

        n2v.train()
        if uses_loader:
            # Newer API: we sample (pos_rw, neg_rw) batches from a loader.
            loader = n2v.loader(batch_size=N2V_BATCH_SIZE, shuffle=True)
            for ep in range(1, N2V_EPOCHS + 1):
                total = 0.0
                steps = 0
                for pos_rw, neg_rw in loader:
                    pos_rw = pos_rw.to(device)
                    neg_rw = neg_rw.to(device)
                    loss = n2v.loss(pos_rw, neg_rw)
                    opt_n2v.zero_grad()
                    loss.backward()
                    opt_n2v.step()
                    total += float(loss.detach())
                    steps += 1
                print(f"[n2v] epoch {ep:02d}/{N2V_EPOCHS} | loss {total / max(1, steps):.4f}")
        else:
            # Older API: loss() may generate its own random walks internally.
            for ep in range(1, N2V_EPOCHS + 1):
                try:
                    loss = n2v.loss(num_negative_samples=5)
                except TypeError:
                    loss = n2v.loss()
                opt_n2v.zero_grad()
                loss.backward()
                opt_n2v.step()
                print(f"[n2v] epoch {ep:02d}/{N2V_EPOCHS} | loss {float(loss.detach()):.4f}")

        # Extract the final embeddings.
        with torch.no_grad():
            emb = n2v.embedding.weight.detach().cpu().numpy().astype(np.float32)
        # Save to disk so future runs can reuse them.
        if N2V_CACHE:
            np.save(cache_path, emb)
            print(f"[n2v] saved cache -> {cache_path}")

    # Concatenate Node2Vec embeddings to the existing features.
    X = np.concatenate([X, emb.astype(np.float32)], axis=1)
    print(f"[n2v] concatenated features: X -> {X.shape}")
else:
    print("[n2v] disabled (USE_N2V=0)")

# Final tensors
# Convert features and labels into PyTorch tensors used by PyTorch Geometric.
x = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)
# Enforce 3 classes (low, medium, high) even if one is rare/missing in a split.
NUM_CLASSES = 3
LABELED_MASK_GLOBAL = (y >= 0)  # Boolean mask of nodes that actually have labels.

# Device & AMP
use_amp = AMP and device.type == "cuda"

# Model
import torch.nn as nn

class SAGE(nn.Module):
    """
    GraphSAGE node classification model with:
    - Multiple SAGEConv layers
    - LayerNorm and GELU between layers
    - A projection head that maps last hidden layer to the final label logits
    - Optional residual connections when shapes match
    """
    def __init__(self, in_dim, hidden, out_dim, num_layers=4, dropout=0.35, aggr='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden] * (num_layers - 1)
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr=aggr))
            self.norms.append(nn.LayerNorm(dims[i + 1]))
        # Final projection into logits (one per class).
        self.proj = nn.Sequential(
            nn.Linear(dims[-1], hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Run through stacked GraphSAGE layers with optional residual connections.
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)
            if x.shape == x_res.shape:
                x = x + x_res  # Residual connection if dimensions line up.
        return self.proj(x)

# EMA
class EMA:
    """
    Exponential Moving Average (EMA) wrapper for model parameters.
    Keep a smoothed copy of all floating-point parameters and can temporarily
    swap them into the model for evaluation.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.float_keys = []
        self.shadow = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if torch.is_floating_point(v):
                    self.float_keys.append(k)
                    self.shadow[k] = v.detach().clone()
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        # Update the moving average with current model parameters.
        sd = model.state_dict()
        for k in self.float_keys:
            v = sd[k]
            if self.shadow[k].device != v.device:
                self.shadow[k] = self.shadow[k].to(v.device)
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        """
        Temporarily copy EMA weights into the model (for evaluation).
        Keep a backup of the original weights to restore later.
        """
        sd = model.state_dict()
        self.backup = {k: sd[k].detach().clone() for k in self.float_keys}
        for k in self.float_keys:
            tgt = sd[k]
            src = self.shadow[k]
            if tgt.device != src.device: 
                src = src.to(tgt.device)
            tgt.copy_(src)

    @torch.no_grad()
    def restore(self, model):
        """
        Restore the model parameters that were present before apply_to() was called.
        """
        if self.backup is None: 
            return
        sd = model.state_dict()
        for k in self.float_keys:
            tgt = sd[k]
            src = self.backup[k]
            if tgt.device != src.device: 
                src = src.to(tgt.device)
            tgt.copy_(src)
        self.backup = None

# Helper Functions
def make_masks(n, train_idx, val_idx, test_idx):
    """
    Turn sets of node indices into boolean masks for train/val/test splits.
    """
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[torch.tensor(train_idx)] = True
    val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[torch.tensor(val_idx)]   = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[torch.tensor(test_idx)] = True
    return train_mask, val_mask, test_mask

def build_data(train_idx, val_idx, test_idx):
    """
    Build a single PyG Data object containing:
    - Node features (x)
    - Edge index (edge_index)
    - Labels (y)
    - Boolean masks for train/val/test
    """
    train_mask, val_mask, test_mask = make_masks(n, train_idx, val_idx, test_idx)
    # Only nodes with a non-negative label are considered "labeled".
    labeled = LABELED_MASK_GLOBAL
    train_mask &= labeled
    val_mask &= labeled
    test_mask &= labeled
    d = Data(x=x, edge_index=edge_index, y=y,
             train_mask=train_mask, val_mask=val_mask, test_mask=test_mask).to(device)
    return d

def build_loaders(data):
    """
    Create neighbor-sampling DataLoaders for train/val/test.
    - Train uses NEI_FANOUTS for sampled neighborhoods.
    - Val/test use 'full' neighborhoods (fanout = -1) for exact evaluation.
    """
    train_nodes = torch.where(data.train_mask)[0]
    val_nodes   = torch.where(data.val_mask)[0]
    test_nodes  = torch.where(data.test_mask)[0]
    train_loader = NeighborLoader(data, num_neighbors=NEI_FANOUTS, batch_size=BATCH_SIZE, input_nodes=train_nodes)
    full_fan = [-1] * NUM_LAYERS
    val_loader = NeighborLoader(data, num_neighbors=full_fan, batch_size=BATCH_SIZE, input_nodes=val_nodes)
    test_loader = NeighborLoader(data, num_neighbors=full_fan, batch_size=BATCH_SIZE, input_nodes=test_nodes)
    return train_loader, val_loader, test_loader, NEI_FANOUTS

def compute_class_weight_tensor(data):
    """
    Compute a vector of class weights based on training frequency, so that
    rare classes get higher weight in the loss.
    """
    tr = torch.where(data.train_mask & (data.y >= 0))[0].cpu().numpy()
    if tr.size == 0: 
        return None
    hist = np.bincount(data.y[tr].cpu().numpy(), minlength=NUM_CLASSES).astype(np.float32)
    if hist.sum() == 0: 
        return None
    w = hist.sum() / np.maximum(hist, 1.0)
    w = w / (w.mean() + 1e-8)
    return torch.tensor(w, device=device, dtype=torch.float32)

def smooth_one_hot(targets, n_classes, smoothing=0.0):
    """
    Turn integer labels into soft one-hot vectors with label smoothing.
    """
    with torch.no_grad():
        true_dist = torch.zeros((targets.size(0), n_classes), device=targets.device)
        true_dist.fill_(smoothing / max(n_classes - 1, 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return true_dist

def ce_with_optional_smoothing(logits, targets, weight=None, smoothing=0.0):
    """
    Cross-entropy loss that can optionally use label smoothing and class weights.
    """
    if smoothing > 0.0:
        log_probs = torch.log_softmax(logits, dim=-1)
        y_sm = smooth_one_hot(targets, logits.size(-1), smoothing=smoothing)
        if weight is not None:
            # Reweight the log_probs and labels by class weights.
            w = weight[None, :].clamp(min=1e-8)
            log_probs = log_probs * w
            y_sm = y_sm / w
        return -(y_sm * log_probs).sum(dim=-1).mean()
    else:
        return F.cross_entropy(logits, targets, weight=weight)

# AMP-safe Label Propagation
@torch.no_grad()
def label_propagation_logits(edge_index, logits, alpha=0.5, steps=10):
    """
    Label propagation on top of raw logits:
    - Treat softmax(logits) as initial label distributions.
    - Repeatedly spread them through the graph using a simple symmetric
      normalized adjacency.
    - Returns log probabilities (logits-like values).
    """
    dtype = logits.dtype
    dev = logits.device
    N = logits.size(0)
    row, col = edge_index
    row = row.to(dev)
    col = col.to(dev)
    deg = torch.bincount(row, minlength=N).to(dtype).clamp(min=1)
    norm = (deg[row] * deg[col]).sqrt()
    w = (1.0 / norm).to(dtype)
    z = logits.softmax(dim=-1).to(dtype)
    init = z.clone()
    for _ in range(steps):
        agg = torch.zeros_like(z)
        agg.index_add_(0, row, (w.unsqueeze(1) * z[col]).to(dtype))
        z = (1 - alpha) * init + alpha * agg
    return (z + 1e-12).log()

def forward_with_dropedge(model, x, edge_index, p=0.0, train=True):
    """
    Run the model forward with optional DropEdge regularization.
    - During training, randomly drop edges with probability p.
    - At evaluation time keep all edges.
    """
    if train and p > 0.0:
        m = torch.rand(edge_index.size(1), device=edge_index.device) >= p
        eidx = edge_index[:, m]
    else:
        eidx = edge_index
    return model(x, eidx)

# Metrics
def _present_classes(y_true, num_classes):
    """
    Return a list of classes that actually appear in y_true.
    This protects metrics from crashing on missing classes.
    """
    y_true = np.asarray(y_true)
    present = [c for c in range(num_classes) if np.any(y_true == c)]
    return present

def safe_macro_auc(y_true, y_prob, num_classes):
    """
    Macro AUC that skips classes with only one label type (all 0s or all 1s).
    """
    try:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        aucs = []
        for c in range(num_classes):
            y_bin = (y_true == c).astype(int)
            if y_bin.min() == y_bin.max():
                continue
            aucs.append(roc_auc_score(y_bin, y_prob[:, c]))
        return float(np.mean(aucs)) if len(aucs) else float("nan")
    except Exception:
        return float("nan")

def safe_micro_auc(y_true, y_prob, num_classes):
    """
    Micro AUC that works with multi-class targets by turning them into
    one-vs-rest format internally.
    """
    try:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_onehot = np.zeros((y_true.size, num_classes), dtype=np.int32)
        y_onehot[np.arange(y_true.size), y_true] = 1
        return float(roc_auc_score(y_onehot, y_prob, average="micro", multi_class="ovr"))
    except Exception:
        return float("nan")

def safe_pr_aucs(y_true, y_prob, num_classes):
    """
    Macro and micro average precision scores for PR curves.
    """
    try:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_onehot = np.zeros((y_true.size, num_classes), dtype=np.int32)
        y_onehot[np.arange(y_true.size), y_true] = 1
        ap_macro = average_precision_score(y_onehot, y_prob, average="macro")
        ap_micro = average_precision_score(y_onehot, y_prob, average="micro")
        return float(ap_macro), float(ap_micro)
    except Exception:
        return float("nan"), float("nan")

def plot_multiclass_roc(y_true, y_prob, num_classes, out_png):
    """
    Plot ROC curves per class, plus micro and macro averages, and save to file.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or y_prob.size == 0: 
        return
    present = _present_classes(y_true, num_classes)
    if not present: 
        return
    y_onehot = np.zeros((y_true.size, num_classes), dtype=np.int32)
    y_onehot[np.arange(y_true.size), y_true] = 1
    fpr = {}
    tpr = {}
    roc_auc = {}
    for c in present:
        try:
            fpr[c], tpr[c], _ = roc_curve(y_onehot[:, c], y_prob[:, c])
            roc_auc[c] = auc(fpr[c], tpr[c])
        except Exception:
            continue
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    except Exception:
        pass
    try:
        import numpy as _np
        all_fpr = _np.unique(_np.concatenate([fpr[c] for c in present if c in fpr]))
        mean_tpr = _np.zeros_like(all_fpr)
        valid = 0
        for c in present:
            if c in fpr:
                mean_tpr += _np.interp(all_fpr, fpr[c], tpr[c])
                valid += 1
        if valid > 0:
            mean_tpr /= valid
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    except Exception:
        pass
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        for c in present:
            if c in roc_auc:
                ax.plot(fpr[c], tpr[c], label=f"class {c} (AUC={roc_auc[c]:.3f})", linewidth=1)
        if "micro" in roc_auc:
            ax.plot(fpr["micro"], tpr["micro"], linestyle="--", linewidth=2, label=f"micro (AUC={roc_auc['micro']:.3f})")
        if "macro" in roc_auc:
            ax.plot(fpr["macro"], tpr["macro"], linestyle="-.", linewidth=2, label=f"macro (AUC={roc_auc['macro']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC (OvR)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass

def plot_multiclass_pr(y_true, y_prob, num_classes, out_png):
    """
    Plot precision-recall curves per class, plus summary AP scores.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or y_prob.size == 0: 
        return
    present = _present_classes(y_true, num_classes)
    if not present: 
        return
    y_onehot = np.zeros((y_true.size, num_classes), dtype=np.int32)
    y_onehot[np.arange(y_true.size), y_true] = 1
    try:
        ap_macro, ap_micro = safe_pr_aucs(y_true, y_prob, num_classes)
        fig, ax = plt.subplots(figsize=(7, 6))
        for c in present:
            try:
                precision, recall, _ = precision_recall_curve(y_onehot[:, c], y_prob[:, c])
                ap_c = average_precision_score(y_onehot[:, c], y_prob[:, c])
                ax.plot(recall, precision, label=f"class {c} (AP={ap_c:.3f})", linewidth=1)
            except Exception:
                continue
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR curves (macro AP={ap_macro:.3f}, micro AP={ap_micro:.3f})")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass

def per_class_report(y_true, y_hat, num_classes):
    """
    Per-class precision, recall, F1, and support.
    """
    if len(y_true) == 0: 
        return None
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_hat, labels=np.arange(num_classes), zero_division=0
    )
    return {"precision": p.tolist(), "recall": r.tolist(), "f1": f1.tolist(), "support": support.tolist()}

def compute_report(y_true, y_hat, y_prob, num_classes, out_png=None, title="Confusion Matrix",
                   roc_png=None, pr_png=None):
    """
    Compute a dictionary of metrics plus optional plots (confusion matrix, ROC, PR).
    """
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_hat)
    y_prob = np.asarray(y_prob)
    metrics = {}
    metrics["acc"] = float((y_true == y_hat).mean()) if y_true.size else 0.0
    metrics["balanced_acc"] = float(balanced_accuracy_score(y_true, y_hat)) if y_true.size else 0.0
    metrics["f1_macro"] = float(f1_score(y_true, y_hat, average="macro")) if y_true.size else 0.0
    metrics["f1_micro"] = float(f1_score(y_true, y_hat, average="micro")) if y_true.size else 0.0
    metrics["auc_macro"] = safe_macro_auc(y_true, y_prob, num_classes) if (y_true.size and y_prob.size) else float("nan")
    metrics["auc_micro"] = safe_micro_auc(y_true, y_prob, num_classes) if (y_true.size and y_prob.size) else float("nan")
    ap_macro, ap_micro = safe_pr_aucs(y_true, y_prob, num_classes) if (y_true.size and y_prob.size) else (float("nan"), float("nan"))
    metrics["ap_macro"] = ap_macro
    metrics["ap_micro"] = ap_micro
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=np.arange(num_classes)))
    except Exception:
        metrics["log_loss"] = float("nan")
    try:
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_hat))
    except Exception:
        metrics["cohen_kappa"] = float("nan")
    try:
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_hat))
    except Exception:
        metrics["mcc"] = float("nan")
    metrics["per_class"] = per_class_report(y_true, y_hat, num_classes)

    # Plots
    if out_png is not None and y_true.size:
        cm = confusion_matrix(y_true, y_hat, labels=np.arange(num_classes))
        disp = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
    if roc_png is not None and y_true.size and y_prob.size:
        plot_multiclass_roc(y_true, y_prob, num_classes, roc_png)
    if pr_png is not None and y_true.size and y_prob.size:
        plot_multiclass_pr(y_true, y_prob, num_classes, pr_png)
    return metrics

def save_cv_summary(metrics_list, out_png):
    """
    For cross-validation mode, make a line plot of accuracy/F1/AUC values per fold.
    """
    accs = [m["acc"] for m in metrics_list]
    f1s = [m["f1_macro"] for m in metrics_list]
    aucs = [m.get("auc_macro", float("nan")) for m in metrics_list]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(1, len(metrics_list) + 1)
    ax.plot(xs, accs, marker='o', label='Acc')
    ax.plot(xs, f1s,  marker='s', label='F1-macro')
    ax.plot(xs, aucs, marker='^', label='AUC-macro')
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("CV metrics per fold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

# ---------- Train/Eval ----------
def train_one_split(data, epochs=EPOCHS):
    """
    Train the GraphSAGE model on a single train/val/test split.
    Returns:
      - trained model
      - training history (loss/acc/F1)
      - last train/val losses
      - final test metrics/raw outputs
    """
    model = SAGE(data.num_features, hidden=HIDDEN, out_dim=NUM_CLASSES,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, aggr='mean').to(device)
    opt_cls = torch.optim.AdamW if USE_ADAMW else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=LR, weight_decay=WD)
    # ReduceLROnPlateau reduces learning rate when val F1 stops improving.
    try:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10, min_lr=1e-5, verbose=True)
        _SCHED_VERBOSE = True
    except TypeError:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10, min_lr=1e-5)
        _SCHED_VERBOSE = False

    # Class weights and mixed-precision scaler.
    weight = compute_class_weight_tensor(data) if USE_CLASS_WEIGHTS else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None

    def run(loader, train=True, use_lp=False, tta=1, return_preds=False, is_val=False):
        """
        Common loop used for both training and evaluation.
        - train=True: do gradient updates.
        - use_lp=True: apply label propagation at the end (usually for val/test).
        - tta=k: average k noisy forward passes (test-time augmentation).
        """
        model.train() if train else model.eval()
        total_ce = 0.0
        total_n = 0
        correct = 0
        all_y = []
        all_pred = []
        all_prob = []
        for batch in loader:
            batch = batch.to(device)
            # Choose which mask to use depending on split.
            if train:
                mask = batch.train_mask
            else:
                mask = batch.val_mask if is_val else batch.test_mask
            idx = mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0: 
                continue
            yb = batch.y[idx]
            valid = (yb >= 0) & (yb < NUM_CLASSES)
            if valid.sum() == 0: 
                continue
            idx = idx[valid]

            if train: 
                opt.zero_grad(set_to_none=True)

            logits_accum = None
            K = max(1, tta)
            last_logits_k2 = None
            # We may do multiple forward passes for TTA.
            for _ in range(K):
                with amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    # Slight dropout on features for robustness.
                    xb1 = F.dropout(batch.x, p=0.1, training=train)
                    logits_k = forward_with_dropedge(model, xb1, batch.edge_index, p=EDGE_DROP_P, train=train)
                    if train and USE_CONSISTENCY:
                        # Second noisy view for consistency regularization.
                        xb2 = F.dropout(batch.x, p=0.1, training=True)
                        logits_k2 = forward_with_dropedge(model, xb2, batch.edge_index, p=EDGE_DROP_P, train=True)
                        last_logits_k2 = logits_k2
                if (not train) and use_lp:
                    # Optionally apply label propagation at the end of each forward.
                    logits_k = label_propagation_logits(batch.edge_index, logits_k, alpha=LP_ALPHA, steps=LP_STEPS)
                logits_accum = logits_k if (logits_accum is None) else (logits_accum + logits_k)
            logits = logits_accum / float(K)

            # Compute loss with optional consistency regularization.
            with amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss = ce_with_optional_smoothing(logits[idx], batch.y[idx], weight=weight, smoothing=LABEL_SMOOTH)
                if train and USE_CONSISTENCY and (last_logits_k2 is not None):
                    p1 = F.log_softmax(logits[idx], dim=-1)
                    q2 = F.softmax(last_logits_k2[idx].detach(), dim=-1)
                    p2 = F.log_softmax(last_logits_k2[idx], dim=-1)
                    q1 = F.softmax(logits[idx].detach(), dim=-1)
                    cons = 0.5 * (F.kl_div(p1, q2, reduction="batchmean") + F.kl_div(p2, q1, reduction="batchmean"))
                    loss = loss + CONS_LAMBDA * cons

            if train:
                # Backward pass with gradient scaling and clipping.
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(opt)
                scaler.update()
                if ema is not None: 
                    ema.update(model)

            total_ce += float(loss.item()) * idx.numel()
            total_n  += int(idx.numel())

            proba = logits[idx].softmax(dim=1)
            pred  = proba.argmax(dim=1)
            correct += int((pred == batch.y[idx]).sum())

            if return_preds:
                all_y.append(batch.y[idx].detach().cpu().numpy())
                all_pred.append(pred.detach().cpu().numpy())
                all_prob.append(proba.detach().cpu().numpy())

        avg_loss = total_ce / max(1, total_n)
        acc = correct / max(1, total_n)
        if return_preds:
            y_true = np.concatenate(all_y) if all_y else np.array([])
            y_hat  = np.concatenate(all_pred) if all_pred else np.array([])
            y_prob = np.concatenate(all_prob) if all_prob else np.array([])
            return avg_loss, acc, y_true, y_hat, y_prob
        return avg_loss, acc

    train_loader, val_loader, test_loader, fanouts_used = build_loaders(data)
    print(f"[model] SAGE(mean) layers={NUM_LAYERS} hidden={HIDDEN} dropout={DROPOUT} fanouts={fanouts_used}")
    print(f"[opt] {opt_cls.__name__} lr={LR} wd={WD} | warmup={WARMUP_EPOCHS} | EMA={'on' if ema else 'off'} "
          f"| class_w={'on' if weight is not None else 'off'} | ls={LABEL_SMOOTH} | dropedge={EDGE_DROP_P} "
          f"| LP={'on' if USE_LP else 'off'} (val={'on' if USE_LP_VAL else 'off'}) | TTA={TTA_K if not USE_CV else 1} "
          f"| Consistency={'on' if USE_CONSISTENCY else 'off'} | grad_clip={GRAD_CLIP_NORM} | "
          f"N2V={'on' if USE_N2V else 'off'} (dim={N2V_DIM})")

    best_f1 = -float('inf')
    best_state = None
    bad = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    # Main training loop.
    for epoch in range(1, epochs + 1):
        # Simple linear LR warmup.
        for g in opt.param_groups:
            if epoch <= WARMUP_EPOCHS:
                g['lr'] = LR * (epoch / max(1, WARMUP_EPOCHS))

        tr = run(train_loader, True, use_lp=False, tta=1)

        # Evaluate on validation with EMA and (optionally) label propagation.
        if ema is not None: 
            ema.apply_to(model)
        va = run(val_loader, False, use_lp=USE_LP_VAL, tta=1, return_preds=True, is_val=True)
        if ema is not None: 
            ema.restore(model)

        va_loss, va_acc, vy, vhat, vprob = va
        val_report = compute_report(vy, vhat, vprob, NUM_CLASSES, out_png=None, title="")
        val_f1 = val_report["f1_macro"]

        # Step LR scheduler based on validation F1.
        prev_lr = opt.param_groups[0]['lr']
        sched.step(val_f1)
        new_lr = opt.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"[lr scheduler] reducing lr: {prev_lr:.6f} -> {new_lr:.6f} (val F1={val_f1:.4f})")

        history["train_loss"].append(tr[0]); history["train_acc"].append(tr[1])
        history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
        history["val_f1"].append(val_f1)

        # Track best model by validation macro-F1.
        improved = val_f1 > best_f1 + 1e-4
        if improved:
            best_f1 = val_f1
            bad = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | train loss {tr[0]:.4f} acc {tr[1]:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f} F1 {val_f1:.3f}")
        if bad >= EARLY_PATIENCE:
            print(f"[early] stopping at epoch {epoch} (best val F1 {best_f1:.4f})")
            break

    # Load back the best model weights.
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final test — EMA + LP + TTA
    effective_tta = 1 if USE_CV else TTA_K
    if ema is not None: 
        ema.apply_to(model)
    te = run(test_loader, False, use_lp=USE_LP, tta=effective_tta, return_preds=True, is_val=False)
    if ema is not None: 
        ema.restore(model)
    print(f"Test: loss {te[0]:.4f} acc {te[1]:.3f}")
    return model, history, {"train": history["train_loss"][-1], "val": history["val_loss"][-1]}, te

# Modes
# Chose which mode to run:
# - Single run (train/val/test given)
# - K-fold cross-validation
if not USE_CV:
    # ----- Single fixed split mode -----
    data_single = build_data(train_i, val_i, test_i)
    model, history, _, te = train_one_split(data_single, EPOCHS)

    # Recompute validation predictions on the best model for saving metrics.
    _, val_loader, _, _ = build_loaders(data_single)
    def _eval_val_preds():
        model.eval()
        all_y = []
        all_pred = []
        all_prob = []
        for batch in val_loader:
            batch = batch.to(device)
            idx = batch.val_mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0: 
                continue
            yb = batch.y[idx]
            valid = (yb >= 0) & (yb < NUM_CLASSES)
            if valid.sum() == 0: 
                continue
            idx = idx[valid]
            with amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits = model(batch.x, batch.edge_index)
            if USE_LP_VAL:
                logits = label_propagation_logits(batch.edge_index, logits, alpha=LP_ALPHA, steps=LP_STEPS)
            proba = logits[idx].softmax(dim=1)
            pred  = proba.argmax(dim=1)
            all_y.append(batch.y[idx].detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
            all_prob.append(proba.detach().cpu().numpy())
        if all_y:
            return np.concatenate(all_y), np.concatenate(all_pred), np.concatenate(all_prob)
        return np.array([]), np.array([]), np.array([])

    vy, vhat, vprob = _eval_val_preds()

    # Warn if any class is absent in validation or test splits (metrics can be
    # misleading if some classes never appear).
    for split_name, ysplit in [("val", vy), ("test", te[2])]:
        present = set(np.unique(ysplit)) if ysplit.size else set()
        missing = set(range(NUM_CLASSES)) - present
        if missing:
            print(f"[warn] classes missing in {split_name} split: {sorted(missing)}")

    # Metrics & plots (node_gnn-prefixed outputs)
    val_cm_path  = os.path.join(OUTDIR, "node_gnn_val_confmat.png")
    val_roc_path = os.path.join(OUTDIR, "node_gnn_val_roc.png")
    val_pr_path  = os.path.join(OUTDIR, "node_gnn_val_pr.png")
    test_cm_path = os.path.join(OUTDIR, "node_gnn_test_confmat.png")
    test_roc_path= os.path.join(OUTDIR, "node_gnn_test_roc.png")
    test_pr_path = os.path.join(OUTDIR, "node_gnn_test_pr.png")

    val_json_path  = os.path.join(OUTDIR, "node_gnn_val_metrics.json")
    test_json_path = os.path.join(OUTDIR, "node_gnn_test_metrics.json")

    # Create full metric reports + plots.
    val_metrics = compute_report(
        vy, vhat, vprob, NUM_CLASSES,
        out_png=val_cm_path, title="Validation Confusion Matrix (Best F1)",
        roc_png=val_roc_path, pr_png=val_pr_path
    )
    test_metrics = compute_report(
        te[2], te[3], te[4], NUM_CLASSES,
        out_png=test_cm_path, title="Test Confusion Matrix",
        roc_png=test_roc_path, pr_png=test_pr_path
    )

    print(f"Val:  F1-macro {val_metrics['f1_macro']:.3f} | Acc {val_metrics['acc']:.3f} | "
          f"AUC-macro {val_metrics['auc_macro']:.3f} | AP-macro {val_metrics['ap_macro']:.3f}")
    print(f"Test: F1-macro {test_metrics['f1_macro']:.3f} | Acc {test_metrics['acc']:.3f} | "
          f"AUC-macro {test_metrics['auc_macro']:.3f} | AP-macro {test_metrics['ap_macro']:.3f}")

    # Save JSON metrics to disk so they can be used for additional analysis.
    with open(val_json_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(test_json_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    # VIS: training curves (loss/accuracy/F1 over epochs)
    if VIS:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["train_loss"], label="train loss")
        ax.plot(history["val_loss"],   label="val loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "node_gnn_train_curves.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["train_acc"], label="train acc")
        ax.plot(history["val_acc"],   label="val acc")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "node_gnn_acc_curves.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["val_f1"], label="val macro-F1")
        ax.set_xlabel("epoch"); ax.set_ylabel("macro-F1"); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "node_gnn_val_f1_curve.png"), dpi=150)
        plt.close(fig)

else:
    # Cross-validation mode
    print(f"Running {CV_FOLDS}-fold cross-validation "
          f"({'fixed test set' if FIXED_TEST else 'full CV with rotating test'})")

    labeled = np.where(labels >= 0)[0]
    y_lab = labels[labeled]
    fold_metrics = []
    fold_id = 0

    if FIXED_TEST:
        # Here we keep the test set fixed, and only re-split train+val across folds.
        base_pool = np.union1d(train_i, val_i)
        base_pool = base_pool[labels[base_pool] >= 0]
        te_nodes_fixed = test_i[labels[test_i] >= 0]
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        for tr_idx, va_idx in skf.split(base_pool, labels[base_pool]):
            fold_id += 1
            tr_nodes = base_pool[tr_idx]
            va_nodes = base_pool[va_idx]
            te_nodes = te_nodes_fixed
            print(f"\n=== Fold {fold_id}/{CV_FOLDS} ===")
            print(f"train {len(tr_nodes)} | val {len(va_nodes)} | test {len(te_nodes)} (fixed)")
            data_fold = build_data(tr_nodes, va_nodes, te_nodes)
            _, _, _, te = train_one_split(data_fold, EPOCHS)
            cm_path  = os.path.join(OUTDIR, f"node_gnn_test_confmat_fold{fold_id:02d}.png")
            roc_path = os.path.join(OUTDIR, f"node_gnn_test_roc_fold{fold_id:02d}.png")
            pr_path  = os.path.join(OUTDIR, f"node_gnn_test_pr_fold{fold_id:02d}.png")
            m = compute_report(te[2], te[3], te[4], NUM_CLASSES, out_png=cm_path,
                               title=f"Test Confusion Matrix (Fold {fold_id})",
                               roc_png=roc_path, pr_png=pr_path)
            print(f"Test: loss {te[0]:.4f} acc {te[1]:.3f} | F1-macro {m['f1_macro']:.3f} | AUC-macro {m['auc_macro']:.3f}")
            fold_metrics.append(m)
    else:
        # Full CV where test sets rotate and we resample train/val for each fold.
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        for test_sel, _ in skf.split(labeled, y_lab):
            fold_id += 1
            te_nodes = labeled[test_sel]
            remain = np.setdiff1d(labeled, te_nodes, assume_unique=False)
            # Split remain into train/val with the requested VAL_RATIO.
            r_train, r_val = train_test_split(remain, test_size=VAL_RATIO, random_state=SEED, stratify=labels[remain])
            tr_nodes, va_nodes = r_train, r_val
            print(f"\n=== Fold {fold_id}/{CV_FOLDS} ===")
            print(f"train {len(tr_nodes)} | val {len(va_nodes)} | test {len(te_nodes)}")
            data_fold = build_data(tr_nodes, va_nodes, te_nodes)
            _, _, _, te = train_one_split(data_fold, EPOCHS)
            cm_path  = os.path.join(OUTDIR, f"node_gnn_test_confmat_fold{fold_id:02d}.png")
            roc_path = os.path.join(OUTDIR, f"node_gnn_test_roc_fold{fold_id:02d}.png")
            pr_path  = os.path.join(OUTDIR, f"node_gnn_test_pr_fold{fold_id:02d}.png")
            m = compute_report(te[2], te[3], te[4], NUM_CLASSES, out_png=cm_path,
                               title=f"Test Confusion Matrix (Fold {fold_id})",
                               roc_png=roc_path, pr_png=pr_path)
            print(f"Test: loss {te[0]:.4f} acc {te[1]:.3f} | F1-macro {m['f1_macro']:.3f} | AUC-macro {m['auc_macro']:.3f}")
            fold_metrics.append(m)

    # Aggregate CV metrics and display summary.
    accs = [m["acc"] for m in fold_metrics]
    f1s = [m["f1_macro"] for m in fold_metrics]
    aucs = [m.get("auc_macro", float("nan")) for m in fold_metrics]
    print("\n===== CV Results =====")
    for i, (a, f, u) in enumerate(zip(accs, f1s, aucs), 1):
        print(f"Fold {i:02d}: acc {a:.4f} | F1-macro {f:.4f} | AUC-macro {u:.4f}")
    print(f"AVG:   acc {np.mean(accs):.4f} ± {np.std(accs):.4f} | "
          f"F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | "
          f"AUC {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}")

    # Save CV summary and per-fold metrics for later analysis.
    with open(os.path.join(OUTDIR, "node_gnn_cv_metrics.json"), "w") as f:
        json.dump({"folds": fold_metrics,
                   "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                   "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                   "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs))}, f, indent=2)

    save_cv_summary(fold_metrics, os.path.join(OUTDIR, "node_gnn_cv_summary.png"))

