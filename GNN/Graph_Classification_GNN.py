"""

This script trains and evaluates a Graph Neural Network (GNN) using PyTorch
and PyTorch Geometric on graph data stored in CSV files.

1. Load data
   - Reads node and edge CSVs (optionally noisy).
   - Extracts numeric node features and parent–child relationships.

2. Build graph samples
   - Finds each LOB (line of business) node and its descendants.
   - Creates a subgraph per LOB using those nodes and edges.
   - Computes a graph label (`hf`): the fraction of descendants labeled "high".

3. Train/val/test splits
   - Splits subgraphs into train, validation, and test sets.
   - Supports binary labels via median or quantile splits.
   - Standardizes node features and normalizes simple graph-level stats.

4. Model
   - 3-layer GIN with GraphNorm.
   - Pools node embeddings using attention, mean, and max pooling.
   - Optionally appends graph-level stats.
   - Provides heads for classification and regression.

5. Training
   - Adam optimizer + BCEWithLogits (with class weights and optional smoothing).
   - Edge dropout for regularization.
   - Early stopping using validation metrics.

6. Calibration & evaluation
   - Temperature scaling on validation set.
   - Threshold tuning via validation F1.
   - Test metrics: AUC, AP, accuracy, F1, etc.
   - Optional extraction of learned embeddings.

7. Diagnostics & plots
   - ROC and PR curves, probability histograms, calibration curves.
   - Confusion matrices per fold.

Configuration is driven by environment variables so hyperparameters and mode
(classification vs. regression) can be changed without editing the code.

"""

import os, random, numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torch import nn

# Data preprocessing / splitting / metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    precision_recall_curve, roc_curve, r2_score, mean_absolute_error, mean_squared_error,
    balanced_accuracy_score, matthews_corrcoef, brier_score_loss, ConfusionMatrixDisplay
)

from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# PyTorch Geometric components for graph data
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GINConv,
    GraphNorm,
    AttentionalAggregation,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.utils import to_undirected, dropout_edge
from IPython.display import display


# Config

ART        = os.environ.get("ARTIFACTS", "../Data Generation")  
USE_NOISY  = os.environ.get("USE_NOISY", "1") == "1" 
LOB_MAX    = int(os.environ.get("LOB_MAX", 300))  
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))  
EPOCHS     = int(os.environ.get("EPOCHS", 160))  
HIDDEN     = int(os.environ.get("HIDDEN", 64))  
LR         = float(os.environ.get("LR", 3e-4))   
WD         = float(os.environ.get("WD", 1e-3))  
USE_CV     = os.environ.get("USE_CV", "1") == "1"  
CV_FOLDS   = int(os.environ.get("CV_FOLDS", 5))   
VAL_RATIO  = float(os.environ.get("VAL_RATIO", 0.2)) 
SEED       = int(os.environ.get("SEED", 42))  
CLIP_NORM  = float(os.environ.get("CLIP_NORM", "0.5")) 
USE_U      = os.environ.get("USE_U", "1") == "1"  
TASK       = os.environ.get("TASK", "cls")   
QUANTILES  = os.environ.get("QUANTILES", "0") == "1"  
BAND       = os.environ.get("BAND", "none")   
PATIENCE   = int(os.environ.get("PATIENCE", 40))   
DROPEDGE_P = float(os.environ.get("DROPEDGE_P", "0.1"))
DROPOUT    = float(os.environ.get("DROPOUT", "0.6"))  
SMOOTH_EPS = float(os.environ.get("SMOOTH_EPS", "0.05"))    


def seed_all(seed: int = 0):
    """
    Set seeds for Python, NumPy, and PyTorch so results are (mostly) reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(SEED)

# Device selection: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | HIDDEN={HIDDEN} | TASK={TASK} | BAND={BAND} | QUANTILES={QUANTILES}")


# Data loading
# Choose which CSVs to load based on the USE_NOISY flag.
nodes_file = "nodes_noisy.csv" if USE_NOISY else "nodes.csv"
edges_file = "edges_noisy.csv" if USE_NOISY else "edges.csv"

# Load node and edge tables into pandas DataFrames.
nodes = pd.read_csv(os.path.join(ART, nodes_file), low_memory=False)
edges = pd.read_csv(os.path.join(ART, edges_file), low_memory=False)

# Node feature columns expected to exist in the nodes CSV.
# Only use the ones that are actually present.
feat_cols = [c for c in ["deg_in", "deg_out", "deg", "level"] if c in nodes.columns]
assert len(feat_cols) > 0, "Expected feature columns not found."

# Basic arrays for node IDs and labels
node_ids = nodes["id"].astype(int).values 
labels_str = nodes["label"].astype(str).values
node_id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}

# Make sure edges have "source" and "target" columns
assert {"source", "target"}.issubset(edges.columns), "Edges must contain source/target."
E = edges[["source", "target"]].astype(int).values

# If "note" column exists, consider rows where note == "parent_of" as parent-child relations.
parent_mask = (
    edges.get("note", "").astype(str).eq("parent_of")
    if "note" in edges.columns
    else pd.Series([True] * len(edges))
)

# Parent–child edges: only those with note == "parent_of" (or all edges if note missing)
po = edges.loc[parent_mask, ["source", "target"]].astype(int).values

# Build a dictionary: parent node ID -> list of child node IDs
child_by_parent = {}
for s, t in po:
    child_by_parent.setdefault(int(s), []).append(int(t))


def descendants(root: int):
    """
    Given a root node ID, return the set of all descendant node IDs
    based on the parent-child relationships (using a DFS-like search).
    """
    stack = [int(root)]
    seen = set([int(root)])
    out = set()

    while stack:
        u = stack.pop()
        # For each child of u, if we haven't seen it yet, add it and keep exploring
        for v in child_by_parent.get(u, []):
            if v not in seen:
                seen.add(v)
                out.add(v)
                stack.append(v)
    return out


# Find up to LOB_MAX nodes whose type_name is "LOB" to use as our graph roots.
lob_ids = (
    nodes.loc[nodes.type_name.astype(str) == "LOB", "id"]
    .astype(int)
    .values[:LOB_MAX]
)

# This list will hold a description of each subgraph centered around a LOB.
graph_descs = []
for lob in lob_ids:
    # All descendants (in terms of parent-child relationships)
    desc_full = sorted(list(descendants(int(lob))))
    if not desc_full:
        # If an LOB has no descendants, we skip it (no graph to build)
        continue

    # Filter descendants to those that actually appear in our node ID map
    nodes_local = [nid for nid in desc_full if nid in node_id_to_idx]
    if not nodes_local:
        continue

    # Select edges where both endpoints are in our local node set
    mask = np.isin(E[:, 0], nodes_local) & np.isin(E[:, 1], nodes_local)
    sub_e = E[mask]
    if len(sub_e) == 0:
        # No edges among the descendants -> not a meaningful graph
        continue

    # Indices into the global node feature array
    desc_idx = [node_id_to_idx[nid] for nid in nodes_local]

    # Compute fraction of descendant nodes whose label is "high"
    high_frac = (labels_str[desc_idx] == "high").mean()

    graph_descs.append(
        {
            "lob": int(lob),
            "nodes_local": nodes_local,
            "desc_idx": desc_idx,
            "sub_e": sub_e,
            "hf": float(high_frac),
        }
    )

# Node-level feature matrix for all nodes (in the original graph)
X_full = nodes[feat_cols].fillna(0).astype(np.float32).values

# Build actual PyTorch Geometric graph objects (but not yet Data objects)
raw_graphs = []
for gd in graph_descs:
    nodes_local, desc_idx, sub_e, hf = (
        gd["nodes_local"],
        gd["desc_idx"],
        gd["sub_e"],
        gd["hf"],
    )

    # Node features for this subgraph
    x = torch.tensor(X_full[desc_idx], dtype=torch.float32)

    # Map original node IDs to a dense [0, num_nodes-1] index range
    nid_map = {nid: i for i, nid in enumerate(nodes_local)}

    # Convert edges to local indices
    edge_index = torch.tensor(
        [[nid_map[int(s)], nid_map[int(t)]] for s, t in sub_e],
        dtype=torch.long,
    ).t().contiguous()

    # Make edges undirected (each edge appears twice: u->v and v->u)
    edge_index = to_undirected(edge_index, num_nodes=x.size(0))

    # Simple graph-level features:
    # n       = number of nodes
    # m_undir = number of undirected edges
    # den     = density (number of edges / max possible edges)
    n = int(x.size(0))
    m_undir = int(edge_index.size(1) // 2)  # each edge counted twice in undirected graph
    den = (m_undir / (n * (n - 1) / 2)) if n > 1 else 0.0

    u = torch.tensor([float(n), float(m_undir), float(den)], dtype=torch.float32)

    raw_graphs.append({"x": x, "edge_index": edge_index, "u": u, "hf": hf})

# Summary of constructed graphs
hf_all = np.array([g["hf"] for g in raw_graphs], float)
print(
    f"Built {len(raw_graphs)} graphs | hf range [{hf_all.min():.5f},{hf_all.max():.5f}] "
    f"| median {np.median(hf_all):.5f}"
)


# Split builder (per-train scalers; val/test labels fixed)
def build_split_datas(
    indexes_tr,
    indexes_val=None,
    indexes_te=None,
    task="cls",
    use_quantiles=False,
    y_fixed=None,
):
    """
    Build DataLoader objects for train/val/test splits.

    Key points:
    - Node-level features are standardized using only the training graphs.
    - Graph-level features `u` are transformed (log + standardization) using
      only the training graphs.
    - For classification:
        * Training labels may be defined by median or quantiles.
        * Validation/test labels can be fixed.
    """

    # Get the fraction-of-high labels for training graphs
    hf_tr = np.array([raw_graphs[i]["hf"] for i in indexes_tr], float)

    # Decide how to turn hf values into labels (for training)
    if task == "cls":
        if use_quantiles:
            # Quantile-based labeling: 1 for >= Q3, 0 for <= Q1, -1 for the middle (ignored)
            q1, q3 = np.quantile(hf_tr, [0.25, 0.75])

            def labeler_train(h):
                return 1 if h >= q3 else (0 if h <= q1 else -1)

            print(f"[LABEL] Train Q1/Q3: {q1:.5f}/{q3:.5f}")
        else:
            # Median-based labeling: 1 if above median, 0 otherwise
            thr = float(np.median(hf_tr))

            def labeler_train(h):
                return 1 if h > thr else 0

            print(f"[LABEL] Train median: {thr:.5f}")
    else:
        # For regression, the label is the raw hf (continuous value)
        def labeler_train(h):
            return float(h)

        print("[LABEL] Regression on high_frac")

    # Compute StandardScaler for node-level features using training graphs only
    X_train_rows = torch.cat([raw_graphs[i]["x"] for i in indexes_tr], dim=0).numpy()
    x_scaler = StandardScaler().fit(X_train_rows)

    # Collect graph-level feature vectors u from training graphs and normalize them
    U = []
    for i in indexes_tr:
        u = raw_graphs[i]["u"].clone().float()
        # Log-transform node and edge counts to reduce scale differences
        u[0] = torch.log1p(torch.clamp(u[0], min=0))
        u[1] = torch.log1p(torch.clamp(u[1], min=0))
        U.append(u.unsqueeze(0))
    U = torch.cat(U, dim=0)
    u_mean, u_std = U.mean(0), U.std(0).clamp(min=1e-6)  # avoid division by zero

    def materialize(idxs, split_kind):
        """
        Turn a list of indices into a list of Data objects and the indices kept.

        - Applies feature scaling and u normalization.
        - Derives labels using labeler_train or y_fixed (for val/test).
        - For classification + quantiles, drops middle graphs from training.
        """
        datas = []
        kept = []

        # Convert indices to a simple Python list
        iter_ids = [] if idxs is None else (idxs.tolist() if hasattr(idxs, "tolist") else list(idxs))

        for i in iter_ids:
            g = raw_graphs[i]

            # Determine label
            if task == "cls":
                if split_kind == "train":
                    # During training, ignore graphs in the middle quantile range
                    yv = labeler_train(g["hf"])
                    if yv == -1:  # middle quantile graphs are dropped from training
                        continue
                else:
                    # For validation/test, use fixed labels
                    if y_fixed is None or i not in y_fixed:
                        yv = 1 if labeler_train(g["hf"]) == 1 else 0
                    else:
                        yv = int(y_fixed[i])
            else:
                # Regression label: just the hf value
                yv = float(g["hf"])

            # Scale node-level features
            x = torch.tensor(
                x_scaler.transform(g["x"].numpy()).astype("float32")
            )

            # Normalize graph-level features u
            u = g["u"].clone().float()
            u[0] = torch.log1p(torch.clamp(u[0], min=0))
            u[1] = torch.log1p(torch.clamp(u[1], min=0))
            u = (u - u_mean) / u_std

            # Wrap label in a one-element tensor
            if task == "cls":
                y = torch.tensor([int(yv)], dtype=torch.long)
            else:
                y = torch.tensor([float(yv)], dtype=torch.float32)

            datas.append(Data(x=x, edge_index=g["edge_index"], y=y, u=u))
            kept.append(i)

        return datas, kept

    # Build train, validation, and test datasets
    tr_set, _ = materialize(indexes_tr, "train")
    va_set, _ = materialize(indexes_val, "val") if indexes_val is not None else ([], [])
    te_set, _ = materialize(indexes_te, "test") if indexes_te is not None else ([], [])

    # Wrap them in DataLoaders
    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False) if len(va_set) else None
    te_loader = DataLoader(te_set, batch_size=BATCH_SIZE, shuffle=False) if len(te_set) else None

    return tr_loader, va_loader, te_loader


# ----- Model: 3x GIN + GraphNorm + (Attn || Mean || Max) readout -----
class GNN(nn.Module):
    """
    Graph Neural Network model.

    Architecture:
    - 3 GINConv layers, each followed by GraphNorm and ReLU.
    - Residual sum of all three layers (h1 + h2 + h3).
    - Graph-level embedding created by concatenating:
        * AttentionalAggregation
        * global_mean_pool
        * global_max_pool
      plus optional graph-level features u.
    - Two heads:
        * head_cls for binary classification (1 logit).
        * head_reg for regression (1 value).
    """

    def __init__(self, in_dim, hidden, dropout=0.6):
        super().__init__()

        # First GIN layer: maps input features to hidden dimension
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
        )
        self.norm1 = GraphNorm(hidden)

        # Second GIN layer
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
        )
        self.norm2 = GraphNorm(hidden)

        # Third GIN layer
        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
        )
        self.norm3 = GraphNorm(hidden)

        self.dropout = nn.Dropout(dropout)

        # Attention-based readout over nodes
        self.readout = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden, 1)))

        # After pooling, concatenate:
        #   - attention pooled vector (hidden)
        #   - mean pooled vector (hidden)
        #   - max pooled vector (hidden)
        #   - graph-level stats u (3-dimensional)
        # So input to heads is of size 3 * hidden + 3.
        self.head_cls = nn.Sequential(
            nn.Linear(3 * hidden + 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.head_reg = nn.Sequential(
            nn.Linear(3 * hidden + 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _normalize_u(self, u, B, like_t):
        """
        Make sure u has shape [B, 3] and matches the batch size.

        If u is missing or has a different shape, this function reshapes/expands it so we always end up with exactly one 3D vector per graph in the batch.
        """
        if u is None:
            return torch.zeros(B, 3, device=like_t.device, dtype=like_t.dtype)

        # Handle different possible shapes for u
        if u.dim() == 1:
            if u.numel() == 3:
                u = u.unsqueeze(0)  # [3] -> [1, 3]
            elif u.numel() % 3 == 0:
                u = u.view(-1, 3)
            else:
                # Fallback: only keep first 3 elements
                u = u[:3].unsqueeze(0)
        elif u.dim() == 2 and u.size(1) != 3 and (u.numel() % 3 == 0):
            u = u.contiguous().view(-1, 3)

        # Adjust batch dimension if needed
        if u.size(0) == 1 and B > 1:
            # Single vector shared across all graphs
            u = u.expand(B, -1)
        elif u.size(0) > B:
            # Truncate if too many
            u = u[:B]
        elif u.size(0) < B:
            # Pad by repeating the last row
            pad = u[-1:].expand(B - u.size(0), -1)
            u = torch.cat([u, pad], dim=0)

        return u.to(like_t.device, dtype=like_t.dtype)

    def trunk(self, x, edge_index, batch):
        """
        Core GNN layers that operate on nodes and edges.

        Returns:
            h (tensor): Node embeddings after 3 GIN layers, combined with a
                        simple residual connection (sum of h1, h2, h3).
        """
        h1 = self.dropout(F.relu(self.norm1(self.conv1(x, edge_index), batch)))
        h2 = self.dropout(F.relu(self.norm2(self.conv2(h1, edge_index), batch)))
        h3 = self.dropout(F.relu(self.norm3(self.conv3(h2, edge_index), batch)))

        # Residual-type sum: combine information from all layers
        return h1 + h2 + h3

    def forward(self, x, edge_index, batch, u=None):
        """
        Forward pass.

        Args:
            x          : Node feature matrix.
            edge_index : Edge list.
            batch      : Batch vector mapping each node to its graph index.
            u          : Optional graph-level features.

        Returns:
            1D tensor of predictions (logits for classification or raw values for regression).
        """
        # Get node embeddings from the GNN trunk
        x = self.trunk(x, edge_index, batch)
        att = self.readout(x, batch)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)

        # Concatenate them into one graph-level embedding
        z = torch.cat([att, mean_pool, max_pool], dim=1)

        B = z.size(0)

        # Append normalized graph-level features u
        u = self._normalize_u(u, B, z) if USE_U else torch.zeros(
            B, 3, device=z.device, dtype=z.dtype
        )
        z = torch.cat([z, u], dim=1)

        # Classification or regression head
        if TASK == "cls":
            # One logit per graph
            return self.head_cls(z).view(-1)
        else:
            # Continuous output per graph
            return self.head_reg(z).view(-1)


def graph_embedding(self, x, edge_index, batch, u=None):
    """
    Utility method that returns the pooled graph embedding z

    """
    x = self.trunk(x, edge_index, batch)
    att = self.readout(x, batch)
    mean_pool = global_mean_pool(x, batch)
    max_pool = global_max_pool(x, batch)
    z = torch.cat([att, mean_pool, max_pool], dim=1)
    B = z.size(0)
    u = self._normalize_u(u, B, z) if USE_U else torch.zeros(
        B, 3, device=z.device, dtype=z.dtype
    )
    z = torch.cat([z, u], dim=1)
    return z


# Calibration (temperature scaling)
def learn_temperature(model, va_loader):
    """
    Learn a temperature parameter T for probability calibration
    using the validation set.

    - We keep the model fixed and optimize T so that the predicted
      probabilities are better calibrated (via BCEWithLogitsLoss).
    """
    if va_loader is None or len(va_loader.dataset) == 0:
        return 1.0

    # Start with T = 1.0 and optimize it
    T = torch.tensor([1.0], device=device, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.15, max_iter=60)

    def closure():
        opt.zero_grad()
        loss_sum, n = 0.0, 0

        for batch in va_loader:
            batch = batch.to(device)

            # Raw logits from the model
            logits = model(batch.x, batch.edge_index, batch.batch, getattr(batch, "u", None))

            # Apply temperature scaling: dividing logits by T
            logits = logits / T.clamp_min(0.05)

            y = batch.y.view(-1).float()

            # Calibration uses standard BCEWithLogitsLoss
            loss = nn.BCEWithLogitsLoss()(logits, y)
            loss.backward()

            loss_sum += float(loss.item()) * batch.num_graphs
            n += batch.num_graphs

        # LBFGS expects a single scalar tensor
        return torch.tensor(loss_sum / max(1, n), device=device)

    opt.step(closure)

    # Return a small-but-not-too-small temperature
    return float(T.detach().clamp_min(0.05).cpu())


# ----- Eval helpers (use sigmoid + temperature) -----
def eval_cls(model, loader, threshold=0.5, temp=1.0):
    """
    Evaluate a classification model on a DataLoader.

    Steps:
    - Run model on all graphs.
    - Apply temperature scaling and sigmoid to obtain probabilities.
    - Threshold probabilities to get predicted labels.
    - Compute a variety of classification metrics.

    Returns:
        A dictionary that includes:
            AUC, AP, ACC, F1, BAL_ACC, MCC, BRIER, y, p, threshold, temp
    """
    if loader is None or len(loader.dataset) == 0:
        return {}

    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch, getattr(batch, "u", None))
            prob = torch.sigmoid(logits / temp)  # apply temperature and sigmoid
            ys.append(batch.y.view(-1).cpu().numpy())
            ps.append(prob.cpu().numpy())

    if not ys:
        return {}

    y = np.concatenate(ys)
    p = np.concatenate(ps)

    # Some metrics (AUC, AP) only make sense if both classes are present
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    ap = average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan

    pred = (p >= threshold).astype(int)  # convert probs to 0/1 labels

    # Standard metrics
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)
    bal = balanced_accuracy_score(y, pred)
    mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0
    brier = brier_score_loss(y, p)

    return {
        "AUC": float(auc),
        "AP": float(ap),
        "ACC": float(acc),
        "F1": float(f1),
        "BAL_ACC": float(bal),
        "MCC": float(mcc),
        "BRIER": float(brier),
        "y": y,
        "p": p,
        "threshold": threshold,
        "temp": float(temp),
    }


def tune_threshold_from_val(model, va_loader, temp=1.0):
    """
    Find a good classification threshold using the validation set.

    Logic:
    - Try thresholds between 0.1 and 0.9 (17 values).
    - Pick the one that gives the best F1 score.
    - If two thresholds tie on F1, pick the one closer to 0.5.
    """
    if va_loader is None or len(va_loader.dataset) == 0:
        return 0.5

    # First compute probabilities once
    vm = eval_cls(model, va_loader, threshold=0.5, temp=temp)
    yv, pv = vm["y"], vm["p"]

    ts = np.linspace(0.1, 0.9, 17)
    best = (0.5, -1.0, 0.0)  # (threshold, F1, tie-break score)

    for t in ts:
        f1 = f1_score(yv, (pv >= t).astype(int), zero_division=0)
        cand = (t, f1, -abs(t - 0.5))
        # Choose better F1, or if equal, closer to 0.5
        if (f1 > best[1]) or (f1 == best[1] and cand[2] > best[2]):
            best = cand

    return float(best[0])


# Train one split
def train_one_split(idx_tr, idx_val, idx_te, y_fixed_map, epochs=EPOCHS):
    """
    Train and evaluate the model on one train/val/test split.

    Returns:
        model : trained (and best checkpoint restored) model
        tm    : test metrics dictionary from eval_cls
        thr   : chosen classification threshold
        T     : learned temperature for calibration
    """
    # Build DataLoaders for this particular split
    tr_loader, va_loader, te_loader = build_split_datas(
        idx_tr, idx_val, idx_te, task=TASK, use_quantiles=False, y_fixed=y_fixed_map
    )

    if len(tr_loader.dataset) == 0:
        raise RuntimeError("Empty train set after filtering; relax settings or check data.")

    # Infer input dimension from a single example
    in_dim = tr_loader.dataset[0].x.size(1)

    # Create model and optimizer
    model = GNN(in_dim=in_dim, hidden=HIDDEN, dropout=DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # Compute class weights for BCE loss (to compensate class imbalance)
    ys_tr = torch.cat([g.y.view(-1) for g in tr_loader.dataset]).long()
    binc = torch.bincount(ys_tr, minlength=2).float()
    pos_weight = (binc[0] / binc[1]).clamp(min=0.5, max=5.0) if binc[1] > 0 else torch.tensor(1.0)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_score = -1e9  # best validation metric so far
    best_state = None  # best model state dict
    patience = PATIENCE

    def forward_train(batch):
        """
        Forward pass during training with edge dropout applied
        for regularization (randomly dropping edges).
        """
        e_aug, _ = dropout_edge(
            batch.edge_index,
            p=DROPEDGE_P,
            force_undirected=True,
            training=True,
        )
        return model(batch.x, e_aug, batch.batch, getattr(batch, "u", None))

    # Main training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total, loss_sum = 0, 0.0

        for batch in tr_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)

            logits = forward_train(batch)
            y = batch.y.view(-1).float()

            # Optional label smoothing
            if SMOOTH_EPS > 0:
                y = y * (1 - SMOOTH_EPS) + (1 - y) * SMOOTH_EPS

            loss = bce(logits, y)
            loss.backward()

            # Gradient clipping to avoid exploding gradients
            if CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

            opt.step()

            total += batch.num_graphs
            loss_sum += float(loss.item()) * batch.num_graphs

        tr_loss = loss_sum / max(1, total)

        # Validation step for early stopping
        if va_loader and len(va_loader.dataset):
            vm = eval_cls(model, va_loader, temp=1.0)
            score = vm.get("AP", -1.0)  # use Average Precision as early-stopping metric

            if score > best_score + 1e-4:
                # New best model found
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = PATIENCE
            else:
                # No improvement -> reduce patience
                patience -= 1

        # Optional logging every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | best val metric {best_score:.4f}")

        # Early stopping if patience runs out
        if va_loader and patience <= 0:
            print(f"Early stopping at epoch {epoch}. Best val metric {best_score:.4f}")
            break

    # Restore the best model weights from validation
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Temperature scaling on validation
    T = learn_temperature(model, va_loader) if TASK == "cls" else 1.0

    # Tune classification threshold on validation
    thr = tune_threshold_from_val(model, va_loader, temp=T) if TASK == "cls" else None

    # Evaluate on test set
    tm = (
        eval_cls(model, te_loader, threshold=thr if thr is not None else 0.5, temp=T)
        if TASK == "cls"
        else {}
    )

    if TASK == "cls":
        print(
            f"[TEST] temp={T:.2f} thr={thr:.2f} ->",
            {k: round(v, 3) for k, v in tm.items() if k not in ["y", "p", "threshold", "temp"]},
        )

    return model, tm, thr, T


# Embedding collector for diagnostics
def collect_embeddings_and_probs(model, loader, temp, fold_id):
    """
    Collect graph-level embeddings and predicted probabilities
    for later analysis or visualization.

    Returns a dictionary with:
        - "fold": list of fold IDs (one per graph)
        - "z"   : stacked embeddings
        - "y"   : true labels
        - "p"   : predictions (probabilities for cls, raw for reg)
    """
    if loader is None or len(loader.dataset) == 0:
        return {"fold": [], "z": [], "y": [], "p": []}

    model.eval()
    out = {"fold": [], "z": [], "y": [], "p": []}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Use the graph_embedding method to get a pooled embedding per graph
            z = model.graph_embedding(
                batch.x, batch.edge_index, batch.batch, getattr(batch, "u", None)
            )

            # Use the appropriate head to get logits or raw values
            logits = (
                model.head_cls(z).view(-1)
                if TASK == "cls"
                else model.head_reg(z).view(-1)
            )

            # Convert logits to probabilities
            prob = torch.sigmoid(logits / temp) if TASK == "cls" else logits

            out["fold"].extend([fold_id] * batch.num_graphs)
            out["z"].append(z.cpu().numpy())
            out["y"].append(batch.y.view(-1).cpu().numpy())
            out["p"].append(prob.view(-1).cpu().numpy())

    # Store results in arrays
    if len(out["z"]):
        out["z"] = np.vstack(out["z"])
        out["y"] = np.concatenate(out["y"])
        out["p"] = np.concatenate(out["p"])
    else:
        # Fallback empty arrays
        out["z"] = np.zeros((0, 1))
        out["y"] = np.zeros((0,), dtype=int)
        out["p"] = np.zeros((0,))

    return out


# Ensure GNN has graph_embedding
def _gnn_graph_embedding(self, x, edge_index, batch, u=None):
    """
    Fallback implementation of graph_embedding, in case GNN doesn't have it.
    """
    x = self.trunk(x, edge_index, batch)
    att = self.readout(x, batch)
    mean_pool = global_mean_pool(x, batch)
    max_pool = global_max_pool(x, batch)
    z = torch.cat([att, mean_pool, max_pool], dim=1)
    B = z.size(0)
    u = self._normalize_u(u, B, z) if USE_U else torch.zeros(
        B, 3, device=z.device, dtype=z.dtype
    )
    z = torch.cat([z, u], dim=1)
    return z


try:
    _ = GNN
    if not hasattr(GNN, "graph_embedding"):
        GNN.graph_embedding = _gnn_graph_embedding
except NameError:
    pass


# Splitting (median by default)
indices = np.arange(len(raw_graphs))
hf_all = np.array([g["hf"] for g in raw_graphs], float)

kept_idx = indices.copy()
y_split = None

if TASK == "cls":
    # For classification, decide which graphs to keep and what their labels are.
    if QUANTILES or BAND in ("quartiles", "deciles"):
        if BAND == "deciles":
            # Keep only the bottom 10% and top 10% graphs by hf
            lo, hi = np.quantile(hf_all, [0.10, 0.90])
            keep_mask = (hf_all <= lo) | (hf_all >= hi)
            kept_idx = np.where(keep_mask)[0]
            y_split = (hf_all[kept_idx] >= hi).astype(int)
            print(f"[SPLIT] Deciles: kept {len(kept_idx)} / {len(indices)} graphs.")
        else:
            # Keep only the bottom 25% and top 25% graphs by hf
            q1_full, q3_full = np.quantile(hf_all, [0.25, 0.75])
            keep_mask = (hf_all <= q1_full) | (hf_all >= q3_full)
            kept_idx = np.where(keep_mask)[0]
            y_split = (hf_all[kept_idx] >= q3_full).astype(int)
            print(f"[SPLIT] Quartiles: kept {len(kept_idx)} / {len(indices)} graphs.")
    else:
        # Default: use all graphs, label by whether hf > median
        kept_idx = indices.copy()
        y_split = (hf_all[kept_idx] > np.median(hf_all[kept_idx])).astype(int)
        print(f"[SPLIT] Median split across all {len(kept_idx)} graphs.")

# For classification, we create a fixed mapping from graph index to label, which is used consistently for validation/testing across folds.
y_fixed_map = (
    {int(idx): int(lbl) for idx, lbl in zip(kept_idx, y_split)} if TASK == "cls" else None
)

# For classification, use stratified K-fold so each fold has balanced classes.
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED) if TASK == "cls" else None
folds = list(skf.split(kept_idx, y_split)) if TASK == "cls" else [(indices, indices)]

# Containers for results across folds
results = []
fold_preds = []
thresholds_per_fold = []
temps_per_fold = []
all_embeds = []

if TASK == "cls":
    # Perform K-fold cross-validation
    for k, (tr_pos, te_pos) in enumerate(folds, 1):
        # Position indices -> actual graph indices
        tr_idx_all = kept_idx[tr_pos]
        te_idx_all = kept_idx[te_pos]

        # For training/validation split inside each fold, we stratify by the split labels
        y_tr_split = y_split[tr_pos]
        tr_idx_all, va_idx_all = train_test_split(
            tr_idx_all,
            test_size=VAL_RATIO,
            stratify=y_tr_split,
            random_state=SEED,
        )

        print(f"\n=== Fold {k} ===")
        model, tm, thr, T = train_one_split(tr_idx_all, va_idx_all, te_idx_all, y_fixed_map)

        thresholds_per_fold.append(thr if thr is not None else 0.5)
        temps_per_fold.append(T)

        # Store main metrics (excluding raw predictions and thresholds)
        results.append(
            {
                "fold": k,
                **{kk: vv for kk, vv in tm.items() if kk not in ["y", "p", "yhat", "threshold", "temp"]},
            }
        )

        # Keep per-fold predictions and labels for later plotting
        if "y" in tm:
            fold_preds.append({"fold": k, "y": tm["y"], "p": tm["p"]})

        # Build a loader just for test indices to collect embeddings
        te_loader_dummy = build_split_datas(
            tr_idx_all, va_idx_all, te_idx_all, task=TASK, use_quantiles=False, y_fixed=y_fixed_map
        )[2]

        emb = collect_embeddings_and_probs(model, te_loader_dummy, temp=T, fold_id=k)
        all_embeds.append(emb)
else:
    print("\n=== Regression mode not used in this run ===")

# Convert fold metrics to a DataFrame for a quick overview
res_df = pd.DataFrame(results)
display(res_df)

if len(res_df):
    print("Means:", res_df.mean(numeric_only=True).to_dict())

print("Chosen thresholds per fold:", thresholds_per_fold)
print("Temperatures per fold:", temps_per_fold)


# Diagnostics / Graphics
if TASK == "cls" and len(fold_preds):
    # ROC / PR curves by fold
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for fp in fold_preds:
        y, p = fp["y"], fp["p"]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)

        axes[0].plot(fpr, tpr, alpha=0.7, label=f"Fold {fp['fold']}")
        axes[1].plot(rec, prec, alpha=0.7, label=f"Fold {fp['fold']}")

    axes[0].plot([0, 1], [0, 1], "--", alpha=0.5)
    axes[0].set_title("ROC by fold")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    axes[1].set_title("Precision-Recall by fold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    plt.show()

    # Histogram of predicted probabilities for each class
    plt.figure(figsize=(6, 4))
    all_y = np.concatenate([fp["y"] for fp in fold_preds])
    all_p = np.concatenate([fp["p"] for fp in fold_preds])

    plt.hist(all_p[all_y == 0], bins=12, alpha=0.6, label="y=0")
    plt.hist(all_p[all_y == 1], bins=12, alpha=0.6, label="y=1")
    plt.title("Predicted probability distribution")
    plt.xlabel("p(class=1)")
    plt.ylabel("count")
    plt.legend()
    plt.show()

    # Calibration curve
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(all_y, all_p, n_bins=8, strategy="quantile")

    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "--", alpha=0.5)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration (all folds)")
    plt.show()

    # Confusion matrices at threshold = 0.5 for reference
    for fp in fold_preds:
        y, p = fp["y"], fp["p"]
        pred = (p >= 0.5).astype(int)
        ConfusionMatrixDisplay.from_predictions(y, pred)
        plt.title(f"Confusion Matrix — Fold {fp['fold']} (thr=0.50)")
        plt.show()
