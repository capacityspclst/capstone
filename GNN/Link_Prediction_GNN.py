"""
This script trains a link prediction model on a directed graph.

Pipeline overview:

1. Configuration
   - Set hyperparameters for data, model, training, and evaluation.
   - Includes a 'quick' preset for fast tests and a 'standard' preset for full runs.

2. Data loading & splits
   - Load train/val/test splits from SPLIT_DIR if available.
   - Otherwise stream the edge CSV in chunks and randomly create splits.
   - Optionally cap training edges for speed.
   - Generate missing validation/test negatives.

3. Graph structure (CSR adjacency)
   - Build a CSR adjacency from all positive edges.
   - Use it to compute degrees, check existing edges, and sample 2-hop “hard” negatives.

4. Negative sampling
   - Easy negatives: corrupt source or target, with optional degree-biased sampling.
   - Hard negatives: nodes two hops from u that are not directly connected.

5. Model
   - BilinearAsym: asymmetric bilinear scoring for directed edges:
       score(u, v) = <src_emb[u] * rel_vector, dst_emb[v]> + bu[u] + bv[v].
   - Embeddings are L2-clipped after each optimizer step.

6. Optional Node2Vec warm-start
   - Run Node2Vec on the training graph (if PyG is available).
   - Initialize embeddings from Node2Vec and optionally freeze before fine-tuning.

7. Training loop
   - For each epoch:
       * Shuffle edges and process minibatches.
       * Sample easy and optional hard negatives.
       * Compute pairwise ranking or BCE loss.
       * Use separate optimizers for sparse and dense params.
       * Track best model via validation ROC-AUC and apply early stopping.

8. Final evaluation
   - Restore best model.
   - Compute ROC-AUC, Average Precision, Precision@K, Recall@K.
   - Find the F1-optimal threshold and show the confusion matrix.

9. Optional plots
   - Generate ROC, PR curves, and score histograms for test positives/negatives.
"""

import os, pathlib, json, time
from dataclasses import dataclass
from typing import Optional



# Config (edit here)

@dataclass
class LinkPredConfig:
    # Preset run mode
    config_preset: str = "standard"   # "quick" | "standard"

    # Data / paths
    relation: str = "depends_on"
    split_dir: Optional[str] = "../Data Generation/edge_splits/depends_on"
    edges_csv: Optional[str] = None

    # Fractions for train / val / test when we have to split edges ourselves
    split_frac_train: float = 0.8
    split_frac_val: float = 0.1
    split_frac_test: float = 0.1

    # seed
    seed: int = 42

    # Device / precision
    device: Optional[str] = None     # "cuda", "cpu" or None for auto
    amp: bool = True

    # Mode (training regime)
    mode: str = "transductive"

    # Model / training hyperparameters
    emb_dim: int = 128
    epochs: int = 20
    lr: float = 1e-3
    l2: float = 1e-4
    clip_norm: float = 1.0
    patience: int = 8
    loss: str = "pairwise"          # "pairwise" | "bce"
    pos_weight: float = 2.0         # for BCE

    # Throughput / batching
    pair_batch: int = 4096
    ns_ratio: int = 6
    max_steps_per_epoch: int = 300
    train_pos_cap: int = 1_600_000
    watchdog_s: float = 15.0

    # CSV reading
    csv_chunksize: int = 2_000_000
    csv_usecols: tuple[int, int] = (0, 1)
    max_edges_debug: Optional[int] = None

    # Negative sampling
    neg_corrupt_p: float = 0.5
    neg_bias_degree: bool = True
    neg_degree_alpha: float = 0.75
    use_hard_neg: bool = True
    hard_neg_per_pos: int = 2

    # Node2Vec pretraining
    pretrain_n2v: bool = True
    n2v_dim: int = 128
    n2v_walk_len: int = 20
    n2v_context: int = 10
    n2v_walks: int = 4
    n2v_p: float = 1.0
    n2v_q: float = 1.0
    n2v_lr: float = 1e-2
    n2v_epochs: int = 5
    freeze_epochs: int = 3

    # Evaluation / visualization
    topk: tuple[int, ...] = (50, 100, 500, 1000)
    plot: bool = True
    verbose: bool = True


CFG = LinkPredConfig()

CONFIG = {
    "CONFIG_PRESET": CFG.config_preset,
    "RELATION": CFG.relation,
    "SPLIT_DIR": CFG.split_dir,
    "EDGES_CSV": CFG.edges_csv,
    "SPLIT_FRAC": {
        "train": CFG.split_frac_train,
        "val": CFG.split_frac_val,
        "test": CFG.split_frac_test,
    },
    "SEED": CFG.seed,
    "DEVICE": CFG.device,
    "AMP": CFG.amp,
    "MODE": CFG.mode,
    "EMB_DIM": CFG.emb_dim,
    "EPOCHS": CFG.epochs,
    "LR": CFG.lr,
    "L2": CFG.l2,
    "CLIP_NORM": CFG.clip_norm,
    "PATIENCE": CFG.patience,
    "LOSS": CFG.loss,
    "POS_WEIGHT": CFG.pos_weight,
    "PAIR_BATCH": CFG.pair_batch,
    "NS_RATIO": CFG.ns_ratio,
    "MAX_STEPS_PER_EPOCH": CFG.max_steps_per_epoch,
    "TRAIN_POS_CAP": CFG.train_pos_cap,
    "WATCHDOG_S": CFG.watchdog_s,
    "CSV_CHUNKSIZE": CFG.csv_chunksize,
    "CSV_USECOLS": list(CFG.csv_usecols),
    "MAX_EDGES_DEBUG": CFG.max_edges_debug,
    "NEG_CORRUPT_P": CFG.neg_corrupt_p,
    "NEG_BIAS_DEGREE": CFG.neg_bias_degree,
    "NEG_DEGREE_ALPHA": CFG.neg_degree_alpha,
    "USE_HARD_NEG": CFG.use_hard_neg,
    "HARD_NEG_PER_POS": CFG.hard_neg_per_pos,
    "PRETRAIN_N2V": CFG.pretrain_n2v,
    "N2V_DIM": CFG.n2v_dim,
    "N2V_WALK_LEN": CFG.n2v_walk_len,
    "N2V_CONTEXT": CFG.n2v_context,
    "N2V_WALKS": CFG.n2v_walks,
    "N2V_P": CFG.n2v_p,
    "N2V_Q": CFG.n2v_q,
    "N2V_LR": CFG.n2v_lr,
    "N2V_EPOCHS": CFG.n2v_epochs,
    "FREEZE_EPOCHS": CFG.freeze_epochs,
    "TOPK": list(CFG.topk),
    "PLOT": CFG.plot,
    "VERBOSE": CFG.verbose,
}

# Apply preset overrides
if CONFIG["CONFIG_PRESET"] == "quick":
    # Quick mode: fewer epochs, smaller batches, less neg sampling.
    CONFIG.update({
        "EPOCHS": 2,
        "PAIR_BATCH": 2048,
        "MAX_STEPS_PER_EPOCH": 50,
        "TRAIN_POS_CAP": 400_000,
        "AMP": False,          
        "PRETRAIN_N2V": False, # skip N2V to save time
    })
elif CONFIG["CONFIG_PRESET"] == "standard":
    # Full/default mode
    pass



# Path resolution and I/O setup

def _abspath(p: str) -> str:
    """Expand ~ and make an absolute, normalized path string."""
    return str(pathlib.Path(p).expanduser().resolve())


# Base directory for data artifacts (edges, splits, etc.)
ART = "../Data Generation"
REL = CONFIG.get("RELATION", "depends_on")

# Tweak PyTorch CUDA memory allocator for large graphs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

# Resolve absolute path for artifact root
ART_EFF = _abspath(ART)

# If SPLIT_DIR not set, try default pattern: <ART>/edge_splits/<REL>
split_auto = os.path.join(ART_EFF, "edge_splits", REL)
if CONFIG["SPLIT_DIR"] is None and os.path.isdir(split_auto):
    CONFIG["SPLIT_DIR"] = split_auto

# Default edges file name to try when using a monolithic CSV
edges_file = "edges_noisy.csv"


def _first_existing(cands):
    """
    Return the first path in 'cands' that exists, else None.
    Used to auto-resolve EDGES_CSV from several likely locations.
    """
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None

# Candidate locations for the edges CSV
edge_cands = [
    CONFIG.get("EDGES_CSV", None),           
    os.path.join("/mnt/data", edges_file),    
    os.path.join(ART_EFF, edges_file),         # artifact directory
]

CONFIG["EDGES_CSV"] = _first_existing(edge_cands)

# Log effective paths for debugging / reproducibility
print("=== Effective Paths ===")
print(json.dumps({
    "ARTIFACTS": ART_EFF,
    "SPLIT_DIR": CONFIG["SPLIT_DIR"],
    "EDGES_CSV": CONFIG["EDGES_CSV"],
    "CUDA_ALLOC_CONF": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
}, indent=2))



# Imports and global RNG/device setup

import os, math, random, warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

warnings.filterwarnings("ignore")

# Discover device
device = torch.device(CONFIG["DEVICE"]) if CONFIG.get("DEVICE") else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = CONFIG["AMP"] and (device.type == "cuda")

def seed_all(seed=0):
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(CONFIG["SEED"])
print("Device:", device, "| AMP:", USE_AMP)



# Load or create train/val/test splits

from pathlib import Path

def maybe_load_splits(split_dir):
    """
    Try to load precomputed split CSVs from split_dir.
    Expected files:
      - train_pos.csv, train_neg.csv
      - val_pos.csv,   val_neg.csv
      - test_pos.csv,  test_neg.csv
    Returns a dict of numpy arrays or None if files are missing.
    """
    if not split_dir:
        return None
    p = Path(split_dir)
    files = ["train_pos.csv","train_neg.csv","val_pos.csv","val_neg.csv","test_pos.csv","test_neg.csv"]
    if all((p/f).exists() for f in files):
        # Load each file as a (num_edges, 2) integer array
        ld = {f.split(".")[0]: pd.read_csv(p/f, dtype=np.int64).values for f in files}
        return ld
    return None

# Try to load curated splits first
loaded = maybe_load_splits(CONFIG["SPLIT_DIR"])

if loaded is None:
    # No precomputed splits; stream the raw edges CSV and split on the fly.
    if CONFIG["EDGES_CSV"] is None or not os.path.exists(CONFIG["EDGES_CSV"]):
        raise FileNotFoundError("Could not resolve EDGES_CSV and no split_dir found.")

    # Chunked reading to handle very large edge lists
    chunksize = int(CONFIG["CSV_CHUNKSIZE"] or 2_000_000)
    usecols = CONFIG.get("CSV_USECOLS", [0, 1])

    # RNG for assigning each edge to train/val/test
    rng = np.random.default_rng(CONFIG["SEED"])
    frac = CONFIG["SPLIT_FRAC"]
    p_train, p_val = frac["train"], frac["val"]

    # We’ll accumulate u and v for each split in lists, then concatenate.
    t_u, t_v = [], []
    v_u, v_v = [], []
    s_u, s_v = [], []

    max_id = 0   # track max node ID to infer total number of nodes
    seen = 0     # total edges processed

    t0 = time.time()

    # Stream over CSV in chunks to avoid memory blowups
    for chunk in pd.read_csv(CONFIG["EDGES_CSV"], usecols=usecols, chunksize=chunksize, header=0):
        # Extract u,v columns as int64 arrays
        u = chunk.iloc[:, 0].to_numpy(dtype=np.int64, copy=False)
        v = chunk.iloc[:, 1].to_numpy(dtype=np.int64, copy=False)

        # Update max node index
        max_id = max(max_id, int(u.max(initial=0)), int(v.max(initial=0)))

        # Draw uniform random number for each edge to assign split
        r = rng.random(len(u))
        m_train = r < p_train
        m_val   = (r >= p_train) & (r < p_train + p_val)
        m_test  = ~(m_train | m_val)

        # Store edges by split
        t_u.append(u[m_train]); t_v.append(v[m_train])
        v_u.append(u[m_val]);   v_v.append(v[m_val])
        s_u.append(u[m_test]);  s_v.append(v[m_test])

        seen += len(u)

        # Optional debug cap: stop after a limited number of edges
        if CONFIG["MAX_EDGES_DEBUG"] and seen >= int(CONFIG["MAX_EDGES_DEBUG"]):
            break

    # Concatenate lists into final arrays of shape (num_edges_split, 2)
    train_pos = np.stack([np.concatenate(t_u), np.concatenate(t_v)], axis=1)
    val_pos   = np.stack([np.concatenate(v_u), np.concatenate(v_v)], axis=1)
    test_pos  = np.stack([np.concatenate(s_u), np.concatenate(s_v)], axis=1)

    # Number of nodes is max ID + 1 (assuming nodes indexed [0, n_nodes-1])
    n_nodes   = max_id + 1

    # Helper to generate random negative examples (used for val/test)
    def gen_negs(n_pairs, seed):
        g = np.random.default_rng(CONFIG["SEED"] + seed)
        U = g.integers(0, n_nodes, size=n_pairs, dtype=np.int64)
        V = g.integers(0, n_nodes, size=n_pairs, dtype=np.int64)
        return np.stack([U, V], axis=1)

    # Simple random negatives for val and test (same count as positives)
    val_neg  = gen_negs(len(val_pos),  2)
    test_neg = gen_negs(len(test_pos), 3)

    print(f"[Stream] splits | n_nodes≈{n_nodes:,} | train_pos={train_pos.shape}")
else:
    # Use precomputed splits; they may already include negatives.
    train_pos = loaded["train_pos"]
    val_pos   = loaded["val_pos"]
    test_pos  = loaded["test_pos"]
    val_neg   = loaded["val_neg"]
    test_neg  = loaded["test_neg"]

    # n_nodes inferred as 1 + max over all edge endpoints in all splits
    n_nodes = int(max(
        train_pos.max(), val_pos.max(), test_pos.max(),
        val_neg.max(),   test_neg.max()
    )) + 1

    print(f"[Load] splits | n_nodes≈{n_nodes:,} | train_pos={train_pos.shape}")

# Optional cap on number of training positives to speed up experiments
if CONFIG["TRAIN_POS_CAP"] and len(train_pos) > CONFIG["TRAIN_POS_CAP"]:
    rng2 = np.random.default_rng(CONFIG["SEED"] + 1234)
    idx = rng2.choice(len(train_pos), size=int(CONFIG["TRAIN_POS_CAP"]), replace=False)
    train_pos = train_pos[idx]
    print(f"[Cap] train_pos -> {train_pos.shape}")



# Build a CSR (Compressed Sparse Row) adjacency structure from ALL positive edges (train + val + test)

t0 = time.time()

# Stack all positives into one (num_all_pos, 2) array
all_pos = np.vstack([train_pos, val_pos, test_pos]).astype(np.int64, copy=False)

# Out-degree for each node (counting edges u->v)
deg = np.bincount(all_pos[:, 0], minlength=n_nodes).astype(np.int64, copy=False)

# CSR layout:
#   - indptr[u] : start index in 'indices' for neighbors of u
#   - indices[indptr[u] : indptr[u+1]] : neighbors of u
indptr = np.zeros(n_nodes + 1, dtype=np.int64)
np.cumsum(deg, out=indptr[1:])

# Fill 'indices' by walking through 'all_pos'
indices = np.empty(len(all_pos), dtype=np.int64)

# 'cursor[u]' tracks where to insert the next neighbor of u
cursor = indptr[:-1].copy()
for u, v in all_pos:
    i = cursor[u]
    indices[i] = v
    cursor[u] = i + 1

# Sort each node's neighbor list for binary search and consistency
for u in np.where(deg > 1)[0]:
    s = indptr[u]
    e = indptr[u + 1]
    if e - s > 1:
        indices[s:e].sort()

# Remove temporary arrays
del cursor, all_pos

print(f"[CSR] built in {time.time()-t0:.2f}s | indptr:{indptr.shape} indices:{indices.shape} | deg>0: {(deg>0).sum()}")

# Degree-biased sampling distribution for negatives
alpha = float(CONFIG.get("NEG_DEGREE_ALPHA", 0.75))

# Weights: w[u] ∝ max(deg[u],1)^alpha to avoid zero-degree nodes having zero prob
w = np.power(np.maximum(deg, 1), alpha).astype(np.float64)
w = w / w.sum()

# CDF for sampling nodes via inverse transform sampling
cdf = np.cumsum(w)



# Graph utility functions

def exists_edge_per_u(u, v, indptr, indices):
    """
    Check if edge (u,v) exists in the graph using CSR data.
    Returns True if v is in the neighbor list of u.
    """
    s = indptr[u]
    e = indptr[u + 1]
    if e <= s:
        return False
    neigh = indices[s:e]
    # Binary search in sorted neigh
    pos = np.searchsorted(neigh, v)
    return (pos < len(neigh)) and (neigh[pos] == v)

def sample_node_degree_biased(rng):
    """
    Sample a node index according to the precomputed degree-biased distribution.
    """
    r = rng.random()
    return int(np.searchsorted(cdf, r))


def sample_easy_negs(
    p_mb, n_nodes, indptr, indices, seed,
    ns_ratio=1, corrupt_p=0.5, degree_bias=True, max_tries=10
):
    """
    Sample "easy" negatives by corrupting either u or v of each positive pair.
    
    Args:
        p_mb         : numpy array of positive pairs in the minibatch, shape (B,2)
        n_nodes      : total number of nodes
        seed         : random seed for this minibatch
        ns_ratio     : number of negatives per positive
        corrupt_p    : probability of corrupting v (otherwise we corrupt u)
        degree_bias  : if True, sample corrupted node according to degree-biased CDF
        max_tries    : max attempts to find a non-existent edge before fallback
    
    Returns:
        neg_pairs: numpy array of shape (B * ns_ratio, 2)
    """
    rng = np.random.default_rng(seed)
    U_out, V_out = [], []

    for (u, v) in p_mb:
        for _ in range(int(ns_ratio)):
            u2, v2 = int(u), int(v)
            for _try in range(max_tries):
                # Decide whether to corrupt source or target
                if rng.random() < corrupt_p:
                    # corrupt tail (v)
                    v2 = sample_node_degree_biased(rng) if degree_bias else int(rng.integers(0, n_nodes))
                else:
                    # corrupt head (u)
                    u2 = sample_node_degree_biased(rng) if degree_bias else int(rng.integers(0, n_nodes))

                # Ensure no self-loop and sampled edge not already present
                if (u2 != v2) and (not exists_edge_per_u(u2, v2, indptr, indices)):
                    U_out.append(u2)
                    V_out.append(v2)
                    break
            else:
                # Fallback: if we fail to find a valid negative after max_tries,
                # we do a simpler random search with more iterations.
                tries = 0
                while tries < 100:
                    v2 = int(rng.integers(0, n_nodes))
                    if (u2 != v2) and (not exists_edge_per_u(u2, v2, indptr, indices)):
                        U_out.append(u2)
                        V_out.append(v2)
                        break
                    tries += 1
                if tries >= 100:
                    # Extremely rare: still no valid negative.
                    # As a last resort, we force something (may rarely collide).
                    U_out.append(u2)
                    V_out.append((v2 + 1) % n_nodes)

    return np.stack(
        [np.array(U_out, dtype=np.int64), np.array(V_out, dtype=np.int64)],
        axis=1
    )


def sample_hard_negs_2hop(p_mb, indptr, indices, seed, per_pos=1):
    """
    Sample "hard" negatives for each positive (u,v) by:
      1. Picking a neighbor n of u (1-hop).
      2. Picking a neighbor w of n (2-hop from u).
      3. Rejecting w if w == u or if (u,w) is already a 1-hop neighbor.
    
    This tends to find nodes that are close in the graph but not directly connected,
    which are harder for the model to distinguish from true positives.
    
    Args:
        p_mb     : minibatch of positives (B,2), we mostly use the source u
        per_pos  : how many hard negatives to sample per positive
    
    Returns:
        numpy array of shape (~B * per_pos, 2) or shape (0,2) if none found.
    """
    rng = np.random.default_rng(seed)
    U_out, V_out = [], []

    for (u, _) in p_mb:
        u = int(u)
        s1, e1 = indptr[u], indptr[u + 1]
        if e1 <= s1:
            # No neighbors -> can't do 2-hop sampling
            continue
        neigh_u = indices[s1:e1]

        for _ in range(int(per_pos)):
            # 1-hop neighbor
            n = int(neigh_u[rng.integers(0, len(neigh_u))])
            s2, e2 = indptr[n], indptr[n + 1]
            if e2 <= s2:
                # That neighbor has no neighbors -> skip
                continue
            neigh_n = indices[s2:e2]

            picked = False
            for _try in range(20):
                w = int(neigh_n[rng.integers(0, len(neigh_n))])
                if w == u:
                    # Avoid trivial 2-hop back to u
                    continue

                # Check if w is already 1-hop neighbor of u (we want 2-hop but not 1-hop)
                pos = np.searchsorted(neigh_u, w)
                is_1hop = (pos < len(neigh_u)) and (neigh_u[pos] == w)
                if not is_1hop:
                    U_out.append(u)
                    V_out.append(w)
                    picked = True
                    break

            if not picked:
                # Could log or handle differently; here we just skip
                pass

    if len(U_out) == 0:
        return np.empty((0, 2), dtype=np.int64)

    return np.stack(
        [np.array(U_out, dtype=np.int64), np.array(V_out, dtype=np.int64)],
        axis=1
    )



# Model definition

class BilinearAsym(nn.Module):
    """
    Asymmetric bilinear scoring model for directed edges (u -> v).

    Score(u,v) = < src_emb[u] * rel_vector, dst_emb[v] > + bu[u] + bv[v]

    where:
      - src_emb[u], dst_emb[v] ∈ R^d
      - rel_vector ∈ R^d (global relation parameter)
      - bu[u], bv[v] are bias terms per node.
    """

    def __init__(self, n_nodes, emb_dim):
        super().__init__()
        # Separate embeddings for source (u) and destination (v) to allow asymmetry
        self.src = nn.Embedding(n_nodes, emb_dim, sparse=True)
        self.dst = nn.Embedding(n_nodes, emb_dim, sparse=True)

        # Initialize with small Gaussian noise
        nn.init.normal_(self.src.weight, std=0.02)
        nn.init.normal_(self.dst.weight, std=0.02)

        # Global relation vector, same dimension as embeddings
        self.rel = nn.Parameter(torch.randn(emb_dim))

        # Node-wise biases for source and destination
        self.bu = nn.Embedding(n_nodes, 1, sparse=True)
        nn.init.zeros_(self.bu.weight)
        self.bv = nn.Embedding(n_nodes, 1, sparse=True)
        nn.init.zeros_(self.bv.weight)

    def scores_for(self, pairs):
        """
        Compute scores for a batch of node pairs.

        Args:
            pairs : tensor or numpy array of shape (B,2) where each row is [u,v].

        Returns:
            scores : tensor of shape (B,)
        """
        s = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=self.rel.device)
        t = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=self.rel.device)

        # Lookup embeddings
        ue = self.src(s)  # (B, D)
        ve = self.dst(t)  # (B, D)

        # Elementwise multiply: ue * rel * ve, then sum over D
        core = (ue * self.rel * ve).sum(dim=1)
        return core + self.bu(s).squeeze(-1) + self.bv(t).squeeze(-1)

    def project(self, clip_norm=1.0):
        """
        Project embeddings and relation vector to have L2 norm ≤ clip_norm.
        This keeps embeddings from exploding and improves stability.
        """
        if clip_norm > 0:
            with torch.no_grad():
                for W in [self.src.weight.data, self.dst.weight.data]:
                    norms = W.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    W.div_(torch.clamp(norms, max=clip_norm))
                rn = self.rel.data.norm().clamp_min(1e-12)
                if rn > clip_norm:
                    self.rel.data.mul_(clip_norm / rn)


def make_optimizers(model, lr, wd):
    """
    Create separate optimizers for sparse and dense parameters.
    
    SparseAdam for embedding matrices (src, dst, bu, bv) and
    Adam for dense parameters (rel, etc).
    """
    sparse_params = [model.src.weight, model.dst.weight, model.bu.weight, model.bv.weight]
    dense_params  = [
        p for n, p in model.named_parameters()
        if ("src" not in n and "dst" not in n and "bu" not in n and "bv" not in n)
    ]
    opt_sparse = torch.optim.SparseAdam(sparse_params, lr=lr)
    opt_dense  = torch.optim.Adam(dense_params, lr=lr, weight_decay=wd)
    return opt_sparse, opt_dense



# Node2Vec pretraining (optional)

def node2vec_pretrain(train_edges, n_nodes, emb_dim):
    """
    Optionally run Node2Vec on train_edges to obtain initial embeddings.
    If PRETRAIN_N2V is False or torch_geometric is unavailable, returns None.

    Returns:
        pre : tensor on CPU of shape (n_nodes, emb_dim) or None.
    """
    if not CONFIG.get("PRETRAIN_N2V", False):
        print("[N2V] skipped (disabled)")
        return None

    # Try importing PyTorch Geometric
    try:
        import torch_geometric
        from torch_geometric.nn import Node2Vec
    except Exception as e:
        print(f"[N2V] PyG not available: {e}. Skipping warm-start.")
        return None

    try:
        # Make undirected edge_index by stacking (u,v) and (v,u)
        e = train_edges.astype(np.int64)
        edge_index_np = np.vstack([e.T, e[:, ::-1].T])
        edge_index = torch.from_numpy(edge_index_np).long().to(device)

        # Create Node2Vec model
        n2v = Node2Vec(
            edge_index,
            embedding_dim=emb_dim,
            walk_length=int(CONFIG["N2V_WALK_LEN"]),
            context_size=int(CONFIG["N2V_CONTEXT"]),
            walks_per_node=int(CONFIG["N2V_WALKS"]),
            p=float(CONFIG["N2V_P"]),
            q=float(CONFIG["N2V_Q"]),
            sparse=True
        ).to(device)

        # PyG provides its own loader for positive/negative random walks
        loader = n2v.loader(batch_size=1024, shuffle=True)
        opt = torch.optim.SparseAdam(n2v.parameters(), lr=float(CONFIG["N2V_LR"]))

        # Train Node2Vec
        n2v.train()
        for ep in range(1, int(CONFIG["N2V_EPOCHS"]) + 1):
            total = 0.0
            for pos_rw, neg_rw in loader:
                opt.zero_grad(set_to_none=True)
                loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                opt.step()
                total += float(loss.item())
            print(f"[N2V] epoch {ep}/{CONFIG['N2V_EPOCHS']} loss {total/len(loader):.4f}")

        # Extract learned embeddings
        with torch.no_grad():
            pre = n2v.embedding_weight().detach().cpu()
        return pre

    except Exception as e:
        print(f"[N2V] runtime error: {e}. Skipping warm-start.")
        return None



# Loss computation

def _forward_loss(model, pos, neg, loss_kind, pos_weight):
    """
    Compute the training loss given a batch of positive and negative edges.

    Args:
        model      : BilinearAsym model
        pos, neg   : tensors of shape (B,2) of positive and negative pairs
        loss_kind  : "pairwise" or "bce"
        pos_weight : positive class weight if using BCE

    Returns:
        scalar loss tensor
    """
    # Scores for positive and negative edges
    s_pos = model.scores_for(pos)
    s_neg = model.scores_for(neg)

    # Ensure equal length (in case we generated different counts)
    m = min(len(s_pos), len(s_neg))
    s_pos, s_neg = s_pos[:m], s_neg[:m]

    if loss_kind == "pairwise":
        # Pairwise ranking loss: encourage s_pos > s_neg
        # Loss = -log(σ(s_pos - s_neg))
        return -torch.log(torch.sigmoid(s_pos - s_neg)).mean()

    # Otherwise, use binary cross-entropy over concatenated scores
    logits  = torch.cat([s_pos, s_neg], dim=0)
    targets = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)], dim=0)

    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=logits.device)
    )
    return bce(logits, targets)



# Single epoch training loop

def train_one_epoch(model, train_pos, opt_sparse, opt_dense, epoch, n_nodes):
    """
    Run one training epoch over train_pos.

    We:
      1. Shuffle training positives.
      2. Slice into minibatches of size PAIR_BATCH.
      3. For each minibatch:
         a) Sample easy negatives.
         b) Optionally sample hard negatives and combine.
         c) Compute loss and apply gradients.
         d) Project embeddings to keep norms bounded.
    
    Returns:
        (average_loss, steps_run)
    """
    # New RNG per epoch (with epoch-dependent seed)
    rng = np.random.default_rng(CONFIG["SEED"] + epoch)
    idx = rng.permutation(len(train_pos))

    bsz       = int(CONFIG["PAIR_BATCH"])
    max_steps = int(CONFIG["MAX_STEPS_PER_EPOCH"])
    steps     = 0
    total_loss = 0.0
    t_epoch0   = time.time()

    for start in range(0, len(idx), bsz):
        end = min(len(idx), start + bsz)
        p_mb = train_pos[idx[start:end]]  # (batch_size, 2)

        # Negative sampling
        t0 = time.time()

        # Easy negatives by corrupting u or v
        easy_neg = sample_easy_negs(
            p_mb, n_nodes, indptr, indices,
            seed=CONFIG["SEED"] + epoch + start,
            ns_ratio=CONFIG["NS_RATIO"],
            corrupt_p=CONFIG["NEG_CORRUPT_P"],
            degree_bias=CONFIG["NEG_BIAS_DEGREE"]
        )

        # Hard negatives via 2-hop sampling
        hard_neg = np.empty((0, 2), dtype=np.int64)
        if CONFIG.get("USE_HARD_NEG", False) and CONFIG.get("HARD_NEG_PER_POS", 0) > 0:
            hard_neg = sample_hard_negs_2hop(
                p_mb, indptr, indices,
                seed=CONFIG["SEED"] + 7 * epoch + start,
                per_pos=int(CONFIG["HARD_NEG_PER_POS"])
            )

        # If we have any hard negatives, concatenate with easy ones
        if len(hard_neg):
            if len(hard_neg) > len(easy_neg):
                hard_neg = hard_neg[:len(easy_neg)]
            neg_mb = np.vstack([easy_neg, hard_neg])
        else:
            neg_mb = easy_neg

        t1 = time.time()

        # Forward + backward
        pos = torch.as_tensor(p_mb,   dtype=torch.long, device=device)
        neg = torch.as_tensor(neg_mb, dtype=torch.long, device=device)

        opt_sparse.zero_grad(set_to_none=True)
        opt_dense.zero_grad(set_to_none=True)

        loss = _forward_loss(
            model, pos, neg,
            CONFIG.get("LOSS", "pairwise"),
            CONFIG.get("POS_WEIGHT", 2.0)
        )
        loss.backward()
        opt_sparse.step()
        opt_dense.step()

        # Projection to keep embedding norms under control
        model.project(CONFIG["CLIP_NORM"])

        t2 = time.time()

        # ---- Logging / watchdogs ----
        total_loss += float(loss.item())
        steps += 1

        if (t1 - t0) > CONFIG["WATCHDOG_S"]:
            print(f"[Watchdog] neg-sampling {t1-t0:.2f}s (batch {len(p_mb)})")
        if (t2 - t1) > CONFIG["WATCHDOG_S"]:
            print(f"[Watchdog] backward+opt {t2-t1:.2f}s (batch {len(p_mb)})")

        if CONFIG["VERBOSE"] and (steps % 10 == 0):
            dt = time.time() - t_epoch0
            print(
                f"  step {steps} | batch_pos={len(p_mb)} batch_neg={len(neg_mb)} "
                f"| loss={total_loss/steps:.4f} | epoch_dt={dt:.1f}s"
            )

        if steps >= max_steps:
            # Limit epoch time by stopping after a fixed number of steps
            break

    return total_loss / max(1, steps), steps



# Evaluation helpers

def eval_auc_ap(y_true, y_pred):
    """
    Compute ROC-AUC, Average Precision, and curve points.
    Returns:
        auc, ap, (fpr,tpr), (prec,rec,thr)
    """
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')
    ap  = average_precision_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    prec, rec, thr = precision_recall_curve(y_true, y_pred)
    return auc, ap, (fpr, tpr), (prec, rec, thr)



# Top-level transductive training loop

def train_transductive(train_pos, val_pos, test_pos, val_neg, test_neg, n_nodes):
    """
    Full training pipeline in transductive mode.

    Steps:
      1. Optional Node2Vec pretraining to get initial embeddings.
      2. Initialize BilinearAsym model, optionally copying N2V embeddings.
      3. Train over epochs, evaluating on validation set.
      4. Early stop based on best val AUC; restore best model state.
      5. Evaluate on test set and return model + metrics.
    """
    # Step 1: Node2Vec pretraining (may return None)
    pre = node2vec_pretrain(train_pos, n_nodes, CONFIG["EMB_DIM"])

    # Step 2: Model initialization
    model = BilinearAsym(n_nodes, CONFIG["EMB_DIM"]).to(device)
    if pre is not None:
        with torch.no_grad():
            sz = min(model.src.num_embeddings, pre.shape[0])
            model.src.weight[:sz].copy_(pre[:sz])
            model.dst.weight[:sz].copy_(pre[:sz])
        freeze_epochs = int(CONFIG.get("FREEZE_EPOCHS", 0) or 0)
    else:
        freeze_epochs = 0

    opt_sparse, opt_dense = make_optimizers(model, CONFIG["LR"], CONFIG["L2"])
    print(f"[Init] model on {device}; emb_dim={CONFIG['EMB_DIM']} | N2V={'yes' if pre is not None else 'no'}")

    # Early stopping tracking
    best_auc   = -1.0
    best_state = None
    patience   = CONFIG["PATIENCE"]

    # Training epochs
    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        # Optionally freeze src/dst embeddings for the first few epochs.
        if pre is not None and epoch <= freeze_epochs:
            for p in model.src.parameters():
                p.requires_grad_(False)
            for p in model.dst.parameters():
                p.requires_grad_(False)
        else:
            for p in model.src.parameters():
                p.requires_grad_(True)
            for p in model.dst.parameters():
                p.requires_grad_(True)

        # One epoch of training
        loss_avg, steps = train_one_epoch(model, train_pos, opt_sparse, opt_dense, epoch, n_nodes)
        print(f"[Epoch {epoch:03d}] steps={steps}  Loss {loss_avg:.4f}")

        # Validation AUC/AP
        with torch.no_grad():
            y_val = np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_neg))])
            yhat_val = torch.cat([
                model.scores_for(val_pos),
                model.scores_for(val_neg)
            ]).float().cpu().numpy()

        v_auc, v_ap, *_ = eval_auc_ap(y_val, yhat_val)
        print(f"[Epoch {epoch:03d}] Val AUC {v_auc:.4f} | AP {v_ap:.4f}")

        # Track best validation AUC and save model state
        if (not np.isnan(v_auc)) and (v_auc > best_auc + 1e-4):
            best_auc = v_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = CONFIG["PATIENCE"]
        else:
            patience -= 1
            if patience <= 0:
                print(f"[EarlyStop] epoch={epoch} bestAUC={best_auc:.4f}")
                break

    # Restore best model weights, if any
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final test evaluation
    with torch.no_grad():
        y_test = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
        yhat_test = torch.cat([
            model.scores_for(test_pos),
            model.scores_for(test_neg)
        ]).float().cpu().numpy()

    t_auc, t_ap, roc_pts, pr_pts = eval_auc_ap(y_test, yhat_test)
    return model, {
        "test_auc": t_auc,
        "test_ap":  t_ap,
        "roc":      roc_pts,
        "pr":       pr_pts,
        "y_test":   y_test,
        "yhat_test": yhat_test
    }, None



# Run training

if CONFIG["MODE"] == "transductive":
    model, results, _ = train_transductive(
        train_pos, val_pos, test_pos,
        val_neg,  test_neg, n_nodes
    )
else:
    # Only transductive mode is implemented in this script but other modes should be explored
    raise NotImplementedError("Inductive mode not wired in this revision.")

# Print headline test metrics
print({k: v for k, v in results.items() if k in ["test_auc", "test_ap"]})



# Detailed Metrics + Plots

y_test = results["y_test"]
yhat_test = results["yhat_test"]

def eval_auc_ap(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')
    ap  = average_precision_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    prec, rec, thr = precision_recall_curve(y_true, y_pred)
    return auc, ap, (fpr, tpr), (prec, rec, thr)

auc, ap, (fpr, tpr), (prec, rec, thr) = eval_auc_ap(y_test, yhat_test)
print(f"Test ROC-AUC={auc:.4f} | AP={ap:.4f}")


def precision_recall_at_k(y_true, y_score, ks):
    """
    Compute precision and recall at various top-K cutoffs based on scores.
    Sort edges by score descending, then for each K compute:

      P@K = (#true positives in top K) / K
      R@K = (#true positives in top K) / (#total positives)
    """
    order = np.argsort(-y_score)  # descending scores
    y_sorted = y_true[order]
    out = []
    for k in ks:
        k = min(k, len(y_sorted))
        topk = y_sorted[:k]
        p = topk.mean() if k > 0 else 0.0
        r = topk.sum() / y_sorted.sum() if y_sorted.sum() > 0 else 0.0
        out.append((k, float(p), float(r)))
    return out

# Print precision/recall at requested K values
for k, p, r in precision_recall_at_k(y_test, yhat_test, CONFIG["TOPK"]):
    print(f"P@{k}: {p:.4f} | R@{k}: {r:.4f}")


def best_f1_threshold(prec, rec, thr):
    """
    Find the threshold that maximizes F1 score based on prec/rec curve.

    """
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    idx = np.nanargmax(f1)
    # Threshold array is shorter by 1; use max(idx-1,0) for indexing thr
    return float(thr[max(idx - 1, 0)]), float(f1[idx]), float(prec[idx]), float(rec[idx])

# Find best threshold and confusion matrix at that threshold
thr_star, f1_star, p_star, r_star = best_f1_threshold(prec, rec, thr)

y_pred_bin = (yhat_test >= thr_star).astype(int)
tp = int(((y_pred_bin == 1) & (y_test == 1)).sum())
fp = int(((y_pred_bin == 1) & (y_test == 0)).sum())
tn = int(((y_pred_bin == 0) & (y_test == 0)).sum())
fn = int(((y_pred_bin == 0) & (y_test == 1)).sum())

print(f"Best-F1 threshold={thr_star:.4f} | F1={f1_star:.4f} | P={p_star:.4f} | R={r_star:.4f}")
print(f"Confusion @bestF1: TP={tp} FP={fp} TN={tn} FN={fn}")

# Optional plots: ROC, PR, and score histograms
if CONFIG["PLOT"]:
    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

    # Precision-Recall curve
    plt.figure()
    plt.plot(rec, prec)
    plt.title("Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    # Score distributions for positive vs negative test edges
    pos_scores = yhat_test[y_test == 1]
    neg_scores = yhat_test[y_test == 0]
    plt.figure()
    plt.hist(pos_scores, bins=40, alpha=0.6, label="pos")
    plt.hist(neg_scores, bins=40, alpha=0.6, label="neg")
    plt.title("Test score distribution")
    plt.xlabel("score")
    plt.ylabel("#pairs")
    plt.legend()
    plt.show()
