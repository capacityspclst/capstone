"""
CMDB Data Generator

This script generates a synthetic CMDB-like graph with:
- Multiple node types (LOB, BusinessService, Application, Server, etc.).
- Multiple edge types (parent_of, depends_on, communicates_with, connected_to, impacts).
- Basic node features (degrees, hierarchy level).
- Incident-based impact scores and labels (low/medium/high).
- Train/validation/test splits for node classification.
- Link prediction splits per relation type.

Outputs written to OUTDIR:
- nodes.csv, edges.csv
- label_int.npy, train_idx.npy, val_idx.npy, test_idx.npy
- type_vocab.json, graph_schema.json
- edge_splits/<relation>/* for link prediction.
"""

# IMPORTS 
import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List
from collections import deque, defaultdict

import numpy as np
import pandas as pd

# ENV + GLOBAL CONFIG 
RNG_SEED = int(os.environ.get("SEED", 42)) # Everyone knows the answer is 42
SCALE    = float(os.environ.get("SCALE", 1.0))
OUTDIR   = os.environ.get("OUTDIR", ".")
LP_HOLDOUT = float(os.environ.get("LP_HOLDOUT", 0.10))

os.makedirs(OUTDIR, exist_ok=True)
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# Counts for each node type (before scaling)
COUNTS_BASE = {
    "LOB": 100,
    "BusinessService": 5000,
    "Application": 50000,
    "ServiceOffering": 300000,
    "AppInstance": 400000,
    "Server": 400000,
    "Database": 300000,
    "NetworkDevice": 400000,
    "Service": 200000,
    "Orphan": 250000,
    "Unclassified": 250000,
    "Change": 333333,
    "Incident": 333333,
    "Alert": 333334,
}

# Apply scale factor
COUNTS = {k: int(v * SCALE) for k, v in COUNTS_BASE.items()}
TOTAL_NODES = sum(COUNTS.values())

# Friendly name lists for each type
INSTANCE_NAMES = {
    "LOB": ["Banking", "Insurance", "Retail", "Healthcare", "Technology"],
    "BusinessService": ["Payments", "Claims", "Inventory", "EHR", "Analytics"],
    "Application": ["AppX", "AppY", "AppZ"],
    "ServiceOffering": ["Basic", "Premium", "Enterprise"],
    "AppInstance": ["Instance1", "Instance2", "Instance3"],
    "Server": ["ServerA", "ServerB"],
    "Database": ["DB1", "DB2"],
    "NetworkDevice": ["Switch1", "Router2"],
    "Service": ["Web", "Auth", "Notification"],
    "Change": ["Patch1", "Update2"],
    "Incident": ["Crash", "Outage"],
    "Alert": ["CPUHigh", "MemoryLow"],
}

# Edge relation names
REL_PARENT_OF    = "parent_of"
REL_DEPENDS_ON   = "depends_on"
REL_COMMUNICATES = "communicates_with"
REL_CONNECTED_TO = "connected_to"
REL_IMPACTS      = "impacts"

# NODE CREATION 

# Order we allocate IDs per type
type_order = list(COUNTS.keys())
type2ids: Dict[str, List[int]] = {}  # mapping type_name -> list of node IDs
node_type_names: List[str] = []      # type name per node index
node_name: List[str] = []            # name per node index
cursor = 0                           # running node ID index

for t in type_order:
    n = COUNTS[t]
    ids = list(range(cursor, cursor + n))
    type2ids[t] = ids
    cursor += n

    # record type for each of these IDs
    node_type_names.extend([t] * n)

    # assign names from small pool per type (or default to type name)
    pool = INSTANCE_NAMES.get(t, [t])
    node_name.extend(np.random.choice(pool, size=n).tolist())

# Integer-encode type names
type_vocab = sorted(COUNTS.keys())
TYPE_INDEX = {t: i for i, t in enumerate(type_vocab)}
node_type_int = np.array([TYPE_INDEX[t] for t in node_type_names], dtype=np.int32)

# TIMESTAMPS FOR EVENTS 

now = datetime.now()
timestamps = [""] * TOTAL_NODES  # default: empty for non-event nodes

# Only Change / Incident / Alert get timestamps within last 365 days
for t in ("Change", "Incident", "Alert"):
    ids = type2ids.get(t, [])
    days = np.random.randint(0, 366, size=len(ids))
    for i, d in zip(ids, days):
        timestamps[i] = (now - timedelta(days=int(d))).isoformat()

# EDGE BUILDING HELPERS 

def one_to_many_parent_edges(parent_ids: List[int], child_ids: List[int]) -> np.ndarray:
    """
    Connect each child to a parent using round-robin assignment:
    - parent_ids: list of parent node IDs
    - child_ids: list of child node IDs
    Returns a (n_child, 2) array with [parent, child] pairs.
    """
    p = np.asarray(parent_ids)
    c = np.asarray(child_ids)
    if len(p) == 0 or len(c) == 0:
        return np.empty((0, 2), dtype=np.int64)

    parents_for_children = p[np.arange(len(c)) % len(p)]
    return np.stack([parents_for_children, c], axis=1)

# Storage for all edges
edges: List[tuple] = []
notes: List[str] = []
weights: List[float] = []

def add_edges(pairs: np.ndarray, note: str, w=np.nan) -> None:
    """
    Append edges plus a relation label and optional weight.
    'pairs' is an (N,2) array of [source, target].
    """
    if len(pairs) == 0:
        return
    edges.extend(map(tuple, pairs.tolist()))
    notes.extend([note] * len(pairs))
    weights.extend([w] * len(pairs))

# HIERARCHICAL & DEPENDENCY EDGES 

# LOB -> BusinessService
add_edges(one_to_many_parent_edges(type2ids["LOB"], type2ids["BusinessService"]),
          REL_PARENT_OF)
# BusinessService -> Application
add_edges(one_to_many_parent_edges(type2ids["BusinessService"], type2ids["Application"]),
          REL_PARENT_OF)
# Application -> ServiceOffering
add_edges(one_to_many_parent_edges(type2ids["Application"], type2ids["ServiceOffering"]),
          REL_PARENT_OF)
# ServiceOffering -> AppInstance
add_edges(one_to_many_parent_edges(type2ids["ServiceOffering"], type2ids["AppInstance"]),
          REL_PARENT_OF)
# AppInstance -> Server
add_edges(one_to_many_parent_edges(type2ids["AppInstance"], type2ids["Server"]),
          REL_PARENT_OF)
# AppInstance -> Database
add_edges(one_to_many_parent_edges(type2ids["AppInstance"], type2ids["Database"]),
          REL_PARENT_OF)

# Service -> AppInstance (logical dependency)
add_edges(one_to_many_parent_edges(type2ids["Service"], type2ids["AppInstance"]),
          REL_DEPENDS_ON)

# RANDOM PAIRS (COMMUNICATION / CONNECTIVITY)
def sample_unique_pairs(ids, m):
    """
    Sample up to 'm' unique (s, t) pairs from list of ids, with:
    - No self loops (s != t)
    - Duplicate pairs removed
    """
    ids = np.asarray(ids)
    if len(ids) < 2 or m <= 0:
        return np.empty((0, 2), dtype=np.int64)

    s = np.random.choice(ids, size=m)
    t = np.random.choice(ids, size=m)
    mask = s != t
    s, t = s[mask], t[mask]
    if not len(s):
        return np.empty((0, 2), dtype=np.int64)

    pairs = np.stack([s, t], axis=1)
    # View as structured array to deduplicate rows
    _, idx = np.unique(
        pairs.view([("s", pairs.dtype), ("t", pairs.dtype)]),
        return_index=True,
    )
    return pairs[np.sort(idx)]

# AppInstance <-> AppInstance communication
add_edges(
    sample_unique_pairs(type2ids["AppInstance"],
                        int(0.15 * len(type2ids["AppInstance"]))),
    REL_COMMUNICATES,
)

# NetworkDevice <-> NetworkDevice connectivity
add_edges(
    sample_unique_pairs(type2ids["NetworkDevice"],
                        int(0.30 * len(type2ids["NetworkDevice"]))),
    REL_CONNECTED_TO,
)

# EVENT IMPACT EDGES 

# Nodes that can be impacted
impact_targets = np.array(
    type2ids["Application"]
    + type2ids["ServiceOffering"]
    + type2ids["AppInstance"]
    + type2ids["Server"]
    + type2ids["Database"]
    + type2ids["NetworkDevice"]
    + type2ids["Service"],
    dtype=np.int64,
)

def add_event_impacts(evt: str) -> None:
    """
    Add 'impacts' edges from event type 'evt' (Change/Incident/Alert) to impact targets.
    - Incidents get severity weights 1–5.
    - Changes / Alerts have NaN weights.
    """
    src = np.asarray(type2ids.get(evt, []))
    if len(src) == 0 or len(impact_targets) == 0:
        return

    # Roughly ~ 5/3 edges per event node
    m = max(1, (len(src) * 5) // 3)
    s = np.random.choice(src, size=m)
    t = np.random.choice(impact_targets, size=m)
    pairs = np.stack([s, t], axis=1)

    # Deduplicate (s, t) pairs
    _, idx = np.unique(
        pairs.view([("s", pairs.dtype), ("t", pairs.dtype)]),
        return_index=True,
    )
    pairs = pairs[np.sort(idx)]

    # Weights: only incidents carry a real severity; others get NaN
    if evt == "Incident":
        sev = np.random.randint(1, 6, size=len(pairs)).astype(float)
    else:
        sev = np.full(len(pairs), np.nan)

    edges.extend(map(tuple, pairs.tolist()))
    notes.extend([REL_IMPACTS] * len(pairs))
    weights.extend(sev.tolist())

# Apply for all event types
for evt in ["Incident", "Change", "Alert"]:
    add_event_impacts(evt)

# Final edge arrays
edges = np.asarray(edges, dtype=np.int64)
notes = np.asarray(notes, dtype=object)
weights = np.asarray(weights, dtype=float)

# NODE FEATURES

# In-degree / out-degree / total degree
deg_out = np.bincount(edges[:, 0], minlength=TOTAL_NODES)
deg_in  = np.bincount(edges[:, 1], minlength=TOTAL_NODES)
deg     = deg_in + deg_out

# Which edges originate from Incident nodes?
incident_src = set(type2ids.get("Incident", []))
mask_inc = np.isin(edges[:, 0], list(incident_src)) if incident_src else np.zeros(len(edges), bool)
inc_targets = edges[mask_inc, 1]
inc_weights = np.nan_to_num(weights[mask_inc], nan=0.0)

# Incident score = sum of incident severities on each node
incident_score = np.zeros(TOTAL_NODES, dtype=np.float32)
if len(inc_targets):
    np.add.at(incident_score, inc_targets, inc_weights)

# Build simple undirected adjacency (for 1-hop propagation)
adj = [[] for _ in range(TOTAL_NODES)]
for s, t in edges:
    adj[s].append(t)
    adj[t].append(s)

# Give neighbors 25% of each incident's severity
for d, w in zip(inc_targets, inc_weights):
    for n in adj[int(d)]:
        incident_score[n] += 0.25 * w

# HIERARCHY LEVELS (BFS FROM LOB)

# Build parent-child tree from parent_of edges
tree = defaultdict(list)
for (s, t), note in zip(edges, notes):
    if note == REL_PARENT_OF:
        tree[s].append(t)

# level[u] = distance in parent_of tree from a LOB node (or -1 if unreachable)
level = np.full(TOTAL_NODES, -1, dtype=np.int32)
dq = deque()

# All LOBs are level 0 roots
for lob in type2ids["LOB"]:
    level[lob] = 0
    dq.append(lob)

# BFS to fill in levels
while dq:
    u = dq.popleft()
    for v in tree.get(u, []):
        if level[v] == -1:
            level[v] = level[u] + 1
            dq.append(v)

# LABELS (LOW / MEDIUM / HIGH)

# Types that never get labels
blocked = {"Unclassified", "Orphan", "Change", "Incident", "Alert"}

# Which nodes are eligible for labels?
eligible = np.array([t not in blocked for t in node_type_names], bool)

# Choose quantile cutpoints among eligible nodes' incident_score
q60, q90 = (
    np.quantile(incident_score[eligible], [0.6, 0.9])
    if eligible.any()
    else (0.0, 0.0)
)

# String labels per node
labels = np.empty(TOTAL_NODES, dtype=object)
labels[:] = None

labels[(incident_score < q60) & eligible] = "low"
labels[(incident_score >= q60) & (incident_score < q90) & eligible] = "medium"
labels[(incident_score >= q90) & eligible] = "high"

# Map string labels -> integers
label_map = {"low": 0, "medium": 1, "high": 2}
label_int = np.full(TOTAL_NODES, -1, dtype=np.int8)

for i, l in enumerate(labels):
    if l is not None:
        label_int[i] = label_map[l]

# Build node-level train/val/test splits using only labeled nodes
labeled_idx = np.where(label_int >= 0)[0]
np.random.shuffle(labeled_idx)
n = len(labeled_idx)

train_idx = labeled_idx[: int(0.70 * n)]
val_idx   = labeled_idx[int(0.70 * n): int(0.85 * n)]
test_idx  = labeled_idx[int(0.85 * n):]

# EXPORT CORE FILES

nodes_df = pd.DataFrame({
    "id":        np.arange(TOTAL_NODES, dtype=np.int64),
    "type_id":   np.array([TYPE_INDEX[t] for t in node_type_names], dtype=int),
    "type_name": node_type_names,
    "name":      node_name,
    "label":     labels,
    "timestamp": timestamps,
    "deg_in":    deg_in.astype(int),
    "deg_out":   deg_out.astype(int),
    "deg":       deg.astype(int),
    "level":     level.astype(int),
})
nodes_df.to_csv(os.path.join(OUTDIR, "nodes.csv"), index=False)

edges_df = pd.DataFrame({
    "source": edges[:, 0],
    "target": edges[:, 1],
    "weight": weights,
    "note":   notes,
})
edges_df.to_csv(os.path.join(OUTDIR, "edges.csv"), index=False)

# Numpy arrays for node classification tasks
np.save(os.path.join(OUTDIR, "label_int.npy"), label_int)
np.save(os.path.join(OUTDIR, "train_idx.npy"), train_idx)
np.save(os.path.join(OUTDIR, "val_idx.npy"),   val_idx)
np.save(os.path.join(OUTDIR, "test_idx.npy"),  test_idx)

# Type vocabulary
with open(os.path.join(OUTDIR, "type_vocab.json"), "w") as f:
    json.dump({"type_vocab": sorted(COUNTS.keys())}, f, indent=2)

# Graph relation schema
with open(os.path.join(OUTDIR, "graph_schema.json"), "w") as f:
    json.dump(
        {
            "relations": {
                "parent_of": {},
                "depends_on": {},
                "communicates_with": {},
                "connected_to": {},
                "impacts": {},
            }
        },
        f,
        indent=2,
    )

print("[✓] nodes.csv, edges.csv, splits & arrays written.")

# LINK PREDICTION SPLITS

splits_root = os.path.join(OUTDIR, "edge_splits")
os.makedirs(splits_root, exist_ok=True)

def write_lp_splits(rel: str, rel_edges: np.ndarray, holdout: float = LP_HOLDOUT) -> None:
    """
    For a given relation 'rel', split its edges into train/val/test for link prediction:
    - Positive edges: actual observed edges of this type.
    - Negative edges: synthetic non-edges with same sources but random targets.
    """
    if len(rel_edges) == 0:
        return

    idx = np.arange(len(rel_edges))
    np.random.shuffle(idx)
    rel_edges = rel_edges[idx]

    n = len(rel_edges)
    n_test = max(1, int(holdout * n))
    n_val  = max(1, int(holdout * n))
    n_train = max(1, n - n_val - n_test)

    tr = rel_edges[:n_train]
    va = rel_edges[n_train : n_train + n_val]
    te = rel_edges[n_train + n_val :]

    pos_set = set(map(tuple, rel_edges.tolist()))

    def neg_of(pos):
        """
        For each positive edge (s, t), generate a negative (s, cand)
        where cand is a random node that is unlikely to be a positive edge.
        """
        out = []
        for s, t in pos:
            cand = np.random.randint(0, len(nodes_df))
            tries = 0
            while (s, cand) in pos_set and tries < 5:
                cand = np.random.randint(0, len(nodes_df))
                tries += 1
            out.append((s, cand))
        return np.asarray(out, dtype=np.int64)

    trn = neg_of(tr)
    van = neg_of(va)
    ten = neg_of(te)

    od = os.path.join(splits_root, rel)
    os.makedirs(od, exist_ok=True)

    pd.DataFrame(tr,  columns=["source", "target"]).to_csv(os.path.join(od, "train_pos.csv"), index=False)
    pd.DataFrame(trn, columns=["source", "target"]).to_csv(os.path.join(od, "train_neg.csv"), index=False)
    pd.DataFrame(va,  columns=["source", "target"]).to_csv(os.path.join(od, "val_pos.csv"),   index=False)
    pd.DataFrame(van, columns=["source", "target"]).to_csv(os.path.join(od, "val_neg.csv"),   index=False)
    pd.DataFrame(te,  columns=["source", "target"]).to_csv(os.path.join(od, "test_pos.csv"),  index=False)
    pd.DataFrame(ten, columns=["source", "target"]).to_csv(os.path.join(od, "test_neg.csv"),  index=False)

    print(f"[LP] {rel} splits written.")

# Build link prediction splits for each relation type
ea = edges_df[["source", "target"]].values
na = edges_df["note"].values

for rel in [
    REL_PARENT_OF,
    REL_DEPENDS_ON,
    REL_COMMUNICATES,
    REL_CONNECTED_TO,
    REL_IMPACTS,
]:
    write_lp_splits(rel, ea[na == rel])

print("[✓] All LP splits complete.")


def main():
    """
    Entry point wrapper.

    NOTE:
    - The generator runs immediately at import time (top-level code above),
      which matches your original script’s behavior.
    - main() is intentionally empty so that running:
          python cmdb_generator.py
      still produces the exact same side effects (files, logs) as before.
    """
    pass


if __name__ == "__main__":
    main()
