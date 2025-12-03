"""
Noisy CMDB Graph Injector

This script reads a CMDB-style graph (nodes.csv + edges.csv) and injects
realistic data quality issues to create a "noisy" version of the graph.

It can simulate:
- Missing edges and broken chains.
- Duplicate nodes with slightly modified metadata.
- Missing/blank metadata fields (name, timestamp, label).
- Mis-labeled node types (e.g., Server vs Database).
- Timestamp anomalies for event nodes (far future or very old).
- Stale edges flagged via a 'stale' column.
- Outlier or non-numeric impact weights.
- Cross-layer "short-circuit" edges between layers that shouldn't connect.
- Broken references where edges point to removed node rows.

The main outputs are updated (noisy) nodes and edges DataFrames and a
NoiseLog plus summary stats.
"""

# IMPORTS
import os
import json
import random
import math
from dataclasses import dataclass, asdict  # kept for possible adjustments
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from IPython.display import display

import numpy as np
import pandas as pd

# Global random seed for Python and NumPy and file locations
RNG_SEED = int(os.environ.get("SEED", 42))
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

ARTIFACTS = os.environ.get("ARTIFACTS", ".")  # where nodes.csv / edges.csv live
OUTDIR = os.environ.get("OUTDIR", os.path.join(ARTIFACTS, "noisy"))
os.makedirs(OUTDIR, exist_ok=True)

# NOISE CONFIG

# Control how much of each type of noise is injected and hard caps.
NOISE_CONFIG = {
    # Graph structure imperfections
    "missing_edges_pct":      0.03,   # % of edges to drop globally
    "duplicate_nodes_pct":    0.01,   # % of nodes to duplicate with name variations
    "cross_layer_links_pct":  0.01,   # % of short-circuit edges
    "broken_refs_pct":        0.005,  # % of edges to break by removing target/source node row

    # Metadata issues
    "missing_metadata_pct":   0.05,   # % of nodes to blank out name/timestamp/label
    "mislabel_type_pct":      0.01,   # % of nodes to assign wrong type_name
    "timestamp_anomaly_pct":  0.02,   # % of event nodes to set future/very-old timestamps

    # Relationship flags & weights
    "stale_link_pct":         0.04,   # % of edges to mark as stale
    "weight_outlier_pct":     0.01,   # % of incident weights to set extreme or string values

    # Safety / bounds (hard limits on operations)
    "max_cross_links":        250_000,   # cap for cross-layer edges
    "max_dupes":              100_000,   # cap for duplicate nodes
    "max_broken_refs":        50_000,    # cap for broken ref operations
    "max_missing_edges":      750_000,   # cap for dropped edges
}

# RELATION & TYPE CONFIG

# Relation names (must match generator)
REL_PARENT_OF     = "parent_of"
REL_DEPENDS_ON    = "depends_on"
REL_COMMUNICATES  = "communicates_with"
REL_CONNECTED_TO  = "connected_to"
REL_IMPACTS       = "impacts"

# Node types (for mislabels and cross-layer link candidates)
TYPE_ORDER = [
    "LOB", "BusinessService", "Application", "ServiceOffering", "AppInstance",
    "Server", "Database", "NetworkDevice", "Service",
    "Orphan", "Unclassified", "Change", "Incident", "Alert",
]

# Type swaps to simulate classification confusion
MISC_TYPE_SWAPS = [
    ("Server", "Database"), ("Database", "Server"),
    ("Application", "Service"), ("Service", "Application"),
    ("NetworkDevice", "Server"), ("Server", "NetworkDevice"),
]

# Cross-layer "short-circuit" link patterns
CROSS_LINK_CANDIDATES = [
    ("Application", "Database"),
    ("Application", "Server"),
    ("BusinessService", "Database"),
    ("LOB", "Application"),
    ("Service", "Database"),
]

print("Config loaded. Output ->", OUTDIR)

# LOAD CLEAN GRAPH

nodes_path = os.path.join(ARTIFACTS, "nodes.csv")
edges_path = os.path.join(ARTIFACTS, "edges.csv")

if not (os.path.exists(nodes_path) and os.path.exists(edges_path)):
    raise FileNotFoundError(
        f"Expected nodes.csv and edges.csv in {ARTIFACTS}. "
        "Run the CMDB Data Generator first or set ARTIFACTS env var."
    )

nodes = pd.read_csv(nodes_path)
edges = pd.read_csv(edges_path)

# Basic sanity checks so downstream logic doesn't break
assert {"id", "type_name", "name"}.issubset(nodes.columns), "nodes.csv missing required columns."
assert {"source", "target", "note"}.issubset(edges.columns), "edges.csv missing required columns."

print("Loaded:", nodes.shape, edges.shape)


# NOISE LOGGING

class NoiseLog:
    """
    Simple in-memory log of what noise was injected.
    - log[kind] = list of event dicts
    """

    def __init__(self):
        self.log = defaultdict(list)

    def add(self, kind: str, **kwargs) -> None:
        """Record a noise event of type 'kind' with arbitrary key-value details."""
        self.log[kind].append(kwargs)

    def summary(self):
        """Return counts per noise type (kind -> number of events recorded)."""
        return {k: len(v) for k, v in self.log.items()}

    def to_json(self, path: str) -> None:
        """Write the noise log (summary + detailed entries) to disk as JSON."""
        with open(path, "w") as f:
            json.dump({"summary": self.summary(), "details": self.log}, f, indent=2)

log = NoiseLog()

# GENERIC HELPERS

def sample_indices(n_total: int, frac: float, cap: int) -> np.ndarray:
    """
    Sample up to frac * n_total sorted indices (capped at 'cap').
    Returns array of indices into a DataFrame.
    """
    k = min(int(n_total * frac), cap)
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.random.choice(np.arange(n_total), size=k, replace=False)
    return np.sort(idx)

def variant_name(base: str) -> str:
    """
    Generate a slightly altered name for a duplicate node to simulate drift:
    - lower/upper case
    - underscores instead of spaces
    - remove dashes
    - add "dup"/"copy_of" suffixes
    - swap 'Server' with 'srv' in the string
    """
    base = str(base)
    variants = [
        base.lower(),
        base.upper(),
        base.replace(" ", "_"),
        base.replace("-", ""),
        f"{base}_dup",
        f"copy_of_{base}",
        base.replace("Server", "srv"),
    ]
    return random.choice(variants) if variants else (base + "_dup")

def safe_new_ids(existing: set, n: int, start_hint: Optional[int] = None) -> List[int]:
    """
    Create 'n' new integer IDs not present in 'existing'.
    Optionally start from 'start_hint', otherwise from max(existing) + 1.
    """
    out = []
    cur = max(existing) + 1 if start_hint is None else start_hint
    while len(out) < n:
        if cur not in existing:
            out.append(cur)
            existing.add(cur)
        cur += 1
    return out

# NOISE OPERATIONS

def inject_missing_edges(edges_df: pd.DataFrame, frac: float, cap: int) -> pd.DataFrame:
    """
    Randomly drop a fraction of edges to simulate missing relationships.
    Returns a new edges_df with some rows removed.
    """
    if frac <= 0:
        return edges_df
    idx = sample_indices(len(edges_df), frac, cap)
    dropped = edges_df.iloc[idx]  # kept for debugging if needed
    kept = edges_df.drop(edges_df.index[idx]).reset_index(drop=True)
    log.add("missing_edges", count=len(idx))
    return kept

def inject_duplicate_nodes(nodes_df: pd.DataFrame, frac: float, cap: int) -> pd.DataFrame:
    """
    Duplicate a fraction of nodes with small metadata differences:
    - New unique IDs.
    - Variant names.
    - Some timestamps blanked.
    - Some labels downgraded or made missing.
    """
    if frac <= 0:
        return nodes_df
    n = min(int(len(nodes_df) * frac), cap)
    if n <= 0:
        return nodes_df

    # Choose random nodes to duplicate
    candidates = nodes_df.sample(n, random_state=RNG_SEED).copy()

    # Allocate new IDs that don't clash with existing
    existing = set(nodes_df["id"].astype(int).tolist())
    dup_ids = safe_new_ids(existing, n)
    candidates.loc[:, "id"] = dup_ids

    # Metadata drift
    if "name" in candidates.columns:
        candidates.loc[:, "name"] = candidates["name"].apply(variant_name)

    # Some duplicates lose their timestamp
    if "timestamp" in candidates.columns:
        mask = np.random.rand(n) < 0.4
        candidates.loc[mask, "timestamp"] = ""

    # Labels are sometimes downgraded or dropped
    if "label" in candidates.columns:
        map_down = {"high": "medium", "medium": "low", "low": None}
        mask = np.random.rand(n) < 0.5
        candidates.loc[mask, "label"] = (
            candidates.loc[mask, "label"].map(map_down).fillna("")
        )

    log.add("duplicate_nodes", count=n)
    out = pd.concat([nodes_df, candidates], ignore_index=True)
    return out

def inject_missing_metadata(nodes_df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    For a random subset of nodes, blank out one or more of:
    - name
    - timestamp
    - label
    """
    if frac <= 0:
        return nodes_df
    n = int(len(nodes_df) * frac)
    if n <= 0:
        return nodes_df

    idx = np.random.choice(nodes_df.index, size=n, replace=False)

    for i in idx:
        if "name" in nodes_df.columns and np.random.rand() < 0.6:
            nodes_df.at[i, "name"] = ""
        if "timestamp" in nodes_df.columns and np.random.rand() < 0.5:
            nodes_df.at[i, "timestamp"] = ""
        if "label" in nodes_df.columns and np.random.rand() < 0.4:
            nodes_df.at[i, "label"] = None

    log.add("missing_metadata_rows", count=n)
    return nodes_df

def inject_mislabel_types(nodes_df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    Change type_name for some nodes to simulate misclassification:
    - Use swap pairs (Server<->Database, etc.) where possible.
    - Otherwise sample from a small set of infra-like types.
    """
    if frac <= 0:
        return nodes_df
    n = int(len(nodes_df) * frac)
    if n <= 0:
        return nodes_df

    idx = np.random.choice(nodes_df.index, size=n, replace=False)
    swaps = dict(MISC_TYPE_SWAPS)

    for i in idx:
        t = nodes_df.at[i, "type_name"]
        if t in swaps:
            nodes_df.at[i, "type_name"] = swaps[t]
        else:
            nodes_df.at[i, "type_name"] = np.random.choice(
                ["Server", "Database", "NetworkDevice", "Service"]
            )

    log.add("mislabel_types", count=n)
    return nodes_df

def inject_timestamp_anomalies(nodes_df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    For event-like nodes (Change/Incident/Alert), adjust timestamps to:
    - Far future (30–365 days ahead), or
    - Very old (5–15 years ago).
    """
    if frac <= 0 or "timestamp" not in nodes_df.columns:
        return nodes_df

    is_event = nodes_df["type_name"].isin(["Change", "Incident", "Alert"])
    cand = nodes_df[is_event]
    n = int(len(cand) * frac)
    if n <= 0:
        return nodes_df

    idx = np.random.choice(cand.index, size=n, replace=False)
    now = datetime.utcnow()

    for i in idx:
        if np.random.rand() < 0.5:
            # Far future
            nodes_df.at[i, "timestamp"] = (
                now + timedelta(days=np.random.randint(30, 365))
            ).isoformat()
        else:
            # Very old
            nodes_df.at[i, "timestamp"] = (
                now
                - timedelta(days=np.random.randint(365 * 5, 365 * 15))
            ).isoformat()

    log.add("timestamp_anomalies", count=n)
    return nodes_df

def inject_stale_links(edges_df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    Mark some edges as 'stale' via a boolean column, but do not remove them.
    """
    if frac <= 0:
        return edges_df
    n = int(len(edges_df) * frac)
    if n <= 0:
        return edges_df

    idx = np.random.choice(edges_df.index, size=n, replace=False)

    if "stale" not in edges_df.columns:
        edges_df["stale"] = False

    edges_df.loc[idx, "stale"] = True
    log.add("stale_links", count=n)
    return edges_df

def inject_weight_outliers(edges_df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    For a fraction of REL_IMPACTS edges, set the 'weight' column to:
    - Extreme numeric values (1000, -50, 9999), or
    - A non-numeric string ("n/a").
    """
    if frac <= 0 or "weight" not in edges_df.columns:
        return edges_df

    is_incident = edges_df["note"] == REL_IMPACTS
    cand = edges_df[is_incident]
    n = int(len(cand) * frac)
    if n <= 0:
        return edges_df

    idx = np.random.choice(cand.index, size=n, replace=False)

    for i in idx:
        if np.random.rand() < 0.5:
            edges_df.at[i, "weight"] = float(np.random.choice([1000, -50, 9999]))
        else:
            edges_df.at[i, "weight"] = "n/a"  # non-numeric

    log.add("weight_outliers", count=n)
    return edges_df

def inject_cross_layer_links(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    frac: float,
    cap: int,
) -> pd.DataFrame:
    """
    Add new 'short_circuit' edges between node types that would not normally
    connect.
    """
    if frac <= 0:
        return edges_df
    n = min(int(len(edges_df) * frac), cap)
    if n <= 0:
        return edges_df

    # Build pools per type
    by_type = {
        t: nodes_df.loc[nodes_df["type_name"] == t, "id"].astype(int).values
        for t in TYPE_ORDER
    }

    new_edges = []
    for _ in range(n):
        a, b = random.choice(CROSS_LINK_CANDIDATES)
        if len(by_type.get(a, [])) == 0 or len(by_type.get(b, [])) == 0:
            continue

        s = int(np.random.choice(by_type[a]))
        t = int(np.random.choice(by_type[b]))
        if s == t:
            continue

        new_edges.append((s, t, np.nan, "short_circuit"))

    if not new_edges:
        return edges_df

    add = pd.DataFrame(new_edges, columns=["source", "target", "weight", "note"])
    log.add("cross_layer_links", count=len(add))
    return pd.concat([edges_df, add], ignore_index=True)

def inject_broken_references(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    frac: float,
    cap: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a subset of edges, remove the source or target node row from nodes_df.
    The edges are left as-is, creating 'broken' references.
    """
    if frac <= 0:
        return nodes_df, edges_df
    n = min(int(len(edges_df) * frac), cap)
    if n <= 0:
        return nodes_df, edges_df

    idx = np.random.choice(edges_df.index, size=n, replace=False)
    chosen = edges_df.loc[idx, ["source", "target"]].copy()

    remove_ids = set()
    for _, row in chosen.iterrows():
        if np.random.rand() < 0.5:
            remove_ids.add(int(row["target"]))
        else:
            remove_ids.add(int(row["source"]))

    before = len(nodes_df)
    nodes_df = nodes_df[
        ~nodes_df["id"].astype(int).isin(remove_ids)
    ].reset_index(drop=True)
    removed = before - len(nodes_df)

    log.add("broken_references", edges=len(idx), nodes_removed=removed)
    return nodes_df, edges_df

# APPLY NOISE PIPELINE

nodes_noisy = nodes.copy()
edges_noisy = edges.copy()

# 1) Drop some edges
edges_noisy = inject_missing_edges(
    edges_noisy,
    NOISE_CONFIG["missing_edges_pct"],
    NOISE_CONFIG["max_missing_edges"],
)

# 2) Duplicate nodes with metadata drift
nodes_noisy = inject_duplicate_nodes(
    nodes_noisy,
    NOISE_CONFIG["duplicate_nodes_pct"],
    NOISE_CONFIG["max_dupes"],
)

# 3) Missing metadata (name/timestamp/label)
nodes_noisy = inject_missing_metadata(
    nodes_noisy,
    NOISE_CONFIG["missing_metadata_pct"],
)

# 4) Mislabel types
nodes_noisy = inject_mislabel_types(
    nodes_noisy,
    NOISE_CONFIG["mislabel_type_pct"],
)

# 5) Timestamp anomalies for event nodes
nodes_noisy = inject_timestamp_anomalies(
    nodes_noisy,
    NOISE_CONFIG["timestamp_anomaly_pct"],
)

# 6) Mark some edges as stale
edges_noisy = inject_stale_links(
    edges_noisy,
    NOISE_CONFIG["stale_link_pct"],
)

# 7) Weight outliers on IMPACTS edges
edges_noisy = inject_weight_outliers(
    edges_noisy,
    NOISE_CONFIG["weight_outlier_pct"],
)

# 8) Cross-layer short-circuit links
edges_noisy = inject_cross_layer_links(
    nodes_noisy,
    edges_noisy,
    NOISE_CONFIG["cross_layer_links_pct"],
    NOISE_CONFIG["max_cross_links"],
)

# 9) Broken references (remove node rows for some edges)
nodes_noisy, edges_noisy = inject_broken_references(
    nodes_noisy,
    edges_noisy,
    NOISE_CONFIG["broken_refs_pct"],
    NOISE_CONFIG["max_broken_refs"],
)

print("Noise summary:", log.summary())

# QUICK EDA / DIAGNOSTICS

def missingness(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Count missing or blank values in selected columns.
    Returns a DataFrame with ('column', 'missing_or_blank').
    """
    data = {
        c: df[c].isna().sum() + (df[c] == "").sum()
        for c in cols
        if c in df.columns
    }
    return (
        pd.DataFrame(
            {"column": list(data.keys()), "missing_or_blank": list(data.values())}
        )
        .sort_values("missing_or_blank", ascending=False)
    )

print("Nodes (clean vs noisy):", nodes.shape, nodes_noisy.shape)
print("Edges (clean vs noisy):", edges.shape, edges_noisy.shape)

# Type distributions (clean vs noisy)
clean_types = nodes["type_name"].value_counts().rename("clean")
noisy_types = nodes_noisy["type_name"].value_counts().rename("noisy")
type_comp = (
    pd.concat([clean_types, noisy_types], axis=1)
    .fillna(0)
    .astype(int)
    .sort_index()
)
display(type_comp.style.format("{:,}"))

node_miss = missingness(nodes_noisy, ["name", "timestamp", "label"])
display(node_miss)

# Edge flags
if "stale" in edges_noisy.columns:
    print("Stale links count:", int(edges_noisy["stale"].sum()))
else:
    print("Stale links column not present (disabled).")

# Non-numeric weights
non_numeric = pd.to_numeric(edges_noisy["weight"], errors="coerce").isna().sum()
print("Non-numeric weights:", non_numeric, " / ", len(edges_noisy))

# Orphan nodes: never appear in edges (source or target)
used_ids = set(edges_noisy["source"].astype(int)).union(
    set(edges_noisy["target"].astype(int))
)
orphan_mask = ~nodes_noisy["id"].astype(int).isin(used_ids)
orphans = nodes_noisy[orphan_mask]
print("Orphan nodes (no incident edges after noise):", len(orphans))

# Degree distribution
deg_series = pd.concat(
    [edges_noisy["source"], edges_noisy["target"]]
).astype(int).value_counts()
deg_stats = deg_series.describe()
print("Degree stats (undirected approx):")
print(deg_stats)

# Optional plotting
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    nodes_noisy["type_name"].value_counts().sort_values(ascending=False).plot(
        kind="bar"
    )
    plt.title("Node Type Distribution (Noisy)")
    plt.tight_layout()
    plt.show()

    # Degree histogram, capped to 99th percentile to avoid extreme tails
    deg_vals = pd.concat(
        [edges_noisy["source"], edges_noisy["target"]]
    ).astype(int).value_counts().values
    cap = np.percentile(deg_vals, 99)

    plt.figure(figsize=(8, 4))
    plt.hist(np.clip(deg_vals, None, cap), bins=50)
    plt.title("Degree Histogram (Noisy, 99th pct capped)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Plotting skipped:", e)


def main():
    """
    Entry point wrapper.

    NOTE:
    - Just like the generator, this script executes its noise pipeline at
      import/run time (above), matching your original behavior.
    - main() is intentionally empty so that:
          python cmdb_noise_injector.py
      gives you the same side effects (noisy DataFrames, prints, displays).
    """
    pass


if __name__ == "__main__":
    main()
