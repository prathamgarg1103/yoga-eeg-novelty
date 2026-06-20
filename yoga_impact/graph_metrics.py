"""Graph/network metrics for connectivity matrices (C2).

networkx is the primary backend (clean modularity + efficiency); a dependency-free
numpy/scipy fallback is provided so the pipeline runs even without networkx. Metrics
are computed on a proportionally-thresholded binary graph (a standard network-
neuroscience choice that controls for edge density across subjects).
"""
from __future__ import annotations

import numpy as np

from yoga_impact import config


def _threshold_proportional(W: np.ndarray, density: float):
    """Keep the strongest ``density`` fraction of edges; return (weighted_thr, binary)."""
    W = np.abs(np.asarray(W, dtype=float)).copy()
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    w = W[iu]
    if w.size == 0:
        return W, (W > 0).astype(float)
    k = max(1, int(round(density * w.size)))
    if k < w.size:
        thr = np.sort(w)[::-1][k - 1]
        W[W < thr] = 0.0
    A = (W > 0).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return W, A


def _fallback_metrics(A: np.ndarray) -> dict:
    from scipy.sparse.csgraph import shortest_path
    n = A.shape[0]
    D = shortest_path(A, method="D", unweighted=True)
    with np.errstate(divide="ignore"):
        invD = 1.0 / D
    invD[~np.isfinite(invD)] = 0.0
    np.fill_diagonal(invD, 0.0)
    geff = float(invD.sum() / (n * (n - 1))) if n > 1 else float("nan")
    finite = D[np.isfinite(D) & (D > 0)]
    cpl = float(finite.mean()) if finite.size else float("nan")
    tri = np.diag(A @ A @ A)
    deg = A.sum(1)
    denom = deg * (deg - 1)
    c = np.where(denom > 0, tri / denom, 0.0)
    return dict(global_efficiency=geff, mean_clustering=float(c.mean()),
                char_path_length=cpl, modularity=float("nan"))


def graph_features(W: np.ndarray, density: float = config.NETWORK_DENSITY,
                   frontal_idx=None) -> dict:
    """Network metrics for one connectivity matrix. Threshold is intra-subject (leak-free)."""
    Wt, A = _threshold_proportional(W, density)
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    out = {
        "mean_connectivity": float(np.mean(np.abs(np.asarray(W, float)[iu]))),
        "density_weight": float(np.mean(Wt[iu])),
    }
    try:
        import networkx as nx
        G = nx.from_numpy_array(A)
        out["global_efficiency"] = float(nx.global_efficiency(G))
        out["mean_clustering"] = float(nx.average_clustering(G))
        if nx.number_connected_components(G) == 1:
            out["char_path_length"] = float(nx.average_shortest_path_length(G))
        else:
            cc = max(nx.connected_components(G), key=len)
            out["char_path_length"] = float(nx.average_shortest_path_length(G.subgraph(cc)))
        comms = nx.community.greedy_modularity_communities(G)
        out["modularity"] = float(nx.community.modularity(G, comms))
    except Exception:  # noqa: BLE001 — networkx missing or graph degenerate
        out.update(_fallback_metrics(A))

    if frontal_idx is not None and len(frontal_idx) > 1:
        fi = np.asarray(frontal_idx)
        sub = np.abs(np.asarray(W, float))[np.ix_(fi, fi)]
        out["frontal_strength"] = float(np.mean(sub[np.triu_indices(len(fi), 1)]))
    else:
        out["frontal_strength"] = float("nan")
    return out
