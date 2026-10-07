"""Per-vertex roadmap embeddings for the cluster pipeline (gnn + unlearned controls).

``compute_embedding`` is the single entry point that ``gnn_cluster_map`` routes
through. For ``method == "gnn"`` it runs the trained encoder exactly as the
pipeline always did (same two lines, float32, no post-processing). The other
methods are *controls* that differ from the GNN only in this step; everything
downstream (start/goal anchor exclusion, K-means, medoids, CTopPRM
reconstruction) is shared and untouched:

* ``euclid``   raw vertex coordinates, float64, no scaling.
* ``isomap``   classical MDS on roadmap shortest-path distances (Euclidean edge
               weights), output dimension ``d``.
* ``spectral`` eigenvectors of the symmetric normalized Laplacian of the
               Gaussian-weighted roadmap (sigma = median edge length) for the
               ``d`` smallest non-zero eigenvalues.

Vertex set: every method embeds *all* ``map_.nodes`` (row i <-> node i), the
same set the GNN embeds, including registered boundary nodes and the
start/goal nodes. isomap/spectral need a connected graph, so they embed the
largest connected component of the symmetrized roadmap and report every other
vertex in ``info["dropped"]`` (row set to NaN); the caller removes those from
the clustering candidates (option (a) of the ablation spec). A start/goal
outside the component is an error (the dense roadmap cannot solve that
instance either).

Cache: isomap/spectral results are content-addressed (vertex coordinates +
symmetric edge list + method + d_max + EMBEDDING_VERSION) so identical source
roadmaps across cluster-fraction roots / embedding dims are computed once.
A cache hit reports the *stored* compute time so runtime accounting always
reflects the real cost of producing the embedding.
"""
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigsh

from path_planning.cluster.kmeans.graph_kmeans import build_symmetric_adjacency

EMBEDDING_METHODS = ("gnn", "euclid", "isomap", "spectral")
EMBEDDING_VERSION = "1"   # bump when the isomap/spectral maths changes (invalidates the cache)
ISOMAP_DMAX = 64          # eigenpairs computed per roadmap; any d <= this is a prefix slice
SPECTRAL_DMAX = 32


def default_method_name(method: str, d: int) -> str:
    """Output-file method label: gnn -> 'gnn', euclid -> 'euclid', else '<method>_<d>'."""
    if method == "gnn":
        return "gnn"
    if method == "euclid":
        return "euclid"
    return f"{method}_{int(d)}"


def node_positions(map_) -> np.ndarray:
    """(|V|, dim) float64 world coordinates, row i <-> map_.nodes[i]."""
    return np.array([n.current for n in map_.nodes], dtype=np.float64)


def symmetric_roadmap_csr(map_, pos: Optional[np.ndarray] = None
                          ) -> Tuple[csr_matrix, np.ndarray]:
    """Undirected CSR adjacency (Euclidean edge lengths) of the roadmap plus
    the canonical (u < v, w) edge array used for content hashing. Uses the
    same symmetrizer as GraphKMeans/GraphEM/CTopPRM (k-NN roadmaps can be
    one-directional)."""
    if pos is None:
        pos = node_positions(map_)
    adj = build_symmetric_adjacency(pos, map_.road_map, map_.road_map_edge_weights)
    rows, cols, vals = [], [], []
    for u, nbrs in enumerate(adj):
        for v, w in nbrs:
            rows.append(u)
            cols.append(v)
            vals.append(w)
    n = len(pos)
    csr = csr_matrix((np.asarray(vals, dtype=np.float64),
                      (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
                     shape=(n, n))
    und = [(u, v, w) for u, nbrs in enumerate(adj) for v, w in nbrs if u < v]
    edges = np.array(und, dtype=np.float64).reshape(-1, 3)
    return csr, edges


def largest_component(csr: csr_matrix) -> Tuple[np.ndarray, int, np.ndarray]:
    """(sorted vertex indices of the largest connected component,
    number of components, per-vertex component labels)."""
    n_comp, labels = connected_components(csr, directed=False)
    if n_comp <= 1:
        return np.arange(csr.shape[0]), int(n_comp), labels
    biggest = int(np.bincount(labels).argmax())
    return np.flatnonzero(labels == biggest), int(n_comp), labels


# ---------------------------------------------------------------------------
# isomap (classical MDS on geodesic distances)
# ---------------------------------------------------------------------------

def _classical_mds_gram(dist: np.ndarray) -> np.ndarray:
    """B = -1/2 J D^2 J via in-place double centering (no explicit J)."""
    b = dist.astype(np.float64, copy=True)
    np.square(b, out=b)
    row_mean = b.mean(axis=1, keepdims=True)
    col_mean = b.mean(axis=0, keepdims=True)
    grand = float(b.mean())
    b -= row_mean
    b -= col_mean
    b += grand
    b *= -0.5
    # symmetrize against round-off so eigsh sees an exactly Hermitian operator
    b += b.T
    b *= 0.5
    return b


def _top_eigsh(mat, k: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Largest-algebraic eigenpairs sorted descending, deterministic start vector."""
    n = mat.shape[0]
    k = int(min(k, n - 1))
    v0 = np.random.default_rng(seed).standard_normal(n)
    vals, vecs = eigsh(mat, k=k, which="LA", v0=v0)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def isomap_component(csr_lcc: csr_matrix, d_max: int = ISOMAP_DMAX,
                     full_spectrum: bool = False) -> Dict[str, np.ndarray]:
    """Classical-MDS embedding of one connected component.

    Returns dict with ``z`` (n, m) = V_m Lambda_m^{1/2} over the m <= d_max
    positive top eigenvalues, ``eigenvalues`` (m,), ``negative_mass``
    (sum|lambda^-| / sum lambda^+, needs the full spectrum else NaN),
    ``spectrum`` (all eigenvalues descending, only with full_spectrum) and
    the two timings ``embed_time`` (Dijkstra + centering + top eigenpairs)
    and ``spectrum_time`` (the diagnostic full eigvalsh)."""
    from scipy.linalg import eigvalsh

    t0 = time.perf_counter()
    dist = shortest_path(csr_lcc, method="D", directed=False)
    if not np.all(np.isfinite(dist)):
        raise RuntimeError("isomap: shortest-path matrix has inf entries (component not connected?)")
    gram = _classical_mds_gram(dist)
    del dist
    vals, vecs = _top_eigsh(gram, d_max)
    tol = 1e-10 * max(float(vals[0]), 1e-300)
    keep = vals > tol
    vals, vecs = vals[keep], vecs[:, keep]
    z = vecs * np.sqrt(vals)[None, :]
    embed_time = time.perf_counter() - t0

    spectrum = None
    negative_mass = np.nan
    spectrum_time = 0.0
    if full_spectrum:
        t1 = time.perf_counter()
        spectrum = eigvalsh(gram)[::-1]
        pos_sum = float(spectrum[spectrum > 0].sum())
        negative_mass = float(-spectrum[spectrum < 0].sum() / pos_sum) if pos_sum > 0 else np.nan
        spectrum_time = time.perf_counter() - t1
    return {"z": z, "eigenvalues": vals, "spectrum": spectrum,
            "negative_mass": negative_mass, "embed_time": embed_time,
            "spectrum_time": spectrum_time}


def explained_variance(eigenvalues_top: np.ndarray, d: int,
                       spectrum: Optional[np.ndarray]) -> float:
    """Fraction of the positive MDS eigenvalue mass captured by the top d
    (Mardia's a_1 on the positive part). Needs the full spectrum for the
    denominator; NaN otherwise."""
    if spectrum is None:
        return float("nan")
    pos_sum = float(spectrum[spectrum > 0].sum())
    if pos_sum <= 0:
        return float("nan")
    return float(eigenvalues_top[:d].sum() / pos_sum)


# ---------------------------------------------------------------------------
# spectral (symmetric normalized Laplacian eigenmaps)
# ---------------------------------------------------------------------------

def spectral_component(csr_lcc: csr_matrix, d_max: int = SPECTRAL_DMAX) -> Dict[str, np.ndarray]:
    """Eigenvectors of L_sym = I - D^-1/2 A D^-1/2 for the d_max smallest
    non-zero eigenvalues, A_ij = exp(-w_ij^2 / 2 sigma^2), sigma = median edge
    length. Computed as the largest eigenpairs of D^-1/2 A D^-1/2 (no
    factorization, no shift), dropping the single trivial vector
    (lambda_L = 0; exactly one because the input is one component).
    Returns ``z`` (n, d_max) = the L_sym eigenvectors (not the random-walk
    rescaling), ``eigenvalues`` (d_max,) of L_sym ascending, ``sigma``."""
    t0 = time.perf_counter()
    w = csr_lcc.data
    sigma = float(np.median(w)) if w.size else 1.0
    if sigma <= 0:
        sigma = 1.0
    aff = csr_lcc.copy()
    aff.data = np.exp(-(w ** 2) / (2.0 * sigma ** 2))
    deg = np.asarray(aff.sum(axis=1)).ravel()
    if np.any(deg <= 0):
        raise RuntimeError("spectral: zero-degree vertex inside the component")
    dinv = 1.0 / np.sqrt(deg)
    normalized = aff.multiply(dinv[:, None]).multiply(dinv[None, :]).tocsr()
    vals, vecs = _top_eigsh(normalized, d_max + 1)
    # vals[0] ~ 1 is the trivial component indicator (lambda_L = 0)
    lam_l = 1.0 - vals[1:]
    z = vecs[:, 1:]
    return {"z": z, "eigenvalues": lam_l, "sigma": sigma,
            "trivial_eigenvalue": float(1.0 - vals[0]),
            "embed_time": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def _cache_key(pos: np.ndarray, edges: np.ndarray, method: str, d_max: int) -> str:
    h = hashlib.sha1()
    h.update(EMBEDDING_VERSION.encode())
    h.update(method.encode())
    h.update(str(int(d_max)).encode())
    h.update(np.ascontiguousarray(pos).tobytes())
    h.update(np.ascontiguousarray(edges).tobytes())
    return h.hexdigest()


def _cache_load(cache_dir, key: str) -> Optional[Dict[str, np.ndarray]]:
    f = Path(cache_dir) / f"{key}.npz"
    if not f.exists():
        return None
    try:
        with np.load(f, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}
    except Exception:  # noqa: BLE001 - corrupt/partial entry: recompute
        return None


def _cache_store(cache_dir, key: str, payload: Dict[str, np.ndarray]) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / f"{key}.npz"
    tmp = cache_dir / f".{key}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, final)   # atomic: concurrent writers never expose a partial file


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def compute_embedding(map_, method: str, d: int = 32, *, model=None, data=None,
                      device=None, cache_dir=None, full_spectrum: bool = False
                      ) -> Tuple[np.ndarray, Dict]:
    """Per-vertex embedding of ``map_`` (row i <-> map_.nodes[i]).

    Returns ``(z, info)``. ``z`` has ``len(map_.nodes)`` rows; rows of
    vertices the method could not embed (outside the largest connected
    component, isomap/spectral only) are NaN and listed in ``info["dropped"]``.
    ``info["compute_time"]`` is the wall-clock cost of producing the embedding
    (stored value on a cache hit).
    """
    if method not in EMBEDDING_METHODS:
        raise ValueError(f"embedding method must be one of {EMBEDDING_METHODS}, got {method!r}")
    n = len(map_.nodes)
    info: Dict = {"method": method, "requested_dim": int(d), "n_embedded": n,
                  "dropped": np.zeros(0, dtype=np.int64), "n_components": 1,
                  "spectrum": None, "eigenvalues": None,
                  "explained_variance": float("nan"), "negative_mass": float("nan"),
                  "compute_time": 0.0, "spectrum_time": 0.0, "cache_hit": False,
                  "sigma": None, "embedding_time_measured_at_dim": int(d)}

    if method == "gnn":
        if model is None or data is None or device is None:
            raise ValueError("compute_embedding('gnn') needs model, data and device")
        import torch
        t0 = time.perf_counter()
        with torch.no_grad():
            ds = data.to(device)
            z = model(dict(ds.x_dict), ds.edge_index_dict, ds.edge_attr_dict)['node'].cpu().numpy()
        info["compute_time"] = time.perf_counter() - t0
        info["dim"] = int(z.shape[1])
        info["embedding_time_measured_at_dim"] = info["dim"]
        return z, info

    t0 = time.perf_counter()
    pos = node_positions(map_)
    if method == "euclid":
        z = pos
        info["compute_time"] = time.perf_counter() - t0
        info["dim"] = int(z.shape[1])
        info["embedding_time_measured_at_dim"] = info["dim"]
        return z, info

    # isomap / spectral: symmetric graph -> largest component -> (cached) solve
    csr, edges = symmetric_roadmap_csr(map_, pos)
    lcc, n_comp, _ = largest_component(csr)
    dropped = np.setdiff1d(np.arange(n), lcc)
    sg = set(map_.start_nodes_index.values()) | set(map_.goal_nodes_index.values())
    bad = sorted(sg & set(dropped.tolist()))
    if bad:
        raise RuntimeError(
            f"{method}: start/goal node(s) {bad} lie outside the largest connected "
            f"component ({n_comp} components); the instance is unsolvable on the dense roadmap")
    d_max = ISOMAP_DMAX if method == "isomap" else max(SPECTRAL_DMAX, int(d))
    if int(d) > d_max:
        raise ValueError(f"{method}: d={d} exceeds d_max={d_max}")
    key = _cache_key(pos, edges, method, d_max)
    entry = _cache_load(cache_dir, key) if cache_dir else None
    need_spectrum = method == "isomap" and full_spectrum
    if entry is not None and need_spectrum and entry.get("spectrum", np.zeros(0)).size == 0:
        entry = None   # cached without the diagnostic spectrum: recompute with it
    cache_hit = entry is not None
    if entry is None:
        csr_lcc = csr[lcc][:, lcc].tocsr()
        if method == "isomap":
            res = isomap_component(csr_lcc, d_max, full_spectrum=need_spectrum)
            entry = {"z": res["z"], "eigenvalues": res["eigenvalues"],
                     "spectrum": (res["spectrum"] if res["spectrum"] is not None
                                  else np.zeros(0)),
                     "negative_mass": np.float64(res["negative_mass"]),
                     "sigma": np.float64(np.nan),
                     "compute_time": np.float64(res["embed_time"]),
                     "spectrum_time": np.float64(res["spectrum_time"])}
        else:
            res = spectral_component(csr_lcc, d_max)
            entry = {"z": res["z"], "eigenvalues": res["eigenvalues"],
                     "spectrum": np.zeros(0),
                     "negative_mass": np.float64(np.nan),
                     "sigma": np.float64(res["sigma"]),
                     "compute_time": np.float64(res["embed_time"]),
                     "spectrum_time": np.float64(0.0)}
        entry["dropped"] = dropped
        entry["n_components"] = np.int64(n_comp)
        if cache_dir:
            _cache_store(cache_dir, key, entry)

    z_full = np.asarray(entry["z"], dtype=np.float64)
    m = min(int(d), z_full.shape[1])
    z = np.full((n, m), np.nan, dtype=np.float64)
    z[lcc] = z_full[:, :m]
    eig = np.asarray(entry["eigenvalues"], dtype=np.float64)
    spectrum = np.asarray(entry["spectrum"], dtype=np.float64)
    spectrum = spectrum if spectrum.size else None
    info.update({
        "dim": int(m),
        "dropped": np.asarray(entry["dropped"], dtype=np.int64),
        "n_components": int(entry["n_components"]),
        "spectrum": spectrum,
        "eigenvalues": eig[:m],
        "explained_variance": (explained_variance(eig, m, spectrum)
                               if method == "isomap" else float("nan")),
        "negative_mass": float(entry["negative_mass"]),
        "compute_time": float(entry["compute_time"]),
        "spectrum_time": float(entry["spectrum_time"]),
        "cache_hit": bool(cache_hit),
        "sigma": (float(entry["sigma"]) if method == "spectral" else None),
        "embedding_time_measured_at_dim": int(d_max),
        "cache_key": key,
    })
    if m < int(d):
        info["dim_shortfall"] = int(d) - m   # fewer positive eigenvalues than requested
    return z, info


def embedding_summary(info: Dict) -> Dict:
    """YAML-friendly subset of ``info`` (no arrays) for the runtime sidecar."""
    keys = ("method", "requested_dim", "dim", "n_embedded", "n_components",
            "explained_variance", "negative_mass", "compute_time", "spectrum_time",
            "cache_hit", "sigma", "embedding_time_measured_at_dim", "cache_key",
            "dim_shortfall")
    out = {k: info[k] for k in keys if k in info}
    out["n_dropped"] = int(len(info.get("dropped", ())))
    out["dropped"] = [int(i) for i in info.get("dropped", ())]
    return out
