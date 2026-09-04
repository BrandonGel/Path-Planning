"""Graph-aware k-means over a ``GraphSampler`` roadmap.

Lloyd's iterations where assignment runs along the roadmap (multi-source
Dijkstra) instead of Euclidean space, with two kinds of centers:

- FIXED centers: caller-supplied seed points (e.g. agent starts/goals) that
  never move — every fixed seed keeps its own cluster, like CTopPRM's
  endpoint seeds.
- FREE centers: additional clusters whose center tracks the mean of the
  member node coordinates. The continuous centroid is kept when it lies in
  free space (boundary-inclusive ``GraphSampler.point_expandable``);
  otherwise the center falls back to the member node nearest the mean, so a
  center never sits inside an obstacle.

Because Dijkstra needs graph sources, every center — free or fixed — is
anchored to a roadmap node (``center_node_indices``); the continuous
``centers`` are reporting/consumer-facing positions. The final
``cluster_labels`` / ``dist`` / ``prev`` state is exactly the wavefront
state CTopPRM's ``_wavefront_fill`` produces for ``center_node_indices``,
so the result can seed the CTopPRM pipeline directly
(``CTopPRM(..., clustering="kmeans")``).
"""

from __future__ import annotations

import heapq
import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from path_planning.common.environment.node import Node

logger = logging.getLogger(__name__)

Endpoint = Union[int, Sequence[float]]

_INIT_MODES = ("kpp", "farthest")


# ---------------------------------------------------------------------------
# Graph helpers (local ports of CTopPRM's, kept here to avoid an import
# cycle: CTopPRMpy.ctopprm imports GraphKMeans for clustering="kmeans")
# ---------------------------------------------------------------------------

def build_symmetric_adjacency(
    points: np.ndarray, road_map, weights=None
) -> List[List[Tuple[int, float]]]:
    """Undirected adjacency from a (possibly directional) ``road_map``.

    Port of ``CTopPRM._build_symmetric_adjacency``: k-NN roadmaps can be
    asymmetric (j in adj[i] without i in adj[j]); every edge is inserted in
    both directions. Weights come from ``weights`` when aligned with
    ``road_map``, else Euclidean distance between the endpoints.
    """
    n = len(points)
    aligned = (
        weights is not None
        and len(weights) == len(road_map)
        and all(len(weights[i]) == len(road_map[i]) for i in range(len(road_map)))
    )
    adj: List[dict] = [dict() for _ in range(n)]
    for u in range(min(len(road_map), n)):
        for k, v in enumerate(road_map[u]):
            if not 0 <= v < n or v == u:
                continue
            w = (
                float(weights[u][k])
                if aligned
                else float(np.linalg.norm(points[u] - points[v]))
            )
            if w < adj[u].get(v, np.inf):
                adj[u][v] = w
                adj[v][u] = w
    return [sorted(d.items()) for d in adj]


def multi_source_dijkstra(
    adj: List[List[Tuple[int, float]]], sources: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-source Dijkstra labelling every node with its nearest source.

    Port of ``CTopPRM._wavefront_fill`` as a pure function. Label ``k``
    is the cluster of ``sources[k]``; ``dist`` is 0.0 exactly at sources;
    the ``prev`` forest terminates (-1) at each node's own source.
    Unreached nodes keep label -1 / dist inf.
    """
    n = len(adj)
    labels = np.full(n, -1, dtype=np.int32)
    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int32)
    heap: List[Tuple[float, int]] = []
    for k, s in enumerate(sources):
        dist[s] = 0.0
        labels[s] = k
        heapq.heappush(heap, (0.0, s))
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                labels[v] = labels[u]
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return labels, dist, prev


# ---------------------------------------------------------------------------
# Graph k-means
# ---------------------------------------------------------------------------

class GraphKMeans:
    """K-means over roadmap nodes with graph distances and fixed anchors.

    Args:
        graph_map: ``GraphSampler`` with a generated roadmap (``nodes`` and
            ``road_map`` populated).
        n_clusters: Total cluster count INCLUDING the fixed seeds passed to
            :meth:`fit` (must be >= their number). May be reduced when the
            roadmap has fewer reachable nodes than requested clusters.
        max_iters: Cap on Lloyd iterations.
        tol: Secondary convergence threshold on the largest center movement;
            the primary criterion is the source set reaching a fixed point.
        init: Top-up seeding for the free centers — ``"kpp"`` samples nodes
            with probability proportional to squared graph distance from the
            current sources (k-means++ style, reproducible via the global
            numpy seed), ``"farthest"`` picks the deterministic argmax.

    Attributes (after :meth:`fit`):
        cluster_labels: int32 (n,) cluster id per node, -1 = unreachable.
        dist: float64 (n,) graph distance to the node's own center node.
        prev: int32 (n,) shortest-path forest parent (-1 at center nodes).
        centers: float (K, dim) center positions — the continuous centroid
            for free clusters whose mean lies in free space, node
            coordinates otherwise (and always for fixed rows).
        center_node_indices: node index anchoring each center on the graph;
            fixed seeds first (label ``k`` == cluster of
            ``center_node_indices[k]`` — same contract as CTopPRM's
            ``seed_indices``).
        center_is_fixed: bool (K,) True for the fixed-seed rows.
        center_snapped: bool (K,) True where a free centroid fell back to a
            graph node (mean not in free space or empty-cluster re-seed);
            False for fixed rows.
        inertia_history: sum of squared member distances per iteration.
        n_iter_: Lloyd iterations run.
    """

    def __init__(
        self,
        graph_map,
        n_clusters: int,
        *,
        max_iters: int = 50,
        tol: float = 1e-6,
        init: str = "kpp",
    ) -> None:
        if init not in _INIT_MODES:
            raise ValueError(f"init must be one of {_INIT_MODES}, got {init!r}")
        if not getattr(graph_map, "nodes", None) or not getattr(graph_map, "road_map", None):
            raise ValueError(
                "graph_map has no roadmap; generate one first "
                "(e.g. generateRandomNodes + generate_roadmap)"
            )
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        self.map = graph_map
        self.n_clusters = int(n_clusters)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.init = init

        self._points = np.asarray([n.current for n in graph_map.nodes], dtype=float)
        self._adj = build_symmetric_adjacency(
            self._points,
            graph_map.road_map,
            getattr(graph_map, "road_map_edge_weights", None),
        )
        self._kd_tree = None

        n = len(self._points)
        self.cluster_labels = np.full(n, -1, dtype=np.int32)
        self.dist = np.full(n, np.inf, dtype=np.float64)
        self.prev = np.full(n, -1, dtype=np.int32)
        self.centers = np.empty((0, self._points.shape[1]))
        self.center_node_indices: List[int] = []
        self.center_is_fixed = np.empty(0, dtype=bool)
        self.center_snapped = np.empty(0, dtype=bool)
        self.inertia_history: List[float] = []
        self.n_iter_ = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, fixed_seeds: Sequence[Endpoint]) -> "GraphKMeans":
        """Cluster the roadmap nodes around the fixed seeds plus free centers.

        Args:
            fixed_seeds: Node indices or node-frame coordinates that become
                the first ``len(fixed_seeds)`` clusters' immovable centers.

        Returns:
            self, with the fitted attributes populated.
        """
        fixed = [self.resolve_endpoint(e) for e in fixed_seeds]
        if len(set(fixed)) != len(fixed):
            raise ValueError(f"duplicate fixed seeds after resolution: {fixed}")
        if self.n_clusters < len(fixed):
            raise ValueError(
                f"n_clusters={self.n_clusters} < {len(fixed)} fixed seeds"
            )

        sources = self._init_sources(fixed)
        k_total = len(sources)
        centers = self._points[sources].copy()
        is_fixed = np.arange(k_total) < len(fixed)
        snapped = np.zeros(k_total, dtype=bool)
        self.inertia_history = []

        labels = dist = prev = None
        for it in range(1, self.max_iters + 1):
            self.n_iter_ = it
            # E-step: graph-Voronoi assignment from the anchor nodes
            # (distance starts at 0 at each anchor — the centroid->anchor
            # offset is dropped to keep the dist==0-at-centers contract).
            labels, dist, prev = multi_source_dijkstra(self._adj, sources)
            finite = np.isfinite(dist)
            self.inertia_history.append(float(np.sum(dist[finite] ** 2)))

            new_sources, moved = self._m_step(
                sources, centers, snapped, labels, dist, len(fixed)
            )
            if new_sources == sources:
                break  # exact fixed point: assignment depends only on sources
            sources = new_sources
            if moved < self.tol:
                # Converged by movement with sources changed: one final
                # E-step keeps the stored state consistent with `sources`.
                labels, dist, prev = multi_source_dijkstra(self._adj, sources)
                break
        else:
            labels, dist, prev = multi_source_dijkstra(self._adj, sources)
            logger.warning(
                "GraphKMeans did not reach a fixed point in %d iterations",
                self.max_iters,
            )

        self.cluster_labels, self.dist, self.prev = labels, dist, prev
        self.centers = centers
        self.center_node_indices = list(sources)
        self.center_is_fixed = is_fixed
        self.center_snapped = snapped
        return self

    def resolve_endpoint(self, endpoint: Endpoint) -> int:
        """Resolve a node index or node-frame coordinate to a node index."""
        if isinstance(endpoint, (int, np.integer)):
            idx = int(endpoint)
            if not 0 <= idx < len(self._points):
                raise IndexError(f"endpoint node index {idx} out of range")
            return idx
        coord = tuple(endpoint)
        node = Node(coord, None, 0, 0)
        idx = self.map.node_index_dict.get(node)
        if idx is not None:
            return int(idx)
        if self._kd_tree is None:
            tree = getattr(self.map, "sample_kd_tree", None)
            if tree is None:
                from scipy.spatial import cKDTree

                tree = cKDTree(self._points)
            self._kd_tree = tree
        _, nn = self._kd_tree.query(np.asarray(coord, dtype=float))
        return int(nn)

    # ------------------------------------------------------------------
    # Lloyd steps
    # ------------------------------------------------------------------

    def _init_sources(self, fixed: List[int]) -> List[int]:
        """Fixed seeds topped up to ``n_clusters`` anchor nodes.

        Each extra source is drawn from the current graph-Voronoi residual:
        squared graph distance to the nearest existing source, sampled
        (``"kpp"``) or maximized (``"farthest"``). A drawn node has positive
        distance, so it can never duplicate an existing source.
        """
        sources = list(fixed)
        while len(sources) < self.n_clusters:
            _, dist, _ = multi_source_dijkstra(self._adj, sources)
            w = np.where(np.isfinite(dist), dist, 0.0) ** 2
            total = float(w.sum())
            if total <= 0.0:
                logger.warning(
                    "only %d distinct reachable clusters possible; "
                    "reducing n_clusters from %d",
                    len(sources), self.n_clusters,
                )
                self.n_clusters = len(sources)
                break
            if self.init == "kpp":
                nxt = int(np.random.choice(len(w), p=w / total))
            else:
                nxt = int(np.argmax(w))
            sources.append(nxt)
        return sources

    def _m_step(
        self,
        sources: List[int],
        centers: np.ndarray,
        snapped: np.ndarray,
        labels: np.ndarray,
        dist: np.ndarray,
        num_fixed: int,
    ) -> Tuple[List[int], float]:
        """Update free centers in place; return (new sources, max movement).

        Free anchors snap to the member node nearest the mean — members
        belong to exactly one cluster, so anchors can never collide across
        clusters (and never with a fixed seed, whose cluster they are not
        members of).
        """
        new_sources = list(sources)
        moved = 0.0
        taken = set(sources[:num_fixed])
        for k in range(num_fixed, len(sources)):
            members = np.flatnonzero(labels == k)
            if members.size == 0:
                cand = self._reseed_node(dist, taken)
                if cand is None:
                    logger.warning("cluster %d is empty and cannot be re-seeded", k)
                    taken.add(new_sources[k])
                    continue
                new_sources[k] = cand
                moved = max(moved, float(np.linalg.norm(self._points[cand] - centers[k])))
                centers[k] = self._points[cand]
                snapped[k] = True
                taken.add(cand)
                continue
            mean = self._points[members].mean(axis=0)
            nearest = int(
                members[np.argmin(np.linalg.norm(self._points[members] - mean, axis=1))]
            )
            new_sources[k] = nearest
            taken.add(nearest)
            if self.map.point_expandable(tuple(mean)):
                new_center = mean
                snapped[k] = False
            else:
                new_center = self._points[nearest]
                snapped[k] = True
            moved = max(moved, float(np.linalg.norm(new_center - centers[k])))
            centers[k] = new_center
        return new_sources, moved

    def _reseed_node(self, dist: np.ndarray, taken: set) -> Optional[int]:
        """Reachable node with the largest distance not already an anchor."""
        order = np.argsort(dist)[::-1]
        for idx in order:
            idx = int(idx)
            if np.isfinite(dist[idx]) and dist[idx] > 0.0 and idx not in taken:
                return idx
        return None
