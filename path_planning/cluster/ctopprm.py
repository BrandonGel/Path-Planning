"""CTopPRM: clustering topological roadmap planner over a GraphSampler.

Python reimplementation of CTopPRM (Novosad, Penicka, Vonasek, "CTopPRM:
Clustering Topological PRM for Planning Multiple Distinct Paths in 3D
Environments", RA-L 2023; C++ reference in ``path_planning/cluster/CTopPRM/``),
generalized to multiple starts/goals:

- ONE shared clustering: every requested start and goal seeds a cluster of a
  single graph-Voronoi partition of the shared roadmap, refined once by the
  centroid-addition loop.
- Distinct paths are then extracted per (start, goal) pair on demand from the
  shared cluster adjacency graph, so any-start-to-any-goal queries are cheap.

The roadmap is NOT built here — any roadmap a ``GraphSampler`` produced
(``prm``, ``halton``, ``grid``, custom, ...) is consumed via ``nodes`` /
``road_map`` / ``road_map_edge_weights``. Deliberate deviations from the C++
(documented at the methods): a clean multi-source Dijkstra replaces the
double-relaxation wavefront; cluster connections are rebuilt by one O(E)
sweep instead of in-loop border bookkeeping + repair; the cluster DFS uses
``continue`` instead of the C++'s enumeration-truncating ``return`` on an
over-budget neighbor, and per-path lengths are recomputed from geometry
instead of the C++'s stale ``path_lengths`` member.
"""

from __future__ import annotations

import heapq
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from path_planning.cluster.shortening import (
    GeometryConfig,
    get_esdf_gradients,
    greedy_shorten_path,
    is_deformable,
    path_length,
    remove_equivalent_paths,
    remove_too_long_paths,
    shorten_path,
)
from path_planning.common.environment.node import Node

logger = logging.getLogger(__name__)

Endpoint = Union[int, Sequence[float]]
PairKey = Tuple[int, int]
# One inter-cluster roadmap edge: (cost, u, v) with u on the lower-cluster side;
# cost = dist[u] + dist[v] + w(u, v), the seed-to-seed path length through it.
Connection = Tuple[float, int, int]

_SHORTENING_MODES = ("gradient", "greedy", "none")


class CTopPRM:
    """Multi-start/goal CTopPRM over an existing ``GraphSampler`` roadmap.

    Args:
        graph_map: ``GraphSampler`` with a generated roadmap (``nodes`` and
            ``road_map`` populated). Node coordinates are used as-is for all
            geometry, matching how the repo's MAPF solvers consume them.
        min_clusters: Force-grow clusters until strictly more than this many
            exist even if all class pairs are deformable. Default:
            ``num_seeds + 2`` (C++ default 4 for 2 seeds).
        max_clusters: Hard cap on cluster count. Default: ``num_seeds + 7``
            (C++ default 9 for 2 seeds), clamped to at least ``min_clusters``.
        max_path_length_ratio: Per-pair DFS length budget as a multiple of
            that pair's shortest roadmap path length (C++ 1.8).
        cutoff_distance_ratio_to_shortest: Post-filter: drop paths longer
            than this multiple of the pair's shortest returned path (C++ 1.5).
        collision_distance_check: Sampling step for all geometric checks;
            default ``0.25 * graph_map.resolution``.
        min_clearance: ESDF clearance for geometric checks; 0 relies on the
            grid's obstacle inflation (exact DDA checks) instead.
        shortening_mode: ``"gradient"`` (C++ ESDF push-out), ``"greedy"``
            (farthest-visible shortcutting, robust on coarse grids), or
            ``"none"``.
        max_paths_per_pair: Optional truncation of each pair's result list.
        max_sequences_per_pair: Safety cap on DFS-emitted cluster sequences.
    """

    def __init__(
        self,
        graph_map,
        *,
        min_clusters: Optional[int] = None,
        max_clusters: Optional[int] = None,
        max_path_length_ratio: float = 1.8,
        cutoff_distance_ratio_to_shortest: float = 1.5,
        collision_distance_check: Optional[float] = None,
        min_clearance: float = 0.0,
        shortening_mode: str = "gradient",
        max_paths_per_pair: Optional[int] = None,
        max_sequences_per_pair: int = 200,
    ) -> None:
        if shortening_mode not in _SHORTENING_MODES:
            raise ValueError(
                f"shortening_mode must be one of {_SHORTENING_MODES}, got {shortening_mode!r}"
            )
        if not getattr(graph_map, "nodes", None) or not getattr(graph_map, "road_map", None):
            raise ValueError(
                "graph_map has no roadmap; generate one first "
                "(e.g. generateRandomNodes + generate_roadmap)"
            )
        self.map = graph_map
        self._min_clusters_arg = min_clusters
        self._max_clusters_arg = max_clusters
        self.max_path_length_ratio = float(max_path_length_ratio)
        self.cutoff_distance_ratio_to_shortest = float(cutoff_distance_ratio_to_shortest)
        self.min_clearance = float(min_clearance)
        self.shortening_mode = shortening_mode
        self.max_paths_per_pair = max_paths_per_pair
        self.max_sequences_per_pair = int(max_sequences_per_pair)

        step = (
            float(collision_distance_check)
            if collision_distance_check is not None
            else 0.25 * float(graph_map.resolution)
        )
        self.geometry = GeometryConfig(
            collision_distance_check=step, min_clearance=self.min_clearance
        )

        self._points = np.asarray([n.current for n in graph_map.nodes], dtype=float)
        self._adj = self._build_symmetric_adjacency()
        self._grads = (
            get_esdf_gradients(graph_map) if shortening_mode == "gradient" else None
        )
        self._kd_tree = None

        n = len(self._points)
        # Wavefront state (graph-Voronoi partition + shortest-path forest).
        self.cluster_labels = np.full(n, -1, dtype=np.int32)
        self.dist = np.full(n, np.inf, dtype=np.float64)
        self.prev = np.full(n, -1, dtype=np.int32)
        self.seed_indices: List[int] = []
        self.min_clusters = 0
        self.max_clusters = 0
        self._connections: Dict[PairKey, List[Connection]] = {}
        self._pair_deform: Dict[PairKey, Tuple[tuple, bool]] = {}
        self._min_cluster_paths: Dict[PairKey, Tuple[np.ndarray, float]] = {}
        # Geometry-keyed memo for path-vs-path deformability verdicts; with
        # unshortened paths the same polylines recur across pairs and passes.
        self._deform_cache: Dict[tuple, bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_distinct_paths(
        self, pairs: List[Tuple[Endpoint, Endpoint]]
    ) -> Dict[PairKey, List[np.ndarray]]:
        """Plan topologically distinct paths for every requested pair.

        Args:
            pairs: (start, goal) pairs; each endpoint is a node index or a
                coordinate in the roadmap-node frame (resolved exactly via
                ``node_index_dict``, else snapped to the nearest node).

        Returns:
            ``{(start_idx, goal_idx): [path, ...]}`` with paths as
            ``np.ndarray (M, dim)`` waypoint polylines sorted by length
            (shortest first); an unreachable pair maps to ``[]``.
        """
        resolved = [
            (self.resolve_endpoint(s), self.resolve_endpoint(g)) for s, g in pairs
        ]
        for s, g in resolved:
            if s == g:
                raise ValueError(f"degenerate pair: start and goal are node {s}")

        # Seeds: dedup'd union of all endpoints, first-appearance order.
        seeds: List[int] = []
        for s, g in resolved:
            for idx in (s, g):
                if idx not in seeds:
                    seeds.append(idx)
        num_seeds = len(seeds)
        self.min_clusters = (
            self._min_clusters_arg
            if self._min_clusters_arg is not None
            else num_seeds + 2
        )
        self.max_clusters = (
            self._max_clusters_arg
            if self._max_clusters_arg is not None
            else num_seeds + 7
        )
        self.max_clusters = max(self.max_clusters, self.min_clusters, num_seeds)

        # Per-pair shortest-path budgets (one Dijkstra per unique start).
        start_dists: Dict[int, np.ndarray] = {}
        for s, _ in resolved:
            if s not in start_dists:
                start_dists[s] = self._single_source_dist(s)
        budgets: Dict[PairKey, float] = {}
        for s, g in resolved:
            shortest = float(start_dists[s][g])
            budgets[(s, g)] = shortest * self.max_path_length_ratio

        self._wavefront_fill(seeds)
        self._collect_cluster_connections()
        self._refine_clusters()
        self._find_min_cluster_tours()

        results: Dict[PairKey, List[np.ndarray]] = {}
        for s, g in resolved:
            key = (s, g)
            if key in results:
                continue
            # The problem is symmetric (undirected roadmap, symmetric budgets
            # and cluster tours), so a pair whose reverse was already planned
            # is answered by reversing those paths instead of recomputing.
            reverse = results.get((g, s))
            if reverse is not None:
                results[key] = [np.ascontiguousarray(p[::-1]) for p in reverse]
                continue
            if not math.isfinite(budgets[key]):
                logger.warning("pair %s: goal unreachable on the roadmap", key)
                results[key] = []
                continue
            results[key] = self._extract_pair_paths(s, g, budgets[key])
        return results

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

    def used_node_indices(
        self, results: Dict[PairKey, List[np.ndarray]]
    ) -> np.ndarray:
        """Roadmap node indices appearing as waypoints in planned paths.

        Waypoints are matched to nodes by exact coordinates. With
        ``shortening_mode="none"`` every waypoint is a roadmap node, so the
        match is complete; shortened paths contain free continuous points
        that belong to no node — those are skipped with a warning.

        Args:
            results: Output of :meth:`find_distinct_paths` (or any subset /
                filtered version of it).

        Returns:
            Sorted unique node indices (np.ndarray of int).
        """
        coord_to_idx = {pt.tobytes(): i for i, pt in enumerate(self._points)}
        used: set = set()
        missed = 0
        for paths in results.values():
            for path in paths:
                for pt in np.ascontiguousarray(np.asarray(path, dtype=float)):
                    idx = coord_to_idx.get(pt.tobytes())
                    if idx is None:
                        missed += 1
                    else:
                        used.add(idx)
        if missed:
            logger.warning(
                "%d waypoints are not roadmap nodes and were skipped "
                "(shortened paths contain free continuous points; use "
                "shortening_mode='none' for node-only paths)", missed
            )
        return np.array(sorted(used), dtype=int)

    def create_pruned_graph(self, results: Dict[PairKey, List[np.ndarray]]):
        """GraphSampler copy keeping only the nodes used by these paths.

        Wraps ``GraphSampler.create_pruned_copy`` with the node indices from
        :meth:`used_node_indices`. With ``shortening_mode="none"`` every
        consecutive waypoint pair is a roadmap edge, so all planned paths
        stay traversable in the pruned copy.
        """
        return self.map.create_pruned_copy(self.used_node_indices(results))

    # ------------------------------------------------------------------
    # Roadmap plumbing
    # ------------------------------------------------------------------

    def _build_symmetric_adjacency(self) -> List[List[Tuple[int, float]]]:
        """Undirected adjacency (like the C++ roadmap) from ``road_map``.

        k-NN roadmaps can be asymmetric (j in adj[i] without i in adj[j]);
        every edge is inserted in both directions here. Weights come from
        ``road_map_edge_weights`` when aligned, else Euclidean distance.
        """
        n = len(self._points)
        road_map = self.map.road_map
        weights = getattr(self.map, "road_map_edge_weights", None)
        aligned = (
            weights is not None
            and len(weights) == len(road_map)
            and all(len(weights[i]) == len(road_map[i]) for i in range(len(road_map)))
        )
        adj: List[Dict[int, float]] = [dict() for _ in range(n)]
        for u in range(min(len(road_map), n)):
            for k, v in enumerate(road_map[u]):
                if not 0 <= v < n or v == u:
                    continue
                w = (
                    float(weights[u][k])
                    if aligned
                    else float(np.linalg.norm(self._points[u] - self._points[v]))
                )
                if w < adj[u].get(v, np.inf):
                    adj[u][v] = w
                    adj[v][u] = w
        return [sorted(d.items()) for d in adj]

    def _single_source_dist(self, source: int) -> np.ndarray:
        """Dijkstra distances from ``source`` over the symmetrized roadmap."""
        n = len(self._points)
        dist = np.full(n, np.inf)
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in self._adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    # ------------------------------------------------------------------
    # Stage 1: wavefront clustering (graph-Voronoi partition)
    # ------------------------------------------------------------------

    def _wavefront_fill(self, seed_indices: List[int]) -> None:
        """Multi-source Dijkstra labelling every node with its nearest seed.

        Clean replacement for the C++ ``wavefrontFill`` (:687), whose
        duplicated relaxation with stale heap positions makes the partition
        only approximately Voronoi; here labels/dist/prev are exact.
        """
        self.seed_indices = list(seed_indices)
        self.cluster_labels.fill(-1)
        self.dist.fill(np.inf)
        self.prev.fill(-1)
        heap: List[Tuple[float, int]] = []
        for k, s in enumerate(self.seed_indices):
            self.dist[s] = 0.0
            self.cluster_labels[s] = k
            heapq.heappush(heap, (0.0, s))
        while heap:
            d, u = heapq.heappop(heap)
            if d > self.dist[u]:
                continue
            for v, w in self._adj[u]:
                nd = d + w
                if nd < self.dist[v]:
                    self.dist[v] = nd
                    self.cluster_labels[v] = self.cluster_labels[u]
                    self.prev[v] = u
                    heapq.heappush(heap, (nd, v))

    def _collect_cluster_connections(self) -> None:
        """Record every roadmap edge whose endpoints lie in different clusters.

        One O(E) sweep after (re-)labelling, replacing the C++ in-expansion
        border bookkeeping and the entire stale-entry repair pass
        (:1220-1263): connection cost ``dist[u] + dist[v] + w`` is always
        consistent with the current forest.
        """
        conns: Dict[PairKey, List[Connection]] = {}
        labels = self.cluster_labels
        for u in range(len(self._adj)):
            for v, w in self._adj[u]:
                if v <= u:
                    continue
                cu, cv = int(labels[u]), int(labels[v])
                if cu == cv or cu < 0 or cv < 0:
                    continue
                cost = float(self.dist[u] + self.dist[v] + w)
                if cu < cv:
                    conns.setdefault((cu, cv), []).append((cost, u, v))
                else:
                    conns.setdefault((cv, cu), []).append((cost, v, u))
        self._connections = conns

    # ------------------------------------------------------------------
    # Stage 2: adding new centroids
    # ------------------------------------------------------------------

    def _refine_clusters(self) -> None:
        """Centroid-addition loop, mirror of the C++ (:869-998).

        While below ``max_clusters``: scan cluster pairs by descending
        max/min connection-cost ratio; split the first pair whose min- and
        max-connection paths are NOT deformable into each other (an obstacle
        separates them). If every pair is deformable, stop — unless still at
        or below ``min_clusters``, then force-split the pair with the
        largest max-connection cost. The new centroid is the max-connection
        endpoint with the smaller distance-from-seed (C++ :988).
        """
        while len(self.seed_indices) < self.max_clusters:
            pair_stats = []
            for pair, conns in self._connections.items():
                cmin = min(conns)
                cmax = max(conns)
                ratio = cmax[0] / max(cmin[0], 1e-12)
                pair_stats.append((-ratio, pair, cmin, cmax))
            pair_stats.sort()

            split = None
            forced_best = None  # (max_cost, pair, cmin, cmax)
            for _, pair, cmin, cmax in pair_stats:
                if self._pair_is_deformable(pair, cmin, cmax):
                    if forced_best is None or cmax[0] > forced_best[0]:
                        forced_best = (cmax[0], pair, cmin, cmax)
                else:
                    split = (pair, cmin, cmax)
                    break

            if split is None:
                if len(self.seed_indices) > self.min_clusters or forced_best is None:
                    break  # all classes deformable (or nothing to split)
                _, pair, cmin, cmax = forced_best
                split = (pair, cmin, cmax)

            _, _, (_, u, v) = split
            new_seed = v if self.dist[u] > self.dist[v] else u
            self.seed_indices.append(int(new_seed))
            self._add_centroid(int(new_seed))
            self._collect_cluster_connections()

    def _pair_is_deformable(
        self, pair: PairKey, cmin: Connection, cmax: Connection
    ) -> bool:
        """Cached wrapper of the deformability test for one cluster pair.

        The cache key includes the min/max connection endpoints and costs;
        a changed signature (some centroid re-labelled the region) forces a
        re-check, the clean equivalent of the C++ ``deformation_checked``
        reset in its repair pass.
        """
        sig = (cmin[1], cmin[2], cmax[1], cmax[2], round(cmin[0], 9), round(cmax[0], 9))
        cached = self._pair_deform.get(pair)
        if cached is not None and cached[0] == sig:
            return cached[1]
        if (cmin[1], cmin[2]) == (cmax[1], cmax[2]):
            deformable = True  # single connection: min and max coincide
        else:
            deformable = self._is_new_homotopy_class(cmin, cmax)
        self._pair_deform[pair] = (sig, deformable)
        return deformable

    def _is_new_homotopy_class(self, cmin: Connection, cmax: Connection) -> bool:
        """Whether the pair's min- and max-connection paths are deformable.

        Mirrors C++ ``isNewHomotopyClass`` (:1471): reconstruct both
        seed-to-seed paths through the connections, shorten (min path
        backward-then-forward, max path forward-then-backward, :1504-1509),
        then run the straight-line homotopy test. Returns True = deformable
        (the C++ return value, despite its name).
        """
        min_path = self._connection_path(cmin)
        max_path = self._connection_path(cmax)
        min_path = self._shorten(self._shorten(min_path, forward=False), forward=True)
        max_path = self._shorten(self._shorten(max_path, forward=True), forward=False)
        return is_deformable(self.map, min_path, max_path, self.geometry)

    def _add_centroid(self, node_idx: int) -> None:
        """Grow-only Dijkstra from a new seed, stealing strictly-closer nodes.

        Port of C++ ``addCentroid`` (:1096) without its border bookkeeping
        (rebuilt wholesale by ``_collect_cluster_connections``). Existing
        seeds have ``dist == 0`` and can never be stolen.
        """
        new_label = len(self.seed_indices) - 1
        self.dist[node_idx] = 0.0
        self.prev[node_idx] = -1
        self.cluster_labels[node_idx] = new_label
        heap = [(0.0, node_idx)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > self.dist[u]:
                continue
            for v, w in self._adj[u]:
                nd = d + w
                if nd < self.dist[v]:
                    self.dist[v] = nd
                    self.cluster_labels[v] = self.cluster_labels[u]
                    self.prev[v] = u
                    heapq.heappush(heap, (nd, v))

    # ------------------------------------------------------------------
    # Stage 3: cluster graph and per-pair path extraction
    # ------------------------------------------------------------------

    def _backtrack_to_seed(self, node_idx: int) -> List[int]:
        """Node indices from ``node_idx`` up to its cluster seed (inclusive)."""
        out = [node_idx]
        while self.prev[out[-1]] >= 0:
            out.append(int(self.prev[out[-1]]))
        return out

    def _connection_path(self, conn: Connection) -> np.ndarray:
        """Seed-to-seed polyline through one inter-cluster connection."""
        _, u, v = conn
        idxs = self._backtrack_to_seed(u)[::-1] + self._backtrack_to_seed(v)
        return self._points[idxs]

    def _find_min_cluster_tours(self) -> None:
        """Cache the shortened min inter-cluster path per connected pair.

        Port of C++ ``findMinClustertours`` (:1533); the keys double as the
        cluster adjacency used by the DFS. Paths are stored in
        low-cluster -> high-cluster direction.
        """
        self._min_cluster_paths = {}
        for pair, conns in self._connections.items():
            path = self._connection_path(min(conns))
            path = self._shorten(self._shorten(path, forward=True), forward=False)
            length = path_length(path)
            if length < 1e-4:  # degenerate (C++ invalidates these too)
                continue
            self._min_cluster_paths[pair] = (path, length)

    def _cluster_adjacency(self) -> Dict[int, List[Tuple[int, float]]]:
        """Cluster-graph adjacency weighted by min-tour lengths."""
        adjacency: Dict[int, List[Tuple[int, float]]] = {}
        for (a, b), (_, length) in self._min_cluster_paths.items():
            adjacency.setdefault(a, []).append((b, length))
            adjacency.setdefault(b, []).append((a, length))
        for nbrs in adjacency.values():
            nbrs.sort()
        return adjacency

    def _cluster_distances(self, source_cl: int) -> Dict[int, float]:
        """Dijkstra distances from one cluster over the cluster graph."""
        adjacency = self._cluster_adjacency()
        dist: Dict[int, float] = {source_cl: 0.0}
        heap = [(0.0, source_cl)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for v, w in adjacency.get(u, []):
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def _shortest_cluster_tour(self, start_cl: int, goal_cl: int) -> float:
        """Shortest cluster-tour length between two clusters."""
        return self._cluster_distances(start_cl).get(goal_cl, float("inf"))

    def _cluster_dfs(
        self, start_cl: int, goal_cl: int, budget: float
    ) -> List[List[int]]:
        """Enumerate cluster sequences start->goal within the length budget.

        Port of C++ ``findPathsRecurseMaxLength`` (:600-650) with two
        deliberate corrections: an over-budget neighbor is skipped
        (``continue``) instead of aborting the whole neighbor loop (the C++
        ``return`` at :638 truncates enumeration nondeterministically), and
        no result-length state is shared across calls.

        The enumeration itself is best-first (A* over the cluster graph,
        heuristic = exact shortest cluster-tour distance to the goal), so
        completed sequences pop in non-decreasing length order and the search
        stops after the ``max_sequences_per_pair`` shortest — the same set
        the exhaustive DFS + sort returned, without enumerating everything.
        As in the DFS, the interior of a sequence must stay within budget
        while the final hop into the goal cluster is unchecked (over-budget
        completions are filtered later by ``remove_too_long_paths``), and
        ties break lexicographically on the cluster sequence. A hard cap on
        expansions remains as a safety valve; because expansion is best-first,
        even a capped run returns the shortest sequences found so far.
        """
        adjacency = self._cluster_adjacency()
        # Admissible heuristic: true shortest tour distance to the goal.
        h = self._cluster_distances(goal_cl)
        if h.get(start_cl, np.inf) == np.inf:
            return []

        max_expansions = 50 * self.max_sequences_per_pair
        sequences: List[List[int]] = []
        expansions = 0
        # Heap entries: (f, sequence, g). Complete sequences end at goal_cl
        # and carry f == g == true total length; interior ones f = g + h.
        heap: List[Tuple[float, List[int], float]] = [
            (h[start_cl], [start_cl], 0.0)
        ]
        while heap and len(sequences) < self.max_sequences_per_pair:
            _, seq, g = heapq.heappop(heap)
            last = seq[-1]
            if last == goal_cl:
                sequences.append(seq)
                continue
            expansions += 1
            if expansions > max_expansions:
                logger.warning(
                    "cluster search %d->%d hit the expansion cap (%d); "
                    "returning the %d shortest sequences found",
                    start_cl, goal_cl, max_expansions, len(sequences),
                )
                break
            for nb, w in adjacency.get(last, []):
                if nb == goal_cl:
                    total = g + w
                    heapq.heappush(heap, (total, seq + [nb], total))
                elif nb not in seq:
                    g2 = g + w
                    hn = h.get(nb, np.inf)
                    if g2 > budget or hn == np.inf:
                        continue
                    heapq.heappush(heap, (g2 + hn, seq + [nb], g2))
        return sequences

    def _assemble_path(self, cluster_seq: List[int]) -> np.ndarray:
        """Concatenate cached min inter-cluster paths along a cluster sequence."""
        parts: List[np.ndarray] = []
        for a, b in zip(cluster_seq[:-1], cluster_seq[1:]):
            seg = (
                self._min_cluster_paths[(a, b)][0]
                if a < b
                else self._min_cluster_paths[(b, a)][0][::-1]
            )
            if parts:
                parts[-1] = parts[-1][:-1]  # drop duplicated junction waypoint
            parts.append(seg)
        return np.concatenate(parts, axis=0)

    def _extract_pair_paths(
        self, start_idx: int, goal_idx: int, budget: float
    ) -> List[np.ndarray]:
        """DFS + assembly + post-processing for one (start, goal) pair.

        Mirrors the C++ pipeline tail (:2264-2303): shorten both directions,
        drop too-long paths, dedup by deformability, shorten and dedup again,
        sort by length. Other agents' seed clusters are ordinary clusters
        here — passing through them is allowed.
        """
        start_cl = int(self.cluster_labels[start_idx])
        goal_cl = int(self.cluster_labels[goal_idx])
        # DFS accumulates min-cluster-tour lengths, which zigzag seed-to-seed
        # and overshoot the geometric path length — increasingly so with many
        # agents' seeds (the C++ only ever has one pair's clusters). Floor the
        # budget at ratio x the shortest TOUR so the best route is never
        # pruned by that overshoot; the geometric cutoff filter below still
        # bounds the returned paths.
        shortest_tour = self._shortest_cluster_tour(start_cl, goal_cl)
        if math.isfinite(shortest_tour):
            budget = max(budget, shortest_tour * self.max_path_length_ratio)
        sequences = self._cluster_dfs(start_cl, goal_cl, budget)

        goal_pt = self._points[goal_idx]
        start_pt = self._points[start_idx]
        paths: List[np.ndarray] = []
        for seq in sequences:
            path = self._assemble_path(seq)
            # With the corrected DFS every sequence reaches the goal cluster's
            # seed, which is the goal node itself.
            if not np.allclose(path[-1], goal_pt, atol=1e-6):
                logger.warning("assembled path does not end at goal; dropped")
                continue
            paths.append(path)
        if not paths:
            return []

        paths = [
            self._shorten(self._shorten(p, forward=False), forward=True) for p in paths
        ]
        endpoints_dist = float(np.linalg.norm(goal_pt - start_pt))
        paths = remove_too_long_paths(
            paths, endpoints_dist, self.cutoff_distance_ratio_to_shortest
        )
        paths.sort(key=path_length)
        paths = remove_equivalent_paths(
            self.map, paths, self.geometry, cache=self._deform_cache
        )
        if self.shortening_mode != "none":
            paths = [
                self._shorten(self._shorten(p, forward=False), forward=True)
                for p in paths
            ]
        # The second dedup pass is NOT redundant even with unchanged paths:
        # a pass compares against the original slot path, not the shorter
        # representative that replaces it, so it can prune further. The
        # cache makes its repeated comparisons free.
        paths.sort(key=path_length)
        paths = remove_equivalent_paths(
            self.map, paths, self.geometry, cache=self._deform_cache
        )
        paths.sort(key=path_length)
        if self.max_paths_per_pair is not None:
            paths = paths[: self.max_paths_per_pair]
        return paths

    # ------------------------------------------------------------------
    # Shortening dispatch
    # ------------------------------------------------------------------

    def _shorten(self, path: np.ndarray, forward: bool = True) -> np.ndarray:
        if self.shortening_mode == "none" or len(path) < 3:
            return path
        if self.shortening_mode == "greedy":
            # Direction-independent; applying it twice is a cheap no-op-ish
            # second pass, kept so call sites stay mode-agnostic.
            return greedy_shorten_path(self.map, path, self.geometry)
        return shorten_path(self.map, path, self.geometry, forward=forward, grads=self._grads)
