"""Graph-distance EM clustering over a ``GraphSampler`` roadmap.

Soft (mixture-model) counterpart of ``GraphKMeans``: each component k is a
full-covariance Gaussian over the graph-warped displacement — anchor node
``a_k``, weight ``pi_k`` and covariance ``Sigma_k``. A node's Euclidean
displacement direction from the anchor is rescaled to GRAPH-distance
length, ``v_nk = d_graph(a_k, n) * u_nk`` with
``u_nk = (x_n - x_a_k) / |x_n - x_a_k|``, and the component likelihood is
``pi_k * N(v_nk; 0, Sigma_k)`` — the metric is still the roadmap
distance, so responsibilities never leak through walls the way a
Euclidean GMM would, while the covariance is free to stretch along
corridors instead of being constrained isotropic. Two kinds of
components:

- FIXED components: caller-supplied seed points (e.g. agent starts/goals)
  whose anchors never move — every fixed seed keeps its own component, like
  CTopPRM's endpoint seeds. Their pi/sigma are still learned.
- FREE components: additional clusters whose center tracks the
  responsibility-weighted mean of the node coordinates. The continuous
  centroid is kept when it lies in free space (boundary-inclusive
  ``GraphSampler.point_expandable``); otherwise the center falls back to
  the member node nearest the mean, so a center never sits inside an
  obstacle.

EM has no repulsion between components — co-located duplicates are a
stationary point of the likelihood — so free anchors additionally keep a
minimum Euclidean spacing of ``max(dup_radius, median free-component
sigma)``, recomputed every M-step: components packed closer than their
own typical width overlap almost entirely and are duplicates whatever
the map resolution. A free anchor that would violate the spacing is
re-seeded at the farthest spacing-respecting node — the same rescue
empty components get.

Because Dijkstra needs graph sources, every component is anchored to a
roadmap node (``center_node_indices``); the continuous ``centers`` are
reporting/consumer-facing positions. After :meth:`fit`, one final
multi-source Dijkstra from the anchors fills ``cluster_labels`` / ``dist``
/ ``prev`` — exactly the wavefront state CTopPRM's ``_wavefront_fill``
produces for ``center_node_indices`` — so the result can seed the CTopPRM
pipeline directly (``CTopPRM(..., clustering="em")``); the soft
assignments stay available as ``responsibilities``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from path_planning.cluster.kmeans.graph_kmeans import (
    build_symmetric_adjacency,
    multi_source_dijkstra,
)
from path_planning.common.environment.node import Node

logger = logging.getLogger(__name__)

Endpoint = Union[int, Sequence[float]]

_INIT_MODES = ("kpp", "farthest")

_LOG_2PI = float(np.log(2.0 * np.pi))


class GraphEM:
    """EM mixture over graph-warped displacements with fixed anchor
    components.

    Args:
        graph_map: ``GraphSampler`` with a generated roadmap (``nodes`` and
            ``road_map`` populated).
        n_clusters: Total component count INCLUDING the fixed seeds passed
            to :meth:`fit` (must be >= their number). May be reduced when
            the roadmap has fewer reachable nodes than requested clusters.
        max_iters: Cap on EM iterations.
        tol: Convergence threshold on the per-node log-likelihood change;
            convergence additionally requires the anchor set to reach a
            fixed point (anchor jumps are discrete, so the log-likelihood
            is not strictly monotone across them).
        min_sigma: Floor on the std deviation along every covariance
            eigendirection (eigenvalues are clamped at ``min_sigma**2``),
            preventing variance collapse onto a single node (default: the
            map resolution).
        dup_radius: Floor on the Euclidean spacing between anchors. The
            effective spacing each M-step is ``max(dup_radius, median
            free-component sigma)`` — components must stay about one
            component-width apart — and a free anchor that would land
            inside it is re-seeded at the farthest spacing-respecting
            node instead of duplicating an existing component (default
            floor: the map resolution; 0 disables the guard entirely).
        init: Top-up seeding for the free components — ``"kpp"`` samples
            nodes with probability proportional to squared graph distance
            from the current anchors (k-means++ style, reproducible via the
            global numpy seed), ``"farthest"`` picks the deterministic
            argmax.

    Attributes (after :meth:`fit`):
        cluster_labels: int32 (n,) hard cluster id per node from the final
            wavefront fill, -1 = unreachable.
        dist: float64 (n,) graph distance to the node's own anchor node.
        prev: int32 (n,) shortest-path forest parent (-1 at anchor nodes).
        centers: float (K, dim) center positions — the responsibility-
            weighted centroid for free components whose mean lies in free
            space, node coordinates otherwise (and always for fixed rows).
        center_node_indices: node index anchoring each component on the
            graph; fixed seeds first (label ``k`` == cluster of
            ``center_node_indices[k]`` — same contract as CTopPRM's
            ``seed_indices``).
        center_is_fixed: bool (K,) True for the fixed-seed rows.
        center_snapped: bool (K,) True where a free centroid fell back to a
            graph node (mean not in free space or empty-component re-seed);
            False for fixed rows.
        responsibilities: float (n, K) soft assignments; rows sum to 1 on
            reachable nodes and are all-zero on unreachable ones.
        weights_: float (K,) mixture weights pi_k (sum to 1).
        covariances_: float (K, dim, dim) per-component covariances over
            the graph-warped displacements (symmetric, eigenvalues >=
            ``min_sigma**2``).
        sigmas_: float (K,) effective isotropic widths
            ``sqrt(trace(Sigma_k) / dim)`` (each >= ``min_sigma``); used
            for the anchor spacing guard and reporting.
        log_likelihood_history: total log-likelihood per EM iteration.
        n_iter_: EM iterations run.
    """

    def __init__(
        self,
        graph_map,
        n_clusters: int,
        *,
        max_iters: int = 50,
        tol: float = 1e-6,
        min_sigma: Optional[float] = None,
        dup_radius: Optional[float] = None,
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
        self.min_sigma = (
            float(min_sigma)
            if min_sigma is not None
            else float(getattr(graph_map, "resolution", 1.0))
        )
        if self.min_sigma <= 0.0:
            raise ValueError(f"min_sigma must be > 0, got {self.min_sigma}")
        self.dup_radius = (
            float(dup_radius)
            if dup_radius is not None
            else float(getattr(graph_map, "resolution", 1.0))
        )
        if self.dup_radius < 0.0:
            raise ValueError(f"dup_radius must be >= 0, got {self.dup_radius}")
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
        self.responsibilities = np.zeros((n, 0))
        self.weights_ = np.empty(0)
        self.covariances_ = np.empty((0, self._points.shape[1], self._points.shape[1]))
        self.sigmas_ = np.empty(0)
        self.log_likelihood_history: List[float] = []
        self.n_iter_ = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, fixed_seeds: Sequence[Endpoint]) -> "GraphEM":
        """Fit the mixture around the fixed seeds plus free components.

        Args:
            fixed_seeds: Node indices or node-frame coordinates that become
                the first ``len(fixed_seeds)`` components' immovable
                anchors.

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
        weights, covs = self._init_params(sources)
        self.log_likelihood_history = []

        # Discrete anchor moves can enter a limit cycle (two anchor sets
        # alternating forever), so alongside the fixed-point test we detect
        # revisited anchor sets and always report the best-likelihood state
        # visited (each snapshot is self-consistent: resp/ll computed from
        # exactly these sources/params).
        best = None
        seen = set()
        prev_ll = -np.inf
        for it in range(1, self.max_iters + 1):
            self.n_iter_ = it
            D = self._distance_matrix(sources)
            U = self._directions(sources)
            resp, reachable, ll = self._e_step(D, U, weights, covs)
            self.log_likelihood_history.append(ll)
            if best is None or ll > best[0]:
                best = (ll, list(sources), weights.copy(), covs.copy(),
                        centers.copy(), snapped.copy(), resp)
            if tuple(sources) in seen:
                logger.debug(
                    "GraphEM anchor limit cycle after %d iterations", it
                )
                break
            seen.add(tuple(sources))

            weights, covs, new_sources = self._m_step(
                sources, D, U, resp, reachable, centers, snapped, len(fixed)
            )
            n_reach = max(int(reachable.sum()), 1)
            if new_sources == sources and abs(ll - prev_ll) <= self.tol * n_reach:
                break
            prev_ll = ll
            sources = new_sources
        else:
            logger.warning(
                "GraphEM did not converge in %d iterations", self.max_iters
            )
        _, sources, weights, covs, centers, snapped, resp = best

        # Hard state: wavefront fill from the final anchors (the exact
        # invariants CTopPRM expects: dist==0 at anchors, positional labels).
        labels, dist, prev = multi_source_dijkstra(self._adj, sources)
        self.cluster_labels, self.dist, self.prev = labels, dist, prev
        self.centers = centers
        self.center_node_indices = list(sources)
        self.center_is_fixed = is_fixed
        self.center_snapped = snapped
        self.responsibilities = resp.T.copy()
        self.weights_ = weights
        self.covariances_ = covs
        self.sigmas_ = self._effective_sigmas(covs)
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
    # Initialization
    # ------------------------------------------------------------------

    def _init_sources(self, fixed: List[int]) -> List[int]:
        """Fixed seeds topped up to ``n_clusters`` anchor nodes.

        Each extra anchor is drawn from the current graph-Voronoi residual:
        squared graph distance to the nearest existing anchor, sampled
        (``"kpp"``) or maximized (``"farthest"``). A drawn node has positive
        distance, so it can never duplicate an existing anchor; nodes
        within ``dup_radius`` of one are also excluded while any other
        candidate remains.
        """
        sources = list(fixed)
        min_euclid = np.full(len(self._points), np.inf)
        for s in sources:
            min_euclid = np.minimum(
                min_euclid, np.linalg.norm(self._points - self._points[s], axis=1)
            )
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
            spaced = np.where(min_euclid >= self.dup_radius, w, 0.0)
            spaced_total = float(spaced.sum())
            if spaced_total > 0.0:
                w, total = spaced, spaced_total
            if self.init == "kpp":
                nxt = int(np.random.choice(len(w), p=w / total))
            else:
                nxt = int(np.argmax(w))
            sources.append(nxt)
            min_euclid = np.minimum(
                min_euclid, np.linalg.norm(self._points - self._points[nxt], axis=1)
            )
        return sources

    def _init_params(self, sources: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform weights; covariances from the initial wavefront partition.

        Sigma_k starts isotropic at ``rms_k**2 * I`` where ``rms_k`` is the
        RMS member distance of component k's hard cluster (global RMS for
        empty clusters), floored at ``min_sigma``; anisotropy is learned by
        the M-step.
        """
        k_total = len(sources)
        dim = self._points.shape[1]
        weights = np.full(k_total, 1.0 / k_total)
        labels, dist, _ = multi_source_dijkstra(self._adj, sources)
        finite = np.isfinite(dist)
        global_rms = (
            float(np.sqrt(np.mean(dist[finite] ** 2))) if finite.any() else 0.0
        )
        covs = np.empty((k_total, dim, dim))
        for k in range(k_total):
            members = finite & (labels == k)
            rms = (
                float(np.sqrt(np.mean(dist[members] ** 2)))
                if members.any()
                else global_rms
            )
            covs[k] = max(rms, self.min_sigma) ** 2 * np.eye(dim)
        return weights, covs

    # ------------------------------------------------------------------
    # EM steps
    # ------------------------------------------------------------------

    def _distance_matrix(self, sources: List[int]) -> np.ndarray:
        """(K, n) graph distances: one single-source Dijkstra per anchor."""
        D = np.empty((len(sources), len(self._points)))
        for k, s in enumerate(sources):
            _, d, _ = multi_source_dijkstra(self._adj, [s])
            D[k] = d
        return D

    def _e_step(
        self,
        D: np.ndarray,
        U: np.ndarray,
        weights: np.ndarray,
        covs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Responsibilities (K, n), reachable mask (n,), total log-likelihood.

        Component k's log density of node n is the multivariate normal of
        the graph-warped displacement ``v = D_kn * u_nk``:
        ``-0.5 * (log|2 pi Sigma_k| + D_kn**2 * u^T Sigma_k^{-1} u)``.
        Log-domain with the log-sum-exp trick; a node unreachable from every
        component gets an all-zero column and is excluded from the
        likelihood.
        """
        dim = self._points.shape[1]
        logp = np.empty_like(D)
        for k in range(len(covs)):
            prec = np.linalg.inv(covs[k])
            _, logdet = np.linalg.slogdet(covs[k])
            quad = np.einsum("nd,de,ne->n", U[k], prec, U[k])
            # inf distance -> -inf log density (inf * 0 at a zero direction
            # would be NaN, so mask explicitly).
            with np.errstate(invalid="ignore"):
                z2 = np.where(np.isfinite(D[k]), np.square(D[k]) * quad, np.inf)
            logp[k] = np.log(weights[k]) - 0.5 * (dim * _LOG_2PI + logdet + z2)
        m = np.max(logp, axis=0)
        reachable = np.isfinite(m)
        resp = np.zeros_like(logp)
        lse = m[reachable] + np.log(
            np.sum(np.exp(logp[:, reachable] - m[reachable]), axis=0)
        )
        resp[:, reachable] = np.exp(logp[:, reachable] - lse)
        ll = float(np.sum(lse))
        return resp, reachable, ll

    def _m_step(
        self,
        sources: List[int],
        D: np.ndarray,
        U: np.ndarray,
        resp: np.ndarray,
        reachable: np.ndarray,
        centers: np.ndarray,
        snapped: np.ndarray,
        num_fixed: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Update pi/Sigma for all components and anchors/centers for free
        ones (``centers``/``snapped`` in place); return (weights,
        covariances, new sources).

        Free anchors snap to the hard-member (argmax-responsibility) node
        nearest the weighted mean — hard members belong to exactly one
        component, so anchors can never collide across components. An
        anchor that would still land within the spacing radius
        ``max(dup_radius, median free sigma)`` of an already-placed one
        is a duplicate (EM never separates components that overlap by
        more than their width) and is re-seeded like an empty component;
        re-seeds honor the same spacing while a candidate exists.
        """
        n_reach = max(int(reachable.sum()), 1)
        dim = self._points.shape[1]
        r = resp[:, reachable]
        mass = r.sum(axis=1)  # (K,) column-stochastic resp -> sums to n_reach
        safe_mass = np.maximum(mass, np.finfo(float).tiny)
        weights = np.maximum(mass / n_reach, 1e-12)
        # Covariance = responsibility-weighted scatter of the graph-warped
        # displacements v = D * u; resp is 0 wherever D is inf, so mask the
        # inf before multiplying.
        d_fin = np.where(np.isfinite(D[:, reachable]), D[:, reachable], 0.0)
        covs = np.empty((len(sources), dim, dim))
        for k in range(len(sources)):
            V = d_fin[k, :, None] * U[k, reachable]
            covs[k] = self._floor_cov((r[k][:, None] * V).T @ V / safe_mass[k])

        hard = np.full(len(reachable), -1, dtype=np.int64)
        hard[reachable] = np.argmax(resp[:, reachable], axis=0)
        new_sources = list(sources)
        taken = set(sources[:num_fixed])
        placed = [self._points[s] for s in sources[:num_fixed]]
        min_dist = np.min(D, axis=0)
        global_rms = float(np.sqrt(np.mean(np.square(min_dist[reachable]))))
        # Adaptive spacing: about one typical component width, floored at
        # dup_radius (0 keeps the guard disabled).
        spacing = 0.0
        if self.dup_radius > 0.0:
            free_widths = self._effective_sigmas(covs[num_fixed:])
            spacing = max(
                self.dup_radius,
                float(np.median(free_widths)) if free_widths.size else 0.0,
            )

        def reseed(k: int, reason: str) -> None:
            cand = self._reseed_node(min_dist, taken, placed, spacing)
            if cand is None:
                logger.warning(
                    "component %d is %s and cannot be re-seeded", k, reason
                )
                taken.add(new_sources[k])
                placed.append(self._points[new_sources[k]])
                return
            new_sources[k] = cand
            centers[k] = self._points[cand]
            snapped[k] = True
            taken.add(cand)
            placed.append(self._points[cand])
            # Give the re-seeded component a fresh chance in the next
            # E-step instead of the starved parameters it converged to.
            weights[k] = 1.0 / len(sources)
            covs[k] = max(global_rms, self.min_sigma) ** 2 * np.eye(dim)

        for k in range(num_fixed, len(sources)):
            members = np.flatnonzero(hard == k)
            if members.size == 0:
                reseed(k, "empty")
                continue
            w_members = resp[k, members]
            mean = (
                (w_members[:, None] * self._points[members]).sum(axis=0)
                / max(float(w_members.sum()), np.finfo(float).tiny)
            )
            nearest = int(
                members[np.argmin(np.linalg.norm(self._points[members] - mean, axis=1))]
            )
            if self._too_close(self._points[nearest], placed, spacing):
                reseed(k, "a duplicate")
                continue
            new_sources[k] = nearest
            taken.add(nearest)
            placed.append(self._points[nearest])
            if self.map.point_expandable(tuple(mean)):
                centers[k] = mean
                snapped[k] = False
            else:
                centers[k] = self._points[nearest]
                snapped[k] = True
        return weights / weights.sum(), covs, new_sources

    def _directions(self, sources: List[int]) -> np.ndarray:
        """(K, n, dim) unit Euclidean displacement direction from each
        anchor to every node (zero vector at the anchor itself)."""
        U = np.zeros((len(sources), len(self._points), self._points.shape[1]))
        for k, s in enumerate(sources):
            disp = self._points - self._points[s]
            norm = np.linalg.norm(disp, axis=1)
            nz = norm > 0.0
            U[k, nz] = disp[nz] / norm[nz, None]
        return U

    def _floor_cov(self, cov: np.ndarray) -> np.ndarray:
        """Symmetrize and clamp the eigenvalues at ``min_sigma**2``."""
        cov = 0.5 * (cov + cov.T)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, self.min_sigma ** 2)
        return (vecs * vals) @ vecs.T

    @staticmethod
    def _effective_sigmas(covs: np.ndarray) -> np.ndarray:
        """Effective isotropic width per component: sqrt(trace/dim)."""
        if len(covs) == 0:
            return np.empty(0)
        dim = covs.shape[-1]
        return np.sqrt(np.trace(covs, axis1=-2, axis2=-1) / dim)

    @staticmethod
    def _too_close(
        pos: np.ndarray, placed: List[np.ndarray], spacing: float
    ) -> bool:
        """True if ``pos`` is within ``spacing`` of a placed anchor."""
        if spacing <= 0.0 or not placed:
            return False
        d = np.linalg.norm(np.asarray(placed) - pos, axis=1)
        return bool(np.min(d) < spacing)

    def _reseed_node(
        self,
        min_dist: np.ndarray,
        taken: set,
        placed: List[np.ndarray],
        spacing: float,
    ) -> Optional[int]:
        """Reachable node farthest from every anchor, not already an anchor.

        Candidates within ``spacing`` of a placed anchor are skipped while
        a spaced alternative exists; if none does, the farthest non-anchor
        node is returned regardless.
        """
        order = np.argsort(min_dist)[::-1]
        fallback = None
        for idx in order:
            idx = int(idx)
            if not np.isfinite(min_dist[idx]) or min_dist[idx] <= 0.0 or idx in taken:
                continue
            if fallback is None:
                fallback = idx
            if not self._too_close(self._points[idx], placed, spacing):
                return idx
        return fallback
