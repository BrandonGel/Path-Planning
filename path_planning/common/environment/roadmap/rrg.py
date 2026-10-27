import warnings
from typing import List, TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_array
from scipy.spatial import KDTree

from path_planning.common.environment.node import Node
from path_planning.common.environment.roadmap.roadmap_base import RoadmapBase, RoadmapConfig

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


class RRG(RoadmapBase):
    """Rapidly-exploring Random Graph (Karaman & Frazzoli 2011).

    Grows the node set from the seeds (the given samples plus
    ``anchor_points``, i.e. start and goal): each round draws a uniform point
    in the map bounds (or a goal with probability ``goal_sample_rate``),
    steers from the nearest node towards it by at most ``max_edge_length``,
    and keeps the new node when it and the segment to its nearest node are
    collision free. Once ``sample_num`` nodes were added, every pair closer
    than ``max_edge_length`` is linked, subject to the usual length band and
    line-of-sight filter. In discrete mode nodes sit on cell centres.

    Samples are drawn in the world frame, so nodes always lie inside the map
    bounds (the base-class ``generate_rrg`` samples in the map frame and
    lands half a cell off). Uses the global numpy RNG unless ``rng`` is given,
    so ``set_global_seed`` makes it reproducible.
    """

    name = "rrg"
    uses_anchors = True

    def __init__(self, graph_map: "GraphSamplerBase", config: dict | RoadmapConfig | None = None):
        super().__init__(graph_map, config)
        self.discrete = bool(graph_map.use_discrete_space)
        self.num_iterations = 0

    @property
    def goal_sample_rate(self) -> float:
        return self.config.goal_sample_rate

    # --------------------------------------------------------------- pieces
    def _snap(self, p: np.ndarray) -> np.ndarray:
        return self.centres_of(self.cells_of(p[None, :]))[0] if self.discrete else p

    def _random_point(self, rng, goals: np.ndarray) -> np.ndarray:
        if len(goals) and rng.random() < self.goal_sample_rate:
            return goals[rng.integers(len(goals))]
        lo, hi = np.asarray(self.graph_map.bounds, dtype=float).T
        return self._snap(lo + rng.random(self.dim) * (hi - lo))

    def _steer(self, near: np.ndarray, target: np.ndarray) -> np.ndarray | None:
        d = float(np.linalg.norm(target - near))
        if d < 1e-12:
            return None
        if d <= self.max_edge_length:
            return target
        return self._snap(near + (self.max_edge_length / d) * (target - near))

    # -------------------------------------------------------------- builder
    def generate_rrg(self, anchor_points=None, sample_num: int | None = None, goal_points=None,
                     max_iters: int | None = None, rng: np.random.Generator | None = None,
                     filter_collision: bool | None = None) -> csr_array:
        """Grow ``sample_num`` nodes (default: ``config.sample_num``) from the
        current nodes plus ``anchor_points``; ``goal_points`` (default: the
        map's ``goal``) receive the goal-sampling bias. ``max_iters`` (default
        ``config.max_iters_ratio * sample_num``) bounds the rejection loop;
        hitting it warns."""
        rng = rng if rng is not None else np.random.default_rng(np.random.randint(0, 2**31 - 1))
        n_target = self.config.sample_num if sample_num is None else int(sample_num)
        max_iters = int(self.config.max_iters_ratio * max(n_target, 1)) if max_iters is None else int(max_iters)
        m = self.graph_map

        seeds = [self.nodes.reshape(-1, self.dim)]
        if anchor_points is not None and len(anchor_points):
            seeds.append(np.asarray(anchor_points, dtype=float).reshape(-1, self.dim))
        seeds = np.vstack(seeds)
        if self.discrete and len(seeds):
            seeds = self.centres_of(self.cells_of(seeds))
        goals = (np.asarray(list(m.goal), dtype=float).reshape(-1, self.dim) if goal_points is None
                 else np.asarray(goal_points, dtype=float).reshape(-1, self.dim))
        if self.discrete and len(goals):
            goals = self.centres_of(self.cells_of(goals))

        # Growing point buffer; nearest-node queries are a vectorised brute force
        # over the filled part (cheap next to the Python-level sample loop).
        pts = np.empty((len(seeds) + n_target, self.dim))
        n = 0
        seen = set()
        for p in seeds:                               # dedupe seeds, keep order
            key = tuple(np.round(p, 9))
            if key not in seen:
                seen.add(key)
                pts[n] = p
                n += 1
        if n == 0:
            raise ValueError("RRG needs at least one seed node (samples or anchor_points)")

        added = 0
        it = 0
        while added < n_target and it < max_iters:
            it += 1
            rand = self._random_point(rng, goals)
            near = pts[int(np.argmin(((pts[:n] - rand) ** 2).sum(axis=1)))]
            new = self._steer(near, rand)
            if new is None:
                continue
            key = tuple(np.round(new, 9))
            if key in seen or m.in_collision(new) or m.in_collision(new, near):
                continue
            seen.add(key)
            pts[n] = new
            n += 1
            added += 1
        self.num_iterations = it
        if added < n_target:
            warnings.warn(f"RRG: stopped after {it} iterations with {added}/{n_target} new nodes "
                          f"(free space saturated or map too constrained)", stacklevel=3)

        self._set_points(pts[:n])
        pairs = KDTree(self.nodes).query_pairs(r=np.nextafter(self.max_edge_length, np.inf), output_type="ndarray")
        if len(pairs) == 0:
            self._adj = csr_array((n, n), dtype=float)
            return self.adj
        return self.filter_edges(pairs[:, 0], pairs[:, 1], filter_collision)

    generate = generate_rrg
