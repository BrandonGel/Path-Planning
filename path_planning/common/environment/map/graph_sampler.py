from typing import Any, List, Tuple
import heapq
import numpy as np
from scipy.spatial import KDTree, Delaunay
from scipy.ndimage import map_coordinates
from path_planning.common.environment.node import Node
from python_motion_planning.common.env.map.grid import Grid, GridTypeMap
from scipy.spatial.distance import cdist
from itertools import product
from python_motion_planning.common import TYPES
from path_planning.utils.cgal_sweep import CGAL_Sweep
from path_planning.common.environment.map.cdt import get_boundary, get_planar_graph
from path_planning.global_planner.sample_search.rrg import RRG
from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase
from path_planning.common.environment.roadmap.roadmap_base import RoadmapBase
from path_planning.common.environment.roadmap.roadmap import Roadmap

import faiss
import pickle

ROADMAP_TYPES = set(["grid","prm","cdt","rrg","midpoints","centroids","voronoi","halton"])
def validate_roadmap_type(roadmap_type: str):
    if roadmap_type not in ROADMAP_TYPES:
        return False
    return True
def snap_to_cell_centers(map_, points) -> np.ndarray:
    """Vectorised map_to_world(world_to_map(p, discrete=True)): snap world points to
    the centres of the cells containing them (clamped to the grid), the frame all
    discrete-space roadmap nodes are stored in."""
    pts = np.asarray(points, dtype=float)
    b = np.asarray(map_.bounds, dtype=float)[:, 0]
    res = float(map_.resolution)
    idx = np.round((pts - b) / res - 0.5 + 1e-10)
    idx = np.clip(idx, 0, np.asarray(map_.shape, dtype=float) - 1)
    return b + res * (idx + 0.5)


class GraphSampler(GraphSamplerBase):
    def __str__(self) -> str:
        return "Graph Sampler"

    def save_graph_sampler(self, path: str):
        data = {
            "start": self.start,
            "goal": self.goal,
            "roadmap_config": self.roadmap_config.as_dict(),
            "use_discrete_space": self.use_discrete_space,
            "nodes": self.nodes,
            # ndarray, not list-of-tuples: pickling 4.5M tuples costs ~10s / ~200MB, the array
            # milliseconds / ~70MB. set_obstacles() accepts either on load.
            "obstacles": np.asarray(self.obstacles, dtype=np.int64) if len(self.obstacles)
                         else [],
            "obs_size": self.obs_size,
            "inflation_radius": self.inflation_radius,
            "track_with_link": self.track_with_link,
            "road_map": self.road_map,
            "road_map_edge_weights": self.road_map_edge_weights,
            "use_constraint_sweep": self.use_constraint_sweep,
            "record_sweep": self.record_sweep,
            "use_exact_collision_check": self.use_exact_collision_check,
            "sweep_backend": getattr(self, "sweep_backend", "cgal"),
            "sampling_dist_dict": self.sampling_dist_dict,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_graph_sampler(self, path: str, args: dict = {}):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._load_from_dict(data, args)

    def _load_from_dict(self, data: dict, args: dict = {}):

        self.use_discrete_space = data["use_discrete_space"]
        self.set_start(data["start"])
        self.set_goal(data["goal"])
        if "roadmap_config" in data:
            self.set_roadmap_config(data["roadmap_config"])
        else:                                   # pickles written before roadmap_config existed
            self.set_roadmap_config(sample_num=data["sample_num"], num_neighbors=data["num_neighbors"],
                                    min_edge_length=data["min_edge_length"], max_edge_length=data["max_edge_length"])
        self.sampling_dist_dict = data.get("sampling_dist_dict", {})


        self.node_sampler.set_nodes(data["nodes"])      # nodes / pos_index_dict are views over the sampler
        self.obstacles = data["obstacles"]
        self.obs_size = data["obs_size"] if "obs_size" in data else 0.5
        self.inflation_radius = data["inflation_radius"]
        self.set_obstacles(self.obstacles)
        self.inflate_obstacles(self.inflation_radius)
        self.track_with_link = data["track_with_link"]
        self.grid_points = data["grid_points"]
        self.boundary_points = data.get("boundary_points", [])

        # Set up node index dicts (pos_index_dict was rebuilt by set_nodes)
        self.node_sampler.sample_nodes(start_list=self.start, goal_list=self.goal)   # registers the start/goal kinds (positions already nodes)

        # Calculate edges (must happen before get_start/goal_nodes_with_all_edges)
        # The CSR roadmap is the source of road_map / road_map_edge_weights.
        self.roadmap = Roadmap(self, config=self.roadmap_config)
        self.roadmap.set_nodes(self.nodes)
        self.roadmap.set_adjacency(data["road_map"], data.get("road_map_edge_weights"))
        # edges / edge_weights / road_map are views over self.roadmap.
        self.get_start_nodes_with_all_edges()
        self.get_goal_nodes_with_all_edges()

        # Set up constraint sweep
        self.use_constraint_sweep = data["use_constraint_sweep"] if not 'use_constraint_sweep' in args else args["use_constraint_sweep"]
        self.record_sweep = data["record_sweep"]
        self.use_exact_collision_check = data["use_exact_collision_check"]
        self.sweep_backend = args.get("sweep_backend", data.get("sweep_backend", "auto"))
        self.constraint_sweep = self._make_constraint_sweep()
        if self.use_constraint_sweep:
            self.set_constraint_sweep()

        points = np.array([samp.current for samp in self.nodes])
        self.sample_kd_tree = KDTree(points)

    def rotate(self, k: int, axes: Tuple[int, int] = (0, 1)):
        """
        Rotate the grid type map and all node / start / goal / obstacle positions
        in-place by k * 90 degrees counterclockwise in the plane of ``axes``.

        ``axes`` is a pair of distinct dimension indices (same convention as ``numpy.rot90``).
        When k is 1 or 3, the world extents along those two axes are swapped.
        Road map topology and cost_matrix are unchanged (rotation preserves distances).

        - **Discrete space** (``use_discrete_space`` is True): grid cell indices are
          rotated with the same mapping as ``numpy.rot90`` on ``type_map``.
        - **Continuous space** (``use_discrete_space`` is False): world coordinates in
          the ``(axes[0], axes[1])`` plane are rotated about the **center of the grid
          bounds** on those axes, using the same corner mapping as the discrete case
          (normalized coordinates in ``[0, 1]`` per axis).

        Example: for a 2D map, ``rotate(1, (0, 1))`` is a 90° CCW rotation in the
        plane of dimensions 0 and 1.
        """
        a, b = int(axes[0]), int(axes[1])
        if a == b or not (0 <= a < self.dim) or not (0 <= b < self.dim):
            raise ValueError(
                f"axes must be two distinct valid dimension indices, got {axes}"
            )

        k %= 4  # 0, 1, 2, 3
        if k == 0:
            return

        old_bounds = np.asarray(self.bounds, dtype=float).copy()
        old_shape = tuple(int(s) for s in self.shape)

        new_bounds = old_bounds.copy()
        if k in (1, 3):
            range_a = old_bounds[a, 1] - old_bounds[a, 0]
            range_b = old_bounds[b, 1] - old_bounds[b, 0]
            new_bounds[a, 1] = old_bounds[a, 0] + range_b
            new_bounds[b, 1] = old_bounds[b, 0] + range_a

        def rotate_grid_nd(coords: tuple) -> tuple:
            ca, cb = coords[a], coords[b]
            if k == 1:
                new_ca, new_cb = old_shape[b] - 1 - cb, ca
            elif k == 2:
                new_ca, new_cb = old_shape[a] - 1 - ca, old_shape[b] - 1 - cb
            else:
                new_ca, new_cb = cb, old_shape[a] - 1 - ca
            lst = list(coords)
            lst[a], lst[b] = new_ca, new_cb
            return tuple(lst)

        def transform_pos_continuous(p: Tuple[float, ...]) -> Tuple[float, ...]:
            """Rotate world coords in the (a,b) plane about the center of bounds."""
            lst = [float(v) for v in p]
            range_a = old_bounds[a, 1] - old_bounds[a, 0]
            range_b = old_bounds[b, 1] - old_bounds[b, 0]
            if range_a <= 0.0 or range_b <= 0.0:
                return tuple(lst)
            ua = (lst[a] - old_bounds[a, 0]) / range_a
            ub = (lst[b] - old_bounds[b, 0]) / range_b
            if k == 1:
                ua, ub = 1.0 - ub, ua
            elif k == 2:
                ua, ub = 1.0 - ua, 1.0 - ub
            else:
                ua, ub = ub, 1.0 - ua
            lst[a] = new_bounds[a, 0] + ua * (new_bounds[a, 1] - new_bounds[a, 0])
            lst[b] = new_bounds[b, 0] + ub * (new_bounds[b, 1] - new_bounds[b, 0])
            return tuple(lst)

        def transform_pos(p: Tuple[float, ...]) -> Tuple[float, ...]:
            if self.use_discrete_space:
                p_grid = tuple(self.world_to_map(p,discrete=True))
                new_coords = rotate_grid_nd(p_grid)
                return tuple(float(v) for v in new_coords)
            return transform_pos_continuous(p)

        # python-motion-planning >= 2.1: ``type_map`` is a read-only property and
        # the numba kernels read ``_shape_array``; a fresh GridTypeMap marks the
        # ESDF / connectivity maps dirty, so they are recomputed lazily on access.
        self._type_map = GridTypeMap(
            np.rot90(self.type_map.data.copy(), k=k, axes=(a, b))
        )
        self._esdf = np.ascontiguousarray(np.rot90(self._esdf, k=k, axes=(a, b)))
        self._connectivity_map = np.ascontiguousarray(
            np.rot90(self._connectivity_map, k=k, axes=(a, b))
        )
        self._shape_array = np.asarray(self.shape, dtype=np.int64)

        self._bounds[:, :] = new_bounds

        # Same nodes, moved: every kind index (start/goal/sample/...) follows its node.
        self.node_sampler.set_positions([transform_pos(node.current) for node in self.nodes])
        if self.roadmap is not None and self.roadmap.num_nodes == len(self.nodes):
            # Same topology, moved coordinates (clearance re-read from the rotated ESDF).
            self.roadmap._set_points(np.array([node.current for node in self.nodes], dtype=np.float64))
        self.start = [list(transform_pos(tuple(s))) for s in self.start]
        self.goal = [list(transform_pos(tuple(g))) for g in self.goal]
        self.grid_points = [transform_pos(tuple(gp)) for gp in self.grid_points]
        self.boundary_points = [transform_pos(tuple(bp)) for bp in self.boundary_points]

        if len(self.obstacles) > 0:
            obs = np.asarray(self.obstacles, dtype=int)
            new_obs = np.empty_like(obs)
            for idx in range(len(obs)):
                rotated = rotate_grid_nd(
                    tuple(int(obs[idx, d]) for d in range(obs.shape[1]))
                )
                new_obs[idx] = rotated
            self.obstacles = [tuple(obs) for obs in new_obs]

        self.obstacle_nodes = [
            Node(
                transform_pos(tuple(float(v) for v in n.current)),
                None,
                0,
                0,
            )
            for n in self.obstacle_nodes
        ]

        # 6. Rebuild index dicts (pos_index_dict was rebuilt by set_nodes)
        # start/goal index views follow the sampler (moved above with set_positions)
        self.grid_nodes_index = {
            Node(tuple(gp), None, 0, 0): self.pos_index_dict[tuple(gp)]
            for gp in self.grid_points
        }
        self._rebuild_boundary_nodes_index()

        # 7. Clear caches and rebuild derived structures
        self.start_to_all_edges_dict = {}
        self.goal_to_all_edges_dict = {}

        if self.nodes:
            points = np.array([samp.current for samp in self.nodes])
            self.sample_kd_tree = KDTree(points)
        else:
            self.sample_kd_tree = None

        if self.use_constraint_sweep and len(self.edges) > 0:
            self.set_constraint_sweep()

    def create_pruned_copy(self, kept_node_indices: np.ndarray):
        """
        Create a pruned copy of the current GraphSampler that only contains
        the nodes whose indices are in kept_node_indices and edges between them.
        """
        if kept_node_indices is None or len(kept_node_indices) == 0:
            raise ValueError("kept_node_indices must be a non-empty array of node indices.")

        kept_node_indices = np.asarray(kept_node_indices, dtype=int)
        kept_set = set(int(i) for i in kept_node_indices.tolist())
        sorted_kept = sorted(kept_set)

        # Mapping from old node index -> new node index
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_kept)}

        # Pruned nodes (preserve order according to sorted_kept)
        pruned_nodes = [self.nodes[old_idx] for old_idx in sorted_kept]

        # Pruned grid points: only keep grid nodes whose indices remain
        pruned_grid_points = []
        for grid_node, idx in self.grid_nodes_index.items():
            if idx in kept_set:
                pruned_grid_points.append(grid_node.current)

        # Same for boundary points
        pruned_boundary_points = []
        for boundary_node, idx in self.boundary_nodes_index.items():
            if idx in kept_set:
                pruned_boundary_points.append(boundary_node.current)

        # Edges and edge weights restricted to kept nodes
        pruned_road_map = []
        pruned_road_map_edge_weights = []
        has_edge_weights = self.road_map_edge_weights is not None and len(self.road_map_edge_weights) > 0

        for old_i in sorted_kept:
            neighbors = []
            weights = []
            if old_i < len(self.road_map):
                for nbr_idx, old_j in enumerate(self.road_map[old_i]):
                    if old_j in kept_set:
                        neighbors.append(old_to_new[old_j])
                        if has_edge_weights:
                            weights.append(self.road_map_edge_weights[old_i][nbr_idx])
            pruned_road_map.append(neighbors)
            if has_edge_weights:
                pruned_road_map_edge_weights.append(weights)

        if not has_edge_weights:
            pruned_road_map_edge_weights = [[] for _ in pruned_road_map]

        # Determine how many non start/goal nodes remain (treat grid_points as non start/goal)
        # This mirrors how num_total_nodes is tracked: sample_num + len(start) + len(goal).
        pruned_sample_num = len(pruned_grid_points)

        pruned_data = {
            "start": self.start,
            "goal": self.goal,
            "roadmap_config": {**self.roadmap_config.as_dict(), "sample_num": pruned_sample_num},
            "use_discrete_space": self.use_discrete_space,
            "grid_points": pruned_grid_points,
            "boundary_points": pruned_boundary_points,
            "nodes": pruned_nodes,
            "obstacles": self.obstacles,
            "inflation_radius": self.inflation_radius,
            "track_with_link": self.track_with_link,
            "road_map": pruned_road_map,
            "road_map_edge_weights": pruned_road_map_edge_weights,
            "use_constraint_sweep": self.use_constraint_sweep,
            "record_sweep": self.record_sweep,
            "use_exact_collision_check": self.use_exact_collision_check,
            "sweep_backend": getattr(self, "sweep_backend", "cgal"),
            "sampling_dist_dict": self.sampling_dist_dict,
        }

        # Create a new GraphSampler instance with the same geometric configuration
        resolution = getattr(self, "resolution", 1.0)
        pruned_sampler = GraphSampler(
            bounds=self.bounds,
            resolution=resolution,
            start=[],
            goal=[],
            roadmap_config={**self.roadmap_config.as_dict(), "sample_num": 0},
            use_discrete_space=self.use_discrete_space,
            use_constraint_sweep=self.use_constraint_sweep,
            record_sweep=self.record_sweep,
            use_exact_collision_check=self.use_exact_collision_check,
            sweep_backend=getattr(self, "sweep_backend", "cgal"),
            sampling_dist_dict=self.sampling_dist_dict,
        )
        pruned_sampler._load_from_dict(pruned_data)
        return pruned_sampler
