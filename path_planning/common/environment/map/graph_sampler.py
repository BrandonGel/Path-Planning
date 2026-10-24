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

    def set_start(self, start):
        self.start = start
        start_pixel = [self.world_to_map(s,discrete=True) for s in start]
        for s in start_pixel:
            self.type_map[tuple(s)] = TYPES.START
        
    def set_goal(self, goal):
        self.goal = goal
        goal_pixel =  [self.world_to_map(g,discrete=True) for g in goal]
        for g in goal_pixel:
            self.type_map[tuple(g)] = TYPES.GOAL
    
    def set_endpoints(self, pickups=None, deliveries=None, parking=None):
        """Store task endpoints (world coords) for visualization only.

        Unlike set_start/set_goal, these do NOT inject nodes into the roadmap
        or touch type_map; they are purely cosmetic metadata that Visualizer2D
        draws on top of the roadmap (pickups, deliveries, parking spots).
        """
        self.pickups = list(pickups) if pickups is not None else []
        self.deliveries = list(deliveries) if deliveries is not None else []
        self.parking = list(parking) if parking is not None else []

    def get_start_nodes(self) -> List[Node]:
        return [self.nodes[i] for i in self.start_nodes_index.values()]
    
    def get_goal_nodes(self) -> List[Node]:
        return [self.nodes[i] for i in self.goal_nodes_index.values()]
    
    def get_grid_nodes(self) -> List[Node]:
        return [self.nodes[i] for i in self.grid_nodes_index.values()]

    def get_boundary_nodes(self) -> List[Node]:
        return [self.nodes[i] for i in self.boundary_nodes_index.values()]

    def get_obstacle_boundary(self):
        """Obstacle/inflation boundary geometry from the CDT extraction
        (path_planning.common.environment.map.cdt.get_boundary): merged wall
        corners/junctions plus the enclosing map rectangle.

        Returns (boundary_points (N,2), boundary_segments (M,2) index pairs,
        holes: one interior point per obstacle component)."""
        mask = (self.type_map.data == TYPES.OBSTACLE) | (self.type_map.data == TYPES.INFLATION)
        return get_boundary(self, mask)

    def get_inflated_obstacles(self):
        return [tuple(int(p) for p in row) for row in np.argwhere(self.type_map.data == TYPES.INFLATION)]

    @staticmethod
    def resample_boundary(bnd_pts: np.ndarray, bnd_segs: np.ndarray, spacing: float) -> np.ndarray:
        """Boundary vertices plus points every ``spacing`` along each segment.

        Every segment of length L gets ``round(L / spacing)`` equal
        subdivisions (at least one), so when ``spacing`` divides the wall
        lengths (e.g. ``spacing == resolution`` on a rectilinear boundary)
        the interior points land exactly on the cell corners. Corner /
        junction vertices come first, in their original order, followed by
        the interior points; duplicates (shared corners) are dropped.
        """
        pts = np.asarray(bnd_pts, dtype=float)
        out = [tuple(float(v) for v in p) for p in pts]
        seen = {tuple(round(float(v), 8) for v in p) for p in out}
        for u, v in np.asarray(bnd_segs, dtype=int).tolist():
            a, b = pts[u], pts[v]
            length = float(np.linalg.norm(b - a))
            n = max(int(round(length / spacing)), 1)
            for k in range(1, n):
                q = a + (b - a) * (k / n)
                key = tuple(round(float(x), 8) for x in q)
                if key in seen:
                    continue
                seen.add(key)
                out.append(tuple(float(x) for x in q))
        return np.asarray(out, dtype=float)

    def _rebuild_boundary_nodes_index(self):
        """Re-point boundary_nodes_index at the current node registry (node
        indices shift whenever nodes are re-registered); boundary points not
        present as nodes are dropped from the index but kept in boundary_points."""
        self.boundary_nodes_index = {}
        for point in self.boundary_points:
            node = Node(tuple(point), None, 0, 0)
            if node in self.node_index_dict:
                self.boundary_nodes_index[node] = self.node_index_dict[node]

    def get_random_nodes(self) -> List[Node]:
        if len(self.nodes) >= self.roadmap_config.sample_num:
            return self.nodes[:self.roadmap_config.sample_num]
        print("Warning: number of nodes is less than sample num")
        return self.nodes

    def get_nodes(self) -> List[Node]:
        return self.nodes
    
    def get_node_index(self, node: Node) -> int:
        return self.node_index_dict[node]
 
    def set_obstacle_map(self, obstacles: np.ndarray):
        obstacles = np.array(obstacles)
        if len(obstacles) == 0:
            return
        if len(obstacles[0]) == 2:
            self.type_map[obstacles[:,0], obstacles[:,1]] = TYPES.OBSTACLE
        elif len(obstacles[0]) == 3:
            self.type_map[obstacles[:,0], obstacles[:,1], obstacles[:,2]] = TYPES.OBSTACLE
        else:
            raise ValueError(f"Unsupported dimensions: {len(obstacles.shape)}")
        for obstacle in obstacles:
            pos = tuple(obstacle)
            node = Node(pos,None,0,0)
            self.obstacle_nodes.append(node)

    def get_obstacle_map(self) -> np.ndarray:
        return self.type_map.data == TYPES.OBSTACLE

    def get_obstacle_nodes(self) -> List[Node]:
        return self.obstacle_nodes

    def get_cost(self, p1_node: tuple, p2_node: tuple) -> float:
        """
        Get the cost between two nodes. (default: distance defined in the map)

        Args:
            p1: Start node.
            p2: Goal node.
        """
        if self.cost_matrix is None:
            return self.get_distance(p1_node.current, p2_node.current)
        else:
            p1_index = self.node_index_dict[p1_node]
            p2_index = self.node_index_dict[p2_node]
            return self.cost_matrix[p1_index, p2_index]

    def get_neighbors(self, node: Node, **kwargs) -> List[Node]:
        """
        Get the neighbors of a node.
        """
        if self.road_map is None:
            return []
        idx = None
        if node in self.node_index_dict:
            idx = self.node_index_dict[node]
        else:
            try:
                if getattr(self, "sample_kd_tree", None) is not None:
                    dist, nn = self.sample_kd_tree.query(
                        np.asarray(getattr(node, "current"), dtype=float).reshape(1, -1), k=1
                    )
                    dist = float(dist.reshape(-1)[0])
                    nn = int(nn.reshape(-1)[0])
                    if dist <= 1e-6 and 0 <= nn < len(self.nodes):
                        idx = nn
            except Exception:
                idx = None
        if idx is None:
            return []
        neighbors = [self.nodes[i] for i in self.road_map[int(idx)]]
        if self.track_with_link:
            neighbors = [neighbor.link(node) for neighbor in neighbors]
        return neighbors

    def _make_constraint_sweep(self):
        """Build the spatial-sweep backend.

        ``sweep_backend``:
        - ``"auto"`` (default): use the Shapely (GEOS) backend in 2D and the CGAL
          backend in 3D+ (Shapely/GEOS is planar). Falls back to CGAL in 2D if
          Shapely is not installed.
        - ``"shapely"``: force the Shapely backend (2D only).
        - ``"cgal"``: force the CGAL backend (2D or 3D).
        Shapely is imported lazily so it is only required when actually selected."""
        backend = getattr(self, "sweep_backend", "auto")
        dim = getattr(self, "dim", None)
        if dim is None:
            dim = len(self.bounds) if getattr(self, "bounds", None) is not None else 2

        if backend == "auto":
            backend = "shapely" if dim == 2 else "cgal"
            if backend == "shapely":
                try:  # 2D auto falls back to CGAL when Shapely is unavailable.
                    import shapely  # noqa: F401
                except Exception:
                    backend = "cgal"

        if backend == "shapely":
            from path_planning.utils.shapely_sweep import ShapelySweep
            return ShapelySweep(
                record_sweep=self.record_sweep,
                use_exact_collision_check=self.use_exact_collision_check,
            )
        if backend != "cgal":
            raise ValueError(f"Unknown sweep_backend {backend!r}; expected 'auto', 'cgal' or 'shapely'.")
        return CGAL_Sweep(
            record_sweep=self.record_sweep,
            use_exact_collision_check=self.use_exact_collision_check,
        )

    def set_constraint_sweep(self, force: bool = False, keep_query_caches: bool = False):
        """(Re)bind the sweep backend to the current roadmap.

        Building the backend's spatial indices (vertex KD-tree / edge R-tree, edge
        geometry arrays) is the expensive part and depends on the roadmap only, so it
        is skipped when the roadmap is unchanged since the last call (content
        fingerprint of nodes + edges); per-query memo caches are still cleared then,
        keeping memory behaviour identical to a full rebuild. Callers that plan many
        times on one roadmap (NeuralATTF builds a SippPlanner per low-level call)
        therefore pay the index build once. ``force`` rebuilds regardless;
        ``keep_query_caches`` also keeps the sweep's memoized query results.
        """
        sig = self._roadmap_signature()
        if not force and getattr(self, "_constraint_sweep_signature", None) == sig:
            if not keep_query_caches:
                self.constraint_sweep.clear_caches()
            self._constraint_sweep_locations_cache = {}
            return
        self.constraint_sweep.set_graph([node.current for node in self.nodes],self.edges)
        self._constraint_sweep_signature = sig
        self._sipp_sweep_memo = {}  # SippGraph's cross-call sweep memo is roadmap-bound
        # Precompute tuples for fast per-index coordinate access — avoids the
        # repeated `self.nodes[idx].current` attribute walk in the hot path.
        self._node_current_tuples = [tuple(n.current) for n in self.nodes]
        # Memoize the coordinate-converted sweep result so identical queries
        # don't re-pay the set-comprehension cost on every CBS conflict check.
        self._constraint_sweep_locations_cache = {}

    def get_constraint_sweep(self, p1: tuple[float,float], p2: tuple[float,float],v: float = 0.0, r: float = 0.5, use_interval: bool = False,get_time_interval: bool = False):
        if not self.use_constraint_sweep:
            return None
        if use_interval:
            overlapping_vertices,overlapping_edges = self.constraint_sweep.overlapping_interval_cgal(p1, p2,v, r,get_time_interval=get_time_interval)
            node_tuples = getattr(self, "_node_current_tuples", None)
            if node_tuples is None:
                node_tuples = [tuple(n.current) for n in self.nodes]
            if get_time_interval:
                vertices_interval = {node_tuples[vi]: iv for vi, iv in overlapping_vertices.items()}
                edges_interval = {(node_tuples[e[0]], node_tuples[e[1]]): iv for e, iv in overlapping_edges.items()}
                return vertices_interval, edges_interval
            vertices_locations = set(node_tuples[vi] for vi in overlapping_vertices)
            edges_locations = set((node_tuples[e[0]], node_tuples[e[1]]) for e in overlapping_edges)
            return vertices_locations, edges_locations

        cache = getattr(self, "_constraint_sweep_locations_cache", None)
        if cache is not None:
            cache_key = (p1, p2, v, r)
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        overlapping_edges = self.constraint_sweep.overlapping_graph_elements_cgal(p1, p2, v, r)
        node_tuples = getattr(self, "_node_current_tuples", None)
        if node_tuples is None:
            node_tuples = [tuple(n.current) for n in self.nodes]
        edges_locations = set((node_tuples[e[0]], node_tuples[e[1]]) for e in overlapping_edges)
        if cache is not None:
            cache[cache_key] = edges_locations
        return edges_locations

    def get_constraint_segment(self, p1a: Tuple[float, ...],p1b: Tuple[float, ...],p2a: Tuple[float, ...],p2b: Tuple[float, ...],v: float = 0.0,r: float = 0.5) -> bool:
        p1a = np.array(p1a)
        p2a = np.array(p2a)
        p1b = np.array(p1b)
        p2b = np.array(p2b)
        if v == 0.0:
            v1 = p1b-p1a
            v2 = p2b-p2a
            t_dur = 1.0
        else:
            dist1 = np.linalg.norm(p1b-p1a)
            dist2 = np.linalg.norm(p2b-p2a)
            t_dur1 = dist1/v
            t_dur2 = dist2/v
            t_dur = min(t_dur1, t_dur2)
            v1 = (p1b-p1a)/t_dur1 if dist1 > 0.0 else np.zeros(self.dim)
            v2 = (p2b-p2a)/t_dur2 if dist2 > 0.0 else np.zeros(self.dim)
        r = 2*r
        r_vec = p2a-p1a
        vel = v2-v1
        tmin = np.clip(-np.dot(vel,r_vec)/(np.dot(vel,vel)+1e-10),0.0,t_dur)
        r_min_vec = r_vec + vel*tmin
        if np.linalg.norm(r_min_vec) < r:
            return True
        return False

    def point_float_to_int(self, point: Tuple[float, ...]) -> Tuple[int, ...]:
        """
        Convert a point from float to integer coordinates.

        Args:
            point: a point in float coordinates
        
        Returns:
            point: a point in integer coordinates
        """
        point_int = []
        for d in range(self.dim):
            point_int.append(max(0, min(self.shape[d] - 1, int(round(point[d]+1e-10)))))
        point_int = tuple(point_int)
        return point_int

    def set_obstacles(self, obstacles: np.ndarray):
        self.obstacles = [tuple[Any, ...](obs) for obs in obstacles]
        self.set_obstacle_map(obstacles)

    def save_graph_sampler(self, path: str):
        data = {
            "start": self.start,
            "goal": self.goal,
            "roadmap_config": self.roadmap_config.as_dict(),
            "use_discrete_space": self.use_discrete_space,
            "grid_points": self.grid_points,
            "boundary_points": self.boundary_points,
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


        self.nodes = data["nodes"]
        self.obstacles = data["obstacles"]
        self.obs_size = data["obs_size"] if "obs_size" in data else 0.5
        self.inflation_radius = data["inflation_radius"]
        self.set_obstacles(self.obstacles)
        self.inflate_obstacles(self.inflation_radius)
        self.track_with_link = data["track_with_link"]
        self.grid_points = data["grid_points"]
        self.boundary_points = data.get("boundary_points", [])

        # Set up node index dict
        self.node_index_dict = {node: i for i, node in enumerate(self.nodes)}
        self.cost_matrix = cdist(np.array([node.current for node in self.nodes]), np.array([node.current for node in self.nodes]), metric='euclidean')
        self.start_nodes_index = {Node(tuple(start),None,0,0):self.node_index_dict[Node(tuple(start),None,0,0)] for start in self.start}
        self.goal_nodes_index = {Node(tuple(goal),None,0,0):self.node_index_dict[Node(tuple(goal),None,0,0)] for goal in self.goal}
        self.grid_nodes_index = {Node(tuple(grid_point),None,0,0):self.node_index_dict[Node(tuple(grid_point),None,0,0)] for grid_point in self.grid_points}
        self._rebuild_boundary_nodes_index()

        # Calculate edges (must happen before get_start/goal_nodes_with_all_edges)
        self.road_map = data["road_map"]
        self.road_map_edge_weights = data["road_map_edge_weights"]
        self.roadmap = None                     # CSR view rebuilt lazily by roadmap_view()
        self.edges, self.edge_indices_dict, self.edge_weights = RoadmapBase.edges_from_road_map(self.road_map, self.road_map_edge_weights)
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

        self.nodes = [
            Node(transform_pos(node.current), None, 0, 0) for node in self.nodes
        ]
        self.roadmap = None                     # node coordinates moved; CSR view rebuilt lazily
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

        # 6. Rebuild index dicts
        self.node_index_dict = {node: i for i, node in enumerate(self.nodes)}
        self.start_nodes_index = {
            Node(tuple(s), None, 0, 0): self.node_index_dict[Node(tuple(s), None, 0, 0)]
            for s in self.start
        }
        self.goal_nodes_index = {
            Node(tuple(g), None, 0, 0): self.node_index_dict[Node(tuple(g), None, 0, 0)]
            for g in self.goal
        }
        self.grid_nodes_index = {
            Node(tuple(gp), None, 0, 0): self.node_index_dict[Node(tuple(gp), None, 0, 0)]
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
