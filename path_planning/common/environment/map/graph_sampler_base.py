from typing import Any, List, Tuple
import heapq
import os
import numpy as np
from scipy.spatial import KDTree, Delaunay
from path_planning.common.environment.node import Node
from python_motion_planning.common.env.map.grid import Grid, GridTypeMap
from scipy.spatial.distance import cdist
from scipy.sparse import csr_array
from itertools import product
from python_motion_planning.common import TYPES
from path_planning.utils.cgal_sweep import CGAL_Sweep
from path_planning.common.environment.map.cdt import get_boundary, get_planar_graph
from path_planning.global_planner.sample_search.rrg import RRG
from python_motion_planning.common.env.map.grid import _njit, _grid_line_of_sight, _grid_in_collision, _grid_is_expandable, _grid_path_map_to_world, _grid_world_to_map_int, _grid_path_world_to_map_int, _grid_path_world_to_map_float
from path_planning.common.environment.map.numbaUtil import _world_line_of_sight, _world_in_collision, _grid_vector_is_expandable, convert_np_to_tuple, _world_min_wall_distance
from path_planning.common.environment.roadmap.roadmap_base import RoadmapConfig
from path_planning.common.environment.roadmap.roadmap import Roadmap
from path_planning.common.environment.map.node_sampler import NodeSampler
from path_planning.common.utils.geometry import GeometryNumpy as Geometry
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


class GraphSamplerBase(Grid):
    def __init__(self,*args,start,goal,roadmap_config: "dict | RoadmapConfig | str | None" = None,use_discrete_space=True,use_constraint_sweep=True,record_sweep=True,use_exact_collision_check=True,sampling_dist_dict = {},sweep_backend="auto",**kwargs):
        super().__init__(*args, **kwargs)

        # Check if start and goal are lists, non-empty, and not None
        assert start is not None, "Start must not be None"
        assert goal is not None, "Goal must not be None"
        assert isinstance(start, list), "Start must be a list"
        assert isinstance(goal, list), "Goal must be a list"
        self.start = start
        self.goal = goal
        self.grid_points = []
        self.boundary_points = []
        self.obstacles = []
        self.obs_size = 0.5
        # Roadmap parameters (sample_num, num_neighbors, edge band, RRG rates,
        # ...) live in one RoadmapConfig: a dict, a RoadmapConfig or the path of
        # a YAML file such as config/roadmap.yaml; see set_roadmap_config.
        self.nodes = []
        self.node_sampler = NodeSampler(self)      # reads the map live; see NodeSampler
        self.roadmap = Roadmap(self, config=roadmap_config)
        self.num_total_nodes = self.roadmap_config.sample_num + len(self.start) + len(self.goal)
        self.cost_matrix = None
        self.pos_index_dict = {}
        self.start_nodes_index = {}
        self.goal_nodes_index = {}
        self.grid_nodes_index = {}
        self.boundary_nodes_index = {}
        self.start_to_all_edges_dict = {}
        self.goal_to_all_edges_dict = {}
        
        self.obstacle_nodes = []
        self.track_with_link = False
        self.edges = []
        self.edge_weights = []
        self.use_discrete_space = use_discrete_space


        

        self.use_constraint_sweep = use_constraint_sweep
        self.record_sweep = record_sweep
        self.use_exact_collision_check = use_exact_collision_check
        self.sweep_backend = sweep_backend
        self.constraint_sweep = self._make_constraint_sweep()
        self.sampling_dist_dict = sampling_dist_dict if sampling_dist_dict else {}
        self._inflated_esdf = None          # cache for the inflated_esdf property
        self._inflated_esdf_source = None   # the esdf array it was computed from
        # Task endpoints, world coords. Visualization-only metadata (no roadmap
        # side effects), populated via set_endpoints() for e.g. MAPD demos.
        self.pickups = []
        self.deliveries = []
        self.parking = []

    def __str__(self) -> str:
        return "Graph Sampler"

    @property
    def size(self) -> int:
        return int(np.prod(self.type_map.shape))

    @property
    def inflated_esdf(self) -> np.ndarray:
        """Largest object radius that can occupy each cell: the largest multiple
        of ``sqrt(dim) * resolution`` strictly below the cell's wall distance,
        since ``inflate_obstacles`` rounds the radius up to that step and blocks
        ``esdf <= radius``. Zero inside obstacles and where nothing fits.

        Cached; recomputed only when ``esdf`` is a new array, which is what
        every ESDF update (lazy recompute, rotation) produces.
        """
        esdf = self.esdf
        if self._inflated_esdf_source is not esdf:
            step = np.sqrt(self.dim) * self.resolution
            self._inflated_esdf = np.maximum(0.0, np.ceil(esdf / step - 1e-9) - 1) * step
            self._inflated_esdf_source = esdf
        return self._inflated_esdf

    @property
    def roadmap_config(self) -> RoadmapConfig:
        """The roadmap parameters: the config of ``self.roadmap`` (one object,
        so the sampler and its builders never disagree)."""
        return self.roadmap.config

    @roadmap_config.setter
    def roadmap_config(self, config: "dict | RoadmapConfig | str | None") -> None:
        self.roadmap.config = RoadmapConfig.load(config)

    # Nested per-node views of the roadmap, read from ``self.roadmap`` (CSR):
    # ``road_map[u]`` are the neighbour indices of node u, ``road_map_edge_weights[u]``
    # the matching edge lengths. ``roadmap_edges`` / ``roadmap_weights`` are the
    # same lists under the Roadmap's names. Empty before a roadmap exists.
    @property
    def road_map(self) -> List[List[int]]:
        return self.roadmap.roadmap_edges if self.roadmap is not None else []

    @property
    def road_map_edge_weights(self) -> List[List[float]]:
        return self.roadmap.roadmap_weights if self.roadmap is not None else []

    @property
    def roadmap_edges(self) -> List[List[int]]:
        return self.road_map

    @property
    def roadmap_weights(self) -> List[List[float]]:
        return self.road_map_edge_weights

    def set_roadmap_config(self, config: "dict | RoadmapConfig | str | None" = None, **fields) -> RoadmapConfig:
        """Replace or update the roadmap parameters.

        A ``RoadmapConfig`` or a YAML path replaces the current config. A dict
        and/or keyword fields (``sample_num``, ``num_neighbors``,
        ``min_edge_length``, ``max_edge_length``, ``goal_sample_rate``, ...)
        are applied on top of the current values. ``num_total_nodes`` follows
        ``sample_num``.
        """
        if isinstance(config, (str, os.PathLike)):
            config = RoadmapConfig.from_yaml(config)
        if isinstance(config, RoadmapConfig):
            if fields:
                raise TypeError("set_roadmap_config: pass either a RoadmapConfig / YAML path or fields, not both")
            self.roadmap_config = config
        else:
            merged = self.roadmap_config.as_dict()
            merged.update(config or {})
            merged.update(fields)
            self.roadmap_config = RoadmapConfig(merged)
        self.num_total_nodes = self.roadmap_config.sample_num + len(self.start) + len(self.goal)
        return self.roadmap_config

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

    def _rebuild_boundary_nodes_index(self):
        """Re-point boundary_nodes_index at the current node registry (node
        indices shift whenever nodes are re-registered); boundary points not
        present as nodes are dropped from the index but kept in boundary_points."""
        self.boundary_nodes_index = {}
        for point in self.boundary_points:
            node = Node(tuple(point), None, 0, 0)
            if tuple(node.current) in self.pos_index_dict:
                self.boundary_nodes_index[node] = self.pos_index_dict[tuple(node.current)]

    def get_nodes(self) -> List[Node]:
        return self.nodes
    
    def get_node_index(self, node: Node) -> int:
        return self.pos_index_dict[tuple(node.current)]

    def _all_edges_from(self, u_indices, type: str = 'Euclidean') -> dict:
        """``{u: [(v, distance), ...]}`` from each node index in ``u_indices`` to the
        other nodes, through ``get_distance``, else straight-line distances to all
        nodes. ``v`` are node indices, in node order."""
        if len(self.nodes) == 0 or len(self.road_map) != len(self.nodes):
            raise ValueError("Edges have not been generated yet. Please generate the roadmap first.")
        pts = [tuple(n.current) for n in self.nodes]
        return {u: [(v, self.get_distance(pts[u], pts[v], type)) for v in range(len(pts))] for u in u_indices}

    def get_start_nodes_with_all_edges(self):
        if len(self.start_to_all_edges_dict) == 0:
            u_indices = [self.start_nodes_index[node] for node in self.get_start_nodes()]
            self.start_to_all_edges_dict = self._all_edges_from(u_indices)
        return self.start_to_all_edges_dict

    def get_goal_nodes_with_all_edges(self):
        if len(self.goal_to_all_edges_dict) == 0:
            u_indices = [self.goal_nodes_index[node] for node in self.get_goal_nodes()]
            self.goal_to_all_edges_dict = self._all_edges_from(u_indices)
        return self.goal_to_all_edges_dict

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

    def _index_of(self, p) -> int:
        """Node index of a world point (exact match in ``pos_index_dict``). An
        integer passes through. Raises ``KeyError`` when the point is not a
        roadmap node."""
        if isinstance(p, (int, np.integer)):
            return int(p)
        idx = self.pos_index_dict.get(tuple(float(x) for x in np.asarray(p).reshape(-1)))
        if idx is not None:
            return int(idx)
        raise KeyError(f"point {tuple(float(x) for x in np.asarray(p).reshape(-1))} is not a roadmap node")

    def get_distance(self, p1, p2=None, type: str = 'Euclidean') -> float | List[Tuple[tuple, float]]:
        """Distance between world points.

        ``type`` ``'Euclidean'`` / ``'Manhattan'``: straight-line distance between
        ``p1`` and ``p2`` (both required). ``type='dijkstra'``: shortest-path length
        over the roadmap; ``p1`` / ``p2`` must be roadmap nodes (world coordinates,
        resolved with ``_index_of``). With ``p2`` the result is a float, ``inf``
        when unreachable; without ``p2`` it is ``[(point, distance), ...]`` for
        every node reachable from ``p1`` (itself at 0), in node order.
        """
        if type == 'dijkstra':
            u = self._index_of(p1)
            if p2 is not None:
                v = self._index_of(p2)
                return float(self.roadmap.dijkstra(u, v))
            return [(tuple(self.nodes[v].current), d) for v, d in self.roadmap.dijkstra(u)]
        if type not in ('Euclidean', 'Manhattan'):
            raise ValueError(f"get_distance: unknown type {type!r}; expected 'Euclidean', 'Manhattan' or 'dijkstra'")
        if p2 is None:
            raise ValueError("get_distance: p2 is required for a straight-line distance")
        return Geometry.dist(p1, p2, type)

    def get_neighbors(self, node: Node, **kwargs) -> List[Node]:
        """
        Get the neighbors of a node.
        """
        try:
            idx = self._index_of(node.current)
        except KeyError:
            return []
        road_map = self.road_map
        if idx >= len(road_map):
            return []
        neighbors = [self.nodes[i] for i in road_map[idx]]
        if self.track_with_link:
            neighbors = [neighbor.link(node) for neighbor in neighbors]
        return neighbors

    def line_of_sight(self, p1: Tuple[float, ...], p2: Tuple[float, ...]) -> List[Tuple[int, ...]]:
        """
        Check if two world points are in line of sight.
        Args:
            p1: Start point of the line (world coordinates).
            p2: End point of the line (world coordinates).

        Returns:
            cells: Grid indices crossed by the segment, start cell first.
        """
        p1_array = np.asarray(p1, dtype=np.float64)
        p2_array = np.asarray(p2, dtype=np.float64)
        points = _world_line_of_sight(p1_array, p2_array, np.asarray(self.bounds, dtype=np.float64), self.resolution, np.asarray(self.shape, dtype=int))
        return [tuple(int(x) for x in points[i]) for i in range(points.shape[0])]

    def in_collision(self, p1: Tuple[float, ...], p2 : Tuple[float, ...] = None) -> bool:
        """
        Check if the line of sight between two points is in collision.
        
        Args:
            p1: Start point of the line.
            p2: End point of the line.
        
        Returns:
            in_collision: True if the line of sight is in collision, False otherwise.
        """
        p1_array = np.asarray(p1, dtype=np.float64)
        if p2 is None:
            p2 = p1
        p2_array = np.asarray(p2, dtype=np.float64)

        return _world_in_collision(
            p1_array,
            p2_array,
            np.asarray(self.bounds, dtype=np.float64),
            self.resolution,
            np.asarray(self.shape, dtype=int),
            self._type_map_flat(),
            self._esdf_flat(),
            TYPES.OBSTACLE,
            TYPES.INFLATION,
            self.strict_collision
        )

    def path_map_to_world(self, path) -> List[Tuple[float, ...]]:
        """
        Convert path from map coordinates to world coordinates.

        Same contract as ``Grid.path_map_to_world`` (cell (r, c) -> its centre),
        but the (N, dim) result is converted with ``convert_np_to_tuple``
        instead of a per-element generator.

        Args:
            path: a list of map coordinates, or an (N, dim) array
        
        Returns:
            path: a list of world coordinates
        """
        is_array = isinstance(path, np.ndarray)
        points = np.asarray(path, dtype=np.float64)
        if points.size == 0:
            return np.zeros((0, self.dim), dtype=np.float64) if is_array else []
        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError("Point dimension does not match map dimension.")

        path_world = _grid_path_map_to_world(points, self.bounds, self.resolution)
        if is_array:
            return path_world
        return convert_np_to_tuple(path_world)

    def path_world_to_map(self, path: List[Tuple[float, ...]], discrete: bool = True) -> List[tuple]:
        """
        Convert path from world coordinates to map coordinates

        Args:
            path: a list of world coordinates
            discrete: whether to round the coordinates to the nearest integer
        
        Returns:
            path: a list of map coordinates
        """
        is_array = isinstance(path, np.ndarray)
        points = np.asarray(path, dtype=np.float64)
        if points.size == 0:
            if not is_array:
                return []
            return np.zeros((0, self.dim), dtype=np.int64 if discrete else np.float64)
        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError("Point dimension does not match map dimension.")

        if discrete:
            path_map = _grid_path_world_to_map_int(points, self.bounds, self.resolution, self._shape_array)
            if is_array:
                return path_map
            return convert_np_to_tuple(path_map)
        else:
            path_map = _grid_path_world_to_map_float(points, self.bounds, self.resolution)
            if is_array:
                return path_map
            return convert_np_to_tuple(path_map)

    def is_expandable(self, point, src_point = None) -> bool | np.ndarray:
        """
        Check if a point is expandable.
        
        Args:
            point: Point to check.
            src_point: Source point.
        
        Returns:
            expandable: True if the point is expandable, False otherwise.
        """
        has_src_point = src_point is not None
        if isinstance(point, np.ndarray):
            assert point.dtype == np.int64, "Point array must be int64"
            assert src_point is None or src_point.dtype == np.int64, "Src array must be int64"
            point_array = point.reshape(-1, self.dim)
            src_array = point_array if src_point is None else src_point.reshape(-1, self.dim)
            return _grid_vector_is_expandable(  
                point_array,
                src_array,
                has_src_point,
                self._shape_array,
                self._type_map_flat(),
                self._esdf_flat(),
                TYPES.OBSTACLE,
                TYPES.INFLATION,
                self.strict_collision,
            )
        point_array = np.asarray(point, dtype=np.int64)
        src_array = point_array if src_point is None else np.asarray(src_point, dtype=np.int64)
        return _grid_is_expandable(
            point_array,
            src_array,
            has_src_point,
            self._shape_array,
            self._type_map_flat(),
            self._esdf_flat(),
            TYPES.OBSTACLE,
            TYPES.INFLATION,
            self.strict_collision,
        )

    def generateRandomNodes(self, generate_grid_nodes = False, generate_boundary_nodes = False, prob_map = None, samp_from_prob_map_ratio = 0.5, roadmap_type: str = None, boundary_node_spacing: float | None = None):
        """Sample this map's node set with a fresh ``NodeSampler`` (see
        ``NodeSampler.sample_nodes`` for the options) and install it here:
        ``roadmap_config.sample_num`` random nodes (Halton when
        ``roadmap_type`` contains ``halton``, prob-map weighted when
        ``prob_map`` is given), then grid, boundary, start and goal nodes.
        ``rrg`` roadmaps sample their own nodes, so nothing is generated for
        them. ``boundary_node_spacing`` is accepted for backward
        compatibility; the boundary spacing is the map resolution.

        Returns:
            nodes: the installed node list.
        """
        if roadmap_type == 'rrg':
            return []
        if roadmap_type is not None and 'halton' in roadmap_type:
            sampling_type = 'halton'
        elif prob_map is not None:
            sampling_type = 'distributed'
        else:
            sampling_type = 'random'
        self.node_sampler.reset()
        self.node_sampler.sample_nodes(
            n_samples=self.roadmap_config.sample_num,
            start_list=self.start,
            goal_list=self.goal,
            generate_grid_nodes=generate_grid_nodes,
            generate_boundary_nodes=generate_boundary_nodes,
            prob_map=prob_map,
            samp_from_prob_map_ratio=samp_from_prob_map_ratio,
            sampling_dist_dict=self.sampling_dist_dict,
            sampling_type=sampling_type,
        )
        self.node_sampler.apply_nodes(self)
        return self.nodes

    def generate_roadmap(self, roadmap_type="prm", samples: List[Node] | None = None, config=None, **options):
        """Build a roadmap with ``path_planning.common.environment.roadmap`` and
        install it here.

        ``roadmap_type`` follows ``generate_map`` (``prm``, ``grid``, ``dt``,
        ``cdt``, ``midpoints``, ``centroids``, ``voronoi``, ``rrg``, ``halton*``;
        plus ``custom`` with ``edge_indices=...``). ``samples`` defaults to
        ``self.nodes``. Extra keyword arguments go to the builder (e.g.
        ``anchor_points``, ``filter_collision``, ``k_query``). ``config`` is a
        ``RoadmapConfig``, or a dict of overrides on top of this sampler's
        ``roadmap_config`` (the default). Afterwards
        ``nodes``, ``edges``, ``edge_weights``, ``cost_matrix`` and the index
        dicts are set (``Roadmap.apply_to``); the ``Roadmap`` object (CSR
        adjacency in ``.adj``) is kept on ``self.roadmap`` and serves the
        ``road_map`` / ``road_map_edge_weights`` views.

        Legacy form ``generate_roadmap(samples)`` (a node list as first
        argument) builds the k-nearest PRM, as before.

        Returns:
            roadmap: the installed ``Roadmap``.
        """
        if not isinstance(roadmap_type, str):          # legacy generate_roadmap(samples)
            roadmap_type, samples = "prm", roadmap_type
        if config is None:                               # this sampler's roadmap_config
            config = self.roadmap_config
        elif not isinstance(config, RoadmapConfig):      # dict: overrides on top of it
            config = RoadmapConfig({**self.roadmap_config.as_dict(), **config})
        roadmap = Roadmap(self, roadmap_type, config=config, **options)
        roadmap.generate_roadmap(self.nodes if samples is None else samples)
        roadmap.apply_to(self)                     # also sets self.roadmap
        return self.roadmap

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

    def _roadmap_signature(self):
        """Cheap content fingerprint of the roadmap the sweep is built for."""
        return (
            len(self.nodes), len(self.edges),
            hash(tuple(tuple(n.current) for n in self.nodes)),
            hash(tuple((int(e[0]), int(e[1])) for e in self.edges)),
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

    def set_obstacles(self, obstacles: np.ndarray):
        self.obstacles = [tuple[Any, ...](obs) for obs in obstacles]
        self.set_obstacle_map(obstacles)

    def expanded_inflation_radius(self, radius: float, use_expand: bool = True) -> float:
        """Radius actually applied by ``inflate_obstacles``.

        With ``use_expand`` the requested radius is rounded UP to the nearest
        multiple of ``sqrt(dim) * resolution`` (the centre-to-centre distance
        of a diagonal neighbour), so an inflated band always includes the
        diagonal neighbours of its outermost obstacle cells. The 1e-9 slack
        makes it idempotent: re-applying an already expanded radius (reload,
        ``cluster_map``) returns the same value.
        """
        if not use_expand or radius <= 0:
            return float(radius)
        step = np.sqrt(self.dim) * self.resolution
        return float(np.ceil(radius / step - 1e-9) * step)

    def inflate_obstacles(self, radius: float = 1.0, use_expand = True) -> None:
        """Euclidean ESDF inflation, see ``expanded_inflation_radius`` for how
        ``radius`` is rounded when ``use_expand`` is True. ``self.inflation_radius``
        ends up holding the applied (expanded) radius.
        """
        radius = self.expanded_inflation_radius(radius, use_expand)
        if len(self.obstacles) == 0:
            # Nothing to inflate; Grid.inflate_obstacles would still record the
            # radius, so keep that behaviour for min_wall_distance & co.
            self.inflation_radius = radius
            return
        super().inflate_obstacles(radius)

    def clear_data(self):
        self.roadmap = Roadmap(self, config=self.roadmap_config)   # empty roadmap, same parameters
        self.edges = []
        self.edge_weights = []
        self.start = []
        self.goal = []
        self.start_to_all_edges_dict = {}
        self.goal_to_all_edges_dict = {}
        self.nodes = []
        self.pos_index_dict = {}
        self.start_nodes_index = {}
        self.goal_nodes_index = {}
        self.grid_nodes_index = {}
        self.boundary_points = []
        self.boundary_nodes_index = {}
        self.cost_matrix = None
        self.obstacles = []
        self.inflation_radius = 0.0
