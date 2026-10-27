from typing import List, Tuple
import os
import numpy as np
from scipy.spatial import KDTree
from path_planning.common.environment.node import Node
from python_motion_planning.common.env.map.grid import GridTypeMap
from python_motion_planning.common import TYPES
from path_planning.utils.cgal_sweep import CGAL_Sweep
from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase, ROADMAP_TYPES, validate_roadmap_type, snap_to_cell_centers  # re-exported: cluster_map, ctopprm, dataset_ground_truth_map import them from here
from path_planning.common.environment.roadmap.roadmap_base import RoadmapConfig
from path_planning.common.environment.node_sampler.node_sampler import NodeSampler
from path_planning.common.utils.geometry import GeometryNumpy as Geometry
from path_planning.common.environment.roadmap.roadmap import Roadmap

import pickle


class GraphSampler(GraphSamplerBase):
    def __init__(self,*args,start,goal,roadmap_config: "dict | RoadmapConfig | str | None" = None,use_discrete_space=True,use_constraint_sweep=True,record_sweep=True,use_exact_collision_check=True,sampling_dist_dict = {},sweep_backend="auto",**kwargs):
        super().__init__(*args, **kwargs)

        # Check if start and goal are lists, non-empty, and not None
        assert start is not None, "Start must not be None"
        assert goal is not None, "Goal must not be None"
        assert isinstance(start, list), "Start must be a list"
        assert isinstance(goal, list), "Goal must be a list"
        self.start = start
        self.goal = goal
        self.obstacles = []
        self.obs_size = 0.5
        # Roadmap parameters (sample_num, num_neighbors, edge band, RRG rates,
        # ...) live in one RoadmapConfig: a dict, a RoadmapConfig or the path of
        # a YAML file such as config/roadmap.yaml; see set_roadmap_config.
        self.node_sampler = NodeSampler(self, sampling_dist_dict=sampling_dist_dict)      # reads the map live; see NodeSampler
        self.roadmap = Roadmap(self, config=roadmap_config)
        
        self.obstacle_nodes = []
        self.track_with_link = False

        self.use_discrete_space = use_discrete_space


        self.use_constraint_sweep = use_constraint_sweep
        self.record_sweep = record_sweep
        self.use_exact_collision_check = use_exact_collision_check
        self.sweep_backend = sweep_backend
        self.constraint_sweep = self._make_constraint_sweep()
        self.sampling_dist_dict = sampling_dist_dict if sampling_dist_dict else {}

        self._inflated_esdf = None          # inflated esdf cache
        self._inflated_esdf_source = None   # inflated esdf source to update inflated esdf


        self.pickups = []
        self.deliveries = []
        self.parking = []

    def __str__(self) -> str:
        return "Graph Sampler"

    @property
    def roadmap_config(self) -> RoadmapConfig:
        """The roadmap parameters: the config of ``self.roadmap`` (one object,
        so the sampler and its builders never disagree)."""
        return self.roadmap.config

    @property
    def road_map_edge_weights(self) -> List[List[float]]:
        return self.roadmap.roadmap_weights if self.roadmap is not None else []

    @property
    def roadmap_edges(self) -> List[List[int]]:
        return self.roadmap.roadmap_edges if self.roadmap is not None else []

    @property
    def road_map(self) -> List[List[int]]:
        """Alias of ``roadmap_edges``: the per-node neighbour lists under the
        name planners, scripts and pickles read."""
        return self.roadmap_edges

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return self.roadmap.edges if self.roadmap is not None else []

    @property
    def edge_weights(self) -> List[float]:
        return self.roadmap.edge_weights if self.roadmap is not None else []

    @property
    def nodes(self) -> List[Node]:
        return self.node_sampler.nodes if self.node_sampler is not None else []

    @property
    def pos_index_dict(self) -> dict:
        """Position tuple -> node index (the sampler's registry)."""
        return self.node_sampler.pos_index_dict if self.node_sampler is not None else {}

    @property
    def points_index_dict(self) -> dict:
        """Kind name -> ``Node -> index`` dict (the sampler's registry)."""
        return self.node_sampler.points_index_dict if self.node_sampler is not None else {}

    @property
    def num_total_nodes(self) -> int:
        return self.node_sampler.num_total_nodes if self.node_sampler is not None else 0

    def get_nodes(self, points: List[tuple] | np.ndarray | None = None, index_dict: dict | None = None, index_type: str = '') -> List[Node]:
        return self.node_sampler.get_nodes(points, index_dict, index_type)

    def set_roadmap_config(self, config: "dict | RoadmapConfig | str | None" = None, **fields) -> RoadmapConfig:
        """Replace or update the roadmap parameters.

        A ``RoadmapConfig`` or a YAML path replaces the current config. A dict
        and/or keyword fields (``sample_num``, ``num_neighbors``,
        ``min_edge_length``, ``max_edge_length``, ``goal_sample_rate``, ...)
        are applied on top of the current values. This is the one mutator:
        ``roadmap_config`` is a read-only view of ``self.roadmap.config``.
        (``num_total_nodes`` is the live node count, not ``sample_num``.)
        """
        if isinstance(config, (str, os.PathLike)):
            config = RoadmapConfig.from_yaml(config)
        if isinstance(config, RoadmapConfig):
            if fields:
                raise TypeError("set_roadmap_config: pass either a RoadmapConfig / YAML path or fields, not both")
            self.roadmap.config = config
        else:
            merged = self.roadmap_config.as_dict()
            merged.update(config or {})
            merged.update(fields)
            self.roadmap.config = RoadmapConfig(merged)
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

    @property
    def start_nodes_index(self) -> dict:
        """``Node -> index`` of the start nodes: the sampler's ``'start_points'`` kind."""
        return self.points_index_dict.get('start_points', {})

    @property
    def goal_nodes_index(self) -> dict:
        """``Node -> index`` of the goal nodes: the sampler's ``'goal_points'`` kind."""
        return self.points_index_dict.get('goal_points', {})

    @property
    def boundary_nodes_index(self) -> dict:
        """``Node -> index`` of the boundary nodes: the sampler's ``'boundary_points'`` kind."""
        return self.points_index_dict.get('boundary_points', {})

    @property
    def start_nodes(self) -> List[Node]:
        return self.node_sampler.get_nodes(index_type='start_points')
    
    @property
    def goal_nodes(self) -> List[Node]:
        return self.node_sampler.get_nodes(index_type='goal_points')
    
    @property
    def boundary_nodes(self):
        """Obstacle/inflation boundary geometry from the CDT extraction
        (path_planning.common.environment.map.cdt.get_boundary): merged wall
        corners/junctions plus the enclosing map rectangle.

        Returns (boundary_points (N,2), boundary_segments (M,2) index pairs,
        holes: one interior point per obstacle component)."""
        return self.node_sampler.get_nodes(index_type='boundary_points')

    def _all_edges_from(self, u_indices, khop: int = -1) -> dict:
        """``{u: [(v, distance), ...]}`` along the roadmap for each node index in
        ``u_indices``: ``RoadmapBase.khop_distances`` (``khop=-1`` every node,
        ``inf`` when unreachable; ``khop=k`` the nodes within ``k`` hops)."""
        n = len(self.nodes)
        if n == 0 or len(self.roadmap_edges) != n:
            raise ValueError("Edges have not been generated yet. Please generate the roadmap first.")
        return self.roadmap.khop_distances(list(u_indices), khop)

    def get_start_nodes_with_all_edges(self, khop: int = -1):
        """``{start index: [(v, distance), ...]}`` along the roadmap (see
        ``_all_edges_from`` for ``khop``), cached per ``khop`` until the
        roadmap is rebuilt (``Roadmap.apply`` resets the cache)."""
        if not getattr(self, "start_to_all_edges_dict", None) or getattr(self, "_start_edges_khop", None) != khop:
            self.start_to_all_edges_dict = self._all_edges_from(list(self.start_nodes_index.values()), khop)
            self._start_edges_khop = khop
        return self.start_to_all_edges_dict

    def get_goal_nodes_with_all_edges(self, khop: int = -1):
        """Goal twin of ``get_start_nodes_with_all_edges``."""
        if not getattr(self, "goal_to_all_edges_dict", None) or getattr(self, "_goal_edges_khop", None) != khop:
            self.goal_to_all_edges_dict = self._all_edges_from(list(self.goal_nodes_index.values()), khop)
            self._goal_edges_khop = khop
        return self.goal_to_all_edges_dict

    def _index_of(self, p) -> int:
        """Node index of a world point (exact match in ``pos_index_dict``). An
        integer passes through. Raises ``KeyError`` when the point is not a
        roadmap node."""
        if isinstance(p, (int, np.integer)):
            return int(p)
        idx = self.pos_index_dict[p]
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

    def get_neighbors(self, node: Node | int, **kwargs) -> List[Node]:
        """
        Get the neighbors of a node.
        """
        try:
            idx = self._index_of(node.current)
        except KeyError:
            return []
        road_map = self.roadmap_edges
        if idx >= len(road_map):
            return []
        neighbors = [self.nodes[i] for i in road_map[idx]]
        if self.track_with_link:
            neighbors = [neighbor.link(node) for neighbor in neighbors]
        return neighbors

    def generateRandomNodes(self,sample_num: int = None, generate_grid_nodes = False, generate_boundary_nodes = False, prob_map = None, samp_from_prob_map_ratio = 0.5, roadmap_type: str = None, boundary_node_spacing: float | None = None):
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
            n_samples=sample_num if sample_num is not None else self.roadmap_config.sample_num,
            start_list=self.start,
            goal_list=self.goal,
            generate_grid_nodes=generate_grid_nodes,
            generate_boundary_nodes=generate_boundary_nodes,
            prob_map=prob_map,
            samp_from_prob_map_ratio=samp_from_prob_map_ratio,
            sampling_dist_dict=self.sampling_dist_dict,
            sampling_type=sampling_type,
        )
        return self.nodes

    def generate_roadmap(self, roadmap_type="prm", config=None, **options):
        """Build a roadmap with ``path_planning.common.environment.roadmap`` and
        install it here.

        ``roadmap_type`` follows ``generate_map`` (``prm``, ``grid``, ``dt``,
        ``cdt``, ``midpoints``, ``centroids``, ``voronoi``, ``rrg``, ``halton*``;
        plus ``custom`` with ``edge_indices=...``). ``samples`` defaults to
        ``self.nodes``. Extra keyword arguments go to the builder (e.g.
        ``anchor_points``, ``filter_collision``, ``k_query``). ``config`` is a
        ``RoadmapConfig``, or a dict of overrides on top of this sampler's
        ``roadmap_config`` (the default). Afterwards
        ``nodes``, ``edges``, ``edge_weights`` and the index
        dicts are set (``Roadmap.apply_to``); the ``Roadmap`` object (CSR
        adjacency in ``.adj``) is kept on ``self.roadmap`` and serves the
        ``road_map`` / ``road_map_edge_weights`` views.

        Legacy form ``generate_roadmap(samples)`` (a node list as first
        argument) builds the k-nearest PRM, as before.

        Returns:
            roadmap: the installed ``Roadmap``.
        """
        samples = None
        if not isinstance(roadmap_type, str):          # legacy generate_roadmap(samples)
            roadmap_type, samples = "prm", roadmap_type
        if config is None:                               # this sampler's roadmap_config
            config = self.roadmap_config
        if not isinstance(config, RoadmapConfig):      # dict: overrides on top of it
            config = RoadmapConfig({**self.roadmap_config.as_dict(), **config})
        self.roadmap = Roadmap(self, roadmap_type, config=config, **options)
        self.roadmap.generate_roadmap(self.nodes if samples is None else samples)
        self.roadmap.apply()
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

    def clear_data(self):
        self.node_sampler.reset()
        self.roadmap.reset()
        self.start = []
        self.goal = []
        self.obstacles = []
        self.inflation_radius = 0.0

    def __str__(self) -> str:
        return "Graph Sampler"

    def save_graph_sampler(self, path: str):
        data = {
            "start": self.start,
            "goal": self.goal,
            "roadmap_config": self.roadmap_config.as_dict(),
            "use_discrete_space": self.use_discrete_space,
            "nodes": self.nodes,
            "boundary_points": [node.current for node in self.boundary_nodes_index],   # the boundary kind, re-registered on load
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

        # Kind indices (pos_index_dict was rebuilt by set_nodes): the stored
        # positions are all nodes already, so this only registers them.
        self.node_sampler.register_points('start_points', self.start)
        self.node_sampler.register_points('goal_points', self.goal)
        self.node_sampler.register_points('boundary_points', data.get("boundary_points", []))

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

        # 6. Index dicts: every kind (start/goal/sample/boundary) followed its
        # node in set_positions above; nothing to rebuild.

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

        # Boundary points: only keep boundary nodes whose indices remain
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

        # sample_num of the copy: the kept nodes that are not a start/goal.
        start_goal = set(self.start_nodes_index.values()) | set(self.goal_nodes_index.values())
        pruned_sample_num = sum(1 for i in sorted_kept if i not in start_goal)

        pruned_data = {
            "start": self.start,
            "goal": self.goal,
            "roadmap_config": {**self.roadmap_config.as_dict(), "sample_num": pruned_sample_num},
            "use_discrete_space": self.use_discrete_space,
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
