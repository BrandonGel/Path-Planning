from typing import List, Tuple
import os
import numpy as np
from scipy.spatial import KDTree
from path_planning.common.environment.node import Node
from python_motion_planning.common.env.map.grid import GridTypeMap
from python_motion_planning.common import TYPES
from path_planning.agent import Agent
from path_planning.common.world.collision_checking.collision_sweep import CollisionSweep
from path_planning.common.world.collision_checking.sweep.disc_sweep import (
    sweep_paths as _disc_sweep_paths, assemble as _assemble_hits, per_agent as _per_agent,
    as_path_list)
from path_planning.common.world.collision_checking.sweep.polytope_sweep import (
    make_body, body_roadmap_hits, body_roadmap_hits_minkowski)
from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase, ROADMAP_TYPES, validate_roadmap_type, snap_to_cell_centers  # re-exported: cluster_map, ctopprm, dataset_ground_truth_map import them from here
from path_planning.common.environment.roadmap.roadmap_base import RoadmapConfig
from path_planning.common.environment.node_sampler.node_sampler import NodeSampler
from path_planning.common.utils.geometry import GeometryNumpy as Geometry
from path_planning.common.environment.roadmap.roadmap import Roadmap

import pickle


class GraphSampler(GraphSamplerBase):
    def __init__(self,*args,start,goal,roadmap_config: "dict | RoadmapConfig | str | None" = None,use_discrete_space=True,use_collision_sweep=True,record_sweep=True,use_exact_collision_check=True,sampling_dist_dict = {},sweep_backend="auto",sweep_spatial_index="grid",**kwargs):
        super().__init__(*args, **kwargs)

        # Check if start and goal are lists, non-empty, and not None
        assert isinstance(start, list), "Start must be a list"
        assert isinstance(goal, list), "Goal must be a list"
        self.obstacles = []
        self.obs_size = 0.5
        # Roadmap parameters (sample_num, num_neighbors, edge band, RRG rates,
        # ...) live in one RoadmapConfig: a dict, a RoadmapConfig or the path of
        # a YAML file such as config/roadmap.yaml; see set_roadmap_config.
        self.node_sampler = NodeSampler(self, sampling_dist_dict=sampling_dist_dict)      # reads the map live; see NodeSampler
        self.roadmap = Roadmap(self, config=roadmap_config)
        
        self.obstacle_nodes = []

        self.use_discrete_space = use_discrete_space

        self.start_to_all_edges_dict = {}
        self.goal_to_all_edges_dict = {}
        self._start_edges_khop = None
        self._goal_edges_khop = None

        self.use_collision_sweep = use_collision_sweep
        self.record_sweep = record_sweep    # legacy flag kept for saved maps; sweeps are cached by planners
        self.use_exact_collision_check = use_exact_collision_check
        self.sweep_backend = sweep_backend
        self.sweep_spatial_index = sweep_spatial_index
        self.collision_sweep = self._make_collision_sweep()
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
    def position(self) -> np.ndarray:
        return self.node_sampler.position if self.node_sampler is not None else np.array([])

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
    def boundary_nodes(self) -> List[Node]:
        """The boundary nodes (the sampler's ``'boundary_points'`` kind) as
        ``Node`` objects: outline vertices when they were generated, otherwise
        the sampled nodes whose cell touches a wall or the map edge."""
        return self.node_sampler.get_nodes(index_type='boundary_points')

    def _all_edges_from(self, u_indices, khop: int = -1) -> dict:
        """``{u: [(v, distance), ...]}`` along the roadmap for each node index in
        ``u_indices``: ``RoadmapBase.khop_distances`` (``khop=-1`` every node,
        ``inf`` when unreachable; ``khop=k`` the nodes within ``k`` hops)."""
        n = len(self.nodes)
        if n == 0 or self.roadmap is None or self.roadmap.num_nodes != n:
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
        if self.roadmap is None or idx >=  self.roadmap.num_nodes:
            return []
        neighbors = [ self.nodes[i] for i in  self.roadmap.neighbors(idx).tolist()]
        return neighbors

    def generateRandomNodes(self,sample_num: int = None, generate_grid_nodes = False, generate_boundary_nodes = False, prob_map = None, samp_from_prob_map_ratio = 0.5, roadmap_type: str = None, boundary_node_spacing: float | None = None):
        """Reset ``node_sampler`` and sample this map's node set (see
        ``NodeSampler.sample_nodes`` for the options); ``nodes``,
        ``pos_index_dict`` and the index dicts are views over the sampler, so
        nothing is copied here. ``sample_num`` (default
        ``roadmap_config.sample_num``) random nodes: Halton when
        ``roadmap_type`` contains ``halton``, prob-map weighted when
        ``prob_map`` is given, uniform otherwise. Then, in order: every
        expandable cell centre (``generate_grid_nodes``, into the sample
        kind), the outline of free space (``generate_boundary_nodes``), the
        start and goal positions, and when no outline was generated the
        existing nodes next to a wall are marked as boundary nodes instead.
        ``rrg`` roadmaps sample their own nodes, so nothing is generated for
        them. ``boundary_node_spacing`` is accepted for backward
        compatibility; the boundary spacing is the map resolution.

        Returns:
            nodes: the map's node list (``self.nodes``).
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
            sampling_type=sampling_type,            # halton / distributed parameters: node_sampler.sampling_dist_dict
        )
        return self.nodes

    def generate_roadmap(self, roadmap_type="prm", config=None, **options):
        """Build a roadmap with ``path_planning.common.environment.roadmap`` and
        install it here.

        ``roadmap_type`` is one of ``prm``, ``grid``, ``dt``, ``cdt``,
        ``midpoints``, ``centroids``, ``voronoi``, ``rrg``, ``halton*`` (a
        triangulation of the Halton samples) or ``custom`` (pass
        ``edge_indices=...``). The nodes are ``self.nodes``, i.e. what ``node_sampler``
        holds. Extra keyword arguments go to the builder (e.g.
        ``anchor_points``, ``filter_collision``, ``k_query``). ``config`` is a
        ``RoadmapConfig``, or a dict of overrides on top of this map's
        ``roadmap_config`` (the default). Afterwards ``Roadmap.apply`` hands
        the builder's nodes back to ``node_sampler`` (planar builders derive
        new points; a snapped or missing start/goal is resolved or appended)
        and clears the start/goal edge caches. The ``Roadmap`` (CSR adjacency
        in ``.adj``) is ``self.roadmap``; ``edges``, ``edge_weights``,
        ``road_map`` and ``road_map_edge_weights`` are views over it.

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

    # ------------------------------------------------------------ collision sweep
    def _make_collision_sweep(self):
        """A ``CollisionSweep`` from ``sweep_backend`` and ``sweep_spatial_index`` (see there)."""
        dim = getattr(self, "dim", None) or (len(self.bounds) if getattr(self, "bounds", None) is not None else 2)
        return CollisionSweep(getattr(self, "sweep_backend", "auto"), getattr(self, "sweep_spatial_index", "grid"), dim)

    def _roadmap_signature(self):
        """Cheap content fingerprint of the roadmap the sweep is built for."""
        return (
            len(self.nodes), len(self.edges),
            hash(tuple(tuple(n.current) for n in self.nodes)),
            hash(tuple((int(e[0]), int(e[1])) for e in self.edges)),
        )

    def set_collision_sweep(self, backend: "str | None" = None, spatial_index: "str | None" = None,
                            force: bool = False):
        """Bind ``collision_sweep`` to the current roadmap (node positions and directed edges).

        - ``backend`` / ``spatial_index``: switch ``sweep_backend`` / ``sweep_spatial_index`` (None keeps the
          current one). A change rebuilds ``collision_sweep``, then binds it if the roadmap has nodes.
        - The index depends on the roadmap only, so it is kept while the roadmap's fingerprint is unchanged;
          ``force`` rebinds regardless.

        Each bind bumps ``sweep_version``, which planner-side sweep caches key on."""
        current = (getattr(self, "sweep_backend", "auto"), getattr(self, "sweep_spatial_index", "grid"))
        wanted = (backend or current[0], spatial_index or current[1])
        if wanted != current:
            self.sweep_backend, self.sweep_spatial_index = wanted
            self.collision_sweep = self._make_collision_sweep()
            self._collision_sweep_signature = None
            if not self.nodes:
                return
        sig = self._roadmap_signature()
        if not force and getattr(self, "_collision_sweep_signature", None) == sig:
            return
        self.collision_sweep.set_graph([node.current for node in self.nodes], self.edges)
        self._collision_sweep_signature = sig
        self._node_current_tuples = [tuple(n.current) for n in self.nodes]
        self.sweep_version = getattr(self, "sweep_version", 0) + 1


    def get_collision_sweep(self, p1: tuple[float,float], p2: tuple[float,float],v=None, r: float = 0.5, use_interval: bool = False,get_time_interval: bool = False):
        """The planners' per-move disc query: the roadmap elements a disc of radius ``r`` moving ``p1 -> p2``
        at speed ``v`` touches, keyed by node coordinates.

        - ``use_interval`` and ``get_time_interval``: ``({vertex: (t1, t2)}, {(a, b): (t1, t2)})``, times
          from 0 at ``p1`` (a wait ``p1 == p2``: ``(0, inf)``).
        - ``use_interval`` only: the two key sets. Neither: the set of directed edge keys.

        Edge keys include a self-loop ``(p, p)`` for every vertex hit. ``v`` of None or 0: the move takes 1
        time unit. Not cached: planners memoize it (see ``sipp.graph_generation.cached_collision_sweep``).
        For agent types and other shapes see ``sweep_paths``."""
        if not self.use_collision_sweep:
            return None
        wait = tuple(p1) == tuple(p2)
        (vid, vt1, vt2), (src, tgt, et1, et2) = _disc_sweep_paths(
            self.collision_sweep, [p1] if wait else [p1, p2], r, v or None, goal_hold=float("inf") if wait else 0.0)[0]
        nt = self._node_current_tuples
        v_pos, v_win = [nt[i] for i in vid.tolist()], list(zip(vt1.tolist(), vt2.tolist()))
        vertices = dict(zip(v_pos, v_win))
        edges = {(nt[a], nt[b]): w for a, b, w in zip(src.tolist(), tgt.tolist(), zip(et1.tolist(), et2.tolist()))}
        edges.update(((p, p), w) for p, w in zip(v_pos, v_win))
        if use_interval and get_time_interval:
            return vertices, edges
        return (set(vertices), set(edges)) if use_interval else set(edges)

    # ------------------------------------------------------------ agent queries (Agent types, any shape)
    def sweep_paths(self, agents, paths, t0=0.0, goal_hold=0.0, other=None, clearance: str = "radius",
                    heading=None, other_heading=None, prev=None, speed=None) -> list:
        """When each agent's body covers each roadmap vertex and directed edge along its path. One entry
        per path: ``((vertex, t1, t2), (src, tgt, t1, t2))`` node-index arrays, one row per window,
        times from ``t0`` at the path's first waypoint. Ball agents use the disc code, other shapes the exact
        polytope sweep. A single move is a 2-point path; a wait a 1-point path with ``goal_hold`` (``inf``:
        it stays).

        - ``agents``: one ``Agent`` (type) for all paths, or one per path; speed and turning come from it.
        - ``paths``: a list of ``(P, D)`` waypoint arrays.
        - ``t0``, ``goal_hold``, ``heading``, ``prev``, ``speed``: one value or one per path.
          ``heading``: a fixed attitude (an angle in 2D; a yaw or 3x3 rotation in 3D).
        - ``prev``: the waypoint the agent arrives from. It then turns in place at the first waypoint for
          ``angle / max_rotation`` (covered as its bounding ball) before moving, a waiting polygon keeps its
          arrival attitude, and elements it still covers on arrival start at ``t0``. Ignored for a
          never-turning agent (``max_rotation = 0``).
        - ``speed``: overrides the agent's ``max_speed``.
        - ``other`` (an ``Agent`` or a radius): who the result will constrain; None: the body alone.
        - ``clearance="radius"``: the body grows by ``other``'s bounding radius. Exact for ball agents; for
          polygons every face moves out by it (never less than the true clearance, more near sharp corners).
        - ``clearance="minkowski"`` (``other`` a polygon / polyhedron ``Agent``): exact. A directed edge is
          covered while ``other``, anywhere on it and facing along it, would overlap the body; a vertex
          while ``other`` standing on it would, with attitude ``other_heading`` or, without it, as its
          bounding ball. A ball ``other`` uses the ``"radius"`` rule.

        Not cached: the planner memoizes what it reuses."""
        if clearance not in ("radius", "minkowski"):
            raise ValueError(f"clearance must be 'radius' or 'minkowski', got {clearance!r}")
        paths = [np.asarray(p, dtype=float).reshape(-1, self.dim) for p in as_path_list(paths)]
        n = len(paths)
        agents = [agents] * n if isinstance(agents, Agent) else list(agents)
        if len(agents) != n:
            raise ValueError(f"expected one Agent or one per path ({n}), got {len(agents)}")
        single_point = prev is not None and np.ndim(prev) == 1 and all(np.isscalar(x) for x in prev)
        prevs = [prev] * n if prev is None or single_point else list(prev)
        t0s, holds, heads, speeds = (_per_agent(x, n) for x in (t0, goal_hold, heading, speed))
        movers = [a if sp is None else Agent(a.shape, sp, a.max_rotation, a.type) for a, sp in zip(agents, speeds)]
        prevs = [None if q is None or a.max_rotation == 0 or tuple(map(float, q)) == tuple(p[0]) else
                 np.asarray(q, dtype=float) for q, a, p in zip(prevs, movers, paths)]
        # A path arriving from ``prev`` is swept with ``prev`` in front, started earlier by its travel time.
        arrive = [0.0 if q is None else (1.0 if a.max_speed is None else float(np.linalg.norm(p[0] - q)) / a.max_speed)
                  for q, a, p in zip(prevs, movers, paths)]
        full = [p if q is None else np.vstack([q, p]) for q, p in zip(prevs, paths)]
        out = self._sweep_batch(movers, full, [t - dt for t, dt in zip(t0s, arrive)], holds, other, clearance,
                                heads, other_heading)
        for i, (dt, t) in enumerate(zip(arrive, t0s)):
            if dt > 0.0:                                         # drop what ended before the arrival
                (vid, t1, t2), (src, tgt, e1, e2) = out[i]
                kv, ke = t2 > t + 1e-12, e2 > t + 1e-12
                out[i] = ((vid[kv], np.maximum(t1[kv], t), t2[kv]),
                          (src[ke], tgt[ke], np.maximum(e1[ke], t), e2[ke]))
        return out

    def _sweep_batch(self, agents, paths, t0s, holds, other, clearance, heads, other_heading) -> list:
        """``sweep_paths`` without ``prev``: one result per path, absolute times."""
        n = len(paths)
        backend, eps = self.collision_sweep, self.collision_sweep.STRICT_EPS
        shapes, speeds, turns = ([getattr(a, f) for a in agents] for f in ("shape", "max_speed", "max_rotation"))
        grow = 0.0
        if isinstance(other, Agent):
            body = make_body(other.shape, self.dim)
            if clearance == "minkowski" and body.ball_r <= 0:
                return body_roadmap_hits_minkowski(backend, paths, shapes, other.shape, speeds, t0s, turns,
                                                   holds, heads, eps, other_heading)
            grow = body.radius
        elif other is not None:
            grow = float(other)
        ball_r = [make_body(sh, self.dim).ball_r for sh in shapes]
        out = [None] * n
        for is_ball in (True, False):
            idx = [i for i in range(n) if (ball_r[i] > 0) == is_ball]
            take = lambda xs: [xs[i] for i in idx]
            if not idx:
                continue
            if is_ball:
                res = _disc_sweep_paths(backend, take(paths), [ball_r[i] + grow for i in idx], take(speeds),
                                        take(t0s), take(turns), take(holds))
            else:
                res = _assemble_hits(backend, len(idx), *body_roadmap_hits(
                    backend, take(paths), take(shapes), take(speeds), take(t0s), take(turns), take(holds),
                    take(heads), eps, grow), True)
            for i, r in zip(idx, res):
                out[i] = r
        return out

    def agent_conflicts(self, agents, paths, t0=0.0, goal_hold=0.0, heading=None, pairs=None,
                        first_only: bool = False, as_arrays: bool = False):
        """When two agents' bodies overlap along their paths: ``{(i, j): [(t1, t2), ...]}`` with ``i < j``.
        ``first_only``: ``{(i, j): first contact time}``. ``as_arrays``: ``(i, j, t1, t2)`` arrays.
        ``pairs``: only check these agent pairs. ``agents``, ``paths``, ``t0``, ``goal_hold`` and ``heading``
        as in ``sweep_paths``. Exact; no roadmap is needed."""
        paths = as_path_list(paths)
        agents = [agents] * len(paths) if isinstance(agents, Agent) else list(agents)
        shapes, speeds, turns = ([getattr(a, f) for a in agents] for f in ("shape", "max_speed", "max_rotation"))
        return self.collision_sweep.agent_conflicts(paths, shapes, speeds, t0, turns, goal_hold, heading, pairs,
                                                    first_only, as_arrays)

    def get_constraint_segment(self, p1a: Tuple[float, ...], p1b: Tuple[float, ...], p2a: Tuple[float, ...],
                               p2b: Tuple[float, ...], v=None, r: float = 0.5, agents=None, heading=None) -> bool:
        """Whether two agents collide while moving ``p1a -> p1b`` and ``p2a -> p2b`` over the same time step,
        both starting at 0: ``agent_conflicts`` on the pair.

        - ``agents``: one ``Agent`` for both moves or one per move, any shape; each move takes its length /
          ``max_speed`` (1 time unit if ``max_speed`` is None). Default: discs of radius ``r`` at speed ``v``
          (None or 0: 1 time unit). A moved agent is gone once it arrives.
        - A wait (``a == b``) lasts the whole step (the longest move's duration).
        - ``heading``: one fixed attitude per move (None: a move faces along itself). A waiting polygon
          without a heading is checked as its bounding ball, since its attitude is unknown."""
        if agents is None:
            if r <= 0:
                return False
            agents = Agent.disc(r, v or None)
        agents = [agents] * 2 if isinstance(agents, Agent) else list(agents)
        heads = [None, None] if heading is None else list(heading)
        moves = [(tuple(p1a), tuple(p1b)), (tuple(p2a), tuple(p2b))]
        waits = [a == b for a, b in moves]
        for i, (ag, w) in enumerate(zip(agents, waits)):
            if w and heads[i] is None and not ag.is_ball:
                agents[i] = Agent(ag.radius, ag.max_speed, ag.max_rotation, ag.type)
        dur = [1.0 if ag.max_speed is None else float(np.linalg.norm(np.subtract(b, a))) / ag.max_speed
               for ag, (a, b), w in zip(agents, moves, waits) if not w]
        step = max(dur, default=1.0)
        paths = [np.array([a]) if w else np.array([a, b]) for (a, b), w in zip(moves, waits)]
        holds = [step if w else 0.0 for w in waits]
        return bool(self.agent_conflicts(agents, paths, goal_hold=holds, first_only=True,
                                         heading=None if all(h is None for h in heads) else heads))

    def clear_data(self):
        self.start_to_all_edges_dict = {}
        self.goal_to_all_edges_dict = {}
        self._start_edges_khop = None
        self._goal_edges_khop = None
        self.node_sampler.reset()
        self.roadmap.reset()
        self.start = []
        self.goal = []
        self.obstacles = []
        self.inflation_radius = 0.0

    def __str__(self) -> str:
        return "Graph Sampler"

    PICKLE_VERSION = 2

    def save_graph_sampler(self, path: str):
        """Pickle the map as plain data (version 2): the settings, the
        obstacles, ``node_sampler.state_dict()`` (nodes, kinds,
        ``sampling_dist_dict``) and ``roadmap.state_dict()`` (nodes, adjacency,
        ``roadmap_config``). Neither object is pickled itself: both reference
        the map (and the CGAL sweep, which cannot be pickled)."""
        data = {
            "version": self.PICKLE_VERSION,
            "start": self.start,
            "goal": self.goal,
            "use_discrete_space": self.use_discrete_space,
            "node_sampler": self.node_sampler.state_dict(),
            "roadmap": self.roadmap.state_dict(),
            # ndarray, not list-of-tuples: pickling 4.5M tuples costs ~10s / ~200MB, the array
            # milliseconds / ~70MB. set_obstacles() accepts either on load.
            "obstacles": np.asarray(self.obstacles, dtype=np.int64) if len(self.obstacles) else [],
            "obs_size": self.obs_size,
            "inflation_radius": self.inflation_radius,
            "use_collision_sweep": self.use_collision_sweep,
            "record_sweep": self.record_sweep,
            "use_exact_collision_check": self.use_exact_collision_check,
            "sweep_backend": getattr(self, "sweep_backend", "auto"),
            "sweep_spatial_index": getattr(self, "sweep_spatial_index", "grid"),
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_graph_sampler(self, path: str, args: dict = {}):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._load_from_dict(data, args)

    def _load_from_dict(self, data: dict, args: dict = {}):
        """Restore a pickle written by ``save_graph_sampler``. Version 2 holds
        ``node_sampler`` / ``roadmap`` state dicts; the legacy layout holds
        ``nodes`` (Node list), ``road_map`` / ``road_map_edge_weights``
        (nested neighbour lists), the config and ``sampling_dist_dict`` at the
        top level and optional ``boundary_points``; it is still read
        (``grid_points``, when present, is ignored: grid nodes are plain
        samples now)."""
        self.use_discrete_space = data["use_discrete_space"]
        self.set_start(data["start"])
        self.set_goal(data["goal"])

        self.obstacles = data["obstacles"]
        self.obs_size = data["obs_size"] if "obs_size" in data else 0.5
        self.inflation_radius = data["inflation_radius"]
        self.set_obstacles(self.obstacles)
        self.inflate_obstacles(self.inflation_radius)

        if "node_sampler" in data:              # version 2: plain-array state dicts carry the config and sampling parameters
            self.node_sampler.load_state_dict(data["node_sampler"])
            self.roadmap = Roadmap(self, config=data["roadmap"].get("config"))
            self.roadmap.load_state_dict(data["roadmap"])
        else:                                   # legacy layout: config and sampling parameters at the top level
            if "roadmap_config" in data:
                roadmap_config = data["roadmap_config"]
            else:                               # pickles written before roadmap_config existed
                roadmap_config = {"sample_num": data["sample_num"], "num_neighbors": data["num_neighbors"],
                                  "min_edge_length": data["min_edge_length"], "max_edge_length": data["max_edge_length"]}
            self.node_sampler.set_sampling_dist_dict(data.get("sampling_dist_dict"))
            self.node_sampler.set_nodes(data["nodes"])      # nodes / pos_index_dict are views over the sampler
            self.node_sampler.register_points('start_points', self.start)
            self.node_sampler.register_points('goal_points', self.goal)
            self.node_sampler.register_points('boundary_points', data.get("boundary_points", []))
            self.roadmap = Roadmap(self, config=roadmap_config)
            self.roadmap.set_nodes(self.nodes)
            self.roadmap.set_adjacency(data["road_map"], data.get("road_map_edge_weights"))
        # nodes / pos_index_dict / edges / edge_weights / road_map are views over the sampler and roadmap.
        # The start/goal edge lists are built lazily by get_start/goal_nodes_with_all_edges.

        # Set up constraint sweep
        # Maps saved before the rename store ``use_constraint_sweep``.
        self.use_collision_sweep = args.get("use_collision_sweep", data.get("use_collision_sweep", data.get("use_constraint_sweep", True)))
        self.record_sweep = data["record_sweep"]
        self.use_exact_collision_check = data["use_exact_collision_check"]
        self.sweep_backend = args.get("sweep_backend", data.get("sweep_backend", "auto"))
        self.sweep_spatial_index = args.get("sweep_spatial_index", data.get("sweep_spatial_index", "grid"))
        self.collision_sweep = self._make_collision_sweep()
        if self.use_collision_sweep:
            self.set_collision_sweep()

        points = np.array([samp.current for samp in self.nodes])
        self.sample_kd_tree = KDTree(points)

    def rotate(self, k: int, axes: Tuple[int, int] = (0, 1)):
        """
        Rotate the grid type map and all node / start / goal / obstacle positions
        in-place by k * 90 degrees counterclockwise in the plane of ``axes``.

        ``axes`` is a pair of distinct dimension indices (same convention as ``numpy.rot90``).
        When k is 1 or 3, the world extents along those two axes are swapped.
        Road map topology and edge weights are unchanged (rotation preserves
        distances); the roadmap's node array and every sampler kind index follow
        their nodes to the new positions.

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

        if self.use_collision_sweep and len(self.edges) > 0:
            self.set_collision_sweep()

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
            "road_map": pruned_road_map,
            "road_map_edge_weights": pruned_road_map_edge_weights,
            "use_collision_sweep": self.use_collision_sweep,
            "record_sweep": self.record_sweep,
            "use_exact_collision_check": self.use_exact_collision_check,
            "sweep_backend": getattr(self, "sweep_backend", "auto"),
            "sweep_spatial_index": getattr(self, "sweep_spatial_index", "grid"),
            "sampling_dist_dict": self.node_sampler.sampling_dist_dict,
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
            use_collision_sweep=self.use_collision_sweep,
            record_sweep=self.record_sweep,
            use_exact_collision_check=self.use_exact_collision_check,
            sweep_backend=getattr(self, "sweep_backend", "auto"),
            sweep_spatial_index=getattr(self, "sweep_spatial_index", "grid"),
            sampling_dist_dict=self.node_sampler.sampling_dist_dict,
        )
        pruned_sampler._load_from_dict(pruned_data)
        return pruned_sampler
