import numpy as np
from path_planning.common.environment.map.numbaUtil import _world_min_wall_distance, convert_np_to_tuple
from path_planning.common.environment.node import Node
from typing import List, Tuple, TYPE_CHECKING
from scipy.stats import qmc
from typing import Final
from path_planning.common.environment.node_sampler.node_sampler_util import validate_sampling_type, density_probability, boundary_vertices
if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase

class NodeSampler:
    """Node samplers over a ``GraphSamplerBase``.

    Holds only the map; everything else is read from it on each call, so the
    sampler stays correct after ``rotate``, an ESDF recompute or a new
    inflation radius (the map's ``shape``, ``bounds`` and ``inflated_esdf``
    all change then).
    """
    def __init__(self, graph_map: "GraphSamplerBase"):
        self.graph_map = graph_map
        self.sampling_dist_dict = {
            'halton': {
                'd_min': 0.3,
                'd_opt': 0.4,
                'sigma': 0.5,
                'floor': 0.2,
            },
            'distributed': {
                'samp_from_prob_map_ratio': 0.5,
            },
        }
        self.reset()

    def reset(self) -> None:
        """Forget every sampled node (and the Halton sequence position).

        The state is rebound, not cleared in place, so a map that received
        the previous lists from ``apply_points`` keeps them."""
        self._halton_sampler = None
        self._nodes: List[Node] = []
        self._pos_index_dict: dict = {}        # position tuple -> index in nodes
        # Node -> index in nodes, one dict per node kind (a node can be in several)
        self._sample_points_index: dict = {}
        self._custom_points_index: dict = {}
        self._grid_points_index: dict = {}
        self._boundary_points_index: dict = {}
        self._start_points_index: dict = {}
        self._goal_points_index: dict = {}
        self._points_index_dict: dict = {
            'sample_points': self._sample_points_index,
            'custom_points': self._custom_points_index,
            'grid_points': self._grid_points_index,
            'boundary_points': self._boundary_points_index,
            'start_points': self._start_points_index,
            'goal_points': self._goal_points_index,
        }

    # ------------------------------------------------------------ read access
    # The sampled state is private; these expose it read-only. ``nodes`` and
    # the dicts are the live objects (not copies): ``apply_points`` hands the
    # same objects to the map, and ``reset`` rebinds rather than clears them.
    @property
    def nodes(self) -> List[Node]:
        """Every node sampled so far, in index order."""
        return self._nodes

    @property
    def num_total_nodes(self) -> int:
        return len(self._nodes)

    @property
    def pos_index_dict(self, pos: tuple | np.ndarray | None = None) -> dict:
        """Position tuple -> index in ``nodes``."""
        if pos is not None:
            if isinstance(pos, tuple):
                return self._pos_index_dict[pos]
            elif isinstance(pos, np.ndarray):
                return [self._pos_index_dict[p] for p in convert_np_to_tuple(pos)]
            else:
                raise ValueError(f"Invalid pos type: {type(pos)} for Node Sampler")
        return self._pos_index_dict

    @property
    def points_index_dict(self, index_type: str = '') -> dict:
        """Kind name (``'sample_points'``, ``'custom_points'``, ``'grid_points'``,
        ``'boundary_points'``, ``'start_points'``, ``'goal_points'``) -> its
        ``Node -> index`` dict."""
        if index_type in self._points_index_dict and index_type != '':
            return self._points_index_dict[index_type]
        elif index_type != '':
            raise ValueError(f"Invalid index type: {index_type} for Node Sampler")
        return self._points_index_dict

    @property
    def pos_points(self) -> np.ndarray:
        """(N, dim) positions of ``nodes``."""
        return np.array([node.current for node in self._nodes], dtype=np.float64).reshape(-1, self.dim)

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(self.graph_map.bounds, dtype=float)

    @property
    def dim(self) -> int:
        return self.graph_map.dim

    @property
    def shape(self) -> tuple:
        return tuple(self.graph_map.shape)

    @property
    def use_discrete_space(self) -> bool:
        return self.graph_map.use_discrete_space

    @property
    def inflated_esdf(self) -> np.ndarray:
        return self.graph_map.inflated_esdf

    @property
    def inflation_radius(self) -> float:
        return self.graph_map.inflation_radius

    @property
    def resolution(self) -> float:
        return self.graph_map.resolution

    def path_world_to_map(self, path, discrete: bool = True):
        return self.graph_map.path_world_to_map(path, discrete)

    def path_map_to_world(self, path):
        return self.graph_map.path_map_to_world(path)

    def is_expandable(self, point, src_point=None):
        return self.graph_map.is_expandable(point, src_point)

    def get_nodes(self, points: List[tuple] | np.ndarray | None = None, index_dict: dict | None = None, index_type: str = '') -> List[Node]:
        """Get nodes from ``points`` or ``index_dict``."""
        nodes = []
        if points is not None:
            points = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
            for pt in convert_np_to_tuple(points):
                if pt in self._pos_index_dict:
                    nodes.append(self._nodes[self._pos_index_dict[pt]])
                else:
                    raise ValueError(f"Invalid point: {pt} for Node Sampler")
            return nodes
        
        if index_dict is not None:
            for index in index_dict.values():
                if 0 <= index < len(self._nodes):
                    nodes.append(self._nodes[index])
                else:
                    raise ValueError(f"Invalid index: {index} for Node Sampler")
            return nodes

        if index_type in self._points_index_dict and index_type != '':
            return [self._nodes[index] for index in self._points_index_dict[index_type].values()]
        elif index_type != '':
            raise ValueError(f"Invalid index type: {index_type} for Node Sampler")

        return self._nodes   

    def set_nodes(self, nodes: List[Node]) -> List[Node]:
        """Replace the node list, e.g. with the nodes a roadmap builder produced
        (the planar family derives new points, Lattice snaps, start/goal may be
        appended). ``pos_index_dict`` is rebuilt (first index wins for a
        repeated position) and every kind index is re-keyed by position; a
        node whose position is gone is dropped from the kind indices."""
        self._nodes = list(nodes)
        self._pos_index_dict = {}
        for i, node in enumerate(self._nodes):
            self._pos_index_dict.setdefault(tuple(node.current), i)
        for index in self._points_index_dict.values():          # same dict objects as self._*_points_index
            kept = {node: self._pos_index_dict[tuple(node.current)]
                    for node in index if tuple(node.current) in self._pos_index_dict}
            index.clear()
            index.update(kept)
        return self._nodes

    def set_positions(self, points) -> List[Node]:
        """Move every node to ``points[i]`` (same count, same order, e.g.
        after ``rotate``), keeping each node's clearance and every kind index:
        the dicts are re-keyed by node index, so a start/goal/grid/... entry
        follows its node to the new position."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
        if len(pts) != len(self._nodes):
            raise ValueError(f"set_positions: {len(pts)} points for {len(self._nodes)} nodes")
        by_index = {kind: {i: None for i in index.values()} for kind, index in self._points_index_dict.items()}
        self._nodes = [Node(p, None, 0, 0, old.max_clearance) for p, old in zip(convert_np_to_tuple(pts), self._nodes)]
        self._pos_index_dict = {}
        for i, node in enumerate(self._nodes):
            self._pos_index_dict.setdefault(tuple(node.current), i)
        for kind, index in self._points_index_dict.items():
            index.clear()
            index.update({Node(self._nodes[i].current, None, 0, 0): i for i in by_index[kind] if i < len(self._nodes)})
        return self._nodes

    def add_node(self, node: Node) -> Tuple[int, bool]:
        """Register ``node``: ``(index, True)`` when its position is new and it
        was appended, ``(index, False)`` when a node already sits there."""
        key = tuple(node.current)
        index = self._pos_index_dict.get(key)
        if index is not None:
            return index, False
        index = len(self._nodes)
        self._nodes.append(node)
        self._pos_index_dict[key] = index
        return index, True

    def make_points(self, points) -> List[Node]:
        """``Node`` per world point, with its clearance (``inflated_esdf`` of the
        cell the point falls in) as the node radius. ``points`` may also be a
        list of ``Node``, whose positions are used."""
        if len(points) and isinstance(points[0], Node):
            points = [node.current for node in points]
        points = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
        if len(points) == 0:
            return []
        cells = self.path_world_to_map(points)
        radii = self.inflated_esdf[tuple(cells.T)]
        return [Node(point, None, 0, 0, float(radius))
                for point, radius in zip(convert_np_to_tuple(points), radii)]

    def _add_points(self, points, index_dict: dict) -> int:
        """Register a node per point in ``index_dict`` (``Node -> index``),
        reusing nodes already at that position. Returns how many were new."""
        added = 0
        for node in self.make_points(points):
            index, is_new = self.add_node(node)
            index_dict[node] = index
            added += is_new
        return added

    def random_sampling(self,
        n_samples,
    ):
        """``n_samples`` uniform points over the map bounds, keeping those in
        expandable cells. Discrete space snaps them to cell centres.

        Returns:
            (nodes, points): parallel lists of accepted nodes and world tuples.
        """
        normalized_points = np.random.random((n_samples,self.dim))
        world_points = normalized_points * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
        grid_points = self.path_world_to_map(world_points)
        expandable_mask = self.is_expandable(grid_points)

        accepted_cells = grid_points[expandable_mask]
        accepted_points = self.path_map_to_world(accepted_cells) if self.use_discrete_space else world_points[expandable_mask]
        return accepted_points

    def distributed_sampling(self,
        n_samples : Final[int],
        prob_map: Final[np.ndarray],
        samp_from_prob_map_ratio: Final[float] = 0.5,
    ):
        """``n_samples`` cells: a ``samp_from_prob_map_ratio`` share drawn from
        ``prob_map`` (one weight per cell), the rest from its complement
        ``1 - p``. Expandable cells are kept; discrete space uses the cell
        centre, continuous space adds a uniform +-0.5 cell jitter.

        Args:
            n_samples: number of samples to draw
            prob_map: probability map
            samp_from_prob_map_ratio: ratio of samples to draw from prob_map

        Returns:
            (points, num_rejected_weighted): accepted world points (K, dim),
            prob-map samples first, and how many prob-map samples were
            rejected (``sample_points`` uses it to re-balance the ratio).
        """
        n_samples = int(n_samples)
        num_prob_map_samples = int(n_samples * samp_from_prob_map_ratio)
        num_unchosen_samples = n_samples - num_prob_map_samples

        prob_map = np.asarray(prob_map, dtype=np.float64)
        if prob_map.shape != tuple(self.shape):
            raise ValueError(f"prob_map shape {prob_map.shape} does not match map shape {tuple(self.shape)}")
        prob_map_flat = prob_map.ravel()
        prob_map_flat = prob_map_flat / prob_map_flat.sum()

        cells = np.zeros((n_samples, self.dim), dtype=np.int64)
        if num_prob_map_samples > 0:
            idx = np.random.choice(prob_map_flat.size, size=num_prob_map_samples, p=prob_map_flat)
            cells[:num_prob_map_samples] = np.column_stack(np.unravel_index(idx, prob_map.shape))

        if num_unchosen_samples > 0:
            unchosen_flat = 1.0 - prob_map_flat
            total = unchosen_flat.sum()
            # Complement is all zeros when one cell holds all the mass: use uniform.
            unchosen_flat = unchosen_flat / total if total > 0 else np.full_like(prob_map_flat, 1.0 / prob_map_flat.size)
            idx = np.random.choice(prob_map_flat.size, size=num_unchosen_samples, p=unchosen_flat)
            cells[num_prob_map_samples:] = np.column_stack(np.unravel_index(idx, prob_map.shape))

        expandable_mask = self.is_expandable(cells)
        accepted_cells = cells[expandable_mask]
        num_rejected_weighted = int(num_prob_map_samples - expandable_mask[:num_prob_map_samples].sum())

        if self.use_discrete_space:
            accepted_points = self.path_map_to_world(accepted_cells)
        else:
            jitter = np.random.random((len(accepted_cells), self.dim)) - 0.5
            accepted_points = self.path_map_to_world(accepted_cells + jitter)
        return accepted_points, num_rejected_weighted

    def sample_halton_points(self,
            n_samples : Final[int],
            d_min: Final[float] = 0.3,
            d_opt: Final[float] = 0.4,
            sigma: Final[float] = 0.5,
            floor: Final[float] = 0.2,
            halton_sampler=None,
            oversample: Final[int] = 1,
        ):
        """``n_samples * oversample`` Halton points over the map bounds, thinned
        by ``density_probability`` of their wall distance, then filtered to
        expandable cells. Discrete space snaps them to cell centres.

        Pass the returned ``halton_sampler`` back in to continue the sequence;
        a fresh sampler repeats the same points.

        Args:
            n_samples: number of samples to draw
            d_min: minimum wall distance
            d_opt: optimal wall distance
            sigma: sigma of the Gaussian
            floor: floor of the Gaussian
            halton_sampler: halton sampler
                - if None, a new halton sampler is created
                - if a halton sampler is passed, it is used to continue the sequence
            oversample: oversample factor
                - number of times to oversample the samples

        Returns:
            (points): accepted world tuples.
        """
        if halton_sampler is None:
            halton_sampler = qmc.Halton(d=self.dim, scramble=False)

        n_draw = int(max(1, n_samples * max(1, oversample)))
        raw_samples = halton_sampler.random(n_draw)
        world_point_samples = qmc.scale(raw_samples, self.bounds[:, 0], self.bounds[:, 1])

        dists = _world_min_wall_distance(
            np.ascontiguousarray(world_point_samples, dtype=np.float64),
            self.graph_map._esdf_flat(),
            float(self.inflation_radius),
            self.bounds,
            float(self.resolution),
            self.graph_map._shape_array,
        )
        probs = density_probability(dists, d_min, d_opt, sigma, floor)
        halton_accept = np.random.random(len(world_point_samples)) < probs
        world_points = world_point_samples[halton_accept]
        grid_points = self.path_world_to_map(world_points)
        expandable_mask = self.is_expandable(grid_points)

        accepted_cells = grid_points[expandable_mask]
        accepted_points = self.path_map_to_world(accepted_cells) if self.use_discrete_space else world_points[expandable_mask]
        return accepted_points, halton_sampler

    def sample_grid_points(self):
        """One node at the centre of every expandable cell.

        Returns:
            (points): accepted world tuples.
        """
        grid_shape = self.shape
        grid_ranges = [np.arange(grid_shape[d]) for d in range(self.dim)]
        grid_points = np.array(np.meshgrid(*grid_ranges, indexing='ij')).reshape(self.dim,-1).T
        world_grid_points = self.path_map_to_world(grid_points)

        expandable_mask = self.is_expandable(grid_points)
        accepted_points = world_grid_points[expandable_mask]
        return accepted_points

    def sample_boundary_points(self, include_map_edge: Final[bool] = True,
                            corners_only: Final[bool] = False, inset: Final[float] = 1e-3):
        """Nodes on the outline of free space: its corners and one point every
        ``resolution`` along its edges (see ``boundary_vertices``).

        Each node is moved ``inset`` cells from its vertex into free space so it
        lies inside a free cell; a point exactly on a corner can round into a
        blocked cell and lose all its edges. ``inset=0`` keeps the exact
        vertices. ``corners_only`` drops the edge vertices.

        Returns:
            (points): accepted world tuples.
        """
        vertices, is_corner, free_dir = boundary_vertices(self.graph_map.type_map.data, include_map_edge)
        if corners_only:
            vertices, free_dir = vertices[is_corner], free_dir[is_corner]
        if vertices.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        accepted_points = self.path_map_to_world(vertices + inset * free_dir)
        return accepted_points

    def sample_nodes(self,
            n_samples: int = 0,
            start_list: List[tuple] | np.ndarray | None = None,
            goal_list: List[tuple] | np.ndarray | None = None,
            pts_list: List[Node] | List[tuple] | np.ndarray | None = None,
            generate_grid_nodes: bool = False,
            generate_boundary_nodes: bool = False,
            prob_map: np.ndarray | None = None,
            samp_from_prob_map_ratio: float | None = None,
            max_outer_iters: int = 20,
            sampling_dist_dict: dict | None = None,
            sampling_type: str = 'random',
        ) -> List[Node]:
        """Sample nodes and add them to this sampler (``nodes`` and the index
        dicts accumulate across calls; ``reset`` clears them). Copy them onto a
        map with ``apply_points``.

        ``n_samples`` random nodes come from ``sample_halton_points``
        (``sampling_type='halton'``, or any roadmap type containing
        ``halton``), ``distributed_sampling`` (``'distributed'``, needs
        ``prob_map``) or ``random_sampling`` (``'random'``); rounds repeat
        until the quota is met or ``max_outer_iters`` rounds have run.
        ``sampling_dist_dict`` overrides the defaults in
        ``self.sampling_dist_dict`` (``halton``: d_min, d_opt, sigma, floor;
        ``distributed``: samp_from_prob_map_ratio), and the
        ``samp_from_prob_map_ratio`` argument overrides both. Then, in order:
        ``pts_list`` (custom points), every expandable cell centre
        (``generate_grid_points``), the outline of free space
        (``generate_boundary_points``), ``start_list`` and ``goal_list``. A
        point already present is reused, not added twice, but is still
        recorded in that kind's index dict.

        Returns:
            nodes: this sampler's full node list.
        """
        sampling_type = validate_sampling_type(sampling_type)
        is_halton = sampling_type == 'halton'
        is_distributed = sampling_type == 'distributed'
        if is_distributed and prob_map is None:
            raise ValueError("sampling_type='distributed' needs a prob_map")
        sampling_dist_dict = sampling_dist_dict or {}
        halton_cfg = {**self.sampling_dist_dict['halton'], **sampling_dist_dict.get('halton', {})}
        distributed_cfg = {**self.sampling_dist_dict['distributed'], **sampling_dist_dict.get('distributed', {})}
        if samp_from_prob_map_ratio is None:
            samp_from_prob_map_ratio = distributed_cfg['samp_from_prob_map_ratio']

        n_samples = int(n_samples)
        target = len(self._nodes) + n_samples
        oversample = 1
        num_rejected_weighted_total = 0
        for outer_iter in range(max_outer_iters + 1):
            if len(self._nodes) >= target:
                break
            if outer_iter == max_outer_iters:
                print(
                    f"Warning: sample_points hit max_outer_iters={max_outer_iters} "
                    f"with {n_samples - (target - len(self._nodes))}/{n_samples} nodes; continuing with what was collected."
                )
                break

            needed = target - len(self._nodes)
            num_rejected_weighted = 0
            if is_halton:
                # The sequence continues across rounds and calls; restarting it
                # would redraw the same points.
                points, self._halton_sampler = self.sample_halton_points(
                    needed,
                    halton_cfg['d_min'],
                    halton_cfg['d_opt'],
                    halton_cfg['sigma'],
                    halton_cfg['floor'],
                    oversample=oversample,
                    halton_sampler=self._halton_sampler,
                )
                n_draw = max(1, needed * max(1, oversample))
            elif is_distributed:
                points, num_rejected_weighted = self.distributed_sampling(needed, prob_map, samp_from_prob_map_ratio)
                n_draw = needed
            else:
                points = self.random_sampling(needed)
                n_draw = needed

            # The samplers return only collision-free points; drop duplicates
            # (snapped cells repeat in discrete space) and stop at the quota.
            accepted_this_round = 0
            for node in self.make_points(points):
                index, is_new = self.add_node(node)
                if not is_new:
                    continue
                self._sample_points_index[node] = index
                accepted_this_round += 1
                if len(self._nodes) == target:
                    break

            # Update oversample / ratio for the next round.
            if is_halton:
                # Draw roughly needed / rate raw Halton points next time.
                rate = accepted_this_round / n_draw
                oversample = min(64, max(2, oversample * 2 if rate <= 0 else int(np.ceil(1.0 / rate))))
            elif is_distributed:
                # Share of prob-map samples rejected so far decides how much of
                # the next draw comes from the prob map again.
                num_rejected_weighted_total += num_rejected_weighted
                samp_from_prob_map_ratio = max(0.0, min(1.0, num_rejected_weighted_total / n_samples))

        if pts_list is not None:
            self._add_points(pts_list, self._custom_points_index)

        if generate_grid_nodes:
            # A snapped random sample already in a cell is reused as its grid node.
            self._add_points(self.sample_grid_points(), self._grid_points_index)

        if generate_boundary_nodes:
            # Vertices on the outline of free space: the corners of the
            # free/blocked interface and one point every resolution along its
            # edges, in world units on the interface itself (not on the
            # cell-centre lattice). They are not filtered for expandability;
            # edge-level collision checks in the roadmap builders decide
            # connectivity.
            self._add_points(self.sample_boundary_points(), self._boundary_points_index)

        if start_list is not None:
            self._add_points(start_list, self._start_points_index)
        if goal_list is not None:
            self._add_points(goal_list, self._goal_points_index)
        return self._nodes