from typing import Any, List, Tuple
import numpy as np
from path_planning.common.environment.node import Node
from python_motion_planning.common.env.map.grid import Grid
from python_motion_planning.common import TYPES
from python_motion_planning.common.env.map.grid import _grid_is_expandable, _grid_path_map_to_world, _grid_path_world_to_map_int, _grid_path_world_to_map_float
from path_planning.common.environment.map.numbaUtil import _world_line_of_sight, _world_in_collision, _grid_vector_is_expandable, convert_np_to_tuple
from path_planning.common.utils.geometry import GeometryNumpy as Geometry

ROADMAP_TYPES = set(["grid", "prm", "cdt", "rrg", "midpoints", "centroids", "voronoi", "halton"])


def validate_roadmap_type(roadmap_type: str) -> bool:
    return roadmap_type in ROADMAP_TYPES


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
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.obstacles = []
        self.obs_size = 0.5
        self._inflated_esdf = None          # inflated esdf cache
        self._inflated_esdf_source = None   # inflated esdf source to update inflated esdf

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

    def get_inflated_obstacles(self):
        return [tuple(int(p) for p in row) for row in np.argwhere(self.type_map.data == TYPES.INFLATION)]

    def set_obstacle_map(self, obstacles: np.ndarray | List[tuple]):
        obstacles = np.array(obstacles).reshape(-1, self.dim)
        if len(obstacles) == 0:
            return
        if len(obstacles[0]) == 2:
            self.type_map[obstacles[:,0], obstacles[:,1]] = TYPES.OBSTACLE
        elif len(obstacles[0]) == 3:
            self.type_map[obstacles[:,0], obstacles[:,1], obstacles[:,2]] = TYPES.OBSTACLE
        else:
            raise ValueError(f"Unsupported dimensions: {len(obstacles.shape)}")

    def get_obstacle_map(self) -> np.ndarray:
        return self.type_map.data == TYPES.OBSTACLE

    def get_distance(self, p1, p2=None, type: str = 'Euclidean') -> float | List[Tuple[tuple, float]]:
        """Straight-line distance between world points ``p1`` and ``p2``
        (``type`` ``'Euclidean'`` or ``'Manhattan'``, both points required).
        ``GraphSampler.get_distance`` adds ``type='dijkstra'``, the
        shortest-path length over its roadmap.
        """
        if type not in ('Euclidean', 'Manhattan'):
            raise ValueError(f"get_distance: unknown type {type!r}; expected 'Euclidean', 'Manhattan' or 'dijkstra'")
        if p2 is None:
            raise ValueError("get_distance: p2 is required for a straight-line distance")
        return Geometry.dist(p1, p2, type)

    def get_neighbors(self, node: Node | int, **kwargs) -> List[Node]:
        """Neighbours of a node along a roadmap. The base map has no roadmap;
        ``GraphSampler.get_neighbors`` reads ``road_map``."""
        raise ValueError("No neighbors")

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
        self.start = []
        self.goal = []
        self.obstacles = []
        self.inflation_radius = 0.0
