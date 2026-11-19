"""``CollisionSweep``: the collision sweep to call. ``CollisionSweep(backend, spatial_index, dim)`` returns this
class with the chosen backend as its parent: ``ShapelyCollisionSweep`` (``ShapelyCollisionHetero``) or
``CGALCollisionSweep`` (``CGALCollisionHetero``). Both backends build on ``sweep.collision_base.CollisionBase``
and serve ball agents (discs / spheres) as well as polygons / polyhedra; results are the same either way.
"""
from __future__ import annotations

from path_planning.common.world.collision_checking.cgal.cgal_collision import SPATIAL_INDEXES
from path_planning.common.world.collision_checking.cgal.cgal_collision_hetero import CGALCollisionHetero
from path_planning.common.world.collision_checking.shapely.shapely_collision_hetero import ShapelyCollisionHetero

BACKENDS = ("auto", "shapely", "cgal")


def resolve_backend(backend: str = "auto", dim: int = 2) -> str:
    """``"shapely"`` or ``"cgal"`` for ``backend`` (``"auto"``: Shapely in 2D, CGAL in 3D)."""
    if backend == "auto":
        return "shapely" if dim == 2 else "cgal"
    if backend not in BACKENDS:
        raise ValueError(f"Unknown sweep backend {backend!r}; expected one of {BACKENDS}.")
    return backend


class CollisionSweep:
    """The roadmap collision sweep; instantiating it gives the backend chosen by ``backend``.

    - ``backend``: ``"auto"`` (Shapely in 2D, CGAL in 3D), ``"shapely"`` or ``"cgal"``.
    - ``spatial_index`` (CGAL's candidate search): ``"grid"`` (fastest for evenly spread roadmaps) or
      ``"rtree"`` (adapts to uneven density). Shapely always uses its STR-tree.
    - ``dim``: the roadmap dimension, for ``"auto"``.

    ``backend`` names the chosen one (``"shapely"`` / ``"cgal"``)."""

    backend: str = ""

    def __new__(cls, backend: str = "auto", spatial_index: str = "grid", dim: int = 2):
        if cls is CollisionSweep:
            cls = _VARIANTS[resolve_backend(backend, dim)]
        return super().__new__(cls)

    def __init__(self, backend: str = "auto", spatial_index: str = "grid", dim: int = 2):
        if backend not in ("auto", self.backend):
            raise ValueError(f"{type(self).__name__} is the {self.backend!r} backend, got backend={backend!r}")
        if spatial_index not in SPATIAL_INDEXES:
            raise ValueError(f"spatial_index must be one of {SPATIAL_INDEXES}, got {spatial_index!r}")
        if self.backend == "cgal":
            super().__init__(spatial_index=spatial_index)
        else:
            super().__init__()
            self.spatial_index = spatial_index       # recorded only; Shapely has one index

    def __repr__(self):
        return f"{type(self).__name__}(spatial_index={self.spatial_index!r})"


class ShapelyCollisionSweep(CollisionSweep, ShapelyCollisionHetero):
    """``CollisionSweep`` on the Shapely backend (2D)."""
    backend = "shapely"


class CGALCollisionSweep(CollisionSweep, CGALCollisionHetero):
    """``CollisionSweep`` on the CGAL backend (2D and 3D)."""
    backend = "cgal"


_VARIANTS = {"shapely": ShapelyCollisionSweep, "cgal": CGALCollisionSweep}
