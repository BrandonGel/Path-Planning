"""Polygon / polyhedron sweep collision checker over a ``GraphSampler`` roadmap (2D or 3D).

``CGALCollisionHetero`` is ``CGALCollision`` for agents of any shape, each with its own speed:

- 2D: discs (a radius or ``Circle``) and polygons (any 2D ``Shape`` or a ``(K, 2)`` outline, concave too).
- 3D: spheres (a radius or ``Sphere``), convex polyhedra (``Box``, ``ConvexPolyhedron`` or ``(K, 3)``
  points) and prisms (a 2D ``Shape`` with a ``z_height``; a ``Circle`` becomes a 32-gon prism).

Results are exact and in the same format as ``ShapelyCollisionHetero``. Touching is not contact; during an
in-place turn the body is its bounding ball. The geometry is in ``polytope_sweep``.

The body follows the direction of travel. 2D: rotated by the direction plus the shape's orientation. 3D:
the travel frame has +x along the direction of travel and no roll (+y horizontal; world +y for vertical
travel), and the shape's orientation is applied in that frame.
"""

from __future__ import annotations

from path_planning.common.world.collision_checking.cgal.cgal_collision import CGALCollision
from path_planning.common.world.collision_checking.sweep.disc_sweep import assemble, as_path_list
from path_planning.common.world.collision_checking.sweep.polytope_sweep import body_roadmap_hits, body_agent_conflicts


class CGALCollisionHetero(CGALCollision):
    """``CGALCollision`` for agents of different shapes and sizes, swept along whole paths."""

    #: Inward shrink of the bodies' boundary faces (roadmap tests) and the penetration two agents need to
    #: collide, so that tangential contact and float noise are not collisions.
    STRICT_EPS = 1e-9

    def overlapping_interval_paths(self, paths, shapes, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0,
                                   as_arrays: bool = True, heading=None) -> list:
        """When each agent's body covers each roadmap element along its path. One entry per path:
        ``((vertex, t1, t2), (src, tgt, t1, t2))`` arrays, one row per window, sorted by element then time,
        absolute times, windows merged. ``as_arrays=False``: ``({vertex: [(t1, t2), ...]},
        {(src, tgt): [(t1, t2), ...]})`` dicts instead.

        A vertex is covered while strictly inside the body, an edge while a positive length of it is. The
        windows are exact; a concave body that leaves an element and returns gives separate windows. During
        an in-place turn the body is its bounding ball; during the goal hold it keeps its last attitude.

        ``paths``: one ``(P, D)`` path, ``(A, P, D)`` paths or a list of paths. The other arguments are one
        value or one per path:

        - ``shapes``: a ``Shape``, a radius, or a ``(K, D)`` outline / point set.
        - ``velocity``: speed; ``None`` means every segment takes 1 time unit.
        - ``t0``: start time.
        - ``turning_speed`` (rad/s): ``None`` or ``inf`` turns instantly; ``0`` never turns (the body keeps
          its first segment's attitude); a positive value turns in place for ``angle / turning_speed``.
        - ``goal_hold``: time spent at the last waypoint (``inf``: forever).
        - ``heading``: fixes the attitude (no turning): an angle in 2D; a yaw or 3x3 rotation in 3D."""
        if self.vertex_positions is None:
            raise RuntimeError("overlapping_interval_paths: call set_graph first")
        paths = as_path_list(paths)
        if len(paths) == 0:
            return []
        v_parts, e_parts = body_roadmap_hits(self, paths, shapes, velocity, t0, turning_speed, goal_hold,
                                             heading, self.STRICT_EPS)
        return assemble(self, len(paths), v_parts, e_parts, as_arrays)

    def agent_conflicts(self, paths, shapes, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0,
                        heading=None, pairs=None, first_only: bool = False, as_arrays: bool = False):
        """When two agents' bodies overlap, exactly: ``{(i, j): [(t1, t2), ...]}`` with ``i < j``.
        ``first_only``: ``{(i, j): first contact time}``. ``as_arrays``: ``(i, j, t1, t2)`` arrays.
        ``pairs``: only check these agent pairs. Inputs as in ``overlapping_interval_paths``; no roadmap is
        needed (the dimension comes from the paths). An agent exists from ``t0`` until it arrives, then for
        ``goal_hold``. See ``polytope_sweep`` for the method."""
        return body_agent_conflicts(paths, shapes, velocity, t0, turning_speed, goal_hold, heading, pairs,
                                    first_only, as_arrays, self.STRICT_EPS)
