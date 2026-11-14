"""
Polygon / polyhedron sweep collision checker over a ``GraphSampler`` roadmap, in 2D or 3D.

``CGALCollisionHetero(config)`` is ``CGALCollision`` (same roadmap, ``set_graph``, spatial index and
config) for agents of any shape and size, each with its own speed:

- 2D: discs (a radius or a ``Circle``) and polygons (any 2D ``agent_shape.Shape``, or a ``(K, 2)``
  outline; concave ones too).
- 3D: spheres (a radius or a ``Sphere``), convex polyhedra (``Box``, ``ConvexPolyhedron``, or ``(K, 3)``
  points: their convex hull) and prisms (a 2D ``Shape`` with a ``z_height``: its outline extruded
  vertically; a ``Circle`` becomes the circumscribed 32-gon prism).

``overlapping_interval_paths`` sweeps whole agent paths, many agents at once, and reports when each body
covers each roadmap vertex and directed edge; ``agent_conflicts`` gives the collision windows between
agents. Both are exact (touching is not contact; during an in-place turn the body is its bounding ball),
in the same format as ``ShapelyCollisionHetero``. The geometry is ``polytope_sweep`` (numba, parallel); the
roadmap candidates come from the inherited spatial index (``spatial_index``: ``"grid"`` or ``"rtree"``).

The body follows the direction of travel: 2D it is rotated by the direction + the shape's orientation;
3D the travel frame has +x along the direction of travel, no roll (+y horizontal; world +y for vertical
travel), and the shape's orientation is applied in that frame.
"""

from __future__ import annotations

from path_planning.common.world.collision_checking.cgal_collision import (   # noqa: F401  (re-exported)
    CGALCollisionConfig, CGALCollision, EdgeDir)
from path_planning.common.world.collision_checking.disc_sweep import assemble, as_path_list
from path_planning.common.world.collision_checking.polytope_sweep import body_roadmap_hits, body_agent_conflicts


class CGALCollisionHetero(CGALCollision):
    """``CGALCollision`` for agents of different sizes and shapes (2D polygons, 3D polyhedra / prisms, or
    balls), swept along whole paths. Vertex indices in the results are the map's node indices; edge keys
    are directed ``(src, tgt)`` node-index pairs, in the directions the roadmap allows."""

    #: Inward shrink of the bodies' boundary faces for the roadmap tests, and the penetration two agents
    #: need to collide: contact along a boundary (a vertex sliding on a face, an edge in a face's plane) is
    #: tangential, not a collision, and float rotation noise must not promote it to one.
    STRICT_EPS = 1e-9

    def __init__(self, config: "CGALCollisionConfig | dict | None" = None, spatial_index: "str | None" = None):
        """Same as ``CGALCollision``: ``spatial_index`` is ``"grid"`` (default) or ``"rtree"``."""
        super().__init__(config, spatial_index)

    def overlapping_interval_paths(self, paths, shapes, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0,
                                   as_arrays: bool = True, heading=None) -> list:
        """When each agent's body, moving along its path, covers each roadmap element:
        ``[((vertex, t1, t2), (src, tgt, t1, t2)), ...]``, one entry per path: NumPy arrays, one row per
        window, sorted by element then time, absolute times from ``t0``, each element's windows merged; edge
        rows are the directed edges the roadmap allows. ``as_arrays=False`` gives dicts instead,
        ``[({vertex: [(t1, t2), ...]}, {(src, tgt): [(t1, t2), ...]}), ...]``.

        A vertex is covered while it is strictly inside the body; an edge while a positive length of it is.
        The windows are exact: over a segment the body is rigid and translates; a concave body that leaves
        an element and comes back gives separate windows. During an in-place turn the body is its bounding
        ball (for the whole turn), during the goal hold its last attitude.

        ``paths``: one path ``(P, D)``, same-length paths ``(A, P, D)`` or a list of paths, D the roadmap's
        dimension. ``shapes`` (one shape, including one ``(K, D)`` outline / point set, or one per path),
        ``velocity``, ``t0``, ``turning_speed``, ``goal_hold`` and ``heading`` are one value for all agents
        or one per path. ``velocity=None``: every segment takes 1 time unit. ``turning_speed`` (rad/s):
        ``None`` or ``inf`` turns instantly at each waypoint; ``0`` never turns, so the body keeps its first
        segment's attitude; a positive value turns in place for ``angle / turning_speed`` (the angle
        between the two segments). ``goal_hold``: the agent stays at its last waypoint that long (``inf``:
        forever). ``heading`` fixes the attitude instead (no turning, no turn time): an angle in 2D; a yaw
        or a 3x3 rotation matrix in 3D."""
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
        """When the bodies of every two agents overlap, exactly (touching is not contact; no roadmap
        needed, the dimension comes from the paths).

        Same inputs as ``overlapping_interval_paths``. Returns ``{(i, j): [(t1, t2), ...]}`` (``i < j``,
        windows sorted and merged) for the pairs that collide; ``first_only=True`` gives
        ``{(i, j): first contact time}``; ``as_arrays=True`` gives ``(i, j, t1, t2)`` arrays (one row per
        window, or per pair with ``first_only``). ``pairs``: only check these ``(i, j)`` agent pairs.

        Each schedule is split into rigid pieces with constant velocity (segments, turns as the bounding
        ball, the goal hold). For two pieces of different agents that overlap in time, the position of one
        relative to the other moves on a line, and the bodies overlap while it lies in ``A (+) (-B)``: one
        clip per pair of convex pieces (face normals and, in 3D, edge cross products; a ball against a
        polytope uses the polytope grown by the radius). An agent exists from ``t0`` until it arrives, then
        for ``goal_hold`` (``inf``: it stays). Candidates come from a sort-and-sweep over each piece's box
        over its window; both it and the exact test run in parallel (numba ``prange``)."""
        return body_agent_conflicts(paths, shapes, velocity, t0, turning_speed, goal_hold, heading, pairs,
                                    first_only, as_arrays, self.STRICT_EPS)
