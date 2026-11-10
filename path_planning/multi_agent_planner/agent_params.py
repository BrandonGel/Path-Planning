"""Per-agent radius / speed bookkeeping shared by the multi-agent planners.

Every planner keeps a scalar ``radius`` and ``velocity`` (the historical API and
the fallback for agents without their own value) plus two optional per-name
dicts. Agent-agent clearance is ``r_i + r_j`` (``2 * r`` when homogeneous, which
is bit-for-bit the same float, so homogeneous runs are unchanged). The scalar
``radius`` is also the "body mode" gate (``radius > 0``) and should be the
LARGEST agent radius, i.e. the one the roadmap was inflated with.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import math

from path_planning.utils.agent_shape import AgentShape, AlignedClearance, AlignedShape, ConvexPolygon, Disc, heading_key, pair_clearance, shape_from_spec


class AgentParamsMixin:
    """Mixin for planner classes that store ``self.radius`` / ``self.velocity``.

    Footprints: an agent may additionally carry a convex-polygon ``shape``
    (``agent_shapes``; rectangles / regular polygons): an ``AlignedShape`` that
    faces its direction of travel (the YAML default) or a ``ConvexPolygon``
    fixed in the world frame. Its ``radius`` is then only the conservative
    bounding radius used for obstacle inflation; agent-agent clearance becomes
    the Minkowski footprint ``P_i (+) (-P_j)`` (``agent_shape.pair_clearance``;
    an ``AlignedClearance`` the sweep resolves per element when a footprint is
    aligned), which the sweep backends evaluate exactly. Disc-disc pairs keep
    the float ``r_i + r_j``.
    """

    agent_radii: Dict[str, float]
    agent_velocities: Dict[str, float]
    agent_shapes: Dict[str, ConvexPolygon | AlignedShape]

    def _init_agent_params(
        self,
        radius: float,
        velocity: float,
        agent_radii: Optional[Mapping[str, float]] = None,
        agent_velocities: Optional[Mapping[str, float]] = None,
        agents: Iterable[dict] = (),
        agent_shapes: Optional[Mapping[str, object]] = None,
        agent_turning_speeds: Optional[Mapping[str, float]] = None,
        agent_headings: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Store the per-agent dicts; ``None`` derives them from the agent dicts.

        ``agents`` entries may carry ``"radius"`` / ``"velocity"`` / ``"shape"``
        keys (see ``path_planning.utils.util.normalize_agent_specs``). Names
        absent from the resulting dicts fall back to the scalars ``radius`` /
        ``velocity`` (and to a disc footprint).
        """
        self.radius = float(radius)
        self.velocity = float(velocity)
        agents = list(agents)
        if agent_shapes is None:
            agent_shapes = {
                a["name"]: a["shape"] for a in agents if a.get("shape") is not None and "name" in a
            }
        if agent_radii is None:
            agent_radii = {
                a["name"]: a["radius"] for a in agents if a.get("radius") is not None and "name" in a
            }
        if agent_velocities is None:
            agent_velocities = {
                a["name"]: a["velocity"] for a in agents if a.get("velocity") is not None and "name" in a
            }
        if agent_turning_speeds is None:
            agent_turning_speeds = {
                a["name"]: a["turning_speed"] for a in agents if a.get("turning_speed") is not None and "name" in a
            }
        if agent_headings is None:
            agent_headings = {a["name"]: a["heading"] for a in agents if a.get("heading") is not None and "name" in a}
        self.agent_radii = {k: float(v) for k, v in agent_radii.items()}
        self.agent_velocities = {k: float(v) for k, v in agent_velocities.items()}
        # Kinematics: finite turning speeds only (absent = instantaneous), initial headings.
        self.agent_turning_speeds = {
            k: float(v) for k, v in agent_turning_speeds.items()
            if v is not None and math.isfinite(float(v)) and float(v) > 0.0
        }
        self.agent_headings = {k: heading_key(float(v)) for k, v in agent_headings.items() if v is not None}
        self.agent_shapes = {}
        for k, v in agent_shapes.items():
            shape = shape_from_spec(v)
            if isinstance(shape, (ConvexPolygon, AlignedShape)):
                self.agent_shapes[k] = shape
                # Bounding radius as the conservative disc fallback for sites that
                # still reason with radii (inflation, bounding-box pruning).
                self.agent_radii.setdefault(k, float(shape.bounding_radius))
            else:
                self.agent_radii.setdefault(k, float(shape.radius))

    # -- accessors ---------------------------------------------------------
    def _radius_for(self, name) -> float:
        return self.agent_radii.get(name, self.radius)

    def _velocity_for(self, name) -> float:
        return self.agent_velocities.get(name, self.velocity)

    def _turning_speed_for(self, name) -> float:
        """Turning speed (rad/s) of ``name``; ``inf`` when it turns instantaneously."""
        return self.agent_turning_speeds.get(name, float("inf"))

    def _heading_for(self, name) -> float:
        """Initial heading of ``name`` (``heading_key``-canonical, default 0 = +x)."""
        return self.agent_headings.get(name, 0.0)

    @property
    def has_turning_agents(self) -> bool:
        return len(self.agent_turning_speeds) > 0

    def _shape_for(self, name) -> AgentShape:
        """Footprint of ``name``: its polygon, else a disc of its radius."""
        shape = self.agent_shapes.get(name)
        return shape if shape is not None else Disc(self._radius_for(name))

    def _plan_key_for(self, name):
        """Hashable key describing the footprint of a PLANNING agent: the float
        radius for discs (unchanged homogeneous behaviour), the polygon /
        aligned shape otherwise."""
        shape = self.agent_shapes.get(name)
        return shape if shape is not None else self._radius_for(name)

    def _footprint_of(self, key) -> AgentShape:
        """Footprint for an agent name or a plan key (float radius / polygon)."""
        if isinstance(key, (ConvexPolygon, AlignedShape)):
            return key
        if isinstance(key, (int, float)):
            return Disc(float(key))
        return self._shape_for(key)

    def _sweep_clearance(self, query, element):
        """Clearance to hand to a sweep whose QUERY agent (the one moving along the
        swept segment) is ``query`` and whose roadmap-element agent is ``element``
        (each an agent name or a plan key): the float ``r_q + r_e`` for a disc pair,
        else the Minkowski polygon ``P_query (+) (-P_element)`` -- the set of
        ``c_element - c_query`` offsets that collide (``agent_shape.pair_clearance``)
        -- or an ``AlignedClearance`` when either footprint is heading-aligned.
        The orientation matters for footprints that are not centrally symmetric
        (triangles, off-centre rectangles); discs and symmetric polygons give the
        same clearance both ways."""
        fq = self._footprint_of(query)
        fe = self._footprint_of(element)
        if isinstance(fq, Disc) and isinstance(fe, Disc):
            return fq.radius + fe.radius
        return pair_clearance(fq, fe)

    def _pair_clearance(self, name_a, name_b):
        """Clearance below which agents ``a`` and ``b`` collide: the float
        ``r_a + r_b`` (disc-disc) or the polygon ``P_a (+) (-P_b)`` (``c_b - c_a``
        inside it). Equals ``_sweep_clearance(a, b)``: use it for sweeps of
        agent ``a``'s motion against elements agent ``b`` may occupy."""
        return self._sweep_clearance(name_a, name_b)

    @property
    def has_polygon_agents(self) -> bool:
        return len(self.agent_shapes) > 0

    def _max_radius(self) -> float:
        return max([self.radius] + list(self.agent_radii.values()))

    @property
    def heterogeneous_radii(self) -> bool:
        return len({self.radius, *self.agent_radii.values()}) > 1

    @property
    def heterogeneous_velocities(self) -> bool:
        return len({self.velocity, *self.agent_velocities.values()}) > 1
