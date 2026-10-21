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


class AgentParamsMixin:
    """Mixin for planner classes that store ``self.radius`` / ``self.velocity``."""

    agent_radii: Dict[str, float]
    agent_velocities: Dict[str, float]

    def _init_agent_params(
        self,
        radius: float,
        velocity: float,
        agent_radii: Optional[Mapping[str, float]] = None,
        agent_velocities: Optional[Mapping[str, float]] = None,
        agents: Iterable[dict] = (),
    ) -> None:
        """Store the per-agent dicts; ``None`` derives them from the agent dicts.

        ``agents`` entries may carry ``"radius"`` / ``"velocity"`` keys (see
        ``path_planning.utils.util.normalize_agent_specs``). Names absent from the
        resulting dicts fall back to the scalars ``radius`` / ``velocity``.
        """
        self.radius = float(radius)
        self.velocity = float(velocity)
        if agent_radii is None:
            agent_radii = {
                a["name"]: a["radius"] for a in agents if a.get("radius") is not None and "name" in a
            }
        if agent_velocities is None:
            agent_velocities = {
                a["name"]: a["velocity"] for a in agents if a.get("velocity") is not None and "name" in a
            }
        self.agent_radii = {k: float(v) for k, v in agent_radii.items()}
        self.agent_velocities = {k: float(v) for k, v in agent_velocities.items()}

    # -- accessors ---------------------------------------------------------
    def _radius_for(self, name) -> float:
        return self.agent_radii.get(name, self.radius)

    def _velocity_for(self, name) -> float:
        return self.agent_velocities.get(name, self.velocity)

    def _pair_clearance(self, name_a, name_b) -> float:
        """Centre-to-centre distance below which agents ``a`` and ``b`` collide."""
        return self._radius_for(name_a) + self._radius_for(name_b)

    def _max_radius(self) -> float:
        return max([self.radius] + list(self.agent_radii.values()))

    @property
    def heterogeneous_radii(self) -> bool:
        return len({self.radius, *self.agent_radii.values()}) > 1

    @property
    def heterogeneous_velocities(self) -> bool:
        return len({self.velocity, *self.agent_velocities.values()}) > 1
