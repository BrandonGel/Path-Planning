"""``Agent``: an agent type (body shape, speed and turn limits, a label) used by the map's collision queries.

It describes a kind of agent, not one agent: many agents share one ``Agent``, and two equal ``Agent``
objects are interchangeable (same hash), which is what the map's sweep cache keys on. An agent's name,
start and goal stay in the planners' agent dicts.
"""
from __future__ import annotations

import json
import math

from path_planning.common.world.collision_checking.sweep import agent_shape
from path_planning.common.world.collision_checking.sweep.agent_shape import Shape, Shape3D, Circle, Sphere


class Agent:
    """An agent type.

    - ``shape``: a ``Shape`` (``agent_shape``), or a radius (a disc in 2D, a sphere in 3D).
    - ``max_speed``: travel speed; ``None`` means every move takes 1 time unit.
    - ``max_rotation`` (radians per time unit): turning rate at waypoints; ``None`` or ``inf`` turns
      instantly, ``0`` never turns (the body keeps its first attitude). CBS / ICBS plan in unit time steps,
      so there a turn through an angle ``a`` takes ``ceil(a / max_rotation)`` steps.
    - ``type``: a label, e.g. ``"forklift"``.
    - ``clearance``: a buffer that expands the body radially: everything within this distance of the shape
      counts as the agent. Two agents then stay at least the sum of their clearances apart.

    Immutable; equality and hashing use ``key``."""

    __slots__ = ("_shape", "_max_speed", "_max_rotation", "_type", "_clearance", "_key")

    def __init__(self, shape, max_speed: "float | None" = None, max_rotation: "float | None" = None,
                 type: str = "agent", clearance: float = 0.0):
        if isinstance(shape, bool) or not isinstance(shape, (Shape, int, float)):
            raise TypeError(f"Agent: shape must be a Shape or a radius, got {shape!r}")
        if not isinstance(shape, Shape):
            shape = float(shape)
            if not (math.isfinite(shape) and shape > 0):
                raise ValueError(f"Agent: a radius must be positive and finite, got {shape}")
        if max_speed is not None and not (math.isfinite(max_speed) and max_speed > 0):
            raise ValueError(f"Agent: max_speed must be positive and finite, or None, got {max_speed}")
        if max_rotation is not None and not max_rotation >= 0:
            raise ValueError(f"Agent: max_rotation must be >= 0, inf or None, got {max_rotation}")
        if not (math.isfinite(clearance) and clearance >= 0):
            raise ValueError(f"Agent: clearance must be >= 0 and finite, got {clearance}")
        self._shape = shape
        self._clearance = float(clearance)
        self._max_speed = None if max_speed is None else float(max_speed)
        self._max_rotation = None if max_rotation is None else float(max_rotation)
        self._type = str(type)
        shape_key = (("radius", shape) if not isinstance(shape, Shape) else
                     (shape.__class__.__name__, json.dumps(shape.to_dict(), sort_keys=True, default=str)))
        self._key = (self._type, shape_key, self._max_speed, self._max_rotation, self._clearance)

    # ------------------------------------------------------------ fields (read-only)
    @property
    def shape(self):
        return self._shape

    @property
    def max_speed(self):
        return self._max_speed

    @property
    def max_rotation(self):
        return self._max_rotation

    @property
    def type(self) -> str:
        return self._type

    @property
    def clearance(self) -> float:
        """The buffer around the shape that still counts as the agent (0: the shape itself)."""
        return self._clearance

    @property
    def key(self) -> tuple:
        """Hashable identity of this type: ``(type, shape, max_speed, max_rotation, clearance)``."""
        return self._key

    @property
    def radius(self) -> float:
        """Bounding radius of the shape about its reference point (in 3D, a prism's includes its height).
        Without the ``clearance``; ``reach`` includes it."""
        s = self._shape
        if not isinstance(s, Shape):
            return s
        r = float(s.radius())
        zh = None if isinstance(s, Shape3D) else s.z_height()
        return r if zh is None else math.hypot(r, 0.5 * zh)

    @property
    def reach(self) -> float:
        """How far the agent extends from its reference point: ``radius + clearance``."""
        return self.radius + self._clearance

    @property
    def is_ball(self) -> bool:
        """Whether the body is a ball in every dimension it can be used in: a radius, a ``Sphere``, or a
        ``Circle`` without a ``z_height``."""
        s = self._shape
        return (not isinstance(s, Shape) or isinstance(s, Sphere) or type(s) is Shape
                or (isinstance(s, Circle) and s.z_height() is None))

    # ------------------------------------------------------------ construction / serialisation
    @classmethod
    def disc(cls, radius: float, max_speed: "float | None" = None, type: str = "disc",
             clearance: float = 0.0) -> "Agent":
        """A ball agent of ``radius`` (a disc in 2D, a sphere in 3D)."""
        return cls(float(radius), max_speed, None, type, clearance)

    def replace(self, **changes) -> "Agent":
        """A copy with some fields changed, e.g. ``agent.replace(max_speed=None)``."""
        fields = dict(shape=self._shape, max_speed=self._max_speed, max_rotation=self._max_rotation,
                      type=self._type, clearance=self._clearance)
        unknown = changes.keys() - fields.keys()
        if unknown:
            raise TypeError(f"Agent.replace: unknown fields {sorted(unknown)}")
        return Agent(**{**fields, **changes})

    def to_dict(self) -> dict:
        s = self._shape
        shape = {"radius": s} if not isinstance(s, Shape) else {"class": s.__class__.__name__, **s.to_dict()}
        return {"type": self._type, "shape": shape, "max_speed": self._max_speed, "max_rotation": self._max_rotation,
                "clearance": self._clearance}

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        shape = dict(data["shape"])
        name = shape.pop("class", None)
        if name is None:
            body = float(shape["radius"])
        else:
            shape_cls = getattr(agent_shape, name, None)
            if not (isinstance(shape_cls, type) and issubclass(shape_cls, Shape)):
                raise ValueError(f"Agent.from_dict: unknown shape class {name!r}")
            body = shape_cls.from_dict(shape)
        return cls(body, data.get("max_speed"), data.get("max_rotation"), data.get("type", "agent"),
                   data.get("clearance", 0.0))

    def __eq__(self, other) -> bool:
        return isinstance(other, Agent) and self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)

    def __repr__(self) -> str:
        s = self._shape
        body = f"radius={s:g}" if not isinstance(s, Shape) else s.__class__.__name__
        buffer = f", clearance={self._clearance:g}" if self._clearance else ""
        return f"Agent({self._type!r}, {body}, max_speed={self._max_speed}, max_rotation={self._max_rotation}{buffer})"
