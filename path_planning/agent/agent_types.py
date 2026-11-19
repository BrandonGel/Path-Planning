"""Agent types from a file (``config/agents.yaml``): a name -> ``Agent`` table.

::

    types:
      amr:      {shape: disc, size: 0.35, max_speed: 1.5}
      forklift: {shape: rectangle, size: [1.6, 0.8], max_speed: 1.0, turning_speed: 90, clearance: 0.05}
      tugger:   {shape: polygon, size: [[0.9, 0], [-0.6, 0.5], [-0.6, -0.5]], max_speed: 0.8}
      hex:      {shape: regular_polygon, size: {sides: 6, side: 0.4}}

- ``shape`` and ``size``: ``disc`` (radius), ``rectangle`` (``[length along travel, width]``), ``triangle``
  (its three sides), ``regular_polygon`` (``{sides, side}``), ``polygon`` (its vertices, in order).
- ``max_speed``: world units per time unit (left out: every move takes one time unit).
- ``turning_speed``: degrees per time unit (left out: turns instantly). Only CBS and ICBS use it.
- ``clearance``: a radial buffer in world units (default 0).

The type's name becomes ``Agent.type``.
"""
import math

import yaml

from path_planning.agent.agent import Agent
from path_planning.common.world.collision_checking.sweep.agent_shape import (CustomPolygon, Rectangle, RegularPolygon,
                                                                             Triangle)

KEYS = {"shape", "size", "max_speed", "turning_speed", "clearance"}


def _numbers(size, count, name, what):
    if not isinstance(size, (list, tuple)) or len(size) != count or not all(isinstance(v, (int, float)) for v in size):
        raise ValueError(f"agent type {name!r}: a {what} needs size = {count} numbers, got {size!r}")
    return [float(v) for v in size]


def _shape(name, kind, size):
    """The ``Agent`` shape argument for ``shape: kind`` / ``size``."""
    if kind == "disc":
        if isinstance(size, bool) or not isinstance(size, (int, float)) or size <= 0:
            raise ValueError(f"agent type {name!r}: a disc needs size = its radius (> 0), got {size!r}")
        return float(size)
    if kind == "rectangle":
        return Rectangle(*_numbers(size, 2, name, "rectangle"))
    if kind == "triangle":
        return Triangle(*_numbers(size, 3, name, "triangle"))
    if kind == "regular_polygon":
        if not isinstance(size, dict) or set(size) != {"sides", "side"}:
            raise ValueError(f"agent type {name!r}: a regular_polygon needs size = {{sides, side}}, got {size!r}")
        return RegularPolygon(float(size["side"]), int(size["sides"]))
    if kind == "polygon":
        if not isinstance(size, (list, tuple)) or len(size) < 3:
            raise ValueError(f"agent type {name!r}: a polygon needs size = its vertices (3 or more), got {size!r}")
        return CustomPolygon([_numbers(p, 2, name, "polygon vertex") for p in size])
    raise ValueError(f"agent type {name!r}: unknown shape {kind!r} (disc, rectangle, triangle, regular_polygon, polygon)")


def agent_type(name: str, spec: dict) -> Agent:
    """The ``Agent`` of one entry of the table."""
    if not isinstance(spec, dict) or "shape" not in spec or "size" not in spec:
        raise ValueError(f"agent type {name!r}: needs 'shape' and 'size', got {spec!r}")
    unknown = set(spec) - KEYS
    if unknown:
        raise ValueError(f"agent type {name!r}: unknown keys {sorted(unknown)} (known: {sorted(KEYS)})")
    speed, turning = spec.get("max_speed"), spec.get("turning_speed")
    if speed is not None and speed <= 0:
        raise ValueError(f"agent type {name!r}: max_speed must be > 0, got {speed!r}")
    if turning is not None and turning <= 0:
        raise ValueError(f"agent type {name!r}: turning_speed must be > 0 degrees per time unit, got {turning!r}")
    return Agent(_shape(name, spec["shape"], spec["size"]), max_speed=None if speed is None else float(speed),
                 max_rotation=None if turning is None else math.radians(float(turning)), type=str(name),
                 clearance=float(spec.get("clearance", 0.0)))


def load_agent_types(source) -> dict:
    """``{type name: Agent}`` from a YAML path or its loaded content (``{"types": {...}}`` or the table)."""
    if not isinstance(source, dict):
        with open(source, "r") as f:
            source = yaml.safe_load(f) or {}
    table = source.get("types", source)
    if not isinstance(table, dict) or not table:
        raise ValueError("agent types: expected `types: {name: {shape, size, ...}}` with at least one type")
    return {str(name): agent_type(str(name), spec) for name, spec in table.items()}
