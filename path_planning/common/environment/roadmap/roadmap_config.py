import math
from dataclasses import dataclass, fields, asdict
import yaml
import os

@dataclass
class RoadmapConfig:
    """Tunable parameters shared by every roadmap builder, built from a dict.

    ``RoadmapConfig({"num_neighbors": 6, "max_edge_length": 12})``; keys left
    out take the defaults below, unknown keys raise. Edge band and k apply to
    the builders that filter edges (PRM, Delaunay, planar family, RRG,
    custom); ``Lattice`` edges are fixed by the grid.

    ``RoadmapConfig.from_yaml(path)`` reads a YAML file (``config/roadmap.yaml``
    is the template) and ``RoadmapConfig.load(x)`` accepts a config, dict,
    YAML path or ``None``, which is what the sampler and builders call.
    """
    YAML_KEY = "roadmap_config"           # optional top-level block in a YAML file
    min_edge_length: float = 0.0          # drop edges shorter than this
    max_edge_length: float = math.inf     # drop edges longer than this; RRG steer / connect radius
    num_neighbors: int = 0                # PRM: accepted edges per node, 0 = unlimited
    sample_num: int = 0                   # RRG: nodes to add
    goal_sample_rate: float = 0.1         # RRG: probability of drawing a goal
    max_iters_ratio: float = 50.0         # RRG: iteration cap = ratio * sample_num
    filter_collision: bool = True         # line-of-sight test on candidate edges
    snap_off_grid: bool = True            # Lattice: snap off-centre nodes instead of raising
    k_query: int | None = None            # PRM: first KD query size (default 4*k+1)
    grow: int = 4                         # PRM: query growth factor per round

    def __init__(self, config: dict | None = None) -> None:
        config = dict(config or {})
        for f in fields(self):                      # every field, default when absent
            setattr(self, f.name, config.pop(f.name, f.default))
        if config:
            raise TypeError(f"RoadmapConfig: unknown keys {sorted(config)}")
        if self.min_edge_length < 0 or self.max_edge_length < self.min_edge_length:
            raise ValueError(f"need 0 <= min_edge_length <= max_edge_length, got {self.min_edge_length}, {self.max_edge_length}")
        if self.num_neighbors < 0 or self.sample_num < 0:
            raise ValueError("num_neighbors and sample_num must be >= 0")
        if not 0.0 <= self.goal_sample_rate <= 1.0:
            raise ValueError(f"goal_sample_rate must be in [0, 1], got {self.goal_sample_rate}")
        if self.max_iters_ratio <= 0 or self.grow < 2:
            raise ValueError("max_iters_ratio must be > 0 and grow >= 2")
        self.min_edge_length = float(self.min_edge_length)
        self.max_edge_length = float(self.max_edge_length)
        self.num_neighbors = int(self.num_neighbors)
        self.sample_num = int(self.sample_num)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: "str | os.PathLike") -> "RoadmapConfig":
        """Read a config from a YAML file: the fields under a top-level
        ``roadmap_config:`` block when present, else the whole document. Use
        ``.inf`` for an unbounded ``max_edge_length`` and ``null`` for ``k_query``."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError(f"{path}: expected a mapping of RoadmapConfig fields")
        if cls.YAML_KEY in data:
            data = data[cls.YAML_KEY] or {}
        return cls(data)

    def to_yaml(self, path: "str | os.PathLike") -> None:
        """Write the config under a ``roadmap_config:`` block (``inf`` is written as ``.inf``)."""
        with open(path, "w") as f:
            yaml.safe_dump({self.YAML_KEY: self.as_dict()}, f, sort_keys=False)

    @classmethod
    def load(cls, source: "RoadmapConfig | dict | str | os.PathLike | None") -> "RoadmapConfig":
        """``RoadmapConfig`` as is; dict or ``None`` through the constructor;
        a str / path through ``from_yaml``."""
        if isinstance(source, cls):
            return source
        if isinstance(source, (str, os.PathLike)):
            return cls.from_yaml(source)
        if source is None or isinstance(source, dict):
            return cls(source)
        raise TypeError(f"RoadmapConfig.load: expected RoadmapConfig, dict, YAML path or None, got {type(source).__name__}")

