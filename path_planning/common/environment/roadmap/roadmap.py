"""General roadmap facade: build any registered roadmap type and hand the
result to a ``GraphSamplerBase``.

    rm = Roadmap(graph_map, "cdt")
    rm.generate_roadmap(nodes)          # anchors default to the map's start + goal
    rm.apply_to(graph_map)              # graph_map.nodes / road_map / edges ... updated

``GraphSamplerBase.build_roadmap(roadmap_type, samples)`` does the three steps.
"""
from typing import Dict, List, Type, TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_array
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from path_planning.common.environment.node import Node
from path_planning.common.environment.map.numbaUtil import convert_np_to_tuple
from path_planning.common.environment.roadmap.roadmap_base import RoadmapBase, RoadmapConfig
from path_planning.common.environment.roadmap.prm import PRM
from path_planning.common.environment.roadmap.lattice import Lattice
from path_planning.common.environment.roadmap.delaunay import DelaunayRoadmap
from path_planning.common.environment.roadmap.cdt import CDT
from path_planning.common.environment.roadmap.midpoints import Midpoints
from path_planning.common.environment.roadmap.centroids import Centroids
from path_planning.common.environment.roadmap.voronoi import Voronoi
from path_planning.common.environment.roadmap.custom import CustomRoadmap
from path_planning.common.environment.roadmap.rrg import RRG

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase

ROADMAP_CLASSES: Dict[str, Type[RoadmapBase]] = {
    cls.name: cls for cls in (PRM, Lattice, DelaunayRoadmap, CDT, Midpoints, Centroids, Voronoi, CustomRoadmap, RRG)
}


def resolve_roadmap_type(roadmap_type: str, dim: int) -> str:
    """Map a ``GraphSamplerBase`` roadmap type onto a registered builder name.
    ``halton*`` means a triangulation of the Halton samples: ``cdt`` in 2D,
    ``dt`` otherwise (as ``generate_map`` does)."""
    if roadmap_type is None:
        raise ValueError("roadmap_type is required")
    if "halton" in roadmap_type:
        return "cdt" if dim == 2 else "dt"
    if roadmap_type not in ROADMAP_CLASSES:
        raise ValueError(f"Invalid roadmap type: {roadmap_type!r}; expected one of {sorted(ROADMAP_CLASSES)}")
    return roadmap_type


def make_roadmap(roadmap_type: str, graph_map: "GraphSamplerBase",
                 config: dict | RoadmapConfig | None = None) -> RoadmapBase:
    """Instantiate the concrete builder registered under ``roadmap_type``."""
    return ROADMAP_CLASSES[resolve_roadmap_type(roadmap_type, graph_map.dim)](graph_map, config)


class Roadmap(RoadmapBase):
    """Roadmap of any registered type, chosen by name.

    ``generate`` instantiates the concrete builder (``PRM``, ``Lattice``,
    ``CDT``, ...) for ``roadmap_type``, runs it on the current nodes and adopts
    its nodes, clearance and adjacency, passing it this roadmap's ``config``.
    For builders that take anchors (the
    planar family and RRG) they default to the map's ``start`` and ``goal``. ``apply_to`` writes the result
    into a ``GraphSamplerBase`` in the layout its planners read.
    """

    name = "general"

    def __init__(self, graph_map: "GraphSamplerBase", roadmap_type: str | None = None,
                 config: dict | RoadmapConfig | None = None, **options):
        super().__init__(graph_map, config)
        self.roadmap_type = roadmap_type
        self.options = dict(options)          # forwarded to the builder's generate()
        self.builder: RoadmapBase | None = None

    # -------------------------------------------------------------- building
    def _default_anchors(self) -> np.ndarray | None:
        pts = [tuple(p) for p in list(self.graph_map.start) + list(self.graph_map.goal)]
        return np.asarray(pts, dtype=float).reshape(-1, self.dim) if pts else None

    def generate(self, roadmap_type: str | None = None, **kwargs) -> csr_array:
        rtype = resolve_roadmap_type(roadmap_type or self.roadmap_type, self.dim)
        roadmap_class = ROADMAP_CLASSES[rtype]
        builder = roadmap_class(self.graph_map, self.config)
        # Hand over the already-installed nodes; no Node list round trip.
        builder._num_nodes = self.num_nodes
        builder.nodes = self.nodes
        builder.clearance = self.clearance
        builder.adj = csr_array((self.num_nodes, self.num_nodes), dtype=float)

        opts = {**self.options, **kwargs}
        if roadmap_class.uses_anchors and opts.get("anchor_points") is None:
            opts["anchor_points"] = self._default_anchors()      # start + goal
        builder.generate(**opts)              # not generate_roadmap: nodes are set

        self.builder = builder
        self.roadmap_type = rtype
        self.snapped = dict(builder.snapped)
        self.nodes = builder.nodes
        self.clearance = builder.clearance
        self._num_nodes = builder.num_nodes
        self.adj = builder.adj
        return self.adj

    # ---------------------------------------------------------------- export
    def to_nodes(self) -> List[Node]:
        """``Node`` objects (position + ``max_clearance``) in roadmap order."""
        return [Node(p, None, 0, 0, float(c)) for p, c in zip(convert_np_to_tuple(self.nodes), self.clearance)]

    def apply_to(self, graph_map: "GraphSamplerBase | None" = None, with_cost_matrix: bool = True) -> "GraphSamplerBase":
        """Install this roadmap into ``graph_map`` in the layout its planners
        read: ``nodes``, ``pos_index_dict``, start/goal/grid/boundary indices,
        ``edges`` / ``edge_weights`` (flat), ``num_total_nodes`` and (unless
        ``with_cost_matrix`` is False) the dense ``cost_matrix``. The nested
        ``road_map`` / ``road_map_edge_weights`` views are properties of the
        map over this object. Start and goal positions missing from the
        roadmap are appended as isolated nodes; a position the builder snapped
        (``self.snapped``) resolves to the node it became, keyed by the
        original position."""
        gm = self.graph_map if graph_map is None else graph_map
        if gm is None:
            return
        old_grid = [n.current for n in gm.nodes if n in gm.grid_nodes_index] if gm.nodes else []

        index: dict = {}
        for i, p in enumerate(convert_np_to_tuple(self.nodes)):
            index.setdefault(p, i)
        extra: list = []                          # isolated nodes to append

        def register(points: list, target: dict) -> None:
            target.clear()
            for p in points:
                key = Node(tuple(p), None, 0, 0)      # start/goal indices stay keyed by the given position
                pos = tuple(p)
                if pos not in index and pos in self.snapped:
                    pos = tuple(self.snapped[pos])    # builder moved it (e.g. Lattice snap)
                if pos not in index:                  # not produced by the builder: isolated node
                    index[pos] = self.num_nodes + len(extra)
                    extra.append(pos)
                target[key] = index[pos]

        register(list(gm.start), gm.start_nodes_index)
        register(list(gm.goal), gm.goal_nodes_index)
        self.append_isolated_nodes(extra)

        gm.roadmap = self
        gm.nodes = self.to_nodes()
        gm.num_total_nodes = len(gm.nodes)
        gm.pos_index_dict = index
        gm.grid_nodes_index = {Node(tuple(p), None, 0, 0): index[tuple(p)]
                               for p in old_grid if tuple(p) in index}
        gm._rebuild_boundary_nodes_index()
        gm.edges = self.edges
        gm.edge_weights = self.edge_weights
        gm.start_to_all_edges_dict = {}
        gm.goal_to_all_edges_dict = {}
        if with_cost_matrix:
            gm.cost_matrix = cdist(self.nodes, self.nodes, metric="euclidean")
        return gm
