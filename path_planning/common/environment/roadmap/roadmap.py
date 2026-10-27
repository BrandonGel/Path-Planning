"""General roadmap facade: build any registered roadmap type and hand the
result to a ``GraphSamplerBase``.

    rm = Roadmap(graph_map, "cdt")
    rm.generate_roadmap(nodes)          # anchors default to the map's start + goal
    rm.update_node_sampler(graph_map.node_sampler)   # graph_map.nodes / road_map / edges ... updated

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
    from path_planning.common.environment.node_sampler.node_sampler import NodeSampler

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
    planar family and RRG) they default to the map's ``start`` and ``goal``. ``update_node_sampler`` writes the result
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
        builder._nodes = self.nodes
        builder._clearance = self.clearance
        builder._adj = csr_array((self.num_nodes, self.num_nodes), dtype=float)

        opts = {**self.options, **kwargs}
        if roadmap_class.uses_anchors and opts.get("anchor_points") is None:
            opts["anchor_points"] = self._default_anchors()      # start + goal
        builder.generate(**opts)              # not generate_roadmap: nodes are set

        self.builder = builder
        self.roadmap_type = rtype
        self._snapped = dict(builder.snapped)
        self._nodes = builder.nodes
        self._clearance = builder.clearance
        self._num_nodes = builder.num_nodes
        self._adj = builder.adj
        return self.adj

    # ---------------------------------------------------------------- export
    def apply(self, node_sampler: "NodeSampler | None" = None) -> None:
        """Hand this roadmap's nodes to ``node_sampler`` (default: the map's),
        so the map's ``nodes`` / ``pos_index_dict`` / ``num_total_nodes`` read
        them in roadmap order, then write the start/goal kinds and drop the
        map's cached start/goal edge lists (``start_to_all_edges_dict`` /
        ``goal_to_all_edges_dict``), which are keyed by node index.
        ``edges`` / ``edge_weights`` / ``road_map`` are views over ``gm.roadmap``.

        The nodes are *replaced*, not appended: planar builders derive new
        points, Lattice snaps and start/goal may be added, so sampler index
        ``i`` must be roadmap node ``i``. A start/goal position the builder did
        not produce is appended as an isolated node; one the builder snapped
        (``self.snapped``) resolves to the node it became, keyed by the
        original position."""
        
        gm = self.graph_map
        node_sampler = gm.node_sampler if node_sampler is None else node_sampler
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

        start_index: dict = {}
        goal_index: dict = {}
        register(list(gm.start), start_index)
        register(list(gm.goal), goal_index)
        self.append_isolated_nodes(extra)
        node_sampler.set_nodes(self.to_nodes())      # re-keys the kinds by position: a snapped start/goal would drop,
        kinds = node_sampler.points_index_dict        # so the start/goal kinds are written afterwards
        kinds['start_points'].clear(); kinds['start_points'].update(start_index)
        kinds['goal_points'].clear(); kinds['goal_points'].update(goal_index)
        gm.start_to_all_edges_dict = {}               # node indices changed: drop the cached start/goal edge lists
        gm.goal_to_all_edges_dict = {}

    def to_nodes(self) -> List[Node]:
        """``Node`` objects (position + ``max_clearance``) in roadmap order."""
        return [Node(p, None, 0, 0, float(c)) for p, c in zip(convert_np_to_tuple(self.nodes), self.clearance)]
