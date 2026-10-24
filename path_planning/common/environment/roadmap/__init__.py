"""Array-backed roadmap classes, one per roadmap type.

    rm = make_roadmap("prm", graph_map)
    rm.generate_roadmap(nodes)                     # PRM / grid / dt
    rm = make_roadmap("cdt", graph_map)
    rm.generate_roadmap(nodes, anchor_points=start_goal)   # planar family
"""
from typing import TYPE_CHECKING

from path_planning.common.environment.roadmap.roadmap_base import RoadmapBase
from path_planning.common.environment.roadmap.prm import PRM
from path_planning.common.environment.roadmap.lattice import Lattice
from path_planning.common.environment.roadmap.delaunay import DelaunayRoadmap
from path_planning.common.environment.roadmap.roadmap_planar import RoadmapPlanar
from path_planning.common.environment.roadmap.cdt import CDT
from path_planning.common.environment.roadmap.midpoints import Midpoints
from path_planning.common.environment.roadmap.centroids import Centroids
from path_planning.common.environment.roadmap.voronoi import Voronoi
from path_planning.common.environment.roadmap.custom import CustomRoadmap

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase

ROADMAP_CLASSES = {cls.name: cls for cls in (PRM, Lattice, DelaunayRoadmap, CDT, Midpoints, Centroids, Voronoi, CustomRoadmap)}


def make_roadmap(roadmap_type: str, graph_map: "GraphSamplerBase") -> RoadmapBase:
    """Instantiate the roadmap class registered under ``roadmap_type``
    (``prm``, ``grid``, ``dt``, ``cdt``, ``midpoints``, ``centroids``, ``voronoi``, ``custom``)."""
    try:
        return ROADMAP_CLASSES[roadmap_type](graph_map)
    except KeyError:
        raise ValueError(f"Invalid roadmap type: {roadmap_type!r}; expected one of {sorted(ROADMAP_CLASSES)}") from None


__all__ = ["RoadmapBase", "PRM", "Lattice", "DelaunayRoadmap", "RoadmapPlanar", "CDT", "Midpoints",
           "Centroids", "Voronoi", "CustomRoadmap", "ROADMAP_CLASSES", "make_roadmap"]
