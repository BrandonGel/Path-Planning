"""
@file: visualizer_3d.py
@author: Ho Brandon 
@update: 2026.01.19
@description: Visualizer for 2D maps (fetch from python_motion_planning by Wu Maojia, Yang Haodong)
"""
from typing import List, Tuple, Union, Iterable, Optional

import numpy as np
import pyvista as pv

from python_motion_planning.common.env import TYPES, Grid, Node
from python_motion_planning.common.visualizer.visualizer_3d import Visualizer3D as BaseVisualizer3D

class Visualizer3D(BaseVisualizer3D):
    """
    Simple 3D visualizer for motion planning using pyvista.

    Args:
        window_size: Window size (width, height) (pyvista window size, unit: pixel).
        off_screen: `off_screen` argument for pyvista. Renders off screen when True. Useful for automated screenshots.
        show_axes: Whether to show axes for pyvista.
        cmap_dict: Color map for 3d voxel visualization.
    """
    def __init__(self,  
                window_size: tuple = (1200, 900),
                off_screen: bool = False,
                show_axes: bool = True,
                cmap_dict: dict = {
                    TYPES.FREE: "#ffffff",
                    TYPES.OBSTACLE: "#000000",
                    TYPES.START: "#ff0000",
                    TYPES.GOAL: "#1155cc",
                    TYPES.INFLATION: "#ffccff",
                    TYPES.EXPAND: "#eeeeee",
                    TYPES.CUSTOM: "#bbbbbb",
                }
            ):
        super().__init__(window_size, off_screen, show_axes, cmap_dict)

    @staticmethod
    def _iter_goal_like(value: Union[None, Tuple[float, ...], List[Tuple[float, ...]]]) -> Iterable[Tuple[float, ...]]:
        """
        Normalize either a single tuple or a list of tuples into an iterator of tuples.
        """
        if value is None:
            return []
        if isinstance(value, list):
            return [v for v in value if v is not None]
        return [value]

    def plot_road_map(
        self,
        map_: Grid,
        nodes: List[Node],
        road_map: List[List[int]],
        node_color: str = "#8c564b",
        edge_color: str = "#e377c2",
        node_size: float = 0.2,
        linewidth: float = 2.0,
        node_alpha: float = 1.0,
        edge_alpha: float = 0.3,
    ) -> None:
        """
        Plot a roadmap (nodes + edges) in 3D using pyvista.

        This mirrors `Visualizer2D.plot_road_map()` but uses pyvista primitives.

        Notes:
        - `nodes[i].current` is assumed to be in world coordinates (floats).
        - If you want to plot map-frame points, convert them before passing in.
        """
        if len(nodes) == 0:
            return

        # ---- Nodes (as spheres) ----
        pts = np.array([n.current for n in nodes], dtype=float)
        cloud = pv.PolyData(pts)
        sphere = pv.Sphere(radius=float(node_size))
        glyph = cloud.glyph(geom=sphere, scale=False)
        self.pv_plotter.add_mesh(
            glyph,
            color=node_color,
            opacity=float(node_alpha),
            show_edges=False,
        )

        # ---- Edges ----
        # Build a set of undirected edges to avoid duplicates.
        edges = []
        for i, nbrs in enumerate(road_map):
            for j in nbrs:
                if 0 <= j < len(nodes) and i != j:
                    a, b = (i, j) if i < j else (j, i)
                    edges.append((a, b))
        if edges:
            edges = sorted(set(edges))
            # Build pyvista PolyData with line cells
            # cells format: [2, p0, p1, 2, p0, p1, ...]
            cells = np.hstack([[2, a, b] for (a, b) in edges]).astype(np.int64)
            poly = pv.PolyData(pts)
            poly.lines = cells
            self.pv_plotter.add_mesh(
                poly,
                color=edge_color,
                opacity=float(edge_alpha),
                line_width=float(linewidth),
            )
