"""
@file: visualizer_2d.py
@author: Ho Brandon 
@update: 2026.01.19
@description: Visualizer for 2D maps (fetch from python_motion_planning by Wu Maojia, Yang Haodong)
"""
from typing import List
from python_motion_planning.common.env import TYPES,  Grid,  Node
from python_motion_planning.common.visualizer.visualizer_2d import Visualizer2D as BaseVisualizer2D

class Visualizer2D(BaseVisualizer2D):
    """
    Simple visualizer for motion planning using matplotlib.

    Args:
        figname: Figure name (window title).
        figsize: Figure size (width, height) (matplotlib figure size, unit: inch).
        cmap_dict: Color map for 2d visualization.
        zorder: Zorder for 2d matplotlib visualization.
    """

    def __init__(self, 
                figname: str = "", 
                figsize: tuple = (10, 8), 
                cmap_dict: dict = {
                    TYPES.FREE: "#ffffff",
                    TYPES.OBSTACLE: "#000000",
                    TYPES.START: "#ff0000",
                    TYPES.GOAL: "#1155cc",
                    TYPES.INFLATION: "#ffccff",
                    TYPES.EXPAND: "#eeeeee",
                    TYPES.CUSTOM: "#bbbbbb",
                },
                zorder: dict = {
                    'grid_map': 10,
                    'voxels': 10,
                    'esdf': 20,
                    'road_map': 25,
                    'expand_tree_edge': 30,
                    'expand_tree_node': 40,
                    'path_2d': 50,
                    'path_3d': 700,
                    'traj': 60,
                    'lookahead_pose_node': 70,
                    'lookahead_pose_orient': 80,
                    'pred_traj': 90,
                    'robot_circle': 100,
                    'robot_orient': 110,
                    'robot_text': 120,
                    'env_info_text': 10000
                }
            ):
        super().__init__(figname, figsize, cmap_dict, zorder) 

    def plot_road_map(self,
                        map_: Grid,nodes: List[Node],
                        road_map: List[List[int]],
                        node_color: str = "#8c564b", 
                        edge_color: str = "#e377c2", 
                        node_size: float = 1, 
                        linewidth: float = 1.0, 
                        node_alpha: float = 1.0,
                        edge_alpha: float = 0.3,) -> None:
        """
        Plot the roadmap.

        Args:
            road_map: List of lists containing edge connections.
            node_color: Color of the nodes.
            edge_color: Color of the edges.
            node_size: Size of the nodes.
            linewidth: Width of the edges.
            node_alpha: Alpha of the nodes.
            edge_alpha: Alpha of the edges.
        """
        # Plot all edges
        x_coords = [node.current[0] for node in nodes]
        y_coords = [node.current[1] for node in nodes]
        for i, edges in enumerate(road_map):
            x1, y1 = x_coords[i], y_coords[i]
            # if not (x1 in points[ind1][:,0] and y1 in points[ind1][:,1]) and not (x1 in points[ind2][:,0] and x1 in points[ind2][:,1]):
            #     continue
            for edge_idx in edges:
                if edge_idx < len(x_coords):  # Safety check
                    x2, y2 = x_coords[edge_idx], y_coords[edge_idx]
                    self.ax.plot([x1, x2], [y1, y2], edge_color, linewidth=linewidth, alpha=edge_alpha, zorder=self.zorder['road_map'])

        # Plot all nodes
        self.ax.scatter(x_coords, y_coords, c=node_color, s=node_size, alpha=node_alpha, zorder=self.zorder['road_map'], label='Sample nodes')
        
        # Plot start nodes (handle both list and single value)
        if hasattr(map_, 'start') and map_.start is not None:
            if isinstance(map_.start, list) and len(map_.start) > 0:
                # Multiple start positions
                for start in map_.start:
                    if start is not None and len(start) >= 2:
                        self.ax.scatter(start[0], start[1], c='red', s=20, alpha=1, zorder=self.zorder['expand_tree_node'], label='Start' if start == map_.start[0] else '')
            else:
                # Single start position (not a list)
                start = map_.start
                if len(start) >= 2:
                    self.ax.scatter(start[0], start[1], c='red', s=20, alpha=1, zorder=self.zorder['expand_tree_node'], label='Start')
        