"""
@file: visualizer_2d.py
@author: Ho Brandon 
@update: 2026.01.19
@description: Visualizer for 2D maps (fetch from python_motion_planning by Wu Maojia, Yang Haodong)
"""
from typing import List
from python_motion_planning.common.env import TYPES,  Grid,  Node
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib import animation
import re
import numpy as np
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
        self.figsize = figsize
        self.fig.subplots_adjust(left=0,right=1,bottom=0,top=1, wspace=None, hspace=None)

    def set_fig_size(self, width: float, height: float, aspect_ratio: float = 0.0):
        if aspect_ratio != 0.0:
            height = width/aspect_ratio
        self.fig.set_size_inches(width, height)

    def close(self):
        plt.close(self.fig)

    def plot_road_map(self,
                        map_: Grid,
                        nodes: List[Node],
                        road_map: List[List[int]],
                        node_color: str = "#8c564b", 
                        edge_color: str = "#e377c2", 
                        node_size: float = 1, 
                        linewidth: float = 1.0, 
                        node_alpha: float = 1.0,
                        edge_alpha: float = 0.3,
                        map_frame: bool = True) -> None:
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
        if map_frame:
            x_coords = [map_.map_to_world(node.current)[0] for node in nodes]
            y_coords = [map_.map_to_world(node.current)[1] for node in nodes]
        else:
            x_coords = [node.current[0] for node in nodes]
            y_coords = [node.current[1] for node in nodes]
        for i, edges in enumerate(road_map):
            if len(edges) == 0:
                continue
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
                    start = map_.map_to_world(start)
                    if start is not None and len(start) >= 2:
                        self.ax.scatter(start[0], start[1], c='red', s=20, alpha=1, zorder=self.zorder['expand_tree_node'], label='Start' if start == map_.start[0] else '')
            else:
                # Single start position (not a list)
                start = map_.map_to_world(map_.start)
                if len(start) >= 2:
                    self.ax.scatter(start[0], start[1], c='red', s=20, alpha=1, zorder=self.zorder['expand_tree_node'], label='Start')

        if hasattr(map_, 'goal') and map_.goal is not None:
            if isinstance(map_.goal, list) and len(map_.goal) > 0:
                for goal in map_.goal:
                    goal = map_.map_to_world(goal)
                    if goal is not None and len(goal) >= 2:
                        self.ax.scatter(goal[0], goal[1], c='blue', s=20, alpha=1, zorder=self.zorder['expand_tree_node'], label='Goal')
            else:
                goal = map_.map_to_world(map_.goal)
                if len(goal) >= 2:
                    self.ax.scatter(goal[0], goal[1], c='blue', s=20, alpha=1, zorder=self.zorder['expand_tree_node'], label='Goal')

    def animate(self,file_name,map, schedule, road_map=None, skip_frames=1, intermediate_frames=3,speed=1):

        combined_schedule = {}
        combined_schedule.update(schedule["schedule"])
        self.set_fig_size(self.figsize[0], self.figsize[1], map.shape[0]/map.shape[1])

        # Draw static map and paths
        Colors = ['orange', 'blue', 'green']
        self.ax.clear()
        self.plot_grid_map(map)
        if road_map is not None and map.nodes != []:
            self.plot_road_map(map,map.nodes,road_map)

        patches = []
        artists = []
        agents = dict()
        agent_names = dict()

        # create agents:
        T = 0
                
        # draw goals first
        for name in schedule["schedule"]:
            start = schedule["schedule"][name][0]
            x,y = start["x"], start["y"]
            agents[name] = Circle((x, y), 0.3, facecolor=Colors[0], edgecolor='black',zorder=self.zorder['robot_circle'])
            agents[name].original_face_color = Colors[0]
            patches.append(agents[name])

            T = max(T, schedule["schedule"][name][-1]["t"])//skip_frames
            text_name = re.findall(r'\d+', name)[-1]
            agent_names[name] = self.ax.text(x, y, text_name ,zorder=self.zorder['robot_text'])
            agent_names[name].set_horizontalalignment('center')
            agent_names[name].set_verticalalignment('center')
            artists.append(agent_names[name])

        colors = ['tab:green', 'tab:orange']
        agent_num = 0
        for agent_name, agent in combined_schedule.items():
            pos = np.array([[state['x'],state['y']] for state in agent])
            self.ax.plot(pos[:,0], pos[:,1], color=colors[agent_num], zorder=self.zorder['traj'],linewidth=3)
            agent_num += 1
            agent_num %= len(colors)

        self.ax.set_axis_off()

        def init_func():
            for p in patches:
                self.ax.add_patch(p)
            for a in artists:
                self.ax.add_artist(a)
            return patches + artists

        def animate_func(i):
            for agent_name, agent in combined_schedule.items():
                pos = getState(i*skip_frames / intermediate_frames, agent)
                p = (pos[0], pos[1])
                agents[agent_name].center = p
                agent_names[agent_name].set_position(p)

            # reset all colors
            for _,agent in agents.items():
                agent.set_facecolor(agent.original_face_color)

            # check drive-drive collisions
            agents_array = [agent for _,agent in agents.items()]
            for i in range(0, len(agents_array)):
                for j in range(i+1, len(agents_array)):
                    d1 = agents_array[i]
                    d2 = agents_array[j]
                    pos1 = np.array(d1.center)
                    pos2 = np.array(d2.center)
                    if np.linalg.norm(pos1 - pos2) < 0.7:
                        d1.set_facecolor('red')
                        d2.set_facecolor('red')
                        print("COLLISION! (agent-agent) ({}, {})".format(i, j))

            return patches + artists

        def getState(t, d):
            idx = 0
            while idx < len(d) and d[idx]["t"] < t:
                idx += 1
            if idx == 0:
                return np.array([float(d[0]["x"]), float(d[0]["y"])])
            elif idx < len(d):
                posLast = np.array([float(d[idx-1]["x"]), float(d[idx-1]["y"])])
                posNext = np.array([float(d[idx]["x"]), float(d[idx]["y"])])
            else:
                return np.array([float(d[-1]["x"]), float(d[-1]["y"])])
            dt = d[idx]["t"] - d[idx-1]["t"]
            t = (t - d[idx-1]["t"]) / dt
            pos = (posNext - posLast) * t + posLast
            return pos
                
        anim = animation.FuncAnimation(self.fig, animate_func,
                                init_func=init_func,
                                frames=int(T+1) * intermediate_frames,
                                interval=100,
                                blit=True)

        anim.save(
            file_name,
            "ffmpeg",
            fps=intermediate_frames * speed,
            dpi=200)
        self.set_fig_size(self.figsize[0], self.figsize[1])

