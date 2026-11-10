"""
@file: visualizer_2d.py
@author: Ho Brandon 
@update: 2026.01.19
@description: Visualizer for 2D maps (fetch from python_motion_planning by Wu Maojia, Yang Haodong)
"""
from typing import List, Mapping
from python_motion_planning.common.env import TYPES,  Grid,  Node
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle as MplRectangle

from path_planning.utils.agent_shape import AlignedShape, ConvexPolygon, Disc, oriented_pair_clearance, resolve_shape, schedule_headings, turn_angle
import matplotlib.pyplot as plt
from matplotlib import animation
import re
import numpy as np
from python_motion_planning.common.visualizer.visualizer_2d import Visualizer2D as BaseVisualizer2D
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from typing import Tuple

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
                    'density_map': 15,
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
        base = plt.cm.YlOrRd
        colors = base(range(256))
        colors[0] = [1, 1, 1, 1]  # force white at the bottom
        self.cmap_density= mcolors.LinearSegmentedColormap.from_list(
            "white_ylorrd", colors
        )
        self.figsize = figsize
        self.fig.tight_layout()
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)

    def set_fig_size(self, width: float, height: float, aspect_ratio: float = 0.0):
        if aspect_ratio != 0.0:
            height = width/aspect_ratio
        self.fig.set_size_inches(width, height)

    def close(self):
        plt.close(self.fig)
        
    def _grid_map_extent(self, grid_map: Grid) -> list:
        """
        imshow ``extent`` for a grid map: the raw world bounds, so cell (r, c)
        is drawn over its true world footprint ``[r, r+1] x [c, c+1]`` in both
        discrete and continuous modes. (An earlier -resolution/2 shift in
        discrete mode centered cells on their node coordinates instead, which
        drew obstacles/inflation half a cell down-left of their true world
        footprints — world obstacle centers live at half-integers — and made
        every discrete figure look offset against the roadmap/paths.)
        """
        return [
            grid_map.bounds[0][0], grid_map.bounds[0][1],
            grid_map.bounds[1][0], grid_map.bounds[1][1],
        ]

    def _overlay_offset(self, grid_map) -> float:
        """World offset placing index-frame overlays at cell centers: a
        discrete node k represents the whole cell [k, k+1], so drawn
        nodes/paths/markers shift by +resolution/2. Zero for continuous."""
        if getattr(grid_map, "use_discrete_space", False):
            return 0.5 * float(getattr(grid_map, "resolution", 1.0))
        return 0.0

    # Note on plot_path (inherited from the base class): with map_frame=True it
    # converts index-frame points via map_to_world, which for discrete maps
    # already returns CELL CENTERS (k + resolution/2) — so no extra overlay
    # offset is needed there. Callers must pass map_frame=False for paths that
    # are already in world coordinates (converting twice shifts them +res/2).

    def plot_grid_map(self, grid_map: Grid, equal: bool = False,
                        show_esdf: bool = False, alpha_esdf: float = 0.5,masked_map = None) -> None:
        '''
        Plot grid map with static obstacles.

        Args:
            map: Grid map or its type map.
            equal: Whether to set axis equal.
            show_esdf: Whether to show esdf.
            alpha_esdf: Alpha of esdf.
        '''
        if grid_map.dim != 2:
            raise ValueError(f"Grid map dimension must be 2.")

        self.grid_map = grid_map
        self.dim = grid_map.dim
        type_data = grid_map.type_map.data.copy()

        # Cells are drawn over their true world footprint [i, i+1] in both
        # modes (see _grid_map_extent): world obstacle centers live at
        # half-integers, so cell i spans [i, i+1] and discrete node i sits at
        # that cell's lower-left corner.
        extent = self._grid_map_extent(grid_map)

        self.ax.imshow(
            np.transpose(type_data),
            cmap=self.cmap,
            norm=self.norm,
            origin='lower',
            interpolation='nearest',
            extent=extent,
            zorder=self.zorder['grid_map'],
            )

        if show_esdf:   # draw esdf hotmap
            mappable=self.ax.imshow(
                np.transpose(grid_map.esdf),
                cmap="jet",
                origin="lower",
                interpolation="nearest",
                extent=extent,
                alpha=alpha_esdf,
                zorder=self.zorder['esdf'],
            )
            self.fig.colorbar(mappable,label="ESDF distance")
            
        if equal: 
            self.ax.axis("equal")

        if getattr(grid_map, "speed_zones", None):
            self.plot_speed_zones(grid_map)

    def plot_speed_zones(self, grid_map, alpha: float = 0.28, color: str = "#ff7f0e", label: bool = True):
        """Shade the map's speed zones (``grid_map.speed_zones``, world-frame
        rectangles) just above the obstacle grid and print each limit."""
        zones = getattr(grid_map, "speed_zones", None) or []
        for z in zones:
            self.ax.add_patch(MplRectangle((z.x0, z.y0), z.width, z.height, facecolor=color, edgecolor=color,
                                           alpha=alpha, linewidth=1.0, linestyle="--",
                                           zorder=self.zorder['grid_map'] + 1))
            if label:
                txt = f"v \u2264 {z.speed_limit:g}" if z.name is None else f"{z.name}: v \u2264 {z.speed_limit:g}"
                self.ax.text(0.5 * (z.x0 + z.x1), z.y1 - 0.15 * z.height, txt, ha="center", va="top",
                             fontsize=7, color="#7f3f00", zorder=self.zorder['grid_map'] + 1)
        
    def plot_road_map(self,
                        map_: Grid,
                        nodes: List[Node],
                        road_map: List[List[int]],
                        node_color: str = "#8c564b", 
                        edge_color: str = "gray", 
                        node_size: float = 0, 
                        start_size: float = 0,
                        goal_size: float = 0,
                        linewidth: float = 1.0, 
                        node_alpha: float = 1.0,
                        edge_alpha: float = 0.3,
                        node_value: List[float] = None,
                        show_edge: bool = True,
                        cmap: mcolors.Colormap = None,
                        map_frame: bool = True,
                        extra_special_specs: List[Tuple[List[Tuple[float, float]], str, str, float, str]] = None,
                        special_patch_radius: float = None) -> None:
        """
        Plot the roadmap.

        special_patch_radius: when set (> 0), the special endpoints (start / goal /
        pickups / ... and extra_special_specs) are drawn as matplotlib patches of
        that radius in WORLD units with a black outline — 'o' -> Circle,
        '^' -> triangle, 's' -> square — so they scale with the map like the agent
        circles do. In this mode the spec's scatter size only acts as a switch:
        size <= 0 hides that spec (as s=0 does for scatter), any positive size
        draws it at special_patch_radius. Default None keeps the scatter markers
        (sizes in points^2).

        Args:
            road_map: List of lists containing edge connections.
            node_color: Color of the nodes.
            edge_color: Color of the edges.
            node_size: Size of the nodes.
            linewidth: Width of the edges.
            node_alpha: Alpha of the nodes.
            edge_alpha: Alpha of the edges.
        """
        # node.current is stored in world coordinates. plot_grid_map draws
        # cell (i, j) over its true world footprint [i, i+1] x [j, j+1], and a
        # DISCRETE node i semantically occupies that whole cell — so with
        # map_frame=True (index-frame roadmaps) overlays shift +resolution/2
        # to the cell centers. Continuous roadmaps (map_frame=False) carry
        # true world coordinates and are plotted directly.
        off = self._overlay_offset(map_) if map_frame else 0.0
        x_coords = np.array([node.current[0] for node in nodes]) + off
        y_coords = np.array([node.current[1] for node in nodes]) + off
        if show_edge:
            for i, edges in enumerate(road_map):
                if len(edges) == 0:
                    continue
                x1, y1 = x_coords[i], y_coords[i]
                for edge_idx in edges:
                    if edge_idx < len(x_coords):  # Safety check
                        x2, y2 = x_coords[edge_idx], y_coords[edge_idx]
                        self.ax.plot([x1, x2], [y1, y2], edge_color, linewidth=linewidth, alpha=edge_alpha, zorder=self.zorder['road_map'])

        # Collect special endpoints (start/goal/task endpoints). Waypoints that
        # coincide with one of these are NOT drawn as generic sample nodes; the
        # special marker/color below stands in for them instead. All coordinates
        # are world coords (same frame as node.current).
        start_size = start_size if start_size > 0 else 3*node_size
        goal_size = goal_size if goal_size > 0 else 3*node_size
        special_specs = []  # (points, color, marker, size, label)

        def _as_point_list(value):
            if value is None:
                return []
            if isinstance(value, list) and (len(value) == 0 or isinstance(value[0], (list, tuple, np.ndarray))):
                return [v for v in value if v is not None and len(v) >= 2]
            return [value] if len(value) >= 2 else []


        # Task endpoints (visualization-only metadata set via GraphSampler.set_endpoints).
        for attr, color, marker, label in (
            ('pickups', 'green', '^', 'Pickup'),
            ('deliveries', 'blue', 'o', 'Delivery'),
            ('parking', 'gray', 's', 'Parking'),
        ):
            special_specs.append((_as_point_list(getattr(map_, attr, None)), color, marker, goal_size, label))
        # Caller-supplied endpoint specs (points, color, marker, size, label), e.g.
        # per-agent coloured starts/goals. A spec labelled 'Start' / 'Goal'
        # (case-insensitive) replaces the default red/blue markers for that kind.
        extra_special_specs = list(extra_special_specs or [])
        extra_labels = {str(spec[4]).lower() for spec in extra_special_specs}
        if 'start' not in extra_labels:
            special_specs.append((_as_point_list(getattr(map_, 'start', None)), 'red', 'o', start_size, 'Start'))
        if 'goal' not in extra_labels:
            special_specs.append((_as_point_list(getattr(map_, 'goal', None)), 'blue', 'o', goal_size, 'Goal'))
        seen_labels = set()
        for pts, color, marker, size, label in extra_special_specs:
            # One legend entry per label even though each agent has its own spec.
            special_specs.append((_as_point_list(pts), color, marker, size,
                                  label if label not in seen_labels else ''))
            seen_labels.add(label)
        
        # Mask out waypoints that coincide with any special endpoint.
        is_special = np.zeros(len(nodes), dtype=bool)
        for pts, _color, _marker, _size, _label in special_specs:
            for pt in pts:
                is_special |= np.isclose(x_coords, pt[0] + off) & np.isclose(y_coords, pt[1] + off)

        keep = ~is_special
        # Plot the remaining (generic) sample nodes.
        if node_value is not None:
            node_value = np.asarray(node_value)
            vmin = min(0, np.min(node_value))
            vmax = max(1, np.max(node_value))
            self.ax.scatter(x_coords[keep], y_coords[keep], c=node_value[keep], edgecolors='black', s=node_size, alpha=node_alpha, zorder=self.zorder['road_map'], label='Sample nodes', cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            # node_color may be one colour for all nodes, or one colour PER NODE (a
            # sequence / (N, 3|4) array aligned with `nodes`, e.g. cluster colours);
            # per-node colours are masked like the coordinates so they stay aligned
            # once the special endpoints are dropped.
            per_node = False
            if not isinstance(node_color, str):
                nc = np.asarray(node_color)
                # (N, 3|4) colour rows, or N colour names; a lone RGB(A) tuple is 1-D numeric.
                per_node = len(nc) == len(nodes) and (nc.ndim == 2 or nc.dtype.kind in 'US')
            c = np.asarray(node_color)[keep] if per_node else node_color
            self.ax.scatter(x_coords[keep], y_coords[keep], c=c, edgecolors='black', s=node_size, alpha=node_alpha, zorder=self.zorder['road_map'], label='Sample nodes')

        # Plot the special endpoints with their respective marker and color.
        use_patches = special_patch_radius is not None and special_patch_radius > 0
        for pts, color, marker, size, label in special_specs:
            if use_patches and size is not None and size <= 0:
                continue  # hidden spec (still suppresses the default marker of its label)
            for k, pt in enumerate(pts):
                xy = (pt[0] + off, pt[1] + off)
                patch = self._special_patch(xy, marker, special_patch_radius) if use_patches else None
                if patch is not None:
                    patch.set(facecolor=color, edgecolor='k', linewidth=1, alpha=1,
                              zorder=self.zorder['expand_tree_node'],
                              label=label if k == 0 else '_nolegend_')
                    self.ax.add_patch(patch)
                else:  # scatter fallback (default mode, or a marker without a patch shape)
                    self.ax.scatter(xy[0], xy[1], color=color, marker=marker, s=size,
                                    alpha=1, zorder=self.zorder['expand_tree_node'],
                                    label=label if k == 0 else '')

    @staticmethod
    def _special_patch(xy, marker: str, radius: float):
        """Patch (world units) for a scatter-style marker, or None if unsupported."""
        if marker == 'o':
            return Circle(xy, radius)
        if marker == '^':
            return RegularPolygon(xy, 3, radius=radius)
        if marker == 'v':
            return RegularPolygon(xy, 3, radius=radius, orientation=np.pi)
        if marker == 's':
            return RegularPolygon(xy, 4, radius=radius, orientation=np.pi / 4)
        if marker == 'D':
            return RegularPolygon(xy, 4, radius=radius)
        return None


    def add_legend(self, loc: str = 'center left', bbox_to_anchor: tuple = (1.01, 0.5),
                   title: str = None, reserve: float = 0.18) -> None:
        """
        Add a legend on the right side of the plot, outside the axes.

        Collects the labeled artists already drawn on the axes (e.g. Start, Goal,
        Pickup, Delivery, Parking, Sample nodes, Trajectory) and de-duplicates
        repeated labels so each entry appears once.

        Args:
            loc: Legend anchor location (relative to bbox_to_anchor).
            bbox_to_anchor: Legend position in axes coordinates; >1 places it
                outside the axes to the right.
            title: Optional legend title.
            reserve: Fraction of the figure width to reserve on the right for the
                legend. Because the figure is saved without bbox_inches='tight'
                (see animate/savefig), the axes must be shrunk so the outside
                legend stays within the figure bounds and is not clipped. Set to
                0 to skip reserving space.
        """
        handles, labels = self.ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l and l not in seen:
                seen[l] = h
        if seen:
            if reserve:
                # Shrink the axes so the legend (drawn outside, to the right)
                # remains inside the figure and is not cut off when saved.
                self.fig.subplots_adjust(right=1 - reserve)
            self.ax.legend(list(seen.values()), list(seen.keys()),
                           loc=loc, bbox_to_anchor=bbox_to_anchor,
                           borderaxespad=0., framealpha=0.9, title=title)

    def animate(self,file_name,map, schedule, road_map=None, skip_frames=1, intermediate_frames=3,speed=1,map_frame=True,radius=0.0,
                show_paths=True, show_legend=True, rack_pts=None, roadmap_specials=True,
                background_img=None, background_extent=None,
                agent_colors=None,extra_special_specs=None,special_patch_radius=None,
                nodes=None, writer=None, dpi=200, view_bounds=None, annotations=None,
                show_time=False):
        """nodes: optional node list aligned with ``road_map`` (e.g. a k-hop
        subgraph from k_hop_subgraph). Defaults to map.nodes.
        writer: matplotlib animation writer name. Defaults to 'ffmpeg' when it
        is available, otherwise 'pillow' (use a .gif file_name in that case).
        view_bounds: optional ``((x_min, x_max), (y_min, y_max))`` world window to
        zoom the animation on (the figure aspect follows the window).
        annotations: optional list of ``(x, y, text)`` or ``(x, y, text, text_kwargs)``
        static labels drawn in world coordinates over the map.
        show_time: draw a ``t = .. s`` stamp; ``True`` = top-left corner, or one of
        ``"top-left" | "top-right" | "bottom-left" | "bottom-right"``."""
        _nodes = map.nodes if nodes is None else nodes

        combined_schedule = {}
        combined_schedule.update(copy.deepcopy(schedule["schedule"]))

        if map_frame:
            # map_to_world returns cell centers for discrete maps — no extra
            # overlay offset on top of it.
            for agent_name, agent in combined_schedule.items():
                for state in agent:
                    state["x"], state["y"] = map.map_to_world((state["x"], state["y"]))
        # Reserve space on the right for the legend only when a legend is drawn;
        # otherwise let the axes fill the whole figure (no white margin).
        _right = 0.82 if show_legend else 1.0
        self.fig.subplots_adjust(left=0,right=_right,bottom=0,top=1, wspace=None, hspace=None)
        if view_bounds is not None:
            (_vx0, _vx1), (_vy0, _vy1) = view_bounds
            self.set_fig_size(self.figsize[0], self.figsize[1], (_vx1 - _vx0) / (_vy1 - _vy0))
        else:
            self.set_fig_size(self.figsize[0], self.figsize[1], map.shape[0]/map.shape[1])

        # Size the agent markers (Circle, in data units) to visually match the rack
        # scatter markers (s=RACK_S points^2). Convert the marker's point-diameter to
        # data units using the axes' data-width / pixel-width.
        # RACK_S = 28.0
        # _data_w = float(map.bounds[0][1] - map.bounds[0][0])
        # _ax_w_pts = self.figsize[0] * _right * 72.0
        # _marker_dia_pts = np.sqrt(RACK_S)
        # agent_radius_vis = (_marker_dia_pts / 2.0) * (_data_w / _ax_w_pts)
        # ``radius``: one value for every agent, or ``{agent_name: radius | ConvexPolygon
        # | AlignedShape}`` for heterogeneous agents (unknown names draw as points). A
        # polygon footprint (rectangle / regular polygon) is drawn as a polygon patch;
        # an ``AlignedShape`` is rotated to the heading the agent holds on each segment
        # (``heading`` keys of the schedule, else carried from its moves). Pairs
        # involving a polygon are collision-tested with the combined footprint
        # ``P_i (+) (-P_j)`` at those headings.
        if isinstance(radius, Mapping):
            _footprint_of = lambda n: radius.get(n, 0.0)
        else:
            _footprint_of = lambda n: radius
        _POLY = (ConvexPolygon, AlignedShape)
        def _radius_of(n):
            fp = _footprint_of(n)
            if isinstance(fp, _POLY):
                return float(fp.bounding_radius)
            if isinstance(fp, Disc):
                return float(fp.radius)
            return float(fp)
        _pair_cache = {}
        def _pair_collides(n1, p1, h1, n2, p2, h2):
            f1, f2 = _footprint_of(n1), _footprint_of(n2)
            if isinstance(f1, _POLY) or isinstance(f2, _POLY):
                key = (n1, h1, n2, h2)
                combo = _pair_cache.get(key)
                if combo is None:
                    s1 = f1 if isinstance(f1, _POLY) else Disc(_radius_of(n1))
                    s2 = f2 if isinstance(f2, _POLY) else Disc(_radius_of(n2))
                    combo = _pair_cache[key] = oriented_pair_clearance(s1, h1, s2, h2)
                rel = np.asarray(p2, float).reshape(1, 2) - np.asarray(p1, float).reshape(1, 2)
                if isinstance(combo, Disc):  # both resolved to discs (e.g. rotating envelope)
                    return bool(np.linalg.norm(rel) < combo.radius)
                return bool(combo.contains(rel)[0])
            return np.linalg.norm(np.asarray(p1) - np.asarray(p2)) < _radius_of(n1) + _radius_of(n2)
        _headings = {
            name: schedule_headings(path) if isinstance(_footprint_of(name), AlignedShape) else None
            for name, path in schedule["schedule"].items()
        }

        # Draw static map and paths
        if agent_colors is None:
            Colors = ['orange', 'blue', 'green']
        else:
            Colors = agent_colors
        self.ax.clear()
        self.plot_grid_map(map)
        for ann in (annotations or []):
            ax_, ay_, atext = ann[0], ann[1], ann[2]
            akw = dict(fontsize=8, ha="left", va="bottom", color="black", zorder=self.zorder['robot_text'])
            if len(ann) > 3 and ann[3]:
                akw.update(ann[3])
            self.ax.text(ax_, ay_, atext, **akw)
        # Optionally draw a detailed background image (e.g. the photographic occupancy
        # map) over the plain obstacle grid. It is already oriented/rotated by the caller
        # and stretched to the env world ``background_extent`` so it aligns with the
        # roadmap; drawn just above the grid_map but below the roadmap/agents.
        if background_img is not None and background_extent is not None:
            self.ax.imshow(background_img, origin='upper', extent=background_extent,
                           interpolation='nearest', zorder=self.zorder['grid_map'] + 2)
        if road_map is not None and len(_nodes) > 0:
            if roadmap_specials:
                # special_patch_radius (world units): draw start/goal/... as outlined
                # patches instead of scatter markers — see plot_road_map.
                self.plot_road_map(map,_nodes,road_map,map_frame=map_frame,extra_special_specs=extra_special_specs,
                                   special_patch_radius=special_patch_radius)
            else:
                # Roadmap nodes + edges only (no start/goal/parking markers, no legend).
                _coords = np.array([n.current for n in _nodes], dtype=float)
                for _i, _nbrs in enumerate(road_map):
                    for _j in _nbrs:
                        if _j > _i and _j < len(_coords):
                            self.ax.plot([_coords[_i,0],_coords[_j,0]],[_coords[_i,1],_coords[_j,1]],
                                         color="#e377c2", linewidth=0.5, alpha=0.3,
                                         zorder=self.zorder['road_map'])
                self.ax.scatter(_coords[:,0], _coords[:,1], c="#8c564b", s=6,
                                zorder=self.zorder['road_map'])
        if rack_pts is not None and len(rack_pts):
            _rp = np.asarray(rack_pts, dtype=float)
            self.ax.scatter(_rp[:,0], _rp[:,1], c="blue", marker="s", s=RACK_S,
                            edgecolors="white", linewidths=0.5,
                            zorder=self.zorder['expand_tree_node'])
        patches = []
        artists = []
        agents = dict()
        agent_names = dict()
        time_text = None
        if show_time:
            corner = show_time if isinstance(show_time, str) else "top-left"
            _v, _h = corner.split("-")
            time_text = self.ax.text(0.01 if _h == "left" else 0.99, 0.99 if _v == "top" else 0.01, "",
                                     transform=self.ax.transAxes, ha=_h, va=_v,
                                     fontsize=10, fontweight="bold", zorder=self.zorder['robot_text'],
                                     bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
            artists.append(time_text)

        # create agents:
        T = 0
                
        # draw agent first
        for ii,name in enumerate(schedule["schedule"]):
            start = schedule["schedule"][name][0]
            x,y = start["x"], start["y"]
            color = Colors[ii%len(Colors)]

            fp = _footprint_of(name)
            if isinstance(fp, _POLY):
                h0 = _headings[name][0] if _headings.get(name) else None
                agents[name] = MplPolygon(resolve_shape(fp, h0).points + np.array([x, y]), closed=True,
                                          facecolor=color, edgecolor='black', zorder=self.zorder['robot_circle'])
                agents[name].footprint = fp
                agents[name].center = (x, y)
                agents[name].heading = h0
            else:
                agents[name] = Circle((x, y), _radius_of(name), facecolor=color, edgecolor='black',zorder=self.zorder['robot_circle'])
            agents[name].original_face_color = color
            patches.append(agents[name])

            T = max(T, schedule["schedule"][name][-1]["t"])//skip_frames
            # Number text removed (overlapped the agent marker); keep an empty artist so
            # the blit machinery (init_func/animate_func) stays unchanged.
            agent_names[name] = self.ax.text(x, y, "" ,zorder=self.zorder['robot_text'])
            agent_names[name].set_horizontalalignment('center')
            agent_names[name].set_verticalalignment('center')
            artists.append(agent_names[name])

        if show_paths:
            for idx, (agent_name, agent) in enumerate(combined_schedule.items()):
                pos = np.array([[state['x'],state['y']] for state in agent])
                color = Colors[idx%len(Colors)]
                self.ax.plot(pos[:,0], pos[:,1], color=color, zorder=self.zorder['traj'],linewidth=3,
                             label='Trajectory' if idx == 0 else '')

        if show_legend:
            self.add_legend()
        self.ax.set_axis_off()
        if view_bounds is not None:
            self.ax.set_xlim(_vx0, _vx1)
            self.ax.set_ylim(_vy0, _vy1)

        def init_func():
            for p in patches:
                self.ax.add_patch(p)
            for a in artists:
                self.ax.add_artist(a)
            return patches + artists

        def animate_func(i):
            t_now = i*skip_frames / intermediate_frames
            if time_text is not None:
                time_text.set_text(f"t = {t_now:5.1f} s")
            for agent_name, agent in combined_schedule.items():
                pos = getState(t_now, agent)
                p = (pos[0], pos[1])
                patch = agents[agent_name]
                if isinstance(patch, MplPolygon):
                    h = getHeading(t_now, agent, _headings.get(agent_name))
                    patch.set_xy(resolve_shape(patch.footprint, h).points + np.asarray(p))
                    patch.center = p
                    patch.heading = getHeading(t_now, agent, _headings.get(agent_name), for_collision=True)
                else:
                    patch.center = p
                agent_names[agent_name].set_position(p)

            # reset all colors
            for _,agent in agents.items():
                agent.set_facecolor(agent.original_face_color)

            # check drive-drive collisions
            agents_array = [agent for _,agent in agents.items()]
            agent_names_arr = list(agents.keys())
            for i in range(0, len(agents_array)):
                for j in range(i+1, len(agents_array)):
                    d1 = agents_array[i]
                    d2 = agents_array[j]
                    pos1 = np.array(d1.center)
                    pos2 = np.array(d2.center)
                    if _pair_collides(agent_names_arr[i], pos1, getattr(d1, "heading", None),
                                      agent_names_arr[j], pos2, getattr(d2, "heading", None)):
                        d1.set_facecolor('red')
                        d2.set_facecolor('red')
                        print("COLLISION! (agent-agent) ({}, {})".format(i, j))

            return patches + artists

        def getHeading(t, d, heads, for_collision=False):
            """Heading held at time ``t`` (the segment starting at or before ``t``).
            On a rotation stretch (``turn_to`` on the segment's start point) the
            drawn heading follows the shortest arc; the collision test uses the
            envelope (``None``) there."""
            if not heads:
                return None
            k = 0
            while k + 1 < len(d) and d[k + 1]["t"] <= t:
                k += 1
            h = heads[k]
            if k + 1 < len(d) and d[k].get("turn_to") is not None:
                if for_collision:
                    return None
                dt = float(d[k + 1]["t"]) - float(d[k]["t"])
                frac = 0.0 if dt <= 1e-12 else min(1.0, max(0.0, (t - float(d[k]["t"])) / dt))
                return h + frac * turn_angle(h, float(d[k]["turn_to"]))
            return h

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

        if writer is None:
            writer = "ffmpeg" if animation.writers.is_available("ffmpeg") else "pillow"
        anim.save(
            file_name,
            writer,
            fps=intermediate_frames * speed,
            dpi=dpi)
        self.set_fig_size(self.figsize[0], self.figsize[1])

    def plot_density_map(self, density_map: np.ndarray, grid_map: Grid=None, equal: bool = False, alpha: float = 1.0,masked_map = None,interpolation: str = 'bilinear',use_fig_colorbar: bool = True) -> None:
        '''
        Plot density map as a heatmap that can be superimposed on other visualizations.

        Args:
            density_map: Density map.
            grid_map: Grid map (optional, can be set separately).
            equal: Whether to set axis equal.
            alpha: Transparency level (0-1) for superimposing on other maps.
        '''

        if grid_map is not None:
            if grid_map.dim != 2:
                raise ValueError(f"Grid map dimension must be 2.")
            self.grid_map = grid_map
            self.dim = grid_map.dim

        if len(density_map.shape) != 2:
            raise ValueError(f"Density map dimension must be 2.")

        assert self.grid_map is not None, "Grid map is not set"

        # Render density map over the full extent; the masked_map overlay is
        # painted on top afterwards at full opacity to segment those cells.
        density_map_plot = density_map.copy()
        cmap_density = self.cmap_density


        im = self.ax.imshow(
            np.transpose(density_map_plot), 
            cmap=cmap_density, 
            origin='lower',
            interpolation=interpolation,
            extent=self._grid_map_extent(self.grid_map),
            vmin=0,
            vmax=max(np.max(density_map),1),
            zorder=self.zorder['density_map'],  # Use esdf zorder to appear above grid_map but below paths
            alpha=alpha,
            )

        # Repaint masked_map==1 cells at full opacity using the underlying grid-map
        # colormap so they are always fully visible regardless of cell type.
        if masked_map is not None and hasattr(self.grid_map, "type_map") and hasattr(self.grid_map.type_map, "data"):
            type_data_visible = np.ma.array(
                self.grid_map.type_map.data.copy(),
                mask=(masked_map == 0),
            )
            visible_cmap = self.cmap.copy()
            visible_cmap.set_bad(alpha=0.0)
            self.ax.imshow(
                np.transpose(type_data_visible),
                cmap=visible_cmap,
                norm=self.norm,
                origin='lower',
                interpolation='nearest',
                extent=self._grid_map_extent(self.grid_map),
                zorder=self.zorder['density_map'] + 0.1,
                alpha=1.0,
            )
        
        if use_fig_colorbar:
            # Adjust axes position to make room for colorbar
            pos = self.ax.get_position()
            # Create a colorbar that matches the height of the plot
            self.ax.set_position([pos.x0, pos.y0, pos.width * 0.92, pos.height])
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            self.fig.colorbar(im, cax=cax, orientation='vertical', label="Frequency")
            
        if equal: 
            plt.axis("equal")