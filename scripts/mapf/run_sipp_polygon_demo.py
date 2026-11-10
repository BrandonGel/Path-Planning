"""SIPP with heading-aligned polygon footprints, speed zones and turning speeds -- animation demo.

Four agents cross the middle room of ``2d.yaml``: a rectangle, a hexagon, a
triangle and a disc, each with its own speed, turning speed and a path that
turns. Per-agent ``shape`` keys make SIPP (and the sweep backends behind it)
use the exact convex-polygon sweep for the agent-agent clearance; the polygons
are heading-aligned (the default): each faces the edge it travels and keeps
that heading while it waits. With a ``turning_speed`` an agent must rotate in
place (shortest arc, bounding disc reserved meanwhile) before it departs along
an edge it is not facing; ``heading_deg`` is its orientation at t=0. A shaded
speed zone (``map_.set_speed_zones``) caps the speed of every edge it touches.
``radius`` falls back to each polygon's bounding radius for obstacle inflation.

Usage:
    python scripts/mapf/run_sipp_polygon_demo.py [--out figs/polygon_sipp] [--backend auto|cgal|shapely]
                                                 [--neighbors 4|8] [--fixed] [--no-zone] [--no-turning]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from copy import deepcopy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.multi_agent_planner.mapf_solver import solve_mapf
from path_planning.utils.agent_shape import AlignedShape, ConvexPolygon
from path_planning.utils.checker import check_collision, check_velocity_anomaly
from path_planning.utils.util import (
    max_agent_radius,
    normalize_agent_specs,
    read_graph_sampler_from_yaml,
    set_global_seed,
)

MAP_YAML = os.path.join(_PROJECT_ROOT, "path_planning/maps/2d/2d.yaml")
COLORS = ["tab:orange", "tab:blue", "tab:green", "tab:purple"]

# name -> (start, goal, speed, shape spec, turning speed [rad/s], initial heading [deg]);
# cell centres of the middle room. Body frames: +x is forward, so the triangle's
# apex leads and the long side of the rectangle lies along its direction of travel.
AGENTS = {
    "rect":     ((23.5, 3.5), (37.5, 11.5), 2.0, {"type": "rectangle", "width": 1.6, "height": 0.6}, math.pi / 2, 90.0),
    "hexagon":  ((37.5, 3.5), (23.5, 11.5), 1.0, {"type": "regular_polygon", "n_sides": 6, "inradius": 0.45}, math.pi, 180.0),
    "triangle": ((30.5, 3.5), (30.5, 12.5), 0.5, {"type": "regular_polygon", "n_sides": 3, "circumradius": 0.6}, math.pi / 4, 0.0),
    "disc":     ((27.5, 12.5), (33.5, 3.5), 1.5, {"type": "disc", "radius": 0.35}, math.pi, 270.0),
}
# A slow band across the middle of the room (world coordinates).
SPEED_ZONES = [{"x": [28.0, 33.0], "y": [1.0, 15.0], "speed_limit": 0.5, "name": "slow"}]


def make_agents(align=True, turning=True):
    agents = []
    for name, (start, goal, speed, shape, omega, heading_deg) in AGENTS.items():
        shape = dict(shape, align=align) if shape["type"] != "disc" else shape
        a = {"name": name, "start": list(start), "goal": list(goal), "velocity": speed, "shape": shape}
        if turning:
            a["turning_speed"] = omega
            a["heading_deg"] = heading_deg
        agents.append(a)
    return normalize_agent_specs(agents)


def build_map(agents, default_radius, backend, neighbors=4):
    set_global_seed(42)
    map_ = read_graph_sampler_from_yaml(MAP_YAML, use_discrete_space=True)
    map_.sweep_backend = backend
    map_.constraint_sweep = map_._make_constraint_sweep()
    map_.inflate_obstacles(radius=max_agent_radius(agents, default_radius) + np.sqrt(2) / 2)
    # 8 neighbours adds the diagonal edges (length sqrt(2)), so the footprints
    # also align at 45 degrees.
    map_.set_parameters(sample_num=0, num_neighbors=float(neighbors), min_edge_len=0.0,
                        max_edge_len=1.1 if neighbors <= 4 else 1.5)
    map_.set_start([a["start"] for a in agents])
    map_.set_goal([a["goal"] for a in agents])
    nodes = map_.generateRandomNodes(generate_grid_nodes=True)
    road_map = map_.generate_roadmap(nodes)
    map_.set_constraint_sweep()
    return map_, road_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_PROJECT_ROOT, "figs/polygon_sipp"))
    ap.add_argument("--backend", default="auto", choices=["auto", "cgal", "shapely"])
    ap.add_argument("--intermediate-frames", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--neighbors", type=int, default=8, choices=[4, 8])
    ap.add_argument("--fixed", action="store_true", help="world-fixed polygons instead of heading-aligned")
    ap.add_argument("--no-zone", action="store_true", help="drop the slow speed zone")
    ap.add_argument("--no-turning", action="store_true", help="instantaneous turns (no turning_speed / heading)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    default_radius, default_velocity = 0.3, 1.0
    agents = make_agents(align=not args.fixed, turning=not args.no_turning)
    map_, road_map = build_map(agents, default_radius, args.backend, args.neighbors)
    if not args.no_zone:
        map_.set_speed_zones(SPEED_ZONES)
        for z in map_.speed_zones:
            print(f"speed zone {z.name!r}: x {z.x0}..{z.x1}, y {z.y0}..{z.y1}, v <= {z.speed_limit}")
    print(f"roadmap: {len(map_.nodes)} nodes, {len(map_.edges)} edges | inflation radius {map_.inflation_radius:.3f}"
          f" | sweep backend {type(map_.constraint_sweep).__name__}")

    summary = solve_mapf(map_, agents, {
        "mapf_solver_name": "sipp", "time_limit": 120, "max_iterations": 10000,
        "agent_radius": default_radius, "agent_velocity": default_velocity, "heuristic_type": "euclidean",
    })
    if not summary["success"]:
        raise SystemExit(f"SIPP failed: {summary.get('failure_reason')}")
    sched = summary["schedule"]
    footprints = summary["agent_footprints"]
    print(f"makespan {summary['makespan']:.2f} | flowtime {summary['flowtime']:.2f}")
    for name, path in sched.items():
        fp = footprints[name]
        if isinstance(fp, (ConvexPolygon, AlignedShape)):
            desc = f"{len(fp.vertices)}-gon{' aligned' if isinstance(fp, AlignedShape) else ''} (R {fp.bounding_radius:.2f})"
        else:
            desc = f"disc r {fp:.2f}"
        turns = [(a, b) for a, b in zip(path, path[1:]) if a.get("turn_to") is not None]
        turn_time = sum(b["t"] - a["t"] for a, b in turns)
        inside, outside = set(), set()
        for a, b in zip(path, path[1:]):
            if b['t'] > a['t'] and (a['x'], a['y']) != (b['x'], b['y']):
                v = round(math.dist((a['x'], a['y']), (b['x'], b['y'])) / (b['t'] - a['t']), 3)
                (inside if map_.edge_speed_limit((a['x'], a['y']), (b['x'], b['y'])) < float("inf") else outside).add(v)
        print(f"  {name:9s} {desc:28s} speed {summary['agent_velocities'][name]:.2f}  observed outside zone {sorted(outside)}"
              f"  inside {sorted(inside)}  turns {len(turns)} ({turn_time:.2f}s)  arrives t={path[-1]['t']:.2f}")
    collisions = check_collision(sched, footprints, verbose=True)
    print(f"exact footprint collisions: {len(collisions)}")
    print("speeds consistent with zone limits:",
          check_velocity_anomaly(sched, velocity=summary["agent_velocities"], verbose=False, speed_zones=map_.speed_zones))

    gif = os.path.join(args.out, "sipp_polygons.gif")
    vis = Visualizer2D()
    vis.animate(gif, map_, {"schedule": deepcopy(sched)}, road_map=road_map, skip_frames=1,
                intermediate_frames=args.intermediate_frames, speed=2, radius=footprints,
                map_frame=False, agent_colors=COLORS, writer="pillow", dpi=args.dpi,
                view_bounds=((20.0, 41.0), (1.0, 15.0)),  # zoom on the middle room
                show_legend=False, show_time=True)
    vis.close()
    print(f"wrote {gif}")

    # Contact sheet of a few frames (static preview of the GIF).
    try:
        from PIL import Image
        im = Image.open(gif)
        n = im.n_frames
        picks = [int(round(k * (n - 1) / 5)) for k in range(6)]
        frames = []
        for k in picks:
            im.seek(k)
            frames.append(im.convert("RGB").copy())
        w, h = frames[0].size
        sheet = Image.new("RGB", (3 * w, 2 * h), "white")
        for i, f in enumerate(frames):
            sheet.paste(f, ((i % 3) * w, (i // 3) * h))
        sheet_path = os.path.join(args.out, "sipp_polygons_frames.png")
        sheet.save(sheet_path)
        print(f"wrote {sheet_path} (frames {picks} of {n})")
    except Exception as exc:  # pragma: no cover - preview only
        print(f"contact sheet skipped: {exc}")


if __name__ == "__main__":
    main()
