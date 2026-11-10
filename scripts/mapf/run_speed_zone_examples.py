"""Several speed-zone layouts on the middle room of ``2d.yaml`` -- animations + overview.

Each example is a different set of ``map.speed_zones`` rectangles (world
coordinates, each with its own speed limit) and runs the same four
heading-aligned agents (rectangle, hexagon, triangle, disc; different speeds
and turning speeds, see ``run_sipp_polygon_demo``) through them with SIPP. An
edge that touches a zone runs at min(agent speed, lowest limit it touches).

Outputs (``--out figs/speed_zones``): one GIF (no legend / margins, time stamp,
makespan vs the zone-free run in the corner) + contact sheet per example, a
static overview ``speed_zone_examples.png`` (zones, paths, arrival times), a
printed table of observed speeds inside every zone and ``speed_zone_metrics.csv``
(per example and agent: arrival, delay vs no zones, time inside zones, path
length, mean / slowest speed).

Usage:
    python scripts/mapf/run_speed_zone_examples.py [--out DIR] [--backend auto|cgal|shapely]
                                                   [--examples single_band,two_bands,...]
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
from path_planning.utils.agent_shape import AlignedShape, ConvexPolygon, resolve_shape, schedule_headings
from path_planning.utils.checker import check_collision, check_velocity_anomaly
from path_planning.utils.util import set_global_seed
from scripts.mapf.run_sipp_polygon_demo import COLORS, build_map, make_agents

VIEW = ((20.0, 41.0), (1.0, 15.0))  # the middle room

# name -> list of zone specs (world coordinates). Limits are chosen so that every
# agent (speeds 2.0 / 1.0 / 0.5 / 1.5) is affected by at least one zone.
EXAMPLES = {
    # one vertical band across the room (the demo's zone)
    "single_band": [
        {"x": [28.0, 33.0], "y": [1.0, 15.0], "speed_limit": 0.5, "name": "band"},
    ],
    # two crossing bands with different limits (the overlap takes the lower one)
    "two_bands": [
        {"x": [24.0, 27.0], "y": [1.0, 15.0], "speed_limit": 0.5, "name": "vertical"},
        {"x": [20.0, 41.0], "y": [9.5, 12.5], "speed_limit": 1.0, "name": "horizontal"},
    ],
    # a slow core nested in a moderate outer zone
    "nested": [
        {"x": [26.0, 36.0], "y": [3.0, 13.0], "speed_limit": 1.0, "name": "outer"},
        {"x": [29.0, 33.0], "y": [6.0, 10.0], "speed_limit": 0.25, "name": "core"},
    ],
    # four patches with different limits and an unlimited corridor between them
    "patchwork": [
        {"x": [21.0, 30.0], "y": [1.0, 7.5], "speed_limit": 0.3, "name": "SW"},
        {"x": [31.0, 40.0], "y": [1.0, 7.5], "speed_limit": 0.6, "name": "SE"},
        {"x": [21.0, 30.0], "y": [8.5, 15.0], "speed_limit": 1.0, "name": "NW"},
        {"x": [31.0, 40.0], "y": [8.5, 15.0], "speed_limit": 1.5, "name": "NE"},
    ],
}


def observed_speeds(path, map_):
    """``{zone name | 'free': sorted speeds}`` of the moving segments of ``path``."""
    out = {}
    for a, b in zip(path, path[1:]):
        pa, pb = (a["x"], a["y"]), (b["x"], b["y"])
        if b["t"] <= a["t"] or pa == pb:
            continue
        v = round(math.dist(pa, pb) / (b["t"] - a["t"]), 3)
        hit = [z.name or f"z{i}" for i, z in enumerate(map_.speed_zones) if z.intersects_segment(pa, pb)]
        out.setdefault("+".join(hit) if hit else "free", set()).add(v)
    return {k: sorted(v) for k, v in out.items()}


def agent_metrics(path, map_):
    """Arrival, driving / rotating / waiting time, time inside any zone, path length, speeds."""
    m = dict(arrival=float(path[-1]["t"]), driving=0.0, rotating=0.0, waiting=0.0, in_zone=0.0,
             length=0.0, min_speed=math.inf)
    for a, b in zip(path, path[1:]):
        dt = float(b["t"]) - float(a["t"])
        pa, pb = (a["x"], a["y"]), (b["x"], b["y"])
        d = math.dist(pa, pb)
        if d > 1e-9:
            m["driving"] += dt
            m["length"] += d
            m["min_speed"] = min(m["min_speed"], d / dt)
            if any(z.intersects_segment(pa, pb) for z in map_.speed_zones):
                m["in_zone"] += dt
        elif a.get("turn_to") is not None and dt > 1e-9:
            m["rotating"] += dt
        else:
            m["waiting"] += dt
    m["mean_speed"] = m["length"] / m["driving"] if m["driving"] > 0 else 0.0
    if math.isinf(m["min_speed"]):
        m["min_speed"] = 0.0
    return m


def write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def draw_overview_panel(map_, road_map, sched, footprints, title, out_png):
    """Static panel: obstacle grid + shaded zones + planned paths + start footprints."""
    vis = Visualizer2D(figsize=(7, 5))
    vis.plot_grid_map(map_)
    (x0, x1), (y0, y1) = VIEW
    vis.set_fig_size(7, 5 * (y1 - y0) / (x1 - x0) * (7 / 5))
    ax = vis.ax
    coords = np.array([n.current for n in map_.nodes], dtype=float)
    for i, nbrs in enumerate(road_map):
        for j in nbrs:
            if j > i:
                ax.plot(coords[[i, j], 0], coords[[i, j], 1], color="#cccccc", linewidth=0.3, zorder=20)
    for k, (name, path) in enumerate(sched.items()):
        color = COLORS[k % len(COLORS)]
        xs = [p["x"] for p in path]
        ys = [p["y"] for p in path]
        ax.plot(xs, ys, color=color, linewidth=2.0, zorder=60, label=f"{name} (t={path[-1]['t']:.1f}s)")
        fp = footprints[name]
        h0 = schedule_headings(path)[0] if isinstance(fp, AlignedShape) else None
        if isinstance(fp, (ConvexPolygon, AlignedShape)):
            pts = resolve_shape(fp, h0).points + np.array([xs[0], ys[0]])
            ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=0.9, zorder=70)
        else:
            ax.add_patch(__import__("matplotlib").patches.Circle((xs[0], ys[0]), float(fp), color=color, zorder=70))
        ax.plot(xs[-1], ys[-1], marker="*", color=color, markersize=12, zorder=70)
    ax.set_xlim(*VIEW[0])
    ax.set_ylim(*VIEW[1])
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.9)
    vis.fig.savefig(out_png, dpi=140, bbox_inches="tight")
    vis.close()


def contact_sheet(gif, out_png, n_pick=6):
    from PIL import Image
    im = Image.open(gif)
    n = im.n_frames
    picks = [int(round(k * (n - 1) / (n_pick - 1))) for k in range(n_pick)]
    frames = []
    for k in picks:
        im.seek(k)
        frames.append(im.convert("RGB").copy())
    w, h = frames[0].size
    sheet = Image.new("RGB", (3 * w, 2 * h), "white")
    for i, f in enumerate(frames):
        sheet.paste(f, ((i % 3) * w, (i // 3) * h))
    sheet.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_PROJECT_ROOT, "figs/speed_zones"))
    ap.add_argument("--backend", default="auto", choices=["auto", "cgal", "shapely"])
    ap.add_argument("--examples", default=",".join(EXAMPLES))
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--no-gif", action="store_true", help="only the static overview")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    agents = make_agents()
    map_, road_map = build_map(agents, 0.3, args.backend, neighbors=8)
    cfg = {"mapf_solver_name": "sipp", "time_limit": 120, "max_iterations": 10000,
           "agent_radius": 0.3, "agent_velocity": 1.0, "heuristic_type": "euclidean"}

    # reference run without any zone (same agents, same map)
    map_.set_speed_zones([])
    set_global_seed(1)  # SIPP shuffles the agent priority order with the global RNG
    base = solve_mapf(map_, agents, cfg)
    if not base["success"]:
        raise SystemExit(f"SIPP failed without zones: {base.get('failure_reason')}")
    base_arrival = {a: float(p[-1]["t"]) for a, p in base["schedule"].items()}
    print(f"=== no zones: makespan {base['makespan']:.2f} | flowtime {base['flowtime']:.2f} | "
          + ", ".join(f"{a} t={t:.1f}" for a, t in base_arrival.items()))

    panels = []
    metric_rows = []
    for name in [e.strip() for e in args.examples.split(",") if e.strip()]:
        zones = EXAMPLES[name]
        map_.set_speed_zones(zones)
        set_global_seed(1)  # same priority order for every example
        print(f"\n=== {name}: " + "; ".join(f"{z.name} v<={z.speed_limit:g} x{z.x0:g}..{z.x1:g} y{z.y0:g}..{z.y1:g}"
                                           for z in map_.speed_zones))
        summary = solve_mapf(map_, agents, cfg)
        if not summary["success"]:
            print(f"  SIPP failed: {summary.get('failure_reason')}")
            continue
        sched = summary["schedule"]
        footprints = summary["agent_footprints"]
        print(f"  makespan {summary['makespan']:.2f} (+{summary['makespan'] - base['makespan']:.2f} vs no zones) | "
              f"flowtime {summary['flowtime']:.2f} (+{summary['flowtime'] - base['flowtime']:.2f})")
        for agent, path in sched.items():
            turns = sum(1 for p in path if p.get("turn_to") is not None)
            print(f"  {agent:9s} v_max {summary['agent_velocities'][agent]:.2f}  arrives t={path[-1]['t']:6.2f}  "
                  f"turns {turns}  speeds {observed_speeds(path, map_)}")
        print("  collisions:", len(check_collision(sched, footprints, verbose=False)),
              "| speeds within limits:", check_velocity_anomaly(sched, velocity=summary["agent_velocities"],
                                                                verbose=False, speed_zones=map_.speed_zones))
        hdr = ["agent", "v_max", "arrival_s", "delay_vs_no_zone_s", "in_zone_s", "in_zone_pct", "rotating_s",
               "waiting_s", "path_length_m", "mean_speed", "min_speed"]
        print("  " + "  ".join(f"{h:>18s}" for h in hdr))
        for agent, path in sched.items():
            am = agent_metrics(path, map_)
            row = {
                "example": name, "agent": agent, "v_max": summary["agent_velocities"][agent],
                "arrival_s": round(am["arrival"], 2),
                "delay_vs_no_zone_s": round(am["arrival"] - base_arrival[agent], 2),
                "in_zone_s": round(am["in_zone"], 2),
                "in_zone_pct": round(100.0 * am["in_zone"] / am["driving"], 1) if am["driving"] > 0 else 0.0,
                "rotating_s": round(am["rotating"], 2), "waiting_s": round(am["waiting"], 2),
                "path_length_m": round(am["length"], 2),
                "mean_speed": round(am["mean_speed"], 3), "min_speed": round(am["min_speed"], 3),
                "makespan_s": round(summary["makespan"], 2), "makespan_no_zone_s": round(base["makespan"], 2),
                "flowtime_s": round(summary["flowtime"], 2), "flowtime_no_zone_s": round(base["flowtime"], 2),
            }
            metric_rows.append(row)
            print("  " + "  ".join(f"{str(row[h]):>18s}" for h in hdr))
        png = os.path.join(args.out, f"{name}_overview.png")
        draw_overview_panel(map_, road_map, sched, footprints, f"{name}: makespan {summary['makespan']:.1f}s", png)
        panels.append(png)
        if not args.no_gif:
            gif = os.path.join(args.out, f"{name}.gif")
            vis = Visualizer2D()
            (vx0, vx1), (vy0, vy1) = VIEW
            title = (f"{name}: makespan {summary['makespan']:.1f} s "
                     f"(no zones {base['makespan']:.1f} s, +{summary['makespan'] - base['makespan']:.1f} s)")
            vis.animate(gif, map_, {"schedule": deepcopy(sched)}, road_map=road_map, skip_frames=1,
                        intermediate_frames=4, speed=2, radius=footprints, map_frame=False,
                        agent_colors=COLORS, writer="pillow", dpi=args.dpi, view_bounds=VIEW,
                        show_legend=False, show_time=True,
                        annotations=[(vx1 - 0.25, vy1 - 0.25, title,
                                      {"fontsize": 9, "fontweight": "bold", "ha": "right", "va": "top",
                                       "bbox": dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)})])
            vis.close()
            contact_sheet(gif, os.path.join(args.out, f"{name}_frames.png"))
            print(f"  wrote {gif}")

    if metric_rows:
        csv_path = os.path.join(args.out, "speed_zone_metrics.csv")
        write_csv(csv_path, metric_rows)
        print(f"\nwrote {csv_path}")

    # 2x2 overview of the static panels
    if panels:
        from PIL import Image
        ims = [Image.open(p).convert("RGB") for p in panels]
        w = max(i.size[0] for i in ims)
        h = max(i.size[1] for i in ims)
        cols = 2
        rows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w, rows * h), "white")
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        out = os.path.join(args.out, "speed_zone_examples.png")
        sheet.save(out)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
