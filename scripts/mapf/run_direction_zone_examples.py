"""Direction-zone layouts on the middle room of ``2d.yaml`` -- animations + metrics.

Each example is a set of ``map.direction_zones`` rectangles (world coordinates)
applied to the same 8-connected grid roadmap and the same four heading-aligned
agents as ``run_sipp_polygon_demo``:

- one-way zones (``direction_deg`` + ``tolerance_deg``) keep only the edges
  heading within the tolerance of the given direction;
- angle zones (``angles_deg``) keep only edges on the given line angles, both
  ways (``[0, 90]`` = horizontal / vertical only).

A zone constrains every edge that touches it. Every solver sees the filtered
roadmap; this script plans with SIPP.

Outputs (``--out figs/direction_zones``): per example a GIF (zones shaded with
their arrows, one-way edges marked with arrowheads, time stamp), a contact
sheet and an overview PNG; ``direction_zone_examples.png`` (all overviews) and
``direction_zone_metrics.csv`` (per example and agent: arrival, delay and extra
path length vs the zone-free run, turns, and how many moves of the zone-free
plan would have broken the rule).

Usage:
    python scripts/mapf/run_direction_zone_examples.py [--out DIR] [--backend auto|cgal|shapely]
                                                       [--examples one_way_halves,...] [--no-gif]
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.multi_agent_planner.mapf_solver import solve_mapf
from path_planning.utils.checker import check_collision, check_direction_anomaly, check_velocity_anomaly
from path_planning.utils.util import set_global_seed
from scripts.mapf.run_sipp_polygon_demo import COLORS, build_map, make_agents
from scripts.mapf.run_speed_zone_examples import VIEW, agent_metrics, contact_sheet, draw_overview_panel, write_csv

# name -> (direction zones, speed zones). The room spans x 21..40, y 2..14; nodes
# sit at cell centres. Agents: rect (23.5,3.5)->(37.5,11.5), hexagon
# (37.5,3.5)->(23.5,11.5), triangle (30.5,3.5)->(30.5,12.5), disc (27.5,12.5)->(33.5,3.5).
EXAMPLES = {
    # the room split into an eastbound upper half and a westbound lower half
    # (tolerance 90: no move with a component against the flow; crossing allowed)
    "one_way_halves": ([
        {"x": [21.0, 40.0], "y": [8.2, 14.0], "direction_deg": 0, "tolerance_deg": 90, "name": "eastbound"},
        {"x": [21.0, 40.0], "y": [2.0, 7.8], "direction_deg": 180, "tolerance_deg": 90, "name": "westbound"},
    ], []),
    # a counter-clockwise ring road around the room; the corners (where the agents
    # start and finish) and the interior stay free
    "ring_road": ([
        {"x": [25.2, 35.8], "y": [2.2, 4.8], "direction_deg": 0, "tolerance_deg": 60, "name": "south: east"},
        {"x": [36.2, 38.8], "y": [5.2, 10.8], "direction_deg": 90, "tolerance_deg": 60, "name": "east: north"},
        {"x": [25.2, 35.8], "y": [11.2, 13.8], "direction_deg": 180, "tolerance_deg": 60, "name": "north: west"},
        {"x": [22.2, 24.8], "y": [5.2, 10.8], "direction_deg": 270, "tolerance_deg": 60, "name": "west: south"},
    ], []),
    # horizontal / vertical edges only in the middle of the 8-connected room
    "axis_only": ([
        {"x": [25.2, 35.8], "y": [5.2, 10.8], "angles_deg": [0, 90], "name": "grid"},
    ], []),
    # diagonal-only core, axis-only frame around it
    "diagonal_core": ([
        {"x": [27.2, 33.8], "y": [5.2, 10.8], "angles_deg": [45, 135], "name": "diagonal"},
    ], []),
    # everything together: a one-way lane, an axis region and a speed zone
    "mixed": ([
        {"x": [22.2, 26.8], "y": [2.2, 13.8], "direction_deg": 90, "tolerance_deg": 60, "name": "northbound"},
        {"x": [28.2, 35.8], "y": [5.2, 10.8], "angles_deg": [0, 90], "name": "grid"},
    ], [{"x": [36.2, 39.8], "y": [2.2, 13.8], "speed_limit": 0.5, "name": "slow"}]),
}
SOLVER_CFG = {"mapf_solver_name": "sipp", "time_limit": 120, "max_iterations": 10000,
              "agent_radius": 0.3, "agent_velocity": 1.0, "heuristic_type": "euclidean"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_PROJECT_ROOT, "figs/direction_zones"))
    ap.add_argument("--backend", default="auto", choices=["auto", "cgal", "shapely"])
    ap.add_argument("--examples", default=",".join(EXAMPLES))
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--no-gif", action="store_true", help="only the static overviews and the metrics")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    agents = make_agents()
    map_, _ = build_map(agents, 0.3, args.backend, neighbors=8)
    n_edges_full = len(map_.edges)

    # reference run on the unrestricted roadmap (same agents, same map)
    map_.set_direction_zones([])
    map_.set_speed_zones([])
    set_global_seed(1)  # SIPP shuffles the agent priority order with the global RNG
    base = solve_mapf(map_, agents, SOLVER_CFG)
    if not base["success"]:
        raise SystemExit(f"SIPP failed without zones: {base.get('failure_reason')}")
    base_m = {a: agent_metrics(p, map_) for a, p in base["schedule"].items()}
    base_one_way = sum(1 for a, b in map_.edges if (b, a) not in map_.edge_indices_dict)
    print(f"=== no zones: {n_edges_full} directed edges ({base_one_way} already one-way from the "
          f"8-neighbour cap) | makespan {base['makespan']:.2f} | flowtime {base['flowtime']:.2f}")

    panels, rows = [], []
    for name in [e.strip() for e in args.examples.split(",") if e.strip()]:
        dzones, szones = EXAMPLES[name]
        map_.set_speed_zones(szones)
        map_.set_direction_zones(dzones)
        one_way = sum(1 for a, b in map_.edges if (b, a) not in map_.edge_indices_dict)
        print(f"\n=== {name}: " + "; ".join(
            f"{z.name} {'angles' if z.bidirectional else 'one-way'} {list(z.headings_deg)} +-{z.tolerance_deg:g}"
            for z in map_.direction_zones))
        print(f"  roadmap: {len(map_.edges)} of {n_edges_full} directed edges kept, {one_way} one-way")
        set_global_seed(1)
        summary = solve_mapf(map_, agents, SOLVER_CFG)
        if not summary["success"]:
            print(f"  SIPP failed: {summary.get('failure_reason')}")
            continue
        sched = summary["schedule"]
        footprints = summary["agent_footprints"]
        violations = check_direction_anomaly(sched, map_.direction_zones, verbose=False)
        base_viol = check_direction_anomaly(base["schedule"], map_.direction_zones, verbose=False)
        print(f"  makespan {summary['makespan']:.2f} (+{summary['makespan'] - base['makespan']:.2f} vs no zones) | "
              f"flowtime {summary['flowtime']:.2f} (+{summary['flowtime'] - base['flowtime']:.2f})")
        print("  collisions:", len(check_collision(sched, footprints, verbose=False)),
              "| direction violations:", len(violations),
              "| speeds within limits:", check_velocity_anomaly(sched, velocity=summary["agent_velocities"],
                                                                verbose=False, speed_zones=map_.speed_zones),
              f"| the zone-free plan would break the rules on {len(base_viol)} moves")
        hdr = ["agent", "arrival_s", "delay_s", "path_length_m", "extra_path_m", "rotating_s", "waiting_s",
               "free_plan_violations"]
        print("  " + "  ".join(f"{h:>20s}" for h in hdr))
        for agent, path in sched.items():
            am = agent_metrics(path, map_)
            row = {
                "example": name, "agent": agent,
                "arrival_s": round(am["arrival"], 2),
                "delay_s": round(am["arrival"] - base_m[agent]["arrival"], 2),
                "path_length_m": round(am["length"], 2),
                "extra_path_m": round(am["length"] - base_m[agent]["length"], 2),
                "rotating_s": round(am["rotating"], 2), "waiting_s": round(am["waiting"], 2),
                "free_plan_violations": sum(1 for v in base_viol if v["agent"] == agent),
                "direction_violations": sum(1 for v in violations if v["agent"] == agent),
                "edges_kept": len(map_.edges), "edges_full": n_edges_full, "one_way_edges": one_way,
                "makespan_s": round(summary["makespan"], 2), "makespan_no_zone_s": round(base["makespan"], 2),
                "flowtime_s": round(summary["flowtime"], 2), "flowtime_no_zone_s": round(base["flowtime"], 2),
            }
            rows.append(row)
            print("  " + "  ".join(f"{str(row[h]):>20s}" for h in hdr))
        png = os.path.join(args.out, f"{name}_overview.png")
        draw_overview_panel(map_, map_.road_map, sched, footprints,
                            f"{name}: makespan {summary['makespan']:.1f}s (no zones {base['makespan']:.1f}s)", png)
        panels.append(png)
        if not args.no_gif:
            gif = os.path.join(args.out, f"{name}.gif")
            (vx0, vx1), (vy0, vy1) = VIEW
            title = (f"{name}: makespan {summary['makespan']:.1f} s "
                     f"(no zones {base['makespan']:.1f} s, +{summary['makespan'] - base['makespan']:.1f} s)")
            vis = Visualizer2D()
            vis.animate(gif, map_, {"schedule": deepcopy(sched)}, road_map=map_.road_map, skip_frames=1,
                        intermediate_frames=4, speed=2, radius=footprints, map_frame=False,
                        agent_colors=COLORS, writer="pillow", dpi=args.dpi, view_bounds=VIEW,
                        show_legend=False, show_time=True,
                        annotations=[(vx1 - 0.25, vy1 - 0.25, title,
                                      {"fontsize": 9, "fontweight": "bold", "ha": "right", "va": "top",
                                       "bbox": dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)})])
            vis.close()
            contact_sheet(gif, os.path.join(args.out, f"{name}_frames.png"))
            print(f"  wrote {gif}")

    if rows:
        csv_path = os.path.join(args.out, "direction_zone_metrics.csv")
        write_csv(csv_path, rows)
        print(f"\nwrote {csv_path}")
    if panels:
        from PIL import Image
        ims = [Image.open(p).convert("RGB") for p in panels]
        w, h = max(i.size[0] for i in ims), max(i.size[1] for i in ims)
        cols = 2
        sheet = Image.new("RGB", (cols * w, ((len(ims) + cols - 1) // cols) * h), "white")
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        out = os.path.join(args.out, "direction_zone_examples.png")
        sheet.save(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
