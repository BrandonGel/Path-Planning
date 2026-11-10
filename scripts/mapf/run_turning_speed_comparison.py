"""Turning-speed comparison: identical rectangles on parallel lanes, different omega.

Four 1.6 x 0.6 heading-aligned rectangles start facing -x (away from their
goal) on separate lanes of an empty 16 x 16 grid and must drive east, then
north to a goal two rows up. A finite turning speed makes SIPP avoid the
180-degree U-turn: it rotates 90 degrees to face north, drives the two rows
first, rotates 90 degrees to face east and drives on (two 90-degree turns =
pi / omega of rotation instead of 270 degrees). The agents only differ in
``turning_speed`` (instantaneous, pi, pi/2, pi/4 rad/s), so their arrival
times differ by exactly the time spent rotating in place.

The reference agent uses ``INSTANT_OMEGA`` (1e6 rad/s, ~3 us for its two
turns) instead of no turning speed at all: with truly instantaneous turning
every staircase route costs the same and SIPP picks one arbitrarily, whereas
a near-infinite turning speed breaks the tie towards the fewest-rotation
route, so all four agents drive the identical path and the comparison isolates
the rotation time.

Outputs (``--out figs/turning_speed``): ``turning_comparison.gif`` (no legend
or margins; lanes labelled with omega and the agent's metrics, a time stamp;
the shapes rotate in place at the corners), ``turning_comparison.png``
(heading vs time, distance vs time, time budget driving / rotating / waiting,
arrival time vs omega from a single-agent sweep), ``turning_metrics.csv`` (the
printed comparison table) and ``turning_sweep.csv`` (the omega sweep).

Usage:
    python scripts/mapf/run_turning_speed_comparison.py [--out DIR] [--backend auto|cgal|shapely]
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.multi_agent_planner.mapf_solver import solve_mapf
from path_planning.utils.agent_shape import schedule_headings, turn_angle
from path_planning.utils.checker import check_collision
from path_planning.utils.util import max_agent_radius, normalize_agent_specs, set_global_seed

COLORS = ["tab:gray", "tab:blue", "tab:green", "tab:red"]
INSTANT_OMEGA = 1e6  # "instant" reference: a near-infinite turning speed (see module docstring)
START_X, GOAL_X = 1.5, 12.5
# (label, turning speed) from fastest to slowest turner
CASES = [("instant", None), ("w = pi rad/s", math.pi), ("w = pi/2 rad/s", math.pi / 2), ("w = pi/4 rad/s", math.pi / 4)]
LANES = [1.5, 5.5, 9.5, 13.5]  # 4 cells apart: the rotating envelopes never interact
# single-agent sweep of the turning speed (rad/s), slowest to fastest
SWEEP_OMEGAS = [math.pi / 8, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, 2 * math.pi / 3,
                math.pi, 1.5 * math.pi, 2 * math.pi, 4 * math.pi]
SOLVER_CFG = {"mapf_solver_name": "sipp", "time_limit": 120, "max_iterations": 10000,
              "agent_radius": 0.3, "agent_velocity": 1.0, "heuristic_type": "euclidean"}


def agent_spec(label, omega, y):
    a = {"name": label, "start": [START_X, y], "goal": [GOAL_X, y + 2.0], "velocity": 1.0,
         "heading_deg": 180.0, "shape": {"type": "rectangle", "width": 1.6, "height": 0.6},
         "turning_speed": INSTANT_OMEGA if omega is None else omega}
    return a


def make_agents():
    return normalize_agent_specs([agent_spec(label, omega, y) for (label, omega), y in zip(CASES, LANES)])


def path_metrics(path):
    """Time budget and geometry of one schedule.

    ``driving`` / ``rotating`` / ``waiting`` partition the arrival time; ``length``
    is the distance driven, ``turns`` the number of in-place rotations and
    ``rotation_deg`` the total angle rotated through."""
    heads = schedule_headings(path)
    m = dict(arrival=float(path[-1]["t"]), driving=0.0, rotating=0.0, waiting=0.0, length=0.0,
             turns=0, rotation_deg=0.0)
    for k, (a, b) in enumerate(zip(path, path[1:])):
        dt = float(b["t"]) - float(a["t"])
        d = math.dist((a["x"], a["y"]), (b["x"], b["y"]))
        if d > 1e-9:
            m["driving"] += dt
            m["length"] += d
        elif a.get("turn_to") is not None and dt > 1e-9:
            m["rotating"] += dt
            m["turns"] += 1
            m["rotation_deg"] += abs(math.degrees(turn_angle(heads[k], float(a["turn_to"]))))
        else:
            m["waiting"] += dt
    return m


def omega_sweep(m, y):
    """Arrival time of one rectangle alone on lane ``y`` for every omega in SWEEP_OMEGAS."""
    rows = []
    for omega in [None] + SWEEP_OMEGAS:
        spec = normalize_agent_specs([agent_spec("sweep", omega, y)])
        summary = solve_mapf(m, spec, SOLVER_CFG)
        if not summary["success"]:
            raise SystemExit(f"SIPP failed in the omega sweep (omega={omega}): {summary.get('failure_reason')}")
        pm = path_metrics(summary["schedule"]["sweep"])
        rows.append(dict(omega=math.inf if omega is None else omega, **pm))
    return rows


def write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_map(agents, backend):
    set_global_seed(1)
    m = GraphSampler(bounds=[[0.0, 16.0], [0.0, 16.0]], resolution=1.0, start=[], goal=[],
                     use_discrete_space=True, sweep_backend=backend)
    m.inflate_obstacles(radius=max_agent_radius(agents, 0.3) + np.sqrt(2) / 2)
    m.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
    m.set_start([a["start"] for a in agents])
    m.set_goal([a["goal"] for a in agents])
    road_map = m.generate_roadmap(m.generateRandomNodes(generate_grid_nodes=True))
    m.set_constraint_sweep()
    return m, road_map


def sample_path(path, ts):
    """Position, heading (deg, turns interpolated) and distance travelled at times ``ts``."""
    heads = schedule_headings(path)
    pos = np.zeros((len(ts), 2))
    hdg = np.zeros(len(ts))
    for i, t in enumerate(ts):
        k = 0
        while k + 1 < len(path) and path[k + 1]["t"] <= t:
            k += 1
        p, q = path[k], path[min(k + 1, len(path) - 1)]
        dt = q["t"] - p["t"]
        frac = 0.0 if dt <= 1e-12 else min(1.0, max(0.0, (t - p["t"]) / dt))
        pos[i] = (p["x"] + frac * (q["x"] - p["x"]), p["y"] + frac * (q["y"] - p["y"]))
        h = heads[k]
        if p.get("turn_to") is not None and k + 1 < len(path):
            h = h + frac * turn_angle(h, p["turn_to"])
        hdg[i] = math.degrees(h) % 360.0
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    return pos, hdg, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_PROJECT_ROOT, "figs/turning_speed"))
    ap.add_argument("--backend", default="auto", choices=["auto", "cgal", "shapely"])
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    agents = make_agents()
    m, road_map = build_map(agents, args.backend)
    set_global_seed(1)  # SIPP shuffles the agent priority order with the global RNG
    summary = solve_mapf(m, agents, SOLVER_CFG)
    if not summary["success"]:
        raise SystemExit(f"SIPP failed: {summary.get('failure_reason')}")
    sched = summary["schedule"]
    footprints = summary["agent_footprints"]
    print("makespan", round(summary["makespan"], 3), "| flowtime", round(summary["flowtime"], 3),
          "| collisions", len(check_collision(sched, footprints, verbose=False)))

    # ---- comparison metrics (the instantaneous agent is the reference)
    rows = []
    ref = path_metrics(sched[CASES[0][0]])["arrival"]
    for label, omega in CASES:
        pm = path_metrics(sched[label])
        rows.append({
            "case": label,
            "omega_rad_s": math.inf if omega is None else round(omega, 4),
            "arrival_s": round(pm["arrival"], 2),
            "driving_s": round(pm["driving"], 2),
            "rotating_s": round(pm["rotating"], 2),
            "waiting_s": round(pm["waiting"], 2),
            "path_length_m": round(pm["length"], 2),
            "turns": pm["turns"],
            "rotation_deg": round(pm["rotation_deg"], 1),
            "min_rotation_s": 0.0 if omega is None else round(math.pi / omega, 2),  # two 90-degree turns
            "delay_vs_instant_s": round(pm["arrival"] - ref, 2),
            "delay_pct": round(100.0 * (pm["arrival"] - ref) / ref, 1),
            "mean_speed_m_s": round(pm["length"] / pm["arrival"], 3),
        })
    hdr = ["case", "omega_rad_s", "arrival_s", "driving_s", "rotating_s", "waiting_s", "path_length_m", "turns",
           "rotation_deg", "delay_vs_instant_s", "delay_pct", "mean_speed_m_s"]
    print("  " + "  ".join(f"{h:>18s}" for h in hdr))
    for r in rows:
        print("  " + "  ".join(f"{str(r[h]):>18s}" for h in hdr))
    csv_path = os.path.join(args.out, "turning_metrics.csv")
    write_csv(csv_path, rows)
    print("wrote", csv_path)

    # ---- single-agent omega sweep
    sweep = omega_sweep(m, LANES[0])
    print("  omega sweep (one rectangle alone): omega [rad/s] -> arrival [s], rotating [s], turns, rotation [deg]")
    for r in sweep:
        om = "inf" if math.isinf(r["omega"]) else f"{r['omega'] / math.pi:.3f} pi"
        print(f"    {om:>10s} -> {r['arrival']:6.2f}  {r['rotating']:5.2f}  {r['turns']}  {r['rotation_deg']:5.0f}")
    sweep_path = os.path.join(args.out, "turning_sweep.csv")
    write_csv(sweep_path, sweep)
    print("wrote", sweep_path)

    # ---- animation (no legend / margins; each lane carries its metrics)
    gif = os.path.join(args.out, "turning_comparison.gif")
    vis = Visualizer2D()
    ann = []
    for r, y in zip(rows, LANES):
        txt = (f"{r['case']}   turn {r['rotating_s']:.1f} s   arrive {r['arrival_s']:.1f} s"
               + (f"   (+{r['delay_vs_instant_s']:.1f} s)" if r["delay_vs_instant_s"] > 0 else "   (reference)"))
        # the row below each lane is never driven on (agents leave north or east)
        ann.append((START_X + 0.8, y - 0.75, txt, {"fontsize": 9, "fontweight": "bold", "va": "center"}))
    vis.animate(gif, m, {"schedule": deepcopy(sched)}, road_map=road_map, skip_frames=1, intermediate_frames=4,
                speed=2, radius=footprints, map_frame=False, agent_colors=COLORS, writer="pillow", dpi=args.dpi,
                view_bounds=((0.0, 16.0), (0.0, 16.0)), show_legend=False, show_time="bottom-right",
                annotations=ann)
    vis.close()
    print("wrote", gif)

    # ---- static comparison
    T = max(p[-1]["t"] for p in sched.values())
    ts = np.linspace(0.0, T, 600)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.ravel()
    for (label, omega), color in zip(CASES, COLORS):
        _, hdg, dist = sample_path(sched[label], ts)
        axes[0].plot(ts, hdg, color=color, label=label)
        axes[1].plot(ts, dist, color=color, label=label)
    axes[0].set(title="heading vs time (rotations in place)", xlabel="t [s]", ylabel="heading [deg]")
    axes[0].set_yticks([0, 90, 180, 270, 360])
    axes[1].set(title="distance travelled vs time (plateaus = turning)", xlabel="t [s]", ylabel="path length [m]")
    for ax in axes[:2]:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    # time budget: driving / rotating / waiting stacked per case
    labels = [r["case"] for r in rows]
    drv = np.array([r["driving_s"] for r in rows])
    rot = np.array([r["rotating_s"] for r in rows])
    wt = np.array([r["waiting_s"] for r in rows])
    axes[2].bar(labels, drv, color="#9ecae1", label="driving")
    axes[2].bar(labels, rot, bottom=drv, color="#fd8d3c", label="rotating")
    axes[2].bar(labels, wt, bottom=drv + rot, color="#bdbdbd", label="waiting")
    for i, r in enumerate(rows):
        axes[2].text(i, r["arrival_s"] + 0.15, f"{r['arrival_s']:.1f} s\n+{r['delay_pct']:.0f}%", ha="center", fontsize=8)
    axes[2].set(title="time budget per agent (arrival = driving + rotating + waiting)", ylabel="t [s]")
    axes[2].text(0.98, 0.97, "identical routes: " + ", ".join(f"{r['path_length_m']:g} m" for r in rows),
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=8)
    axes[2].set_ylim(0, max(r["arrival_s"] for r in rows) * 1.18)
    axes[2].tick_params(axis="x", labelrotation=15)
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend(fontsize=8, loc="upper left")
    # arrival vs omega sweep
    finite = [r for r in sweep if not math.isinf(r["omega"])]
    om = np.array([r["omega"] for r in finite])
    arr = np.array([r["arrival"] for r in finite])
    drive_only = next(r["arrival"] for r in sweep if math.isinf(r["omega"]))
    om_fine = np.geomspace(om.min(), om.max(), 200)
    axes[3].plot(om_fine / math.pi, drive_only + math.pi / om_fine, color="#888888", linestyle="--",
                 label=f"{drive_only:.0f} s driving + pi / omega (two 90-degree turns)")
    axes[3].plot(om / math.pi, arr, "o-", color="tab:red", label="SIPP arrival (single rectangle)")
    axes[3].axhline(drive_only, color="tab:gray", linewidth=1, label="driving only (instant turning)")
    for (label, omega), color in zip(CASES[1:], COLORS[1:]):
        axes[3].plot([omega / math.pi], [next(r["arrival"] for r in sweep if abs(r["omega"] - omega) < 1e-9)],
                     marker="s", color=color, markersize=9, linestyle="none", label=label)
    axes[3].set_xscale("log")
    axes[3].set(title="arrival time vs turning speed", xlabel="omega / pi [rad/s]", ylabel="arrival [s]")
    axes[3].grid(alpha=0.3, which="both")
    axes[3].legend(fontsize=7)
    fig.tight_layout()
    png = os.path.join(args.out, "turning_comparison.png")
    fig.savefig(png, dpi=140)
    print("wrote", png)


if __name__ == "__main__":
    main()
