'''
Check the saved MAPF / MAPD solutions for time anomalies, velocity anomalies and collisions.

Every ``solution*.yaml`` under ``path_planning/maps/2d/<solver>/`` is checked, for the solvers written by
``run_cbs.py``, ``run_icbs.py``, ``run_ccbs.py``, ``run_sipp.py``, ``run_lacam*.py`` and
``run_neural_attf*.py``. Summaries that record each agent's body (``agents``: name -> ``Agent.to_dict()``,
written by ``mapf_solver`` for every solver with bodies) are checked exactly with those bodies; older ones
as discs of the summary's ``agent_radius``. A constant speed is expected when the summary's
``agent_velocity`` is non-zero (velocity 0: every move takes one time unit).

python scripts/mapf/run_mapf_checker.py [solver ...]
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # this repo, ahead of any installed copy

import glob
import os

import yaml

from path_planning.multi_agent_planner.mapf_solver import MAPF_SOLVERS
from path_planning.utils.checker import check_collision, check_collision_bodies, check_time_anomaly, check_velocity_anomaly

SOLVERS = (*MAPF_SOLVERS, "neural_attf")
ROOT = "path_planning/maps/2d"


def check_mapf_solution(file, radius: float = None, velocity: float = None, verbose: bool = False) -> bool:
    """Check one saved solution; True when it has no anomaly and no collision. ``radius`` / ``velocity``
    default to the summary's ``agent_radius`` / ``agent_velocity``; recorded bodies take precedence."""
    with open(file, 'r') as f:
        data = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    solution = data.get('schedule') or {}
    if not solution:
        print(f"{file}: no schedule")
        return False
    radius = float(data.get("agent_radius") or 0.0) if radius is None else radius
    velocity = float(data.get("agent_velocity") or 0.0) if velocity is None else velocity
    no_time = check_time_anomaly(solution, verbose)
    no_speed = check_velocity_anomaly(solution, is_using_constant_speed=(velocity != 0), verbose=verbose)
    if data.get("agents"):
        collisions, how = check_collision_bodies(solution, data["agents"], verbose=verbose), "exact bodies"
    elif radius > 0:
        collisions, how = check_collision(solution, radius, verbose), f"discs r={radius}"
    else:
        collisions, how = {}, "point particles (not checked)"
    ok = no_time and no_speed and not collisions
    print(f"{'ok  ' if ok else 'FAIL'} {file} [{how}]"
          + ("" if no_time else " time anomaly") + ("" if no_speed else " velocity anomaly")
          + (f" {len(collisions)} colliding pair(s)" if collisions else ""))
    return ok


if __name__ == "__main__":
    solvers = sys.argv[1:] or SOLVERS
    results = {}
    for solver in solvers:
        for file in sorted(glob.glob(os.path.join(ROOT, solver, "solution*.yaml"))):
            results[file] = check_mapf_solution(file)
    print(f"{sum(results.values())}/{len(results)} solutions passed")
