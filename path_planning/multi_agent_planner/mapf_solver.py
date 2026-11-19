from path_planning.agent import Agent
from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, CBS
from path_planning.multi_agent_planner.centralized.icbs.icbs import IEnvironment, ICBS
from path_planning.multi_agent_planner.centralized.lacam.lacam import LaCAM
from path_planning.multi_agent_planner.centralized.lacam.lacam_random import LaCAM as LaCAM_random
from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner
from path_planning.multi_agent_planner.centralized.ccbs.ccbs import CCBS, Environment as CCBS_Environment
from path_planning.multi_agent_planner.centralized.lacam.utility import set_starts_goals_config, is_valid_mapf_solution
from typing import List, Tuple
import numpy as np
import time
from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import NeuralATTF
from path_planning.multi_agent_planner.decentralized.neural_attf.simulation import Simulation
from path_planning.utils.checker import _xyz, check_collision, check_collision_bodies
from path_planning.utils.util import agents_yaml_to_roadmap_frame

#: The solvers ``solve_mapf`` runs (``neural_attf`` is a MAPD planner: see ``solve_mapd``).
MAPF_SOLVERS = ("cbs", "icbs", "ccbs", "sipp", "lacam", "lacam_random")
#: Solvers that take ``Agent`` bodies from the agent dicts.
BODY_SOLVERS = ("cbs", "icbs", "ccbs", "sipp", "neural_attf")
#: Solvers restricted to point particles (agent_radius 0, no ``"agent"``): LaCAM.
POINT_SOLVERS = ("lacam", "lacam_random")
#: Of the body solvers, the ones that plan in continuous time: their bodies also carry the speed.
TIMED_SOLVERS = ("ccbs", "sipp", "neural_attf")


def solver_supports(mapf_solver_name: str, agent_radius: float, agent_velocity: float, road_map_type: str) -> bool:
    """Whether ``mapf_solver_name`` can run this setting, so batch scripts can skip the ones it cannot:
    LaCAM plans point particles on a grid in unit steps; CBS / ICBS plan in unit steps (no velocity);
    CCBS needs a body."""
    if mapf_solver_name in POINT_SOLVERS:
        return agent_radius == 0.0 and agent_velocity == 0.0 and road_map_type == "grid"
    if mapf_solver_name in ("cbs", "icbs"):
        return agent_velocity == 0.0
    if mapf_solver_name == "ccbs":
        return agent_radius > 0.0
    return True


def config_agent(mapf_solver_config: dict, mapf_solver_name: str):
    """The ``Agent`` every agent of ``mapf_solver_name`` gets unless its dict carries its own: the config's
    ``"agent"`` (any shape), else a disc / sphere of ``agent_radius`` (with the optional ``agent_clearance``
    buffer and ``agent_max_rotation``: a finite rate makes it turn before changing direction), else None
    (point particles). Continuous-time solvers also get the speed ``agent_velocity`` (0 or None: one time
    unit per move) unless the config's ``"agent"`` already has one."""
    agent_radius = mapf_solver_config.get("agent_radius") or 0.0
    body = mapf_solver_config.get("agent") or (
        Agent(agent_radius, max_rotation=mapf_solver_config.get("agent_max_rotation"), type="disc",
              clearance=float(mapf_solver_config.get("agent_clearance", 0.0))) if agent_radius > 0 else None)
    velocity = mapf_solver_config.get("agent_velocity")
    if body is not None and mapf_solver_name in TIMED_SOLVERS and body.max_speed is None and velocity:
        body = body.replace(max_speed=float(velocity))
    return body


def with_bodies(agents: list, mapf_solver_config: dict, mapf_solver_name: str) -> list:
    """``agents`` with the config's body (:func:`config_agent`) added to every dict without an ``"agent"``,
    for the solvers that take bodies; unchanged otherwise."""
    body = config_agent(mapf_solver_config, mapf_solver_name)
    if body is None or mapf_solver_name not in BODY_SOLVERS:
        return agents
    return [{"agent": body, **a} for a in agents]


def _bodies_of(agents: list) -> dict | None:
    """``{name: Agent}`` when the agents carry bodies, else None (point particles)."""
    bodies = {a["name"]: a.get("agent") for a in agents}
    return bodies if any(b is not None for b in bodies.values()) else None


def _snap_agents_to_nodes(map_, agents_rt):
    """Snap each agent's start/goal to the nearest roadmap node.

    Topology-rebuilding roadmaps (cdt/rrg) and some prm builds do not include the
    requested start/goal as nodes, so a point that isn't a node would make SIPP
    KeyError on ``sipp_graph[start]`` and make CBS/ICBS fail instantly (no neighbors
    from an off-node state). The nearest node is typically <1 cell away. Points that
    are already nodes are left untouched (grid/halton), so this is a no-op there.
    """

    nodes = getattr(map_, "nodes", None)
    if not nodes:
        return agents_rt
    nidx = getattr(map_, "pos_index_dict", {})
    pts = [tuple(float(c) for c in n.current) for n in nodes]
    kdt = None

    def snap(p):
        nonlocal kdt
        p = tuple(float(x) for x in p)
        if p in nidx:
            return p
        if kdt is None:
            from scipy.spatial import KDTree
            kdt = KDTree(pts)
        return pts[int(kdt.query(np.asarray(p))[1])]

    return [{**a, "start": snap(a["start"]), "goal": snap(a["goal"])} for a in agents_rt]


def _agents_off_nodes(map_, agents_rt) -> List[str]:
    """Return a description of every agent start/goal that is not an exact roadmap node."""

    nidx = getattr(map_, "pos_index_dict", {}) or {}
    off = []
    for a in agents_rt:
        s = tuple(float(x) for x in a["start"])
        g = tuple(float(x) for x in a["goal"])
        if s not in nidx:
            off.append(f"{a.get('name', 'agent')} start {s}")
        if g not in nidx:
            off.append(f"{a.get('name', 'agent')} goal {g}")
    return off


def solve_mapf(map_, agents,mapf_solver_config:dict) -> Tuple[dict, float]:
    mapf_solver_name = mapf_solver_config['mapf_solver_name'].lower()
    agent_radius = mapf_solver_config['agent_radius']
    time_limit = mapf_solver_config['time_limit']
    max_iterations = mapf_solver_config['max_iterations']
    heuristic_type = mapf_solver_config['heuristic_type']
    # YAML stores world starts/goals. Convert them into the roadmap's native node
    # frame so the start/goal coincide with actual nodes: discrete/grid maps key
    # nodes by integer cell indices, continuous maps by world coords.
    # agents_yaml_to_roadmap_frame already dispatches on use_discrete_space, so call
    # it unconditionally. (Previously discrete mode passed world ".5" cell-center
    # coords through unchanged; those don't match the integer-index grid nodes, so
    # SIPP KeyErrored on sipp_graph[start] and CBS/ICBS silently failed instantly.)
    agents_rt = agents_yaml_to_roadmap_frame(map_, agents)
    if mapf_solver_name == 'sipp':
        # SIPP runs on the instance's true start/goal nodes only. Snapping to the nearest
        # node silently changes the instance, and on pruned roadmaps it can land on an
        # isolated node or merge two agents onto one node, so an off-node start/goal is
        # reported as a failure (with the reason) instead of being patched over. Roadmaps
        # generated from the instance embed every start/goal as a node, so this only
        # triggers when the roadmap and the instance are out of sync.
        off_nodes = _agents_off_nodes(map_, agents_rt)
        if off_nodes:
            solution_info = {
                "success": False,
                "runtime": 0.0,
                "total_iterations": 0,
                "low_level_iterations": 0,
                "failure_reason": "start/goal not a roadmap node: " + "; ".join(off_nodes),
            }
            solution_info = check_solution_collision({}, agent_radius, solution_info)
            return summarize_solution({}, solution_info, mapf_solver_config, map_,
                                      _bodies_of(with_bodies(agents_rt, mapf_solver_config, mapf_solver_name)))
    else:
        # cdt/rrg/prm roadmaps may not include the exact start/goal as nodes; snap to the
        # nearest node so CBS/ICBS/LaCAM/CCBS have an on-node start/goal. No-op for
        # grid/halton where they already are nodes.
        agents_rt = _snap_agents_to_nodes(map_, agents_rt)
    # Low-level A* (inside CBS/ICBS) may need a larger budget than the high-level CBS iteration cap,
    # especially once constraints are introduced (time-expanded state space).
    num_nodes = len(getattr(map_, "nodes", []))
    low_level_astar_max_iterations = max(int(max_iterations or 0), num_nodes * 5, 10000)   # None: no cap given
    low_level_cap = mapf_solver_config.get("low_level_astar_max_iterations")
    if low_level_cap is not None:
        low_level_astar_max_iterations = min(low_level_astar_max_iterations, int(low_level_cap))

    if mapf_solver_name in POINT_SOLVERS and (
            (mapf_solver_config.get("agent_radius") or 0.0) > 0 or mapf_solver_config.get("agent") is not None
            or any(a.get("agent") is not None for a in agents_rt)):
        raise ValueError(f"{mapf_solver_name} plans point particles only: use agent_radius 0 and no agent bodies")
    # The body goes in each agent's dict (agent dicts may carry their own); see config_agent.
    agents_rt = with_bodies(agents_rt, mapf_solver_config, mapf_solver_name)
    bodies = _bodies_of(agents_rt)

    if mapf_solver_name in ('cbs', 'icbs'):
        env = (Environment if mapf_solver_name == 'cbs' else IEnvironment)(
            map_,
            agents_rt,
            astar_max_iterations=low_level_astar_max_iterations,
            heuristic_type=heuristic_type,
            move_parked_agents=bool(mapf_solver_config.get("move_parked_agents", True)),
            conflict_span=mapf_solver_config.get("conflict_span", "window"),        # agents with bodies: "window" | "step"
            goal_conflicts=bool(mapf_solver_config.get("goal_conflicts", False)),   # agents with bodies
        )
    if mapf_solver_name == 'cbs':
        solution, solution_info = CBS(env, time_limit=time_limit, max_iterations=max_iterations, verbose=False).search()
    elif mapf_solver_name == 'icbs':
        icbs = ICBS(
            env,
            time_limit=time_limit,
            max_iterations=max_iterations,
            verbose=False,
            disjoint=bool(mapf_solver_config.get("icbs_disjoint", False)),          # point agents only
            merge_bound=mapf_solver_config.get("icbs_merge_bound"),                 # point agents only; None: no meta-agents
        )
        solution, solution_info = icbs.search()
    elif mapf_solver_name in POINT_SOLVERS:
        # Point particles on the roadmap (one vertex per agent per step); checked for vertex and swap
        # conflicts and roadmap moves by LaCAM's own validator (the disc / body checks do not apply).
        starts, goals = set_starts_goals_config(
            [a["start"] for a in agents_rt],
            [a["goal"] for a in agents_rt],
            map_,
        )
        planner = LaCAM() if mapf_solver_name == 'lacam' else LaCAM_random()
        solution_config = planner.solve(map_, starts, goals,seed=0,time_limit_ms=1000*time_limit,max_iterations=max_iterations,verbose=0)
        solution, solution_info = planner.get_solution_dict(solution_config, [a["name"] for a in agents_rt])
        solution_info["collision"] = False
        if solution and not is_valid_mapf_solution(map_, starts, goals, solution_config):
            solution_info.update(success=False, collision=True)
        summary = summarize_solution(solution, solution_info, mapf_solver_config, map_)
        return summary
    elif mapf_solver_name == 'sipp':
        dynamic_obstacles = dict()
        sipp_planner = SippPlanner(map_, agents_rt, dynamic_obstacles=dynamic_obstacles, heuristic_type=heuristic_type,
                                   time_limit=time_limit, max_iterations=max_iterations,
                                   exact_vertices=bool(mapf_solver_config.get("sipp_exact_vertices", False)))
        solution,solution_info = sipp_planner.compute_plan()
    elif mapf_solver_name == 'ccbs':
        # One solver with switches: ccbs_use_cardinal (cardinal conflicts first), ccbs_disjoint, ccbs_hlh.
        env = CCBS_Environment(map_, agents_rt)
        ccbs = CCBS(
            env,
            time_limit=time_limit,
            max_iterations=max_iterations,
            verbose=False,
            use_cardinal=bool(mapf_solver_config.get("ccbs_use_cardinal", False)),
            disjoint=bool(mapf_solver_config.get("ccbs_disjoint", False)),
            hlh=mapf_solver_config.get("ccbs_hlh"),                                 # None | "lp" | "greedy"; needs cardinal
        )
        solution, solution_info = ccbs.search()
    else:
        raise ValueError(f"Invalid algorithm: {mapf_solver_name}")
    solution_info = check_solution_collision(solution, agent_radius, solution_info, bodies, map_)
    summary = summarize_solution(solution, solution_info, mapf_solver_config, map_, bodies)
    return summary


def solve_mapd(map_, agents, tasks, non_task_endpoints, mapf_solver_config: dict, rng=None) -> dict:
    """Life-long MAPD with Neural-ATTF (token passing; SIPP or grid low level) driven by ``Simulation``.

    Inputs are in the roadmap frame (see ``util.mapd_agents_to_roadmap_frame`` / ``points_to_roadmap_frame``):
    ``agents`` (``name``, ``start``, optionally ``yaw`` and their own ``"agent"``), ``tasks`` (dicts with
    ``task_name``, ``start_time`` and ``waypoints``, as ``tasks_generator.convert_mapd_tasks`` makes them) and
    the parking spots ``non_task_endpoints``. Config keys: the body as for ``solve_mapf`` (``agent`` /
    ``agent_radius`` / ``agent_velocity`` / ``agent_clearance``), ``low_level`` ("sipp" | "grid"; default
    "sipp"), ``heuristic_type`` (default "euclidean"), ``horizon`` (ticks, default 2000),
    ``timestep_duration`` (default 1), ``delay_probability`` (default 0), ``seed`` (default 0; or pass
    ``rng``, a ``random.Random``, for the simulation's delays) and ``neural_attf_options`` (more
    ``NeuralATTF`` keyword arguments).

    Returns a summary like ``solve_mapf``'s: ``schedule`` (the executed paths), costs, ``completed_tasks`` /
    ``n_tasks``, ``completed_tasks_times``, ``n_replans``, SIPP effort, ``runtime`` (wall) and ``algo_time``
    (planner), ``success`` (every task done, collision-free) and ``collision`` from the exact check."""
    import random

    agents = with_bodies(list(agents), mapf_solver_config, "neural_attf")
    bodies = _bodies_of(agents)
    dt = float(mapf_solver_config.get("timestep_duration", 1.0))
    horizon = int(mapf_solver_config.get("horizon", 2000))
    planner = NeuralATTF(
        map_, agents, non_task_endpoints,
        heuristic_type=mapf_solver_config.get("heuristic_type", "euclidean"),
        low_level=mapf_solver_config.get("low_level", "sipp"),
        timestep_duration=dt,
        **mapf_solver_config.get("neural_attf_options", {}),
    )
    sim = Simulation(tasks=[], agents=agents, delay_probability=float(mapf_solver_config.get("delay_probability", 0.0)),
                     rng=rng or random.Random(mapf_solver_config.get("seed", 0)), timestep_duration=dt)
    st = time.perf_counter()
    while planner.get_completed_tasks() < len(tasks) and sim.get_time() < horizon:
        new_tasks = [t for t in tasks if int(t["start_time"]) == sim.get_time()]
        if new_tasks:
            planner.update_tasks(sim.get_time(), new_tasks)
        sim.time_forward(planner)
    solution = sim.get_actual_paths()
    solution_info = {
        "success": planner.get_completed_tasks() == len(tasks),
        "runtime": time.perf_counter() - st,
        "algo_time": sim.get_algo_time(),
        "steps": sim.get_time(),
        "horizon": horizon,
        "completed_tasks": planner.get_completed_tasks(),
        "n_tasks": len(tasks),
        "completed_tasks_times": planner.get_completed_tasks_times(),
        "n_replans": planner.get_n_replans(),
        "low_level_iterations": planner.get_sipp_iterations(),
        "low_level_calls": planner.get_sipp_calls(),
        "times_agent_delayed": dict(sim.times_agent_delayed),
    }
    solution_info = check_solution_collision(solution, mapf_solver_config.get("agent_radius") or 0.0,
                                             solution_info, bodies, map_, require_success=False)
    if solution_info["collision"]:
        solution_info["success"] = False
    config = {"mapf_solver_name": "neural_attf", "agent_radius": 0.0, "agent_velocity": 0.0, "time_limit": None,
              "max_iterations": None, "heuristic_type": "euclidean", **mapf_solver_config}
    return summarize_solution(solution, solution_info, config, map_, bodies)


def computer_solution_cost(solution:dict):
    cost_dict = {}
    flowtime = 0
    total_travel_time = 0
    total_wait_time = 0
    makespan = 0
    success = True if solution is not None and solution != {} and solution != False else False
    if success:
        for agent, path in solution.items():
            path_array = np.array([_xyz(p) for p in path], dtype=float)  # (N, D): positions only, not yaw / pitch
            dist_travel = np.linalg.norm(path_array[1:] - path_array[:-1], axis=1) # (N-1,)
            time_stamp = np.array([p["t"] for p in path], dtype=float) # (N,)
            delta_time = np.diff(time_stamp) # (N-1,)

            wait_time = delta_time[dist_travel == 0]
            travel_time = delta_time[dist_travel > 0]

            flowtime += dist_travel.sum().item()
            total_travel_time += travel_time.sum().item()
            total_wait_time += wait_time.sum().item()
            makespan = max(makespan, path[-1]["t"] - path[0]["t"])

            cost_dict[agent] = {}
            cost_dict[agent]["path_length"] = dist_travel.sum().item()
            cost_dict[agent]["flowtime"] = travel_time.sum().item()
            cost_dict[agent]["wait_time"] = wait_time.sum().item()
            cost_dict[agent]["makespan"] = path[-1]["t"] - path[0]["t"]
        solution_temp = {}
        for agent, path in solution.items():
            solution_temp[agent] = []
            for state in path:
                for key, value in state.items():
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        state[key] = float(value)
                solution_temp[agent].append(state)
        solution = solution_temp
    all_cost_dict = {}
    all_cost_dict["path_length"] = flowtime
    all_cost_dict["flowtime"] = total_travel_time
    all_cost_dict["wait_time"] = total_wait_time
    all_cost_dict["makespan"] = makespan
    all_cost_dict["success"] = success
    return cost_dict, all_cost_dict

def check_solution_collision(solution, agent_radius, solution_info, bodies=None, map_=None, require_success=True):
    """``solution_info`` plus ``collision`` (and ``success`` False when there is one). Agents with bodies
    (``bodies``: name -> ``Agent``) get the exact check (``check_collision_bodies``, with ``map_``'s sweep),
    which uses the planners' own contact rule; otherwise the disc check at ``agent_radius`` (none at 0;
    LaCAM's point particles are validated by LaCAM itself). Only a successful solution is checked unless
    ``require_success`` is False."""
    solution_info_collision = solution_info.copy()
    solution_info_collision["collision"] = False
    if not solution or (require_success and not solution_info["success"]):
        return solution_info_collision
    if bodies:
        collisions = check_collision_bodies(solution, bodies, map_, verbose=False)
    elif agent_radius:
        collisions = check_collision(solution, agent_radius, verbose=False)
    else:
        collisions = {}
    if collisions:
        solution_info_collision["success"] = False
        solution_info_collision["collision"] = True
        solution_info_collision["collisions"] = {f"{a}|{b}": float(t) if np.isscalar(t) else t
                                                 for (a, b), t in collisions.items()}
    return solution_info_collision

def summarize_solution(solution, solution_info, mapf_solver_config, map_, bodies=None):
    agent_cost, total_cost = computer_solution_cost(solution)
    summary = {}
    summary["schedule"] = solution
    summary["schedule_cost"] = agent_cost
    summary.update(total_cost)
    summary.update(solution_info)
    summary["mapf_solver_name"] = mapf_solver_config["mapf_solver_name"]
    summary["agent_radius"] = mapf_solver_config["agent_radius"]
    summary["agent_velocity"] = mapf_solver_config["agent_velocity"]
    summary["time_limit"] = mapf_solver_config["time_limit"]
    summary["max_iterations"] = mapf_solver_config["max_iterations"]
    summary["heuristic_type"] = mapf_solver_config["heuristic_type"]
    if mapf_solver_config.get("low_level_astar_max_iterations") is not None:
        summary["low_level_astar_max_iterations"] = int(mapf_solver_config["low_level_astar_max_iterations"])
    if bodies:                                   # each agent's body, for checking the schedule later
        summary["agents"] = {name: body.to_dict() for name, body in bodies.items()}
    summary["num_nodes"] = len(map_.nodes)
    summary["num_edges"] = len(map_.edges)
    return summary