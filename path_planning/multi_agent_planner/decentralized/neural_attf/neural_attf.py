"""
Neural-ATTF (Adaptive Token-Passing for Task assignment + Neural STA*) on
arbitrary graphs.

This is a refactor of the original lattice-grid implementation (see
``~/Documents/code/Neural_ATTF/Simulation/ATTF.py``) so that the same
token-passing logic operates on any roadmap exposed by
``path_planning.common.environment.map.graph_sampler.GraphSampler`` — discrete
grid, PRM, CDT, RRG, etc.

Spatial primitives are point tuples taken from ``Node.current``. Obstacles are
checked through ``graph_map.in_collision_point``. The heuristic is Euclidean
over node points. The neural-encoder guidance path is optional and goes through
a ``GridOverlay`` so the U-Net can still run on non-grid graphs (the encoder
itself stays 2D-image-based).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.multi_agent_planner.decentralized.neural_attf.cbs import Environment
from path_planning.multi_agent_planner.decentralized.neural_attf.grid_overlay import GridOverlay
from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner

try:
    import torch  # type: ignore
except ImportError:  # torch is only required when encoder is supplied
    torch = None  # type: ignore


def _as_point(p) -> Tuple[float, ...]:
    if isinstance(p, dict):
        if "z" in p:
            return (float(p["x"]), float(p["y"]), float(p["z"]))
        return (float(p["x"]), float(p["y"]))
    return tuple(float(c) for c in p)


def _point_schedule(p, t) -> dict:
    """A timed-waypoint dict ``{x, y[, z], t}`` for SIPP dynamic obstacles."""
    p = _as_point(p)
    d = {"x": float(p[0]), "y": float(p[1]), "t": float(t)}
    if len(p) == 3:
        d["z"] = float(p[2])
    return d


class _Loc:
    """Minimal stand-in exposing ``.point`` (matches cbs.Location's interface)."""
    __slots__ = ("point",)

    def __init__(self, point):
        self.point = tuple(point)


class _SegState:
    """Minimal path state exposing ``.location.point`` so SIPP-derived paths plug
    into the same join code as the grid Environment solution states."""
    __slots__ = ("location",)

    def __init__(self, point):
        self.location = _Loc(point)


class NeuralATTF:
    def __init__(
        self,
        graph_map: GraphSampler,
        agents: List[dict],
        non_task_endpoints: Sequence[Sequence[float]],
        a_star_max_iter: int = 4000,
        encoder=None,
        grid_overlay: Optional[GridOverlay] = None,
        alpha: float = 0.001,
        num_goal_wait_steps: int = 0,
        heuristic_type: str = "euclidean",
        deadlock_radius: float = 2.0,
        agent_radius: float = 0.0,
        velocity: float = 0.0,
        low_level: str = "grid",
        device: str = "cpu",
        sipp_time_limit: float | None = None,
        timestep_duration: float = 1.0,
        max_waypoint_edge_len: float = 5.0,
        park_retry_cooldown: int = 3,
        parking_goal_clearance: float = 0.0,
        sipp_obstacle_horizon: float | None = None,
    ):
        self.graph_map = graph_map
        self.agents = agents
        self.assigned_tasks = set()
        self.encoder = encoder
        self.grid_overlay = grid_overlay
        self.alpha = alpha
        self.num_goal_wait_steps = num_goal_wait_steps
        self.heuristic_type = heuristic_type
        self.a_star_max_iter = a_star_max_iter
        self.deadlock_radius = float(deadlock_radius)
        self.agent_radius = float(agent_radius)
        self.velocity = float(velocity)
        self.low_level = low_level
        self.device = device
        # Wall-clock cap (s) per SIPP low-level call; None = unbounded (stops only
        # at the iteration cap). Real seconds each simulation step represents:
        # one step covers ~= timestep_duration * velocity world units.
        self.sipp_time_limit = sipp_time_limit
        self.timestep_duration = float(timestep_duration)
        # Max spacing (world units) between consecutive trajectory waypoints. SIPP edges
        # longer than this are subdivided equal-parts so the path is a fine polyline that
        # the simulation walks at constant velocity (multiple waypoints per tick).
        self.max_waypoint_edge_len = float(max_waypoint_edge_len)
        self.park_retry_cooldown = int(park_retry_cooldown)
        # Idle agents avoid parking within this distance of an ACTIVE (pending/assigned)
        # delivery goal, so finished agents leave congested destination regions instead of
        # resting there and walling off the corridor for later tasks. 0 disables the rule.
        self.parking_goal_clearance = float(parking_goal_clearance)
        # Finite blocking horizon (s) for other agents' resting positions in SIPP; None/<=0 =
        # unbounded (block forever). A finite value lets SIPP route through a currently-resting
        # agent for far-future arrivals, breaking corridor deadlocks (the blocker is then moved
        # by the idle-agent re-router, or the conflict is caught by the sim backstop).
        self.sipp_obstacle_horizon = sipp_obstacle_horizon

        # Agent footprint is the set of graph nodes a radius-``agent_radius`` disk
        # covers (computed via the map's CGAL swept-collision query), so it works on
        # any roadmap — grid or continuous PRM. At radius 0 an agent is a point.
        self._coverage_cache: dict = {}
        self._node_kdtree = None       # nearest-node snap (built from graph_map.nodes)
        self._node_kdtree_n = -1
        if self.agent_radius > 0 or self.low_level == "sipp":
            # SIPP / radius coverage both rely on the CGAL constraint sweep.
            self.graph_map.use_constraint_sweep = True
            self.graph_map.set_constraint_sweep()

        non_task_endpoints = [tuple(p) for p in non_task_endpoints]
        if len(agents) > len(non_task_endpoints):
            print(
                "There are more agents than non task endpoints, instance is not well-formed; "
                "padding with agent starts."
            )
            non_task_endpoints = non_task_endpoints + [tuple(a["start"]) for a in agents]
        self.non_task_endpoints = set(non_task_endpoints)

        # Cached extent for fallback "idle" cost used in find_closest_agent.
        bounds = np.asarray(graph_map.bounds, dtype=float)
        self._extent = float(np.sum(bounds[:, 1] - bounds[:, 0]))

        self.token: dict = {}
        self.init_token()

    def init_token(self):
        self.token["agents"] = {}
        self.token["agents_size"] = {}
        self.token["tasks"] = {}
        self.token["start_tasks_times"] = {}
        self.token["completed_tasks_times"] = {}
        self.token["agents_to_tasks"] = {}
        self.token["completed_tasks"] = 0
        self.token["n_replans"] = 0
        self.token["path_ends"] = set()
        self.token["occupied_non_task_endpoints"] = set()
        self.token["delayed_agents"] = []
        self.token["delayed_agents_to_reach_task_start"] = []
        # task name -> time first assigned / owning agent at assignment.
        self.token["assigned_tasks_times"] = {}
        self.token["assigned_tasks_agent"] = {}
        # SIPP low-level effort, sourced from solution_info["low_level_iterations"].
        self.token["sipp_iterations"] = 0
        self.token["sipp_calls"] = 0
        self.token["sipp_iterations_max_seen"] = 0
        # agent name -> earliest step its parking dispatch may run again (back-off).
        self.token["park_retry_after"] = {}
        for a in self.agents:
            start = tuple(a["start"])
            self.token["agents"][a["name"]] = [start]
            self.token["path_ends"].add(start)
            if start in self.non_task_endpoints:
                self.token["occupied_non_task_endpoints"].add(start)
        self.token["deadlock_count_per_agent"] = defaultdict(lambda: 0)
        self.token["assigned_task_pairs"] = {}

    # ------------------------------------------------------------------ tasks

    def update_tasks(self, t: int, new_tasks: Iterable[dict], assigned_task_pairs: Optional[dict] = None):
        if assigned_task_pairs:
            self.token["assigned_task_pairs"].update(assigned_task_pairs)
        for new_task in new_tasks:
            # A task is stored as an ordered list of >=2 waypoints. A multi-leg
            # (pickup->delivery->pickup->delivery...) task is given as ``waypoints``;
            # legacy single-leg tasks give ``start``/``goal`` (kept as 2 waypoints).
            if "waypoints" in new_task:
                wpts = [tuple(p) for p in new_task["waypoints"]]
            else:
                wpts = [tuple(new_task["start"]), tuple(new_task["goal"])]
            self.token["tasks"][new_task["task_name"]] = wpts
            self.token["start_tasks_times"][new_task["task_name"]] = t

    def get_idle_agents(self) -> dict:
        agents = {}
        for name, path in self.token["agents"].items():
            if (
                name in self.token["agents_to_tasks"]
                and self.token["agents_to_tasks"][name]["task_name"] == "safe_idle"
            ):
                agents[name] = path
            if len(path) == 1:
                agents[name] = path
        return agents

    # -------------------------------------------------------------- heuristic

    def admissible_heuristic(self, task_pos, agent_pos) -> float:
        if self.heuristic_type == "manhattan":
            return sum(abs(a - b) for a, b in zip(task_pos, agent_pos))
        d2 = sum((a - b) ** 2 for a, b in zip(task_pos, agent_pos))
        if self.alpha and self.heuristic_type == "manhattan_euclidean":
            mhd = sum(abs(a - b) for a, b in zip(task_pos, agent_pos))
            return mhd + self.alpha * math.sqrt(d2)
        return math.sqrt(d2)

    # ------------------------------------------------------------ assignment

    def find_closest_agent(self, available_tasks, idle_agents, token, agents_size=None):
        pairs = []
        if agents_size is not None:
            self.token["agents_size"] = agents_size

        for agent in idle_agents.keys():
            if (
                agent in token["agents_to_tasks"]
                and token["agents_to_tasks"][agent]["task_name"] != "safe_idle"
            ):
                task_name = token["agents_to_tasks"][agent]["task_name"]
                entry = token["agents_to_tasks"][agent]
                # Full waypoint route of the in-progress task (legacy entries only
                # carried start/goal). NOTE: progress through a partially-completed
                # multi-leg route is not tracked, so a delayed mid-route agent
                # replans the whole route from its current position (acceptable
                # while delay_probability == 0).
                task = entry.get("waypoints", [entry["start"], entry["goal"]])
                pairs.append((agent, task_name, task, -1))
            elif len(available_tasks) != 0:
                agent_position = idle_agents[agent][0]
                for task_name, task_positions in available_tasks.items():
                    if (
                        task_name in self.token["assigned_task_pairs"]
                        and self.token["assigned_task_pairs"][task_name] != agent
                    ):
                        continue
                    task_start = task_positions[0]
                    d = self.admissible_heuristic(task_start, agent_position)
                    pairs.append((agent, task_name, task_positions, d))

            cost = len(token["agents"][agent]) + 2.0 * self._extent
            pairs.append((agent, None, None, cost))

        pairs = sorted(pairs, key=lambda x: x[3])
        assigned_pairs = deque()
        assigned_tasks = set()
        assigned_agents = set()
        valid_pairs = []
        for pair in pairs:
            agent = pair[0]
            task_name = pair[1]
            if task_name:
                if task_name not in assigned_tasks and agent not in assigned_agents:
                    assigned_pairs.append(pair)
                    assigned_tasks.add(task_name)
                    assigned_agents.add(agent)
                    valid_pairs.append(pair)
            else:
                if agent not in assigned_agents:
                    assigned_pairs.append(pair)
                    assigned_agents.add(agent)
        return assigned_pairs, valid_pairs

    # ----------------------------------------------------- dynamic obstacles

    def _covered_nodes(self, p1, p2) -> set:
        """
        Graph nodes a radius-``agent_radius`` disk covers while moving ``p1 -> p2``.

        Uses the map's CGAL swept-collision query (the same machinery as CBS/SIPP),
        which returns the exact set of roadmap vertices/edges the moving disk
        overlaps — real ``node.current`` tuples, so they match A* state points on any
        roadmap (grid or continuous PRM). At ``agent_radius == 0`` (or if the sweep is
        unavailable) the footprint collapses to the single point ``p2``.
        """
        p1 = tuple(float(c) for c in p1)
        p2 = tuple(float(c) for c in p2)
        if self.agent_radius <= 0:
            return {p2}
        key = (p1, p2)
        cached = self._coverage_cache.get(key)
        if cached is not None:
            return cached
        covered = {p2}
        try:
            sweep = self.graph_map.get_constraint_sweep(
                p1, p2, self.velocity, 2.0 * self.agent_radius, use_interval=True
            )
            if sweep is not None:
                verts, edges = sweep
                covered |= set(verts)
                for a, b in edges:
                    covered.add(tuple(a))
                    covered.add(tuple(b))
        except Exception:
            # Fallback: nodes within the disk via the roadmap KD-tree.
            kdt = getattr(self.graph_map, "sample_kd_tree", None)
            if kdt is not None:
                idxs = kdt.query_ball_point(np.asarray(p2, dtype=float), self.agent_radius)
                for i in idxs:
                    covered.add(tuple(self.graph_map.nodes[int(i)].current))
        self._coverage_cache[key] = covered
        return covered

    def get_moving_obstacles_agents(self, agents, time_start: int) -> dict:
        obstacles: dict = {}
        for name, path in agents.items():
            if len(path) > time_start and len(path) > 1:
                for i in range(time_start, len(path)):
                    k = i - time_start
                    # Cover the swept motion arriving at this timestep (radius-aware,
                    # graph-node footprint) rather than crude grid offsets.
                    prev = path[i - 1] if i >= 1 else path[i]
                    for c in self._covered_nodes(prev, path[i]):
                        obstacles[(c[0], c[1], k)] = name
                    if i == len(path) - 1:
                        for c in self._covered_nodes(path[i], path[i]):
                            obstacles[(c[0], c[1], k + 1)] = name
        return obstacles

    def get_idle_obstacles_agents(self, agents_paths, delayed_agents, _time_start: int = 0) -> set:
        obstacles: set = set()
        for _, path in agents_paths.items():
            last = path[-1]
            if len(path) == 1 or last in self.non_task_endpoints:
                obstacles |= self._covered_nodes(last, last)
        for agent_name in delayed_agents:
            pos = self.token["agents"][agent_name][0]
            obstacles |= self._covered_nodes(pos, pos)
        return obstacles

    # ------------------------------------------------------------- idle plan

    def check_safe_idle(self, agent_pos, agent_name) -> bool:
        # Never idle on (or within 2*agent_radius of) a pending task waypoint — a resting agent's
        # footprint there would block that task's pickup/delivery.
        clearance = 2.0 * self.agent_radius if self.agent_radius > 0 else 1e-9
        ap = np.asarray(agent_pos, dtype=float)
        for _, task in self.token["tasks"].items():
            for wp in task:
                if float(np.linalg.norm(ap - np.asarray(wp, dtype=float))) <= clearance:
                    return False
        if len(self.token["agents"][agent_name]) != 1:
            return False
        if self.token["agents"][agent_name][0] not in self.non_task_endpoints:
            return False
        for start_goal in self.get_agents_to_tasks_starts_goals():
            if tuple(start_goal) == tuple(agent_pos):
                return False
        # Resting in a still-active delivery region blocks the corridor for later tasks; force a
        # re-park to an endpoint clear of active goals.
        if self._near_active_goal(agent_pos):
            return False
        return True

    def _blocks_pending_task(self, endpoint) -> bool:
        """True if ``endpoint`` lies on (or within 2*agent_radius of) any pending or assigned task
        waypoint. Parking there would let the resting agent's footprint occupy a task pickup/goal
        and permanently block that leg, so such endpoints are avoided."""
        clearance = 2.0 * self.agent_radius if self.agent_radius > 0 else 1e-9
        ep = np.asarray(endpoint, dtype=float)
        for task in self.token["tasks"].values():
            for wp in task:
                if float(np.linalg.norm(ep - np.asarray(wp, dtype=float))) <= clearance:
                    return True
        for sg in self.get_agents_to_tasks_starts_goals():
            if float(np.linalg.norm(ep - np.asarray(sg, dtype=float))) <= clearance:
                return True
        return False

    def _active_task_goals(self) -> list:
        """Delivery goals of currently ACTIVE work: the last waypoint of each pending task plus
        each assigned (non-safe_idle) task's goal. Used to keep idle agents from parking in a
        still-busy destination region."""
        goals = [tuple(task[-1]) for task in self.token["tasks"].values() if len(task)]
        for info in self.token["agents_to_tasks"].values():
            if info.get("task_name") != "safe_idle" and info.get("goal") is not None:
                goals.append(tuple(info["goal"]))
        return goals

    def _near_active_goal(self, pt) -> bool:
        """True if ``pt`` is within ``parking_goal_clearance`` of any active delivery goal.
        Always False when the clearance is disabled (<= 0)."""
        if self.parking_goal_clearance <= 0:
            return False
        p = np.asarray(pt, dtype=float)
        for g in self._active_task_goals():
            if float(np.linalg.norm(p - np.asarray(g, dtype=float))) <= self.parking_goal_clearance:
                return True
        return False

    def get_closest_non_task_endpoint(self, agent_pos):
        occupied = self.token["occupied_non_task_endpoints"].copy()
        for _, path in self.token["agents"].items():
            if path[-1] in occupied:
                occupied.remove(path[-1])
        # Free any endpoints that no agent is currently sitting on.
        for endpoint in occupied:
            self.token["occupied_non_task_endpoints"].discard(endpoint)

        # Prefer endpoints that are (a) clear of active delivery regions, then (b) not on a
        # pending/assigned task waypoint; fall back to any free endpoint if every candidate is
        # blocked (preserve the must-return contract). ``best`` = nearest endpoint clear of both
        # active goals and task waypoints; ``best_clear`` relaxes the task-waypoint rule;
        # ``best_fallback`` = nearest free endpoint regardless.
        best = None
        best_d = float("inf")
        best_clear = None
        best_clear_d = float("inf")
        best_fallback = None
        best_fallback_d = float("inf")
        for endpoint in self.non_task_endpoints:
            if endpoint in self.token["occupied_non_task_endpoints"]:
                continue
            d = self.admissible_heuristic(endpoint, agent_pos)
            if d < best_fallback_d:
                best_fallback_d = d
                best_fallback = endpoint
            if self._near_active_goal(endpoint):
                continue
            if d < best_clear_d:
                best_clear_d = d
                best_clear = endpoint
            if self._blocks_pending_task(endpoint):
                continue
            if d < best_d:
                best_d = d
                best = endpoint
        best = best if best is not None else (best_clear if best_clear is not None else best_fallback)
        if best is None:
            raise RuntimeError("No free non-task endpoint available; instance not well-formed.")
        return best

    def update_ends(self, agent_pos, agent_name: Optional[str] = None):
        agent_pos = tuple(agent_pos)
        others = [
            tuple(path[-1])
            for name, path in self.token["agents"].items()
            if name != agent_name
        ]
        if agent_pos not in others:
            self.token["path_ends"].discard(agent_pos)
            self.token["occupied_non_task_endpoints"].discard(agent_pos)

    def get_agents_to_tasks_goals(self):
        return {tuple(el["goal"]) for el in self.token["agents_to_tasks"].values()}

    def get_agents_to_tasks_starts_goals(self):
        out = set()
        for el in self.token["agents_to_tasks"].values():
            for wp in el.get("waypoints", [el["start"], el["goal"]]):
                out.add(tuple(wp))
        return out

    def get_completed_tasks(self):
        return self.token["completed_tasks"]

    def get_completed_tasks_times(self):
        return self.token["completed_tasks_times"]

    def get_n_replans(self):
        return self.token["n_replans"]

    def get_assigned_tasks_times(self):
        """task name -> time first assigned."""
        return self.token["assigned_tasks_times"]

    def get_assigned_tasks_agent(self):
        """task name -> owning agent at assignment."""
        return self.token["assigned_tasks_agent"]

    def get_start_tasks_times(self):
        """task name -> time the task entered the system."""
        return self.token["start_tasks_times"]

    def get_sipp_iterations(self):
        """Total SIPP low-level expansions across the run."""
        return self.token["sipp_iterations"]

    def get_sipp_calls(self):
        """Number of SIPP low-level calls."""
        return self.token["sipp_calls"]

    def get_sipp_iterations_max_seen(self):
        """Worst single SIPP low-level call (expansions)."""
        return self.token["sipp_iterations_max_seen"]

    def get_token(self):
        return self.token

    # ------------------------------------------------------------- A* driver

    def plan(self, agent_name, start, goal, all_idle_agents, all_delayed_agents, cost_map, cost: int):
        start = tuple(start)
        goal = tuple(goal)
        if self.low_level == "sipp":
            # Continuous-time, interval-based, radius-aware low-level planning.
            # cost_map / cost (grid time-offset) are not used by SIPP.
            return self._plan_sipp(agent_name, start, goal)
        moving = self.get_moving_obstacles_agents(self.token["agents"], cost)
        idle = self.get_idle_obstacles_agents(all_idle_agents, all_delayed_agents, cost)
        agents = [{"name": agent_name, "start": start, "goal": goal}]
        heur = self.heuristic_type if self.heuristic_type in {"manhattan", "euclidean"} else "euclidean"
        # Scale A* budget with graph size so larger maps still terminate within
        # admissible-heuristic time-expanded search depth.
        budget = max(int(self.a_star_max_iter), 10 * len(self.graph_map.nodes))
        env = Environment(
            self.graph_map,
            agents,
            obstacles=idle,
            moving_obstacles=moving,
            astar_max_iterations=budget,
            heuristic_type=heur,
            cost_map=cost_map,
            alpha=self.alpha,
        )
        if env.ignore_agent_dict.get(agent_name, False):
            return False
        solution, _ = env.compute_solution()
        return solution

    # ----------------------------------------------------- SIPP (continuous t)

    def _snap_to_node(self, pt):
        """Return ``pt`` if it is already a graph node, else the nearest node coord
        (SIPP states must be graph nodes; resampled mid-edge positions may not be).

        Uses a KD-tree built from the authoritative ``graph_map.nodes`` — the map's
        own ``sample_kd_tree`` can be stale for roadmap types (cdt/voronoi/rrg) that
        rebuild the node set, so its indices may not align with ``nodes``."""
        from path_planning.common.environment.node import Node

        pt = tuple(pt)
        if Node(pt) in self.graph_map.node_index_dict:
            return pt
        nodes = self.graph_map.nodes
        if not nodes:
            return pt
        if self._node_kdtree is None or self._node_kdtree_n != len(nodes):
            from scipy.spatial import KDTree

            self._node_kdtree = KDTree([tuple(n.current) for n in nodes])
            self._node_kdtree_n = len(nodes)
        _, idx = self._node_kdtree.query(np.asarray(pt, dtype=float), k=1)
        return tuple(nodes[int(idx)].current)

    def _other_agent_schedules(self, agent_name) -> dict:
        """Other agents' committed paths as SIPP dynamic obstacles: name -> timed
        schedule ``[{x, y, t}]``. SIPP plans in seconds, so step index ``k`` is
        stamped at SIPP time ``k * timestep_duration`` to align other agents'
        committed paths with SIPP's continuous clock."""
        sched = {}
        for name, path in self.token["agents"].items():
            if name == agent_name or not path:
                continue
            sched[name] = [
                _point_schedule(p, k * self.timestep_duration) for k, p in enumerate(path)
            ]
        return sched

    def _plan_sipp(self, agent_name, start, goal):
        start = self._snap_to_node(start)
        goal = self._snap_to_node(goal)
        heur = self.heuristic_type if self.heuristic_type in {"manhattan", "euclidean"} else "euclidean"
        budget = max(int(self.a_star_max_iter), 10 * len(self.graph_map.nodes))
        planner = SippPlanner(
            self.graph_map,
            dynamic_obstacles=self._other_agent_schedules(agent_name),
            agents=[{"name": agent_name, "start": start, "goal": goal}],
            radius=self.agent_radius,
            velocity=self.velocity,
            use_constraint_sweep=True,
            heuristic_type=heur,
            time_limit=self.sipp_time_limit,
            sipp_max_iterations=budget,
            obstacle_horizon=self.sipp_obstacle_horizon,
        )
        solution, info = planner.compute_plan()
        self._record_sipp_metrics(info)
        schedule = solution.get(agent_name) if solution else None
        if not schedule:
            return False
        positions = self._resample_unit(
            schedule, goal, dt=self.timestep_duration, max_edge_len=self.max_waypoint_edge_len
        )
        return {agent_name: [_SegState(p) for p in positions]}

    def _record_sipp_metrics(self, info: dict) -> None:
        """Fold one SIPP low-level call's effort into the run-wide counters."""
        iters = int(info.get("low_level_iterations", 0)) if info else 0
        self.token["sipp_iterations"] += iters
        self.token["sipp_calls"] += 1
        if iters > self.token["sipp_iterations_max_seen"]:
            self.token["sipp_iterations_max_seen"] = iters

    @staticmethod
    def _resample_unit(schedule, goal, dt: float = 1.0, max_edge_len: float = 5.0):
        """Convert a continuous-time SIPP schedule ``[{t,x,y}]`` into a FINE waypoint
        polyline that follows the roadmap edges.

        SIPP plans on the roadmap, so each schedule entry is a roadmap vertex and
        every pair of consecutive distinct entries is exactly one collision-free
        edge. We walk the schedule vertex-to-vertex (so a chord never cuts a corner
        through an obstacle) and subdivide each edge **equal-parts** so consecutive
        waypoints are at most ``max_edge_len`` world units apart:

        - a wait (consecutive equal vertices) becomes ``round((t1-t0)/dt)``
          repeated positions (>= 1), so the hold is never silently dropped;
        - a move of length ``L`` along one edge is split into ``n = ceil(L/max_edge_len)``
          equal sub-segments (each ``L/n <= max_edge_len``), all collinear on that edge.

        This sets the path *shape* only; the per-tick distance an agent travels is the
        simulation's arc-length budget (it walks multiple of these waypoints per tick at
        constant velocity). The result ends exactly on ``goal``. Output guarantee: every
        consecutive pair is equal (a wait) or collinear on one collision-free roadmap
        edge, so no chord crosses a static obstacle.
        """
        dt = float(dt) if dt and dt > 0 else 1.0
        max_edge_len = float(max_edge_len) if max_edge_len and max_edge_len > 0 else 0.0
        pts = sorted(((float(s["t"]), _as_point(s)) for s in schedule), key=lambda x: x[0])
        if not pts:
            return [tuple(goal)]

        out = [tuple(pts[0][1])]
        for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
            if p1 == p0:
                # Wait at the vertex: one repeat per dt of held real time (>= 1).
                n_wait = max(1, int(round((t1 - t0) / dt)))
                out.extend([tuple(p1)] * n_wait)
                continue
            # Move along the single roadmap edge p0 -> p1, equal-parts subdivided.
            if max_edge_len > 0.0:
                n_sub = max(1, int(math.ceil(math.dist(p0, p1) / max_edge_len)))
            else:
                n_sub = 1
            for k in range(1, n_sub + 1):
                frac = k / n_sub
                out.append(tuple(a + frac * (b - a) for a, b in zip(p0, p1)))

        if out[-1] != tuple(goal):
            out.append(tuple(goal))
        return out

    # ---------------------------------------------------- safe-idle dispatch

    def go_to_closest_non_task_endpoint(
        self, agent_name, agent_pos, all_idle_agents, all_delayed_agents, _cost_map=None
    ):
        """Route a stuck/idle agent to its nearest free parking endpoint.

        Returns ``True`` when the agent is already parked or a path was committed,
        ``False`` when no endpoint path exists (the agent is stuck) so the caller
        can apply a parking-dispatch back-off.
        """
        cur = tuple(self.token["agents"][agent_name][-1])
        # Already parked counts as done only if the current endpoint is clear of active delivery
        # regions; otherwise relocate it out of the congested corner.
        if cur in self.non_task_endpoints and not self._near_active_goal(cur):
            return True
        target = self.get_closest_non_task_endpoint(agent_pos)
        path = self.plan(agent_name, agent_pos, target, all_idle_agents, all_delayed_agents, None, 0)
        if not path:
            print(f"Solution to non-task endpoint not found for {agent_name}; trying deadlock recovery.")
            self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, all_delayed_agents, self.deadlock_radius)
            return False
        self.update_ends(agent_pos, agent_name)
        self.token["occupied_non_task_endpoints"].add(tuple(target))
        self.token["agents_to_tasks"][agent_name] = {
            "task_name": "safe_idle",
            "start": tuple(agent_pos),
            "goal": tuple(target),
            "predicted_cost": 0,
        }
        self.token["agents"][agent_name] = [tuple(state.location.point) for state in path[agent_name]]
        return True

    # ------------------------------------------------------ deadlock recovery

    def _interference_footprint(self, agent_name) -> set:
        """Graph-node coords ``agent_name`` must not rest on.

        The union of: every *other* agent's moving + idle footprint (radius-aware),
        all path ends, occupied parking spots, **every waypoint of pending/assigned
        tasks**, and assigned task start/goal cells. The task-waypoint inclusion is
        what breaks the repeated-deadlock loop — a stuck agent will no longer
        evacuate onto a cell some task still needs.
        """
        footprint: set = set()
        for name, path in self.token["agents"].items():
            if name == agent_name or not path:
                continue
            for i in range(len(path)):
                prev = path[i - 1] if i >= 1 else path[i]
                footprint |= self._covered_nodes(prev, path[i])
            # Resting (idle) footprint at the committed path end.
            footprint |= self._covered_nodes(path[-1], path[-1])
        footprint |= {tuple(p) for p in self.token["path_ends"]}
        footprint |= {tuple(p) for p in self.token["occupied_non_task_endpoints"]}
        for task in self.token["tasks"].values():
            for wp in task:
                footprint |= self._covered_nodes(wp, wp)
        footprint |= {tuple(sg) for sg in self.get_agents_to_tasks_starts_goals()}
        return footprint

    def _close_non_interfering_nodes(self, agent_pos, agent_name, r: float):
        """Reachable graph nodes within radius ``r`` of ``agent_pos`` whose resting
        footprint is clear of :meth:`_interference_footprint`, sorted nearest-first.

        Local-only — an empty list means the agent should stay put. A
        ``scipy.spatial.KDTree`` ball query (clearance ``2 * agent_radius``) replaces
        the old per-candidate CGAL sweep, which dominated runtime over the ~1500
        nodes inspected per call.
        """
        nodes = self.graph_map.nodes
        if not nodes:
            return []
        pt = np.asarray(agent_pos, dtype=float)
        # Reachable nodes within r (reuse the authoritative-node KD-tree).
        if self._node_kdtree is None or self._node_kdtree_n != len(nodes):
            from scipy.spatial import KDTree

            self._node_kdtree = KDTree([tuple(n.current) for n in nodes])
            self._node_kdtree_n = len(nodes)
        near_idxs = self._node_kdtree.query_ball_point(pt, r)

        footprint = self._interference_footprint(agent_name)
        clearance = 2.0 * self.agent_radius if self.agent_radius > 0 else 1e-9
        fp_tree = None
        if footprint:
            from scipy.spatial import KDTree

            fp_tree = KDTree([tuple(p) for p in footprint])

        candidates = []
        for i in near_idxs:
            cand = tuple(float(c) for c in nodes[int(i)].current)
            d = float(np.linalg.norm(np.asarray(cand) - pt))
            if d <= 1e-9:
                continue
            if self.graph_map.in_collision_point(cand):
                continue
            if fp_tree is not None and fp_tree.query_ball_point(np.asarray(cand, dtype=float), clearance):
                continue
            candidates.append((d, cand))
        candidates.sort(key=lambda x: x[0])
        return [c for _, c in candidates]

    def deadlock_recovery(self, agent_name, agent_pos, all_idle_agents, all_delayed_agents, r: float):
        self.token["deadlock_count_per_agent"][agent_name] += 1
        if self.token["deadlock_count_per_agent"][agent_name] < 2:
            return
        self.token["deadlock_count_per_agent"][agent_name] = 0
        candidates = self._close_non_interfering_nodes(agent_pos, agent_name, r)
        if not candidates:
            print(f"Deadlock recovery: no non-interfering nearby node for {agent_name}; staying put.")
            return
        # Try candidates nearest-first; commit the first whose trajectory also
        # plans collision-free (SIPP/A* treat other agents' paths as dynamic
        # obstacles, so a successful plan == no interference).
        for target in candidates:
            path = self.plan(agent_name, agent_pos, target, all_idle_agents, all_delayed_agents, None, 0)
            if not path:
                continue
            self.update_ends(agent_pos, agent_name)
            self.token["agents"][agent_name] = [tuple(state.location.point) for state in path[agent_name]]
            return
        print(f"Deadlock recovery: no collision-free escape for {agent_name}.")

    # ----------------------------------------------- neural cost-map (option)

    def _maybe_compute_cost_maps(self, valid_pairs) -> Optional[List]:
        """Run the U-Net encoder on a 2-channel image per (agent, task) pair.

        Returns a list of point-callable lookups parallel to ``valid_pairs``
        (two per pair: agent->task_start, task_start->task_goal), or None when
        the encoder is not configured.
        """
        if self.encoder is None or self.grid_overlay is None or not valid_pairs or torch is None:
            return None

        overlay = self.grid_overlay
        h, w = overlay.shape
        device = self.device

        # Start from the static obstacle maze; mask out idle-agent occupied cells.
        maze_np = overlay.maze.copy()
        for o in self.get_idle_obstacles_agents(self.token["agents"], self.token["delayed_agents"], 0):
            ci, cj = overlay.point_to_cell(o)
            maze_np[ci, cj] = 0.0
        maze = torch.tensor(maze_np, device=device).unsqueeze(0).unsqueeze(0).float()

        start_goals = torch.zeros((2 * len(valid_pairs), 1, h, w), device=device)
        for i, (agent_name, _, closest_task, _) in enumerate(valid_pairs):
            agent_pt = self.token["agents"][agent_name][0]
            ci, cj = overlay.point_to_cell(agent_pt)
            start_goals[2 * i, 0, ci, cj] = 1
            ci, cj = overlay.point_to_cell(closest_task[0])
            start_goals[2 * i, 0, ci, cj] = 1
            ci, cj = overlay.point_to_cell(closest_task[0])
            start_goals[2 * i + 1, 0, ci, cj] = 1
            ci, cj = overlay.point_to_cell(closest_task[1])
            start_goals[2 * i + 1, 0, ci, cj] = 1

        batch = torch.cat([maze.expand(2 * len(valid_pairs), -1, -1, -1), start_goals], dim=1)
        t0 = time.time()
        with torch.no_grad():
            cost_maps = self.encoder(batch).squeeze(1).detach().cpu().numpy()
        print(f"Encoder time: {time.time() - t0:.3f}s")
        return [overlay.make_cost_map_lookup(cost_maps[k]) for k in range(cost_maps.shape[0])]

    # -------------------------------------------------------- main step loop

    def time_forward(self, t: int, position: dict, delayed_agents: List[str], agents_size=None):
        if agents_size is not None:
            self.token["agents_size"] = agents_size

        # 1) check task completions
        for agent_name in self.token["agents"]:
            pos = _as_point(position[agent_name])
            if (
                agent_name in self.token["agents_to_tasks"]
                and pos == tuple(self.token["agents_to_tasks"][agent_name]["goal"])
                and len(self.token["agents"][agent_name]) == 1
            ):
                if self.token["agents_to_tasks"][agent_name]["task_name"] != "safe_idle":
                    self.token["completed_tasks"] += 1
                    self.token["completed_tasks_times"][
                        self.token["agents_to_tasks"][agent_name]["task_name"]
                    ] = t
                self.token["agents_to_tasks"].pop(agent_name)

        # 2) replan delayed agents
        self.token["delayed_agents"] = list(delayed_agents)
        for agent_name in self.token["delayed_agents"]:
            path = self.token["agents"][agent_name]
            self.token["n_replans"] += 1
            self.update_ends(path[-1], agent_name)
            if path[0] in self.non_task_endpoints:
                self.token["occupied_non_task_endpoints"].add(tuple(path[0]))
            else:
                self.token["path_ends"].add(tuple(path[0]))
            if agent_name in self.token["agents_to_tasks"]:
                if tuple(self.token["agents_to_tasks"][agent_name]["start"]) not in path:
                    self.token["delayed_agents_to_reach_task_start"].append(agent_name)
            self.token["agents"][agent_name] = [path[0]]

        # 3) assign idle agents -> tasks
        idle_agents = self.get_idle_agents()
        available_tasks = {
            name: task for name, task in self.token["tasks"].items() if name not in self.assigned_tasks
        }
        assigned_pairs, valid_pairs = self.find_closest_agent(available_tasks, idle_agents, self.token)
        cost_lookups = self._maybe_compute_cost_maps(valid_pairs)

        # 4) plan each assigned pair
        cost_map_idx = 0
        while len(idle_agents) > 0:
            agent_name, closest_task_name, closest_task, _ = assigned_pairs.popleft()

            all_idle_agents = {k: v for k, v in self.token["agents"].items() if k != agent_name}
            all_delayed_agents = [a for a in self.token["delayed_agents"] if a != agent_name]
            agent_pos = idle_agents.pop(agent_name)[0]

            if closest_task:
                # Plan the full multi-leg route agent_pos -> wp0 -> wp1 -> ... as a
                # chain of A* segments. ``cum`` is the running time offset (matches
                # the old 0 / cost1-1 offsets for the 2-waypoint case); each
                # non-final segment drops its last point when joined (it equals the
                # next segment's first point). Cost maps are consumed per segment
                # when an encoder is configured (None otherwise).
                waypoints = [tuple(p) for p in closest_task]
                segments = []
                seg_start = agent_pos
                cum = 0
                ok = True
                for wp in waypoints:
                    cost_map = None
                    if cost_lookups is not None and cost_map_idx < len(cost_lookups):
                        cost_map = cost_lookups[cost_map_idx]
                        cost_map_idx += 1
                    seg = self.plan(
                        agent_name, seg_start, wp, all_idle_agents, all_delayed_agents, cost_map, cum
                    )
                    if not seg:
                        ok = False
                        break
                    for _ in range(self.num_goal_wait_steps):
                        seg[agent_name].append(seg[agent_name][-1])
                    segments.append(seg[agent_name])
                    cum += len(seg[agent_name]) - 1
                    seg_start = wp
                if not ok:
                    if len(self.token["delayed_agents"]) == 0:
                        self.deadlock_recovery(
                            agent_name, agent_pos, all_idle_agents, all_delayed_agents, self.deadlock_radius
                        )
                    continue
                joined = []
                for i, seg_pts in enumerate(segments):
                    pts = [tuple(s.location.point) for s in seg_pts]
                    joined += pts[:-1] if i < len(segments) - 1 else pts
                last_pos = joined[-1]
                self.assigned_tasks.add(closest_task_name)
                # Assignment-time bookkeeping (first-assignment time is sticky; the
                # owning agent reflects the latest assignment).
                if closest_task_name not in self.token["assigned_tasks_times"]:
                    self.token["assigned_tasks_times"][closest_task_name] = t
                self.token["assigned_tasks_agent"][closest_task_name] = agent_name
                if agent_name not in self.token["agents_to_tasks"]:
                    self.token["tasks"].pop(closest_task_name, None)
                    task = available_tasks.pop(closest_task_name, closest_task)
                else:
                    task = closest_task
                if agent_name in self.token["delayed_agents_to_reach_task_start"]:
                    self.token["delayed_agents_to_reach_task_start"].remove(agent_name)
                self.update_ends(agent_pos, agent_name)
                if last_pos in self.non_task_endpoints:
                    self.token["occupied_non_task_endpoints"].add(last_pos)
                if len(self.token["agents"][agent_name]) > 1:
                    self.update_ends(self.token["agents"][agent_name][-1], agent_name)
                self.token["path_ends"].add(last_pos)
                wpts = [tuple(p) for p in task]
                self.token["agents_to_tasks"][agent_name] = {
                    "task_name": closest_task_name,
                    "start": wpts[0],
                    "goal": wpts[-1],
                    "waypoints": wpts,
                    "predicted_cost": len(joined),
                }
                self.token["agents"][agent_name] = joined

            elif self.check_safe_idle(agent_pos, agent_name):
                if agent_name in self.token["delayed_agents"]:
                    if (
                        agent_name in self.token["agents_to_tasks"]
                        and self.token["agents_to_tasks"][agent_name]["task_name"] == "safe_idle"
                    ):
                        goal = tuple(self.token["agents_to_tasks"][agent_name]["goal"])
                        self.token["occupied_non_task_endpoints"].discard(goal)
            else:
                # Parking back-off: a stuck agent's parking dispatch is skipped for
                # ``park_retry_cooldown`` steps after a failure, instead of re-running
                # SIPP toward an unreachable endpoint every step. A new task
                # assignment still routes through the task branch above, so the agent
                # is never starved.
                if t < self.token["park_retry_after"].get(agent_name, 0):
                    continue
                parked = self.go_to_closest_non_task_endpoint(
                    agent_name, agent_pos, all_idle_agents, all_delayed_agents
                )
                if not parked:
                    self.token["park_retry_after"][agent_name] = t + self.park_retry_cooldown

        # 5) re-route idle / just-finished agents that are blocking a mover. With
        #    SIPP's check_goal_safe_forever removed, an agent may commit to rest on a
        #    node that another agent's committed path needs; nudge such blockers aside.
        self._reroute_interfering_idle_agents(t)

    def _free_parking_endpoints_sorted(self, agent_pos, exclude):
        """Free (unoccupied), non-task-blocking parking endpoints, nearest-first, excluding
        ``exclude``. Refreshes occupancy first (an endpoint nobody currently rests on is free)."""
        occupied = self.token["occupied_non_task_endpoints"].copy()
        for _, path in self.token["agents"].items():
            occupied.discard(tuple(path[-1]))
        for endpoint in occupied:
            self.token["occupied_non_task_endpoints"].discard(endpoint)
        cands = []
        for ep in self.non_task_endpoints:
            ep = tuple(ep)
            if ep in self.token["occupied_non_task_endpoints"] or ep == tuple(exclude):
                continue
            if self._blocks_pending_task(ep):
                continue
            cands.append((self.admissible_heuristic(ep, agent_pos), ep))
        cands.sort(key=lambda x: x[0])
        return [c for _, c in cands]

    def _reroute_interfering_idle_agents(self, t: int) -> None:
        """Move any idle/just-finished agent (committed path length 1) that is sitting on
        another agent's *moving* committed path to its nearest free parking endpoint.

        A just-finished agent becomes idle the same step (its task is popped in phase 1,
        leaving a length-1 path), so this covers both idle and finished agents. Runs after
        the assignment/parking phase so every mover's path for this step is committed before
        blockers are evaluated. Routing blockers to a free parking endpoint (rather than the
        nearest open node) clears them out of the aisles entirely, avoiding the mid-aisle
        oscillation that stalled the all-nearest-node variant. Falls back to the nearest
        non-interfering node when no parking endpoint is reachable."""
        # Nodes covered by every agent's *moving* path (len > 1), keyed by their owner.
        moving_cover: dict = {}
        for name, path in self.token["agents"].items():
            if len(path) <= 1:
                continue
            for i in range(len(path)):
                prev = path[i - 1] if i >= 1 else path[i]
                for c in self._covered_nodes(prev, path[i]):
                    moving_cover.setdefault(tuple(c), name)
        if not moving_cover:
            return

        from scipy.spatial import KDTree

        cover_pts = list(moving_cover.keys())
        cover_tree = KDTree([tuple(p) for p in cover_pts])
        clearance = 2.0 * self.agent_radius if self.agent_radius > 0 else 1e-9

        for name, path in list(self.token["agents"].items()):
            if len(path) != 1:
                continue  # only idle / just-finished agents
            pos = tuple(path[0])
            hits = cover_tree.query_ball_point(np.asarray(pos, dtype=float), clearance)
            # Interfering only if some overlapping moving-path node belongs to ANOTHER agent.
            if not hits or all(moving_cover[cover_pts[h]] == name for h in hits):
                continue
            all_idle_agents = {k: v for k, v in self.token["agents"].items() if k != name}
            all_delayed_agents = [a for a in self.token["delayed_agents"] if a != name]
            # Prefer free parking endpoints (off-aisle); fall back to nearest non-interfering node.
            parking = self._free_parking_endpoints_sorted(pos, exclude=pos)
            committed = False
            for target in parking:
                new_path = self.plan(name, pos, target, all_idle_agents, all_delayed_agents, None, 0)
                if not new_path:
                    continue
                self.update_ends(pos, name)
                self.token["occupied_non_task_endpoints"].discard(pos)
                self.token["occupied_non_task_endpoints"].add(tuple(target))
                self.token["agents_to_tasks"][name] = {
                    "task_name": "safe_idle",
                    "start": tuple(pos),
                    "goal": tuple(target),
                    "predicted_cost": 0,
                }
                self.token["agents"][name] = [tuple(s.location.point) for s in new_path[name]]
                self.token["n_replans"] += 1
                committed = True
                break
            if committed:
                continue
            for target in self._close_non_interfering_nodes(pos, name, self.deadlock_radius):
                new_path = self.plan(name, pos, target, all_idle_agents, all_delayed_agents, None, 0)
                if not new_path:
                    continue
                self.update_ends(pos, name)
                self.token["occupied_non_task_endpoints"].discard(pos)
                self.token["agents_to_tasks"].pop(name, None)
                self.token["agents"][name] = [tuple(s.location.point) for s in new_path[name]]
                self.token["n_replans"] += 1
                break
