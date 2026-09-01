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
from path_planning.multi_agent_planner.centralized.sipp.graph_generation import cached_constraint_sweep
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


class _Waypoint(tuple):
    """A committed per-tick waypoint that remembers the roadmap vertices the agent
    passes *within* the tick (``via``, in order, strictly between the previous
    waypoint and this one) and *when* it reaches each of them (``via_t``, the
    fraction of the tick, in (0, 1), aligned with ``via``). Behaves exactly like
    a position tuple (equality, hashing, indexing), so all position bookkeeping
    is unaffected; the simulation drives through ``via`` at those times so the
    executed motion follows the roadmap - including any wait SIPP inserted
    inside the tick, which appears as the same vertex twice with its arrival and
    departure fractions - and other agents plan against exactly that motion.
    ``via_t=None`` (legacy) means "unknown": consumers fall back to constant
    speed within the tick (arc-length fractions).
    """

    def __new__(cls, pos, via=(), via_t=None):
        obj = super().__new__(cls, tuple(float(c) for c in pos))
        obj.via = tuple(tuple(float(c) for c in v) for v in via)
        obj.via_t = None if via_t is None else tuple(float(f) for f in via_t)
        if obj.via_t is not None and len(obj.via_t) != len(obj.via):
            raise ValueError("via_t must align with via")
        return obj


def _via_fracs_from(prev, wp):
    """Tick fractions of ``wp``'s vias: its own ``via_t`` when it carries one, else
    arc-length fractions of the polyline ``prev -> vias -> wp`` (constant speed)."""
    vias = list(getattr(wp, "via", ()))
    if not vias:
        return []
    fr = getattr(wp, "via_t", None)
    if fr is not None:
        return list(fr)
    legs = [tuple(prev)] + vias + [tuple(wp)]
    seg = [math.dist(legs[i], legs[i + 1]) for i in range(len(legs) - 1)]
    total = sum(seg)
    out, acc = [], 0.0
    for i in range(len(vias)):
        acc += seg[i]
        out.append(acc / total if total > 0 else (i + 1) / (len(vias) + 1))
    return out


def _wp(p):
    """Position tuple for a path entry, keeping ``_Waypoint`` (and its vias) intact."""
    return p if isinstance(p, _Waypoint) else tuple(p)


class _Loc:
    """Minimal stand-in exposing ``.point`` (matches cbs.Location's interface)."""
    __slots__ = ("point",)

    def __init__(self, point):
        self.point = _wp(point)


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
        park_retry_cooldown: int = 3,
        parking_goal_clearance: float = 0.0,
        sipp_obstacle_horizon: float | None = None,
        sipp_clearance_margin: float = 0.005,
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
        # at the iteration cap). Real seconds each simulation step represents. SIPP
        # schedules are resampled once per timestep (see _resample_by_time), so the
        # committed waypoint list has exactly one entry per simulation tick and
        # waypoint index k == time k * timestep_duration, for the agent's own path and
        # for the other agents' paths SIPP plans against.
        self.sipp_time_limit = sipp_time_limit
        self.timestep_duration = float(timestep_duration)
        self.park_retry_cooldown = int(park_retry_cooldown)
        # Idle agents avoid parking within this distance of an ACTIVE (pending/assigned)
        # delivery goal, so finished agents leave congested destination regions instead of
        # resting there and walling off the corridor for later tasks. 0 disables the rule.
        self.parking_goal_clearance = float(parking_goal_clearance)
        # Fallback resting horizon (s) for other agents in SIPP. Normally unused:
        # _other_agent_schedules assigns every other agent an explicit horizon - a moving
        # agent's final waypoint is held for exactly one timestep (it is reassigned to a
        # task or a parking spot at the next time_forward), while an idle, unassigned
        # agent (committed path of length 1) blocks its footprint until it is moved.
        self.sipp_obstacle_horizon = sipp_obstacle_horizon
        # Extra agent-agent clearance (world units) handed to SIPP on top of agent_radius.
        # The sweep clears other agents at exactly 2 * radius and the solution checker
        # flags distance <= 2 * radius, so tangential passes would sit on the threshold.
        self.sipp_clearance_margin = float(sipp_clearance_margin)

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
        # agent -> static roadmap route of a task leg that could not be planned this
        # step (SIPP mode); idle agents resting on it are nudged aside like blockers
        # of a committed path (see _reroute_interfering_idle_agents).
        self.token["blocked_routes"] = {}
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
                # Remaining waypoint route of the in-progress task (legacy entries
                # only carried start/goal); ``next_wp`` is advanced in time_forward
                # step 1 as the agent reaches each waypoint, so a mid-route replan
                # resumes from the legs still to do.
                task = self._remaining_waypoints(entry)
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

    def plan(
        self, agent_name, start, goal, all_idle_agents, all_delayed_agents, cost_map, cost: int,
        rest_forever: bool = False,
        final_leg: bool = False,
    ):
        """Low-level plan ``start -> goal`` for one agent.

        ``cost`` is the tick at which this leg starts relative to now (0 for a leg
        starting immediately; the running offset for later legs of a multi-waypoint
        task). ``rest_forever`` marks a leg whose end is a permanent rest (parking,
        deadlock relocation, idle reroute): SIPP then only accepts a goal whose safe
        interval is unbounded, so no already-committed path crosses the resting spot.
        Task legs leave it False - the agent rests one tick and is reassigned;
        ``final_leg`` marks the last leg of a task route, whose goal must then stay
        safe through that one-tick hold (see :meth:`_plan_sipp`).
        """
        start = tuple(start)
        goal = tuple(goal)
        if self.low_level == "sipp":
            # Continuous-time, interval-based, radius-aware low-level planning.
            # cost_map is grid-only; cost is the leg's start-tick offset.
            return self._plan_sipp(
                agent_name, start, goal, offset_steps=int(cost), rest_forever=rest_forever,
                final_leg=final_leg,
            )
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

    def _snap_start(self, pt):
        """Nearest ``(node, t_reach)`` of :meth:`_snap_start_candidates`."""
        return self._snap_start_candidates(pt)[0]

    def _snap_start_candidates(self, pt):
        """Roadmap nodes a SIPP plan may start from, nearest first, each with the
        time (s) needed to reach it: ``[(node, t_reach), ...]``.

        A re-planned agent usually sits mid-edge (its committed path is sampled per
        tick, so a hold or a re-validation catches it between vertices). Planning
        from the nearest node and then jumping to it would execute an unchecked,
        off-roadmap chord; instead the start is snapped to the nearer ENDPOINT of the
        edge the agent is on, so the first move stays on the roadmap, and the
        returned ``t_reach`` lets the caller shift the world by the time that move
        takes. Both endpoints of the edge are offered (nearest first) so the caller
        can fall back to the other one when the move to the nearest is not safe
        (see :meth:`_reach_segment_safe`). Points already on a node (or off every
        edge) snap as before with ``t_reach = 0``.
        """
        from path_planning.common.environment.node import Node

        pt = tuple(float(c) for c in pt)
        if Node(pt) in self.graph_map.node_index_dict:
            return [(pt, 0.0)]
        nodes = self.graph_map.nodes
        snapped = self._snap_to_node(pt)
        if not nodes or self.velocity <= 0:
            return [(snapped, 0.0)]
        p = np.asarray(pt, dtype=float)
        # The containing edge's endpoints can be far from ``pt`` (long aisle edges)
        # while many unrelated nodes are closer, so search every node within one
        # maximum edge length rather than a fixed k nearest.
        idxs = self._node_kdtree.query_ball_point(p, self._max_edge_len() + 1e-6)
        found = {}  # endpoint -> (distance to edge, endpoint distance)
        for i in idxs:
            i = int(i)
            a = np.asarray(nodes[i].current, dtype=float)
            for j in self.graph_map.road_map[i]:
                b = np.asarray(nodes[int(j)].current, dtype=float)
                ab = b - a
                denom = float(ab @ ab)
                if denom <= 0:
                    continue
                s = max(0.0, min(1.0, float((p - a) @ ab) / denom))
                d_edge = float(np.linalg.norm(p - (a + s * ab)))
                if d_edge > 1e-6:
                    continue
                for end in (a, b):
                    key = tuple(float(c) for c in end)
                    d_end = float(np.linalg.norm(p - end))
                    if key not in found or (d_edge, d_end) < found[key]:
                        found[key] = (d_edge, d_end)
        if not found:
            # Off every edge (should not happen for a committed-path position): drive
            # the chord to the nearest node, at least time-consistently.
            return [(snapped, math.dist(pt, snapped) / self.velocity)]
        ordered = sorted(found.items(), key=lambda kv: kv[1])
        return [(end, d_end / self.velocity) for end, (_, d_end) in ordered]

    def _idle_blocked_nodes(self, agent_name, start) -> set:
        """Roadmap nodes inside the resting footprint of some OTHER idle agent
        (length-1 path) - blocked forever as far as SIPP is concerned. Idle agents
        whose footprint covers ``start`` are skipped (escaping from inside one is
        allowed, see _plan_sipp)."""
        blocked = set()
        clearance = 2.0 * self.agent_radius if self.agent_radius > 0 else 1e-9
        for name, path in self.token["agents"].items():
            if name == agent_name or len(path) != 1:
                continue
            p = tuple(path[0])
            if math.dist(p, start) <= clearance:
                continue
            blocked |= self._covered_nodes(p, p)
        return blocked

    def _statically_reachable(self, agent_name, start, goal) -> bool:
        """Cheap necessary condition for a SIPP plan ``start -> goal`` to exist: a
        roadmap route avoiding every other idle agent's (forever-blocked) footprint.
        Used before expensive low-level calls that would otherwise exhaust their
        iteration budget on an unreachable goal."""
        if self.low_level != "sipp":
            return True
        goal = tuple(goal)
        blocked = self._idle_blocked_nodes(agent_name, tuple(start))
        blocked.discard(self._snap_to_node(goal))
        return bool(self._static_route(start, goal, blocked))

    def _static_route(self, start, goal, blocked=frozenset()) -> list:
        """Shortest roadmap route (node tuples, directed edges, Euclidean weights)
        from the node nearest ``start`` to the node nearest ``goal``, ignoring time
        and other agents except the nodes in ``blocked``; ``[]`` when unreachable."""
        import heapq

        nodes = self.graph_map.nodes
        if not nodes:
            return []
        pos2idx = {tuple(n.current): i for i, n in enumerate(nodes)}
        s = pos2idx.get(self._snap_to_node(start))
        g = pos2idx.get(self._snap_to_node(goal))
        if s is None or g is None:
            return []
        dist = {s: 0.0}
        prev = {}
        heap = [(0.0, s)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == g:
                break
            if d > dist.get(u, float("inf")):
                continue
            pu = nodes[u].current
            for v in self.graph_map.road_map[u]:
                v = int(v)
                if blocked and tuple(nodes[v].current) in blocked and v != g:
                    continue
                nd = d + math.dist(pu, nodes[v].current)
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if g not in dist:
            return []
        route = [g]
        while route[-1] != s:
            route.append(prev[route[-1]])
        return [tuple(nodes[i].current) for i in reversed(route)]

    def _note_blocked_route(self, agent_name, agent_pos, waypoints, legs_tried) -> None:
        """Record the static route of the task leg that failed to plan (leg index
        ``legs_tried - 1``) so idle agents resting on it are nudged aside."""
        idx = max(0, int(legs_tried) - 1)
        if idx >= len(waypoints):
            return
        leg_start = tuple(agent_pos) if idx == 0 else tuple(waypoints[idx - 1])
        route = self._static_route(leg_start, tuple(waypoints[idx]))
        if route:
            self.token["blocked_routes"][agent_name] = route

    def _max_edge_len(self) -> float:
        """Longest roadmap edge (cached per roadmap size)."""
        nodes = self.graph_map.nodes
        key = (len(nodes), len(self.graph_map.edges))
        cached = getattr(self, "_max_edge_len_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        best = 0.0
        for e in self.graph_map.edges:
            d = math.dist(nodes[int(e[0])].current, nodes[int(e[1])].current)
            if d > best:
                best = d
        self._max_edge_len_cache = (key, best)
        return best

    def _reach_segment_safe(self, actual, start, other_end, t_reach, sched, horizons) -> bool:
        """Whether driving ``actual -> start`` over ``[0, t_reach]`` (world clock) is
        clear of the other agents. ``actual`` lies on roadmap edge
        ``(other_end, start)``; SIPP validates a plan only from ``start`` on, so this
        pre-segment is the one part of a re-planned agent's motion it never checks.

        Two stages. The swept-footprint machinery SIPP blocks its graph with is the
        pre-filter: every other-agent schedule segment overlapping the window is
        swept (memoized on the roadmap, see :func:`cached_constraint_sweep`) and
        only segments whose sweep touches this edge, in a contact window that
        intersects ``[0, t_reach]``, are considered further - waits and the resting
        hold after a schedule's last point included. Because that window is for the
        WHOLE edge (an idle agent 20 m down a long aisle edge "touches" it forever,
        which would pin a held agent for good), the decision is then the exact
        closest approach between our motion along the driven sub-segment and the
        obstacle's motion inside that window. An obstacle already inside the
        clearance at t=0 (we were held inside its footprint) is ignored when the
        move takes us away from it - that is the escape.
        """
        if t_reach <= 0.0 or other_end is None:
            return True
        keys = {(tuple(other_end), tuple(start)), (tuple(start), tuple(other_end))}
        r_sweep = 2.0 * (self.agent_radius + self.sipp_clearance_margin)
        v_nom = float(self.velocity)
        a = np.asarray(actual, dtype=float)
        b = np.asarray(start, dtype=float)

        def edge_hit(edges):
            for k in keys:
                iv = edges.get(k)
                if iv is not None:
                    return iv
            return None

        def overlaps(t0, t1):
            return t0 <= t_reach + 1e-9 and t1 >= -1e-9

        v_a = (b - a) / t_reach

        def approaches(q0, t0, q1, t1, w0, w1):
            """Closest approach < clearance between us (a + v_a t) and the obstacle
            (q0 -> q1 over [t0, t1], resting if q0 == q1) inside [w0, w1]."""
            ta, tb = max(0.0, w0, t0), min(t_reach, w1, t1)
            if tb < ta:
                return False
            q0 = np.asarray(q0, dtype=float)
            q1 = np.asarray(q1, dtype=float)
            v_q = (q1 - q0) / (t1 - t0) if (t1 - t0) > 1e-9 and np.isfinite(t1) else np.zeros_like(q0)
            r0 = (a - q0) + v_q * t0
            v_rel = v_a - v_q
            vv = float(v_rel @ v_rel)
            t_star = ta if vv <= 1e-12 else min(tb, max(ta, -float(r0 @ v_rel) / vv))
            return min(float(np.linalg.norm(r0 + v_rel * t)) for t in (ta, t_star, tb)) < r_sweep

        for name, pts in sched.items():
            if not pts:
                continue
            timed = [(float(p["t"]), tuple(_as_point(p))) for p in pts]
            q0 = np.asarray(timed[0][1], dtype=float)
            d0 = float(np.linalg.norm(a - q0))
            if d0 < r_sweep and float(np.linalg.norm(b - q0)) >= d0:
                continue  # already inside its clearance and moving away: escaping
            h = horizons.get(name)
            h = float("inf") if h is None else float(h)
            t_last, p_last = timed[-1]
            segs = list(zip(timed, timed[1:])) + [((t_last, p_last), (t_last + h, p_last))]
            for (t0, p0), (t1, p1) in segs:
                if not overlaps(t0, t1):
                    continue
                if p0 == p1:
                    _, edges = cached_constraint_sweep(self.graph_map, p0, p0, v_nom, r_sweep)
                    if edge_hit(edges) is not None and approaches(p0, t0, p1, t1, t0, t1):
                        return False  # resting footprint on our sub-segment during [t0, t1]
                    continue
                dur = t1 - t0
                v_seg = v_nom
                if v_nom > 0 and dur > 1e-9:
                    implied = math.dist(p0, p1) / dur
                    if abs(implied - v_nom) > 1e-6:
                        v_seg = implied
                _, edges = cached_constraint_sweep(self.graph_map, p0, p1, v_seg, r_sweep)
                iv = edge_hit(edges)
                if iv is not None and overlaps(t0 + iv[0], t0 + iv[1]):
                    if approaches(p0, t0, p1, t1, t0 + iv[0], t0 + iv[1]):
                        return False
        return True

    def _other_agent_schedules(
        self, agent_name, offset_steps: int = 0, exclude=(), offset_time: float = 0.0
    ) -> tuple:
        """Other agents' committed paths as SIPP dynamic obstacles.

        Returns ``(schedules, horizons)``: ``schedules`` maps name -> timed schedule
        ``[{x, y, t}]`` and ``horizons`` maps name -> how long (s) the agent's final
        position stays blocked after it arrives there.

        Committed paths hold one waypoint per simulation tick (see
        :meth:`_resample_by_time`), so waypoint ``k`` is stamped at SIPP time
        ``k * timestep_duration``. ``offset_steps`` is the tick the leg being planned
        starts at: each other path is sliced from that index so SIPP's ``t = 0``
        coincides with the leg's real start.

        Resting semantics:
        - a MOVING agent (path length > 1) is assumed to stay at its final waypoint
          for exactly one timestep - it is reassigned to a task or a parking spot
          at the next ``time_forward`` - so its horizon is ``timestep_duration``;
          a mover whose path ends before the leg starts is kept as one resting
          point with the same one-tick hold;
        - an IDLE, unassigned agent (path length 1) blocks its footprint until it
          is told to move (horizon ``None`` -> forever).
        ``exclude`` drops names entirely (used to let an agent escape from inside an
        idle agent's footprint). ``offset_time`` adds a fractional shift (s) on top
        of ``offset_steps`` - the time the planning agent needs to reach the roadmap
        node its plan starts from (see :meth:`_snap_start`).
        """
        dt = self.timestep_duration
        shift = offset_steps * dt + float(offset_time)
        sched = {}
        horizons = {}
        for name, path in self.token["agents"].items():
            if name == agent_name or not path or name in exclude:
                continue
            if len(path) == 1:
                sched[name] = [_point_schedule(path[0], 0.0)]
                horizons[name] = None
                continue
            # Absolute timeline of the committed path: waypoint k at k*dt, with the
            # roadmap vertices passed inside tick k (its vias) stamped at the tick
            # fractions the waypoint carries (``via_t``; arc-length fallback).
            timed = [(tuple(path[0]), 0.0)]
            for k in range(1, len(path)):
                vias = list(getattr(path[k], "via", ()))
                if vias:
                    for q, frac in zip(vias, _via_fracs_from(path[k - 1], path[k])):
                        timed.append((q, (k - 1 + frac) * dt))
                timed.append((tuple(path[k]), k * dt))
            # Re-base onto the leg's own clock: drop what happens before the leg
            # starts, interpolating the position at exactly t = 0.
            rebased = [(q, tq - shift) for q, tq in timed]
            first_future = next((i for i, (_, tq) in enumerate(rebased) if tq > 1e-9), None)
            if first_future is None:
                rebased = [(rebased[-1][0], 0.0)]
            elif first_future == 0:
                pass
            else:
                (qa, ta), (qb, tb) = rebased[first_future - 1], rebased[first_future]
                frac = 0.0 if tb - ta <= 1e-12 else (0.0 - ta) / (tb - ta)
                q0 = tuple(a + frac * (b - a) for a, b in zip(qa, qb))
                rebased = [(q0, 0.0)] + rebased[first_future:]
            sched[name] = [_point_schedule(q, tq) for q, tq in rebased]
            horizons[name] = dt
        return sched, horizons

    def _plan_sipp(
        self, agent_name, start, goal, offset_steps: int = 0, rest_forever: bool = False,
        final_leg: bool = False,
    ):
        actual_start = tuple(float(c) for c in start)
        # Mid-edge start: SIPP checks the plan from the snapped node on, so the move
        # to that node is validated here; fall back to the edge's other endpoint,
        # and give up (the caller re-plans next tick) when neither is safe now.
        chosen = None
        world_sched = None
        candidates = self._snap_start_candidates(actual_start)
        for cand, cand_reach in candidates:
            if cand_reach > 0.0 and offset_steps == 0:
                if world_sched is None:
                    world_sched = self._other_agent_schedules(agent_name, 0, (), 0.0)
                other_end = next((c for c, _ in candidates if c != cand), None)
                if not self._reach_segment_safe(actual_start, cand, other_end, cand_reach, *world_sched):
                    continue
            chosen = (cand, cand_reach)
            break
        if chosen is None:
            return False
        start, t_reach = chosen
        goal = self._snap_to_node(goal)
        dt = float(self.timestep_duration)
        # The committed path is sampled per tick, so after reaching the goal at
        # ``t`` the agent rests there until the next tick boundary (the following
        # leg starts on a boundary); after a route's final leg the other agents
        # additionally assume a one-tick hold. SIPP only validates the arrival, so
        # ask it for a goal interval that also covers that rest - otherwise another
        # committed path may legally sweep through the spot while we sit on it.
        hold = self.num_goal_wait_steps * dt + (dt if final_leg else 0.0)

        def goal_safe_until(t_arrive):
            world = t_arrive + t_reach  # SIPP clock -> tick-aligned clock (offset is a multiple of dt)
            boundary = math.ceil(world / dt - 1e-9) * dt
            return boundary - t_reach + hold

        heur = self.heuristic_type if self.heuristic_type in {"manhattan", "euclidean"} else "euclidean"
        budget = max(int(self.a_star_max_iter), 10 * len(self.graph_map.nodes))
        radius = self.agent_radius + self.sipp_clearance_margin if self.agent_radius > 0 else 0.0

        def build(exclude=()):
            # SIPP's t=0 is the moment the agent stands on ``start``; the world is
            # shifted by the time it takes to get there from its true position.
            sched, horizons = self._other_agent_schedules(
                agent_name, offset_steps, exclude, offset_time=t_reach
            )
            return SippPlanner(
                self.graph_map,
                dynamic_obstacles=sched,
                agents=[{"name": agent_name, "start": start, "goal": goal}],
                radius=radius,
                velocity=self.velocity,
                use_constraint_sweep=True,
                heuristic_type=heur,
                time_limit=self.sipp_time_limit,
                sipp_max_iterations=budget,
                obstacle_horizon=horizons,
                # Task legs accept a bounded goal interval: the agent rests one tick
                # and is reassigned. Legs ending in a permanent rest (parking etc.)
                # need an unbounded interval so no committed path crosses the spot.
                require_goal_safe_forever=rest_forever,
                goal_safe_until=None if rest_forever else goal_safe_until,
            )

        planner = build()
        if offset_steps == 0 and not planner.sipp_graph[start].interval_list:
            # Our own start lies inside an idle agent's (forever-blocked) footprint,
            # e.g. two agents were held next to each other by the simulation. SIPP
            # cannot even seed its search then, so plan the escape ignoring those
            # idle agents; the idle-agent re-router / deadlock recovery separate them.
            clearance = 2.0 * radius
            covering = {
                name
                for name, path in self.token["agents"].items()
                if name != agent_name and len(path) == 1 and math.dist(path[0], start) <= clearance
            }
            if covering:
                print(
                    f"[NeuralATTF] {agent_name} starts inside idle {sorted(covering)}; "
                    "planning escape ignoring them"
                )
                planner = build(exclude=covering)
        solution, info = planner.compute_plan()
        self._record_sipp_metrics(info)
        schedule = solution.get(agent_name) if solution else None
        if not schedule:
            return False
        if t_reach > 0.0:
            # Put the plan back on the simulation clock: the agent first drives from
            # its true (mid-edge) position to ``start`` along the edge it is on.
            schedule = [dict(s, t=float(s["t"]) + t_reach) for s in schedule]
            schedule = [_point_schedule(actual_start, 0.0)] + schedule
        positions = self._resample_by_time(schedule, goal, dt=self.timestep_duration)
        return {agent_name: [_SegState(p) for p in positions]}

    def _record_sipp_metrics(self, info: dict) -> None:
        """Fold one SIPP low-level call's effort into the run-wide counters."""
        iters = int(info.get("low_level_iterations", 0)) if info else 0
        self.token["sipp_iterations"] += iters
        self.token["sipp_calls"] += 1
        if iters > self.token["sipp_iterations_max_seen"]:
            self.token["sipp_iterations_max_seen"] = iters

    @staticmethod
    def _resample_by_time(schedule, goal, dt: float = 1.0):
        """Convert a continuous-time SIPP schedule ``[{t,x,y}]`` into one waypoint
        per simulation tick: ``out[k]`` is the position on the timed schedule at
        ``t0 + k * dt`` (linear interpolation along the segment being traversed).

        The simulation consumes exactly one waypoint per tick, so this is what
        makes "waypoint index k" mean "time k * dt" - for the agent's own path and
        for the schedules other agents plan against. Waits become repeated points;
        the last sample lands on the first tick at or after the schedule's end and
        is forced onto ``goal``. A single-point schedule yields ``[point]``.

        Each sample is a :class:`_Waypoint` whose ``via`` lists the schedule
        entries strictly inside the preceding tick and ``via_t`` the tick
        fraction at which each is reached, so the executed motion (and what
        other agents plan against) follows the roadmap *with SIPP's timing*:
        a wait inside the tick is kept as the vertex at its arrival and departure
        times instead of being smeared into slow uniform motion, which would put
        the agent in the lane while whatever it waited for is still there.
        """
        dt = float(dt) if dt and dt > 0 else 1.0
        pts = sorted(((float(s["t"]), _as_point(s)) for s in schedule), key=lambda x: x[0])
        if not pts:
            return [_Waypoint(goal)]

        t0, t_end = pts[0][0], pts[-1][0]
        n_steps = max(0, int(math.ceil((t_end - t0) / dt - 1e-9)))
        out = []
        j = 0
        prev_tk = t0
        for k in range(n_steps + 1):
            tk = t0 + k * dt
            # Schedule entries strictly inside (prev_tk, tk) are vias of this sample,
            # stamped by their fraction of the tick.
            via = [(q, (tq - prev_tk) / dt) for (tq, q) in pts if prev_tk + 1e-9 < tq < tk - 1e-9]
            while j + 1 < len(pts) and pts[j + 1][0] <= tk + 1e-9:
                j += 1
            if j + 1 >= len(pts):
                p = pts[-1][1]
            else:
                (ta, pa), (tb, pb) = pts[j], pts[j + 1]
                if tb - ta <= 1e-12 or pa == pb:
                    p = pa
                else:
                    frac = (tk - ta) / (tb - ta)
                    p = tuple(a + frac * (b - a) for a, b in zip(pa, pb))
            # Drop exact duplicates (same point at the same instant); a repeated
            # point at a later fraction is a wait and is kept.
            cleaned = []
            for q, f in via:
                if cleaned and cleaned[-1][0] == q and abs(cleaned[-1][1] - f) <= 1e-9:
                    continue
                cleaned.append((q, f))
            out.append(_Waypoint(p, [q for q, _ in cleaned], [f for _, f in cleaned]))
            prev_tk = tk

        if tuple(out[-1]) != tuple(goal):
            out.append(_Waypoint(goal))
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
        path = None
        if self._statically_reachable(agent_name, agent_pos, target):
            path = self.plan(
                agent_name, agent_pos, target, all_idle_agents, all_delayed_agents, None, 0,
                rest_forever=True,
            )
        if not path:
            if self.low_level == "sipp":
                if self._blocks_pending_task(tuple(agent_pos)):
                    # A squatter stuck on a cell some task still needs: publish its
                    # escape corridor like a failed task leg's, so idle agents resting
                    # on it are nudged aside (step 5) and others stop resting on it
                    # (footprints).
                    esc = self._static_route(tuple(agent_pos), tuple(target))
                    if esc:
                        self.token["blocked_routes"][agent_name] = esc
                else:
                    # Not blocking any task cell: rest where it is (back-off retries
                    # parking later). Recovery-hopping unparkable agents around a
                    # congested drain region reseals random corridors every few ticks
                    # and keeps the region from ever settling; if this agent does
                    # block a mover or a corridor, the step-5 re-router moves it.
                    return False
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
        self.token["agents"][agent_name] = [_wp(state.location.point) for state in path[agent_name]]
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
        for owner, route in self.token.get("blocked_routes", {}).items():
            if owner == agent_name:
                continue
            for i in range(len(route)):
                footprint |= self._covered_nodes(route[i - 1] if i else route[i], route[i])
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

    def _hold_off_squatted_goal(self, name, pos, all_idle_agents, all_delayed_agents) -> bool:
        """Handle a task-leg failure whose target is squatted by an idle agent.

        The hot-cell standoff: a just-finished agent rests on a delivery cell it
        cannot leave, while the carriers queuing for that cell rest on the cell's
        only exit corridors — the squatter is sealed by the very agents waiting on
        it, and deadlock recovery just ejects the waiters far away. Instead:

        - if the squatter can still statically drain to a free parking endpoint,
          the waiter simply stays where it is (no recovery ejection);
        - if it is sealed, but clearing the waiters' resting footprints opens a
          drain, this waiter relocates to a nearby non-interfering rest spot that
          is also OFF its own failed leg's corridor (its assignment is kept — a
          repair pair re-plans the leg once the cell frees);
        - if it is sealed by agents that are not waiters, there is nothing this
          agent can do; it waits in place (the squatter's own escape route is
          published via blocked_routes, so the step-5 re-router works on them).

        The same holding logic applies when the corridor is sealed MID-route by
        resting agents while the goal itself is free: recovery ejection would move
        this agent (and its recorded route, and with it the nudge pressure) every
        tick, so it holds position and lets the re-router work the resters off
        the published route instead.

        Returns True when the failure is rester-caused (handled here: relocated
        or deliberately waiting), False when it is not — the caller then falls
        back to plain deadlock recovery.
        """
        route = self.token["blocked_routes"].get(name)
        if not route:
            return False
        wp = tuple(route[-1])
        clearance = 2.0 * self.agent_radius if self.agent_radius > 0 else 1e-9
        # Idle agents resting on the failed leg's corridor. If there are none, the
        # failure is not rester-caused (moving congestion, SIPP timing): fall back
        # to plain deadlock recovery.
        blockers = [
            a for a, p in self.token["agents"].items()
            if a != name and len(p) == 1
            and any(math.dist(tuple(p[0]), tuple(q)) <= clearance for q in route)
        ]
        if not blockers:
            return False
        squatter = next(
            (a for a in blockers if math.dist(tuple(self.token["agents"][a][0]), wp) <= clearance),
            None,
        )
        if squatter is None:
            # Corridor sealed mid-route while the goal itself is free. Holding
            # position is QUEUE behaviour and only earns its keep at the jam
            # itself: recovery ejection there would move this agent (and its
            # recorded route, and with it the step-5 nudge pressure) every tick.
            # An agent far from every blocking rester is not queuing — freezing
            # it across the plant starves its task, so it keeps the old recovery
            # behaviour instead.
            near_jam = min(
                math.dist(tuple(pos), tuple(self.token["agents"][a][0])) for a in blockers
            ) <= self.deadlock_radius
            return near_jam
        exits = self._free_parking_endpoints_sorted(wp, exclude=wp)[:3]
        if not exits:
            return True  # nowhere for the squatter to drain to anyway; just wait
        blocked_all = self._idle_blocked_nodes(squatter, wp)
        if any(self._static_route(wp, e, frozenset(blocked_all)) for e in exits):
            return True  # squatter can leave on its own; wait instead of recovering away
        # Sealed. Would clearing the waiters (this tick's blocked-route owners,
        # this agent included) open a drain?
        yielders = {name, squatter} | set(self.token["blocked_routes"])
        open_blocked: set = set()
        for a, p in self.token["agents"].items():
            if a in yielders or len(p) != 1:
                continue
            q = tuple(p[0])
            if math.dist(q, wp) <= clearance:
                continue
            open_blocked |= self._covered_nodes(q, q)
        if not any(self._static_route(wp, e, frozenset(open_blocked)) for e in exits):
            return True  # sealed by non-waiters; nothing this agent can fix by moving
        # This waiter's rest is (part of) the seal: step aside, off its own corridor.
        # The corridor stand-off gets an extra half-clearance pad: parallel twin
        # lanes sit barely one clearance apart, and a rest that clears the corridor
        # by centimetres leaves the squatter no margin to actually drive it.
        for target in self._close_non_interfering_nodes(pos, name, self.deadlock_radius):
            if any(math.dist(target, q) <= 1.5 * clearance for q in route):
                continue
            if not self._statically_reachable(name, pos, target):
                continue
            path = self.plan(
                name, pos, target, all_idle_agents, all_delayed_agents, None, 0, rest_forever=True
            )
            if not path:
                continue
            self.update_ends(pos, name)
            self.token["agents"][name] = [_wp(s.location.point) for s in path[name]]
            self.token["n_replans"] += 1
            print(
                f"[NeuralATTF] {name} yields the corridor to {tuple(round(c, 1) for c in wp)} "
                f"(squatter {squatter}); waits at {tuple(round(c, 1) for c in target)}"
            )
            return True
        return True  # could not step aside; wait rather than recovery-eject

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
            if not self._statically_reachable(agent_name, agent_pos, target):
                continue
            path = self.plan(
                agent_name, agent_pos, target, all_idle_agents, all_delayed_agents, None, 0,
                rest_forever=True,
            )
            if not path:
                continue
            self.update_ends(agent_pos, agent_name)
            self.token["agents"][agent_name] = [_wp(state.location.point) for state in path[agent_name]]
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

    @staticmethod
    def _remaining_waypoints(entry: dict) -> list:
        """Waypoints of an in-progress task still to be visited (never empty)."""
        wps = [tuple(p) for p in entry.get("waypoints", [entry["start"], entry["goal"]])]
        rest = wps[int(entry.get("next_wp", 0)):]
        return rest if rest else [tuple(entry["goal"])]

    def _plan_task_route(
        self, agent_name, agent_pos, waypoints, all_idle_agents, all_delayed_agents, cost_maps=None
    ):
        """Plan ``agent_pos -> waypoints[0] -> waypoints[1] -> ...`` as a chain of
        low-level segments and join them into one committed position list.

        ``cum`` is the tick each leg starts at (one committed waypoint == one tick,
        for both low levels), so every leg is checked against the other agents'
        paths at the right time; each non-final segment drops its last point when
        joined (it equals the next segment's first point). Returns
        ``(joined_or_None, legs_attempted)``.
        """
        segments = []
        seg_start = agent_pos
        cum = 0
        for i, wp in enumerate(waypoints):
            cost_map = cost_maps[i] if cost_maps is not None and i < len(cost_maps) else None
            if not self._statically_reachable(agent_name, seg_start, wp):
                return None, i + 1  # sealed off by idle footprints: SIPP would only burn its budget
            seg = self.plan(
                agent_name, seg_start, wp, all_idle_agents, all_delayed_agents, cost_map, cum,
                final_leg=(i == len(waypoints) - 1),
            )
            if not seg:
                return None, i + 1
            for _ in range(self.num_goal_wait_steps):
                # A plain copy: re-appending the last sample would replay its vias.
                seg[agent_name].append(_SegState(tuple(seg[agent_name][-1].location.point)))
            segments.append(seg[agent_name])
            cum += len(seg[agent_name]) - 1
            seg_start = wp
        joined = []
        prev_last = None
        for i, seg_pts in enumerate(segments):
            pts = [_wp(s.location.point) for s in seg_pts]
            if prev_last is not None:
                # The previous leg's final sample is the same position as this leg's
                # first, but it is the one that carries the vias / via_t of the tick
                # that reaches it; keep it so the join does not turn that tick into a
                # straight chord executed (and seen by others) at the wrong times.
                pts[0] = prev_last
            joined += pts[:-1] if i < len(segments) - 1 else pts
            prev_last = pts[-1]
        return joined, len(waypoints)

    def _replan_mover(self, name: str) -> None:
        """Re-plan a moving agent from its current position because a blocker it was
        planned around (assumed to leave after its one-tick hold) is still resting
        there. Its committed path is dropped first; a task agent re-plans the legs it
        has left, anything else is sent to parking. If no plan exists the agent simply
        stays put (a length-1 path), i.e. it stops short of the blocker and becomes a
        resting obstacle itself instead of driving into one."""
        pos = tuple(self.token["agents"][name][0])
        self.token["agents"][name] = [pos]
        self.token["n_replans"] += 1
        all_idle_agents = {k: v for k, v in self.token["agents"].items() if k != name}
        all_delayed_agents = [a for a in self.token["delayed_agents"] if a != name]
        entry = self.token["agents_to_tasks"].get(name)
        if entry and entry.get("task_name") != "safe_idle":
            joined, _ = self._plan_task_route(
                name, pos, self._remaining_waypoints(entry), all_idle_agents, all_delayed_agents
            )
            if joined is not None:
                self.token["agents"][name] = joined
                entry["predicted_cost"] = len(joined)
            return
        if entry:
            self.token["occupied_non_task_endpoints"].discard(tuple(entry["goal"]))
            self.token["agents_to_tasks"].pop(name, None)
        self.go_to_closest_non_task_endpoint(name, pos, all_idle_agents, all_delayed_agents)

    def time_forward(self, t: int, position: dict, delayed_agents: List[str], agents_size=None):
        if agents_size is not None:
            self.token["agents_size"] = agents_size

        # 1) check task completions (and waypoint progress of multi-leg tasks)
        for agent_name in self.token["agents"]:
            pos = _as_point(position[agent_name])
            entry = self.token["agents_to_tasks"].get(agent_name)
            if entry and entry.get("task_name") != "safe_idle" and "waypoints" in entry:
                # Committed paths pass exactly through each leg's waypoint at a tick
                # boundary, so "standing on waypoint k" means legs 0..k are done. A
                # later replan (delay, blocker re-validation) then resumes from the
                # remaining legs instead of redoing the whole route.
                wps = entry["waypoints"]
                k = entry.get("next_wp", 0)
                while k < len(wps) and pos == tuple(wps[k]):
                    k += 1
                entry["next_wp"] = k
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
        self.token["blocked_routes"] = {}
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
                # Plan the multi-leg route agent_pos -> wp0 -> wp1 -> ... (see
                # _plan_task_route). Cost maps are consumed per segment when an
                # encoder is configured (None otherwise).
                waypoints = [tuple(p) for p in closest_task]
                cost_maps = None
                if cost_lookups is not None:
                    cost_maps = cost_lookups[cost_map_idx:cost_map_idx + len(waypoints)]
                joined, legs_tried = self._plan_task_route(
                    agent_name, agent_pos, waypoints, all_idle_agents, all_delayed_agents, cost_maps
                )
                cost_map_idx += legs_tried
                if joined is None:
                    if self.low_level == "sipp":
                        self._note_blocked_route(agent_name, agent_pos, waypoints, legs_tried)
                        if self._hold_off_squatted_goal(
                            agent_name, agent_pos, all_idle_agents, all_delayed_agents
                        ):
                            continue
                    if len(self.token["delayed_agents"]) == 0:
                        self.deadlock_recovery(
                            agent_name, agent_pos, all_idle_agents, all_delayed_agents, self.deadlock_radius
                        )
                    continue
                last_pos = joined[-1]
                self.assigned_tasks.add(closest_task_name)
                # Assignment-time bookkeeping (first-assignment time is sticky; the
                # owning agent reflects the latest assignment).
                if closest_task_name not in self.token["assigned_tasks_times"]:
                    self.token["assigned_tasks_times"][closest_task_name] = t
                self.token["assigned_tasks_agent"][closest_task_name] = agent_name
                # Always retire the task from the pending pool. An agent that is
                # parked (or driving to parking) carries a "safe_idle" entry in
                # agents_to_tasks, so gating the pop on membership leaked every task
                # assigned to such an agent: it stayed "pending" forever, and by the
                # drain phase dozens of long-finished deliveries still counted as
                # active goals — parking clearance, safe-idle checks and interference
                # footprints then walled off the whole delivery strip. For a repair
                # pair (agent resuming its own in-progress task) both pops are no-ops.
                self.token["tasks"].pop(closest_task_name, None)
                task = available_tasks.pop(closest_task_name, closest_task)
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
                    "next_wp": 0,
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
        # Nodes covered by every agent's *moving* path (len > 1) -> set of owners,
        # plus the static route of every task leg that could not be planned this
        # step: an idle agent parked on the only corridor into a goal makes that
        # leg unplannable in the first place, so it must be nudged off it too, or
        # the leg (and the task) never gets a path at all.
        moving_cover: dict = {}
        # Owners with a COMMITTED path over each node, kept separate from mere
        # blocked-route owners: an immovable blocker invalidates only committed
        # motion. Re-planning a blocked-route owner would cancel whatever it just
        # committed instead (e.g. its yield sidestep off a squatted goal's
        # corridor) and freeze it in place tick after tick.
        path_owner: dict = {}
        for name, path in self.token["agents"].items():
            if len(path) <= 1:
                continue
            for i in range(len(path)):
                prev = path[i - 1] if i >= 1 else path[i]
                for c in self._covered_nodes(prev, path[i]):
                    moving_cover.setdefault(tuple(c), set()).add(name)
                    path_owner.setdefault(tuple(c), set()).add(name)
        for name, route in self.token.get("blocked_routes", {}).items():
            for i in range(len(route)):
                prev = route[i - 1] if i >= 1 else route[i]
                for c in self._covered_nodes(prev, route[i]):
                    moving_cover.setdefault(tuple(c), set()).add(name)
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
            movers = set()
            for h in hits:
                movers |= moving_cover[cover_pts[h]]
            movers.discard(name)
            if not movers:
                continue
            all_idle_agents = {k: v for k, v in self.token["agents"].items() if k != name}
            all_delayed_agents = [a for a in self.token["delayed_agents"] if a != name]
            # An idle agent that still owns a task (its route could not be planned this
            # step) keeps that assignment while it is nudged aside; it is re-planned from
            # its remaining legs once it is idle again. Only a parking assignment is
            # replaced by the new parking target.
            entry = self.token["agents_to_tasks"].get(name)
            has_task = bool(entry) and entry.get("task_name") != "safe_idle"
            # Prefer free parking endpoints (off-aisle) that are not themselves on a
            # covered node (else the blocker is just nudged again next step); fall
            # back to nearest non-interfering node.
            parking = [
                ep for ep in self._free_parking_endpoints_sorted(pos, exclude=pos)
                if not cover_tree.query_ball_point(np.asarray(ep, dtype=float), clearance)
            ]
            committed = False
            for target in parking:
                if has_task and self.admissible_heuristic(target, pos) > self.deadlock_radius:
                    # A nudged task owner stays local: shipping the last carrier to a
                    # far endpoint costs the whole drain another round trip. Nearby
                    # endpoints only; otherwise the close-node fallback below keeps
                    # it beside its goal.
                    continue
                if not self._statically_reachable(name, pos, target):
                    continue
                new_path = self.plan(
                    name, pos, target, all_idle_agents, all_delayed_agents, None, 0, rest_forever=True
                )
                if not new_path:
                    continue
                self.update_ends(pos, name)
                self.token["occupied_non_task_endpoints"].discard(pos)
                self.token["occupied_non_task_endpoints"].add(tuple(target))
                if not has_task:
                    self.token["agents_to_tasks"][name] = {
                        "task_name": "safe_idle",
                        "start": tuple(pos),
                        "goal": tuple(target),
                        "predicted_cost": 0,
                    }
                self.token["agents"][name] = [_wp(s.location.point) for s in new_path[name]]
                self.token["n_replans"] += 1
                committed = True
                break
            if not committed:
                for target in self._close_non_interfering_nodes(pos, name, self.deadlock_radius):
                    if not self._statically_reachable(name, pos, target):
                        continue
                    new_path = self.plan(
                        name, pos, target, all_idle_agents, all_delayed_agents, None, 0, rest_forever=True
                    )
                    if not new_path:
                        continue
                    self.update_ends(pos, name)
                    self.token["occupied_non_task_endpoints"].discard(pos)
                    if not has_task:
                        self.token["agents_to_tasks"].pop(name, None)
                    self.token["agents"][name] = [_wp(s.location.point) for s in new_path[name]]
                    self.token["n_replans"] += 1
                    committed = True
                    break
            if committed or self.low_level != "sipp":
                continue
            # The blocker cannot move. SIPP movers were planned on the assumption that
            # it would leave after its one-tick hold; that assumption is now false, so
            # their committed paths are invalid: re-plan each of them against the
            # blocker's (now permanent) footprint before they drive into it. (The grid
            # low level already treats idle agents as static obstacles when planning.)
            # Only owners whose COMMITTED path covers the blocker qualify — a
            # blocked-route owner's committed motion (its yield sidestep) is
            # elsewhere and must not be cancelled.
            path_movers = set()
            for h in hits:
                path_movers |= path_owner.get(cover_pts[h], set())
            path_movers.discard(name)
            for mover in sorted(path_movers):
                if len(self.token["agents"].get(mover, [])) > 1:
                    self._replan_mover(mover)
