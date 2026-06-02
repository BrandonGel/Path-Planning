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
import random
import time
from collections import defaultdict, deque
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.multi_agent_planner.decentralized.neural_attf.cbs import Environment
from path_planning.multi_agent_planner.decentralized.neural_attf.grid_overlay import GridOverlay

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
        device: str = "cpu",
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
        self.device = device

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

    def get_moving_obstacles_agents(self, agents, time_start: int) -> dict:
        obstacles: dict = {}
        for name, path in agents.items():
            if len(path) > time_start and len(path) > 1:
                for i in range(time_start, len(path)):
                    k = i - time_start
                    pt = path[i]
                    obstacles[(pt[0], pt[1], k)] = name
                    if name in self.token["agents_size"]:
                        for dx, dy in self.token["agents_size"][name]:
                            obstacles[(pt[0] + dx, pt[1] + dy, k)] = name
                    if i == len(path) - 1:
                        obstacles[(pt[0], pt[1], k + 1)] = name
                        if name in self.token["agents_size"]:
                            for dx, dy in self.token["agents_size"][name]:
                                obstacles[(pt[0] + dx, pt[1] + dy, k + 1)] = name
        return obstacles

    def get_idle_obstacles_agents(self, agents_paths, delayed_agents, _time_start: int = 0) -> set:
        obstacles: set = set()
        for name, path in agents_paths.items():
            last = path[-1]
            if len(path) == 1 or last in self.non_task_endpoints:
                obstacles.add((last[0], last[1]))
                if name in self.token["agents_size"]:
                    for dx, dy in self.token["agents_size"][name]:
                        obstacles.add((last[0] + dx, last[1] + dy))
        for agent_name in delayed_agents:
            obstacles.add(tuple(self.token["agents"][agent_name][0]))
        return obstacles

    # ------------------------------------------------------------- idle plan

    def check_safe_idle(self, agent_pos, agent_name) -> bool:
        for _, task in self.token["tasks"].items():
            if any(tuple(wp) == tuple(agent_pos) for wp in task):
                return False
        if len(self.token["agents"][agent_name]) != 1:
            return False
        if self.token["agents"][agent_name][0] not in self.non_task_endpoints:
            return False
        for start_goal in self.get_agents_to_tasks_starts_goals():
            if tuple(start_goal) == tuple(agent_pos):
                return False
        return True

    def get_closest_non_task_endpoint(self, agent_pos):
        occupied = self.token["occupied_non_task_endpoints"].copy()
        for _, path in self.token["agents"].items():
            if path[-1] in occupied:
                occupied.remove(path[-1])
        # Free any endpoints that no agent is currently sitting on.
        for endpoint in occupied:
            self.token["occupied_non_task_endpoints"].discard(endpoint)

        best = None
        best_d = float("inf")
        for endpoint in self.non_task_endpoints:
            if endpoint in self.token["occupied_non_task_endpoints"]:
                continue
            d = self.admissible_heuristic(endpoint, agent_pos)
            if d < best_d:
                best_d = d
                best = endpoint
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

    def get_token(self):
        return self.token

    # ------------------------------------------------------------- A* driver

    def plan(self, agent_name, start, goal, all_idle_agents, all_delayed_agents, cost_map, cost: int):
        start = tuple(start)
        goal = tuple(goal)
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

    # ---------------------------------------------------- safe-idle dispatch

    def go_to_closest_non_task_endpoint(
        self, agent_name, agent_pos, all_idle_agents, all_delayed_agents, _cost_map=None
    ):
        if tuple(self.token["agents"][agent_name][-1]) in self.non_task_endpoints:
            return
        target = self.get_closest_non_task_endpoint(agent_pos)
        path = self.plan(agent_name, agent_pos, target, all_idle_agents, all_delayed_agents, None, 0)
        if not path:
            print(f"Solution to non-task endpoint not found for {agent_name}; trying deadlock recovery.")
            self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, all_delayed_agents, self.deadlock_radius)
            return
        self.update_ends(agent_pos, agent_name)
        self.token["occupied_non_task_endpoints"].add(tuple(target))
        self.token["agents_to_tasks"][agent_name] = {
            "task_name": "safe_idle",
            "start": tuple(agent_pos),
            "goal": tuple(target),
            "predicted_cost": 0,
        }
        self.token["agents"][agent_name] = [tuple(state.location.point) for state in path[agent_name]]

    # ------------------------------------------------------ deadlock recovery

    def _random_close_node_point(self, agent_pos, r: float):
        """Pick a random reachable graph node within Euclidean radius ``r`` of ``agent_pos``.

        Excludes nodes that would land on a path end, an occupied parking spot,
        a currently assigned task goal, or an obstacle.
        """
        nodes = self.graph_map.nodes
        if not nodes:
            return None
        pt = np.asarray(agent_pos, dtype=float)
        forbidden = (
            self.token["path_ends"]
            | self.token["occupied_non_task_endpoints"]
            | self.get_agents_to_tasks_goals()
        )
        candidates = []
        for node in nodes:
            cand = tuple(float(c) for c in node.current)
            if cand in forbidden:
                continue
            if self.graph_map.in_collision_point(cand):
                continue
            d = float(np.linalg.norm(np.asarray(cand) - pt))
            if 1e-9 < d <= r:
                candidates.append(cand)
        if not candidates:
            return None
        return random.choice(candidates)

    def deadlock_recovery(self, agent_name, agent_pos, all_idle_agents, all_delayed_agents, r: float):
        self.token["deadlock_count_per_agent"][agent_name] += 1
        if self.token["deadlock_count_per_agent"][agent_name] < 2:
            return
        self.token["deadlock_count_per_agent"][agent_name] = 0
        target = self._random_close_node_point(agent_pos, r)
        if target is None:
            print(f"Deadlock recovery: no free nearby node for {agent_name}.")
            return
        path = self.plan(agent_name, agent_pos, target, all_idle_agents, all_delayed_agents, None, 0)
        if not path:
            print(f"Deadlock recovery: no path for {agent_name} -> {target}.")
            return
        self.update_ends(agent_pos, agent_name)
        self.token["agents"][agent_name] = [tuple(state.location.point) for state in path[agent_name]]

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
                self.go_to_closest_non_task_endpoint(
                    agent_name, agent_pos, all_idle_agents, all_delayed_agents
                )
