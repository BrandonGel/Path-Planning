"""
Prioritised safe-interval path planning (SIPP) for agents with bodies.

Based on the SIPP implementation by Ashwin Bose (@atb033); see DOI: 10.1109/ICRA.2011.5980306.
"""
import heapq

import numpy as np
import math
import random
import time
from collections import deque

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.multi_agent_planner.centralized.cbs.data_types import _ori_between, put_heading, start_heading
from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippGraph, SippNode, State
from path_planning.multi_agent_planner.data_type import HEURISTIC_TYPE


class SippPlanner:
    """Plans the agents one after another (in up to ``len(agents)`` random orders, keeping the cheapest
    plan), each with SIPP against the dynamic obstacles and the agents planned before it.

    ``agents`` is a list of dicts: ``name``, ``start``, ``goal`` and optionally ``agent`` (an ``Agent``: the
    body) and ``yaw`` / ``pitch`` (the start heading, degrees; default 0). Everything about a body comes from its
    ``Agent``, as in CBS / ICBS:

    - Size: its shape (2D or 3D) and ``clearance``. When no dict has one, the agents are point particles:
      every move takes one time unit and a vertex is taken for one step. Bodies and point particles cannot
      be mixed; different bodies can.
    - Timing: a move takes ``length / max_speed``; ``max_speed=None`` means 1 time unit per move.
    - Turning is instant (``max_rotation`` is not used): a body faces along each move and keeps that
      heading while it waits. Agents with a body get ``yaw`` (and ``pitch`` in 3D) in the plan.

    ``dynamic_obstacles``: timed schedules of other bodies, see ``SippGraph`` (a plain schedule has the
    body of the agent being planned). ``exact_vertices`` (default False): an agent that is not a ball is
    checked at a vertex as its bounding ball (faster, more cautious); True checks it at the heading it
    arrived with (slower, lets slim bodies pass closer). ``obstacle_horizon``, ``require_goal_safe_forever`` and
    ``goal_safe_until``: how long obstacles rest at their last point and how long a goal has to stay safe.
    ``max_iterations`` is not used: every call tries ``len(agents)`` orders.
    """

    def __init__(self, graph_map: GraphSampler, agents: list, dynamic_obstacles: dict | None = None,
                 heuristic_type: str = 'manhattan', time_limit: float | None = None, max_iterations: int | None = None,
                 verbose: bool = False, sipp_max_iterations: int = 10000, obstacle_horizon: float | dict | None = None,
                 require_goal_safe_forever: bool = True, goal_safe_until=None,
                 exact_vertices: bool = False, use_numba: bool = True):
        self.graph_map = graph_map
        self.agents = agents
        self.dynamic_obstacles = dynamic_obstacles or {}
        self.obstacle_horizon = obstacle_horizon
        self.bodies = {a["name"]: a.get("agent") for a in agents}
        self.has_bodies = any(body is not None for body in self.bodies.values())
        if self.has_bodies and None in self.bodies.values():
            raise ValueError('SIPP: every agent dict needs an "agent" (an Agent) once any of them has one')
        self.start_ori = {a["name"]: start_heading(a) for a in agents}   # the given heading, else 0
        # A finished agent occupies its goal for all future time, so a goal arrival is only truly
        # collision-free if the accepted safe interval extends to infinity - otherwise a later-planned
        # agent may legally transit through that spot. Callers that replan around finished agents
        # themselves (e.g. NeuralATTF) can disable this to accept a goal in any safe interval.
        self.require_goal_safe_forever = require_goal_safe_forever
        # Optional refinement of the bounded-goal case: ``goal_safe_until(t_arrive)`` returns the time the
        # goal must stay safe until for an arrival at ``t_arrive``. A goal reached inside an interval that
        # ends earlier is not accepted; the search keeps waiting/detouring for a later interval.
        self.goal_safe_until = goal_safe_until
        # Agents that are not balls: exact safe intervals per arrival heading at vertices (False: the bounding
        # ball, faster and more cautious).
        self.exact_vertices = exact_vertices
        # Agents with bodies: run the low-level search compiled (fast_search). Same plans; False keeps the
        # Python search (also used with ``goal_safe_until``, which it calls back into).
        self.use_numba = use_numba
        self._h_arrays = {}                  # (goal, speed) -> per-vertex estimate
        self.time_limit = time_limit if time_limit is not None and time_limit > 0 else float('inf')
        self.verbose = verbose
        self.total_time = 0
        self.total_iterations = 0
        self._goal_dist_maps = {}            # (goal index, speed) -> {node index: time to the goal}
        self._max_edge = None                # longest roadmap edge (estimates for unit-time moves)
        self._neighbours = {}                # position -> neighbour positions
        self.agent_names = [agent["name"] for agent in agents]
        self.plan, self.plan_cost, self.action_cost = {}, {}, {}
        self._mtime_cache = {}
        self.sipp_max_iterations = sipp_max_iterations if sipp_max_iterations is not None and sipp_max_iterations > 0 else float("inf")
        self.heuristic_type = HEURISTIC_TYPE.get(heuristic_type, HEURISTIC_TYPE["manhattan"]) if heuristic_type else HEURISTIC_TYPE["manhattan"]
        self._new_round()

    # ------------------------------------------------------------------ tables, one per body
    def _new_round(self):
        """Forget the planned agents: only the dynamic obstacles remain."""
        self._tables = {}                    # body key (None: points) -> SippGraph
        self._committed = []                 # (name, body, plan, action costs), in planning order

    def _graph(self, name) -> SippGraph:
        """The safe-interval table for agent ``name``'s body, with every agent planned so far in it."""
        body = self.bodies.get(name)
        key = None if body is None else body.key
        graph = self._tables.get(key)
        if graph is None:
            graph = self._tables[key] = SippGraph(self.graph_map, self.dynamic_obstacles, body, self.obstacle_horizon,
                                                  self.exact_vertices)
            for other, other_body, plan, costs in self._committed:
                graph.update_intervals([plan], [costs], [other], [other_body], [self.start_ori[other]])
        return graph

    def _commit(self, name, plan, costs):
        """Agent ``name`` now follows ``plan``: block it in every table."""
        body = self.bodies.get(name)
        self._committed.append((name, body, plan, costs))
        for graph in self._tables.values():
            graph.update_intervals([plan], [costs], [name], [body], [self.start_ori[name]])

    @property
    def sipp_graph(self) -> dict:
        """The table of the first agent's body (``{position: SippNode, (p, q): SippEdge}``)."""
        return self._graph(self.agent_names[0]).sipp_graph

    @staticmethod
    def _interval_containing(node, t: float):
        """The safe interval of a vertex's ``SippNode`` that contains time ``t`` (None if blocked)."""
        for interval in node.interval_list:
            if interval[0] <= t <= interval[1]:
                return interval
        return None

    def shuffle_agents(self):
        random.shuffle(self.agents)

    def get_mtime(self, position1, position2, name):
        """How long agent ``name`` takes over the move (one time unit without a ``max_speed``)."""
        body = self.bodies.get(name)
        speed = None if body is None else body.max_speed
        key = (position1, position2, speed)
        m_time = self._mtime_cache.get(key)
        if m_time is None:
            m_time = 1.0 if speed is None else float(self.graph_map.get_distance(tuple(position1), tuple(position2))) / speed
            self._mtime_cache[key] = m_time
        return m_time

    def get_earliest_no_collision_arrival_time_point(self, start_t, interval, start_pos, neighbour, graph):
        """A point particle's arrival at ``neighbour``: the earliest in ``interval``, or None when an obstacle
        steps the other way along the same edge over that step (a swap)."""
        arrive_t = max(start_t, interval[0])
        if arrive_t == interval[0]:
            if (neighbour, start_pos, arrive_t - 1, arrive_t) in graph.swaps:
                return None
        return arrive_t

    @staticmethod
    def get_earliest_no_collision_arrival_time_body(start_t, vertex_interval, edge_interval, m_time):
        """A body's arrival in ``vertex_interval`` leaving inside the departure window ``edge_interval``:
        the earliest, or None."""
        arrive_t = max(start_t, vertex_interval[0], edge_interval[0] + m_time)
        depart_t = arrive_t - m_time
        if not (depart_t >= edge_interval[0] - 1e-9 and depart_t <= edge_interval[1]):
            return None
        if not(arrive_t >= vertex_interval[0] and arrive_t <= vertex_interval[1]):
            return None
        return arrive_t

    def get_valid_neighbours(self, position):
        neighbours = self._neighbours.get(position)
        if neighbours is None:
            index = self.graph_map.pos_index_dict.get(tuple(position))
            neighbours = [] if index is None else [n.current for n in self.graph_map.get_neighbors(int(index))]
            self._neighbours[position] = neighbours
        return neighbours

    def _moves_out(self, graph, name, position) -> list:
        """The moves out of ``position`` for agent ``name``: ``[(next position, duration, heading on arrival,
        the edge's unsafe-interval record)]``, built once per table and speed."""
        body = self.bodies.get(name)
        cache = graph.__dict__.setdefault("_moves_out", {}).setdefault(None if body is None else body.max_speed, {})
        moves = cache.get(position)
        if moves is None:
            table = graph.sipp_graph
            moves = cache[position] = [(q, self.get_mtime(position, q, name), _ori_between(position, q) if graph.oriented else None,
                                        table.get((position, q))) for q in self.get_valid_neighbours(position)]
        return moves

    def get_successors(self, state, graph, name):
        """The states reachable from ``state`` for agent ``name``, with their costs and ``(wait, move)``
        times, under the safe intervals of ``graph`` (its ``SippGraph``).
        An agent that is not a ball arrives facing along the edge, so the next vertex's intervals are the
        ones for that heading."""
        table, oriented = graph.sipp_graph, graph.oriented
        successors = []
        costs = []
        time_taken = []
        start_pos, now, leave_by = state.position, state.time, state.interval[1]
        moves = self._moves_out(graph, name, start_pos)
        if oriented:                               # every neighbour's intervals for its arrival heading, in one go
            graph.prepare([(q, ori) for q, _, ori, _ in moves])
        for neighbour_pos, m_time, arrive_ori, edge in moves:
            start_t = now + m_time                 # earliest possible arrival
            end_t = leave_by + m_time              # latest possible arrival
            arrival = graph.vertex_node(neighbour_pos, arrive_ori) if oriented else table[neighbour_pos]
            intervals = [i for i in arrival.interval_list if i[0] <= end_t and i[1] >= start_t]
            if not intervals:
                continue
            if not self.has_bodies:
                for i in intervals:
                    t = self.get_earliest_no_collision_arrival_time_point(start_t, i, start_pos, neighbour_pos, graph)
                    if t is None:                  # a swap along the edge
                        continue
                    successors.append(State(neighbour_pos, t, i))
                    costs.append(t - now)
                    time_taken.append((t - now - m_time, m_time))
                continue
            # When the agent may start along the edge: from now until it has to leave, less the times
            # it would be on the edge while something covers it (on the edge during [d, d + m_time],
            # so an unsafe window [t1, t2] rules out departures in [t1 - m_time, t2]). The same for
            # every safe interval of the next vertex, so worked out once per edge.
            unsafe = edge.get_unsafe_intervals(now, end_t) if edge else ()
            if unsafe:
                departures = SippNode()
                departures.interval_list = [(now, leave_by)]
                for t1_unsafe, t2_unsafe in unsafe:
                    departures.split_interval(t1_unsafe - m_time, t2_unsafe)
                departures = [w for w in departures.interval_list if w[0] <= leave_by and w[1] >= now]
            else:
                departures = [(now, leave_by)]
            for i in intervals:
                for window in departures:
                    t = self.get_earliest_no_collision_arrival_time_body(start_t, i, window, m_time)
                    if t is None:
                        continue
                    successors.append(State(neighbour_pos, t, i))
                    costs.append(t - now)
                    time_taken.append((t - now - m_time, m_time))
        return successors, costs, time_taken

    def get_heuristic(self, position, goal, name):
        """A lower bound on agent ``name``'s time from ``position`` to ``goal``: the roadmap's quickest time
        (``dijkstra``), or the straight-line / Manhattan distance over its speed (with one time unit per
        move, over the longest edge)."""
        body = self.bodies.get(name)
        speed = None if body is None else body.max_speed
        if self.heuristic_type == HEURISTIC_TYPE["dijkstra"]:
            return self._dijkstra_heuristic(position, goal, speed)
        if self.heuristic_type == HEURISTIC_TYPE["manhattan"]:
            dist = sum(math.fabs(a - b) for a, b in zip(position, goal))
        elif self.heuristic_type == HEURISTIC_TYPE["euclidean"]:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, goal)))
        else:
            raise ValueError(f"Invalid heuristic type: {self.heuristic_type}")
        return dist / speed if speed is not None else dist / self._longest_edge()

    def _longest_edge(self) -> float:
        if self._max_edge is None:
            nodes = self.graph_map.nodes
            self._max_edge = max((float(self.graph_map.get_distance(nodes[u].current, nodes[v].current))
                                  for u, out in enumerate(self.graph_map.road_map) for v in out), default=1.0) or 1.0
        return self._max_edge

    def _goal_distance_map(self, goal_idx: int, speed=None):
        """``{node index: quickest time to goal_idx}`` (one Dijkstra from the goal, cached per goal and speed;
        roadmap edges are two-way with equal lengths). With one time unit per move: the number of moves."""
        key = (goal_idx, speed)
        cached = self._goal_dist_maps.get(key)
        if cached is not None:
            return cached
        road_map = getattr(self.graph_map, "road_map", None)
        dist = {goal_idx: 0.0}
        heap = [(0.0, goal_idx)]
        seen = set()
        while heap and road_map is not None and len(road_map):
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            u_node = self.graph_map.nodes[u]
            for v in road_map[u]:
                step = 1.0 if speed is None else float(self.graph_map.get_distance(u_node.current, self.graph_map.nodes[v].current)) / speed
                nd = d + step
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._goal_dist_maps[key] = dist
        return dist

    def _dijkstra_heuristic(self, position, goal, speed=None):
        """The quickest roadmap time from ``position`` to ``goal``; the straight-line bound when either is not
        a roadmap node or the goal cannot be reached."""
        index = self.graph_map.pos_index_dict
        u, g = index.get(tuple(float(x) for x in position)), index.get(tuple(float(x) for x in goal))
        if u is not None and g is not None:
            d = self._goal_distance_map(int(g), speed).get(int(u))
            if d is not None:
                return d
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, goal)))
        return dist / speed if speed is not None else dist / self._longest_edge()

    def _goal_interval_ok(self, state) -> bool:
        """Whether arriving at the goal in ``state`` leaves it safe for as long as the
        caller needs: forever (``require_goal_safe_forever``), until
        ``goal_safe_until(t)`` when given, else any safe interval."""
        end = state.interval[1]
        if self.require_goal_safe_forever:
            return math.isinf(end)
        if self.goal_safe_until is not None:
            return end >= self.goal_safe_until(state.time) - 1e-9
        return True

    def compute_plan(self):
        solution_info = {}
        st = time.perf_counter()
        best_solution = None
        best_solution_cost = float('inf')
        iterations = 0
        total_low_level_iterations = 0
        start_blocked = set()                    # agents whose start is never safe (blocked for good)
        for _ in range(len(self.agents)):
            self.shuffle_agents()
            self._new_round()
            self.plan = {}
            self.plan_cost = {}
            self.action_cost = {}
            iterations += 1
            success = True
            total_cost = 0
            for agent in self.agents:
                start = tuple(agent["start"])
                goal = tuple(agent["goal"])
                name = agent["name"]
                graph = self._graph(name)
                # At its start the agent stands with its start heading (given, else 0).
                start_node = graph.vertex_node(start, self.start_ori[name])
                if len(start_node.interval_list) == 0:
                    start_blocked.add(name)
                if len(start_node.interval_list) == 0 or (not graph.oriented and len(graph.sipp_graph[goal].interval_list) == 0):
                    success = False
                    break
                # The agent is at ``start`` at t=0, so its initial safe interval must be the
                # one containing t=0. Seeding with interval_list[0] regardless would let the
                # search "wait" at the start through a blocked window (another agent
                # sweeping over it) and emit a plan that collides before it even moves.
                start_interval = self._interval_containing(start_node, 0.0)
                if start_interval is None:
                    success = False
                    break
                if start == goal:                        # already there: a single-state plan
                    found, plan, costs, goal_cost, expanded = True, [State(start, 0, start_interval)], [(0.0, 0.0)], 0.0, 0
                elif self.use_numba and self.has_bodies and self.goal_safe_until is None:
                    # The same search, compiled (fast_search); identical plans.
                    found, plan, costs, goal_cost, expanded = self._fast_search(graph, name, start, goal, start_interval, st)
                else:
                    found, plan, costs, goal_cost, expanded = self._python_search(graph, name, start, goal, start_interval, st)
                # Low-level expansions over every round and agent, so callers can report the search effort.
                total_low_level_iterations += expanded
                if not found:
                    success = False
                    break
                total_cost += goal_cost
                self.plan[name] = plan
                self.plan_cost[name] = goal_cost
                self.action_cost[name] = costs
                self._commit(name, plan, costs)
            if success and total_cost < best_solution_cost:
                best_solution_cost = total_cost
                best_solution = self.get_plan()
        self.total_time += time.perf_counter() - st
        self.total_iterations = iterations
        solution_info["runtime"] = self.total_time
        solution_info["total_iterations"] = self.total_iterations
        solution_info["low_level_iterations"] = total_low_level_iterations
        solution_info["start_blocked"] = sorted(start_blocked)
        solution_info["success"] = best_solution is not None
        return (best_solution if best_solution is not None else {}), solution_info

    def _python_search(self, graph, name, start, goal, start_interval, st):
        """The low-level search for ``name`` (A* over (vertex, safe interval) states): ``(found, plan,
        action costs, arrival time, expansions)``."""
        initial_state = State(start, 0, start_interval)
        initial_state_key = (start, start_interval)
        # Min-heap: (f, counter, state); counter ensures we never compare State objects
        open_heap = []
        counter = 0
        closed_set = set()  # (position, interval) already expanded
        g_score = {initial_state_key: 0.0}      # (position, interval) -> g
        came_from = {}      # (position, interval) -> State
        action_cost = {}    # (position, interval) -> (wait_cost, move_cost)
        # A body that is not a ball has a safe-interval list per arrival heading, so one vertex can
        # be reached in overlapping intervals. Turns are instant, so all that matters after arriving
        # is when, and until when it may stay: an arrival no later that may stay at least as long
        # makes another redundant. position -> [(arrival time, end of its interval)].
        arrivals = {start: [(0.0, start_interval[1])]} if graph.oriented else None

        heapq.heappush(open_heap, (self.get_heuristic(start, goal, name), counter, initial_state))
        counter += 1
        low_level_iterations = 0
        while open_heap and low_level_iterations < self.sipp_max_iterations and time.perf_counter() - st < self.time_limit:
            low_level_iterations += 1
            _, _, current = heapq.heappop(open_heap)
            current_state_key = (current.position, current.interval)
            if current_state_key in closed_set:
                continue
            closed_set.add(current_state_key)

            successors, costs, time_takens = self.get_successors(current, graph, name)
            for cost, time_taken, successor in zip(costs, time_takens, successors):
                succ_key = (successor.position, successor.interval)
                if succ_key in closed_set:
                    continue
                tentative_g_score = g_score[current_state_key] + cost
                if arrivals is not None:
                    end = successor.interval[1]
                    seen = arrivals.get(successor.position)
                    if seen and any(g0 <= tentative_g_score + 1e-9 and e0 >= end - 1e-9 for g0, e0 in seen):
                        continue
                    arrivals[successor.position] = [(g0, e0) for g0, e0 in (seen or ())
                                                    if not (tentative_g_score <= g0 + 1e-9 and end >= e0 - 1e-9)] + [(tentative_g_score, end)]
                if tentative_g_score < g_score.get(succ_key, float('inf')):
                    came_from[succ_key] = current
                    g_score[succ_key] = tentative_g_score
                    action_cost[succ_key] = time_taken
                    if successor.position == goal and self._goal_interval_ok(successor):
                        # Accept the goal only in an interval it may stay in as long as needed (see
                        # _goal_interval_ok). Otherwise the state is pushed like any other: the vertex's
                        # later intervals are separate states, reachable by waiting or detouring.
                        if self.verbose:
                            print("Plan successfully calculated!!")
                        plan, plan_action_cost = self.reconstruct_path(came_from, action_cost, successor)
                        return True, plan, plan_action_cost, successor.time, low_level_iterations
                    f_score = g_score[succ_key] + self.get_heuristic(successor.position, goal, name)
                    heapq.heappush(open_heap, (f_score, counter, successor))
                    counter += 1
        return False, None, None, float('inf'), low_level_iterations

    def _fast_search(self, graph, name, start, goal, start_interval, st):
        """The low-level search for ``name`` with ``fast_search.search_chunk``: ``(found, plan, action
        costs, arrival time, expansions)``, the same as the Python search gives."""
        from path_planning.multi_agent_planner.centralized.sipp import fast_search as F
        nodes = self.graph_map.nodes
        speed = self.bodies[name].max_speed
        duration = (lambda L: 1.0) if speed is None else (lambda L: L / speed)
        data = graph.arrays(speed, duration)
        vid = graph._arr["vid"]
        s, gv = vid[start], vid[goal]
        key = (goal, speed, name if self.heuristic_type == HEURISTIC_TYPE["dijkstra"] else None)
        h = self._h_arrays.get(key)
        if h is None:
            h = self._h_arrays[key] = np.array([self.get_heuristic(tuple(n.current), goal, name) for n in nodes],
                                               dtype=np.float64)
        cap = 4096
        S = {"st_v": np.zeros(cap, np.int64), "st_lo": np.zeros(cap), "st_hi": np.zeros(cap), "g": np.zeros(cap),
             "closed": np.zeros(cap, np.bool_), "parent": np.zeros(cap, np.int64), "parent_t": np.zeros(cap),
             "act_w": np.zeros(cap), "act_m": np.zeros(cap), "hf": np.zeros(cap), "hc": np.zeros(cap, np.int64),
             "hs": np.zeros(cap, np.int64), "ht": np.zeros(cap), "ar_next": np.zeros(cap, np.int64),
             "ar_g": np.zeros(cap), "ar_e": np.zeros(cap), "ar_alive": np.zeros(cap, np.bool_)}
        ar_head = np.full(len(nodes), -1, np.int64)
        lo, hi = float(start_interval[0]), float(start_interval[1])
        index = F.new_index(np.int64(s), lo, hi)
        S["st_v"][0], S["st_lo"][0], S["st_hi"][0], S["g"][0], S["parent"][0] = s, lo, hi, 0.0, -1
        S["hf"][0], S["hc"][0], S["hs"][0], S["ht"][0] = self.get_heuristic(start, goal, name), 0, 0, 0.0
        meta = np.array([1, 1, 1, 0, 0, -1], dtype=np.int64)  # states, heap, counter, expansions, arrivals, goal
        if graph.oriented:
            S["ar_g"][0], S["ar_e"][0], S["ar_alive"][0], S["ar_next"][0] = 0.0, hi, True, -1
            ar_head[s] = 0
            meta[4] = 1
        need = np.zeros(max(64, len(nodes) + 1), np.int64)
        goal_t = np.zeros(1)
        limit = int(min(self.sipp_max_iterations, 2 ** 62))
        while True:
            status = F.search_chunk(*data, h, gv, self.require_goal_safe_forever, graph.oriented, index,
                                    S["st_v"], S["st_lo"], S["st_hi"], S["g"], S["closed"], S["parent"], S["parent_t"],
                                    S["act_w"], S["act_m"], S["hf"], S["hc"], S["hs"], S["ht"], ar_head, S["ar_next"],
                                    S["ar_g"], S["ar_e"], S["ar_alive"], meta, limit, 20000, need, goal_t)
            if status == F.NEED:                         # work out those vertices' per-heading intervals
                wanted = []
                for v in need:
                    if v < 0:
                        break
                    position = tuple(nodes[v].current)
                    wanted.append((position, graph._arrival_headings(position)[0]))
                graph.prepare(wanted)
                data = graph.arrays(speed, duration)
            elif status == F.GROW:
                for key, arr in S.items():
                    grown = np.zeros(2 * len(arr), arr.dtype)
                    grown[:len(arr)] = arr
                    S[key] = grown
            elif status == F.CHUNK:
                if time.perf_counter() - st >= self.time_limit:
                    return False, None, None, float("inf"), int(meta[3])
            else:
                break
        if status != F.FOUND:
            return False, None, None, float("inf"), int(meta[3])
        k = int(meta[5])
        position = lambda v: tuple(nodes[v].current)
        plan = deque([State(position(S["st_v"][k]), float(goal_t[0]), (float(S["st_lo"][k]), float(S["st_hi"][k])))])
        costs = deque([(0, 0)])
        while S["parent"][k] >= 0:
            p = int(S["parent"][k])
            plan.appendleft(State(position(S["st_v"][p]), float(S["parent_t"][k]), (float(S["st_lo"][p]), float(S["st_hi"][p]))))
            costs.appendleft((float(S["act_w"][k]), float(S["act_m"][k])))
            k = p
        return True, list(plan), list(costs), float(goal_t[0]), int(meta[3])

    def reconstruct_path(self, came_from, came_from_action_cost, current):
        total_path = deque([current])
        total_path_action_cost = deque([(0, 0)])
        key = (current.position, current.interval)
        while key in came_from:
            current = came_from[key]
            total_path.appendleft(current)
            total_path_action_cost.appendleft(came_from_action_cost[key])
            key = (current.position, current.interval)
        return list(total_path), list(total_path_action_cost)
                
    @staticmethod
    def _setpoint(t, position) -> dict:
        """One schedule entry: ``t``, ``x``, ``y`` and, in 3D, ``z``."""
        d = {"t": t, "x": position[0], "y": position[1]}
        if len(position) > 2:
            d["z"] = position[2]
        return d

    def get_plan(self):
        """``{agent: [{t, x, y[, z][, yaw[, pitch]]}, ...]}``. Point particles: one entry per time step.
        Bodies: one per state and one at the end of each wait, with the heading of the last move made
        (before the first: the start heading, given or 0)."""
        solution = {}
        for agent in self.agent_names:
            if agent not in self.plan or agent not in self.action_cost:
                continue
            plan = self.plan[agent]
            action_cost = self.action_cost[agent]
            path_list = []
            if not self.has_bodies:
                path_list.append(self._setpoint(plan[0].time, plan[0].position))
                for i in range(len(plan)-1):
                    for j in range(int(plan[i+1].time - plan[i].time-1)):
                        path_list.append(self._setpoint(plan[i].time, plan[i].position))
                    path_list.append(self._setpoint(plan[i+1].time, plan[i+1].position))
            else:
                points = [state.position for state in plan]
                moves = [_ori_between(a, b) if a != b else None for a, b in zip(points, points[1:])]
                facing = self.start_ori[agent]
                for k, (action, state) in enumerate(zip(action_cost, plan)):
                    if k and moves[k - 1] is not None:
                        facing = moves[k - 1]
                    path_list.append(self._with_heading(self._setpoint(state.time, state.position), facing))
                    action_wait_cost, action_move_cost = action
                    if action_wait_cost > 1e-10:
                        path_list.append(self._with_heading(self._setpoint(state.time + action_wait_cost, state.position), facing))
            solution[agent] = path_list
        return solution

    @staticmethod
    def _with_heading(entry: dict, facing) -> dict:
        return put_heading(entry, facing)
