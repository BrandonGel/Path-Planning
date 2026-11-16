"""Roadmap distances for CBS: each agent's best path without constraints, the cost of a step, and the
estimates of the remaining cost that guide the low-level search. Mixed into ``cbs.Environment``."""
import heapq
from math import fabs

from path_planning.multi_agent_planner.centralized.cbs.data_types import (
    FREE, Location, State, _ori_between, _turn_steps)
from path_planning.multi_agent_planner.data_type import HEURISTIC_TYPE


class RoadmapCosts:
    """The routing part of ``Environment``. Uses its ``graph_map``, ``agent_dict``, ``heuristic_type``,
    ``turn_rate`` and the caches ``_moves_cache``, ``_links``, ``_dijkstra_cache``, ``_goal_dist_maps``,
    ``_turn_dist_maps`` and ``_heuristic_cache``."""

    def _node_index(self, point):
        """The roadmap node index of ``point``, or None when it is not a node."""
        index = self.graph_map.pos_index_dict.get(tuple(float(x) for x in point))
        return None if index is None else int(index)

    def _moves_from(self, point):
        """The roadmap moves out of ``point``: ``[(neighbour point, its Location, ori facing it)]``, cached."""
        moves = self._moves_cache.get(point)
        if moves is None:
            index = self._node_index(point)
            targets = [] if index is None else [tuple(n.current) for n in self.graph_map.get_neighbors(index)]
            moves = self._moves_cache[point] = [(q, Location(q), _ori_between(point, q)) for q in targets]
        return moves

    def _roadmap_links(self):
        """The directed roadmap edges by node index, built once: ``(successors, predecessors)`` lists of
        ``(other node, heading along the edge, length)`` per node."""
        if self._links is None:
            nodes, index = self.graph_map.nodes, self.graph_map.pos_index_dict
            succ, pred = [[] for _ in nodes], [[] for _ in nodes]
            for u, node in enumerate(nodes):
                p = tuple(node.current)
                for q, _, facing in self._moves_from(p):
                    v, length = int(index[q]), float(self.graph_map.get_distance(p, q))
                    succ[u].append((v, facing, length))
                    pred[v].append((u, facing, length))
            self._links = (succ, pred)
        return self._links

    def _dijkstra(self, source: int, target: "int | None" = None):
        """Shortest roadmap distances from node ``source``: ``(dist, prev)`` dicts by node index. Stops once
        ``target`` is settled."""
        road_map, nodes = self.graph_map.road_map, self.graph_map.nodes
        dist, prev, seen, heap = {source: 0.0}, {}, set(), [(0.0, source)]
        if not len(road_map):
            return dist, prev
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            if u == target:
                break
            here = nodes[u].current
            for v in road_map[u]:
                nd = d + float(self.graph_map.get_distance(here, nodes[v].current))
                if nd < dist.get(v, float("inf")):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        return dist, prev

    def _static_shortest_path_states(self, agent_name: str):
        """The agent's shortest roadmap path, ignoring time: ``(states, cost)`` as the low-level search
        returns them, or ``(False, inf)``. Only valid for an agent without constraints or turn steps."""
        start_pt = tuple(float(x) for x in self.agent_dict[agent_name]["start"].location.point)
        s, g = self._node_index(start_pt), self._node_index(self.agent_dict[agent_name]["goal"].location.point)
        if s is None or g is None:
            return False, float("inf")
        if s == g:
            return [State(0, Location(start_pt))], 0.0
        if (s, g) not in self._dijkstra_cache:
            dist, prev = self._dijkstra(s, g)
            if g not in dist:
                return False, float("inf")
            indices = [g]
            while indices[-1] != s:
                indices.append(prev[indices[-1]])
            states = [State(t, Location(tuple(float(x) for x in self.graph_map.nodes[i].current)))
                      for t, i in enumerate(reversed(indices))]
            self._dijkstra_cache[(s, g)] = (states, float(dist[g]))
        return self._dijkstra_cache[(s, g)]

    def _goal_distance_map(self, goal_idx: int) -> dict:
        """``{node: shortest distance to the goal}``, computed once per goal. Roadmap edges are two-way
        with equal costs, so distances from the goal are distances to it."""
        if goal_idx not in self._goal_dist_maps:
            self._goal_dist_maps[goal_idx] = self._dijkstra(goal_idx)[0]
        return self._goal_dist_maps[goal_idx]

    def _dijkstra_heuristic(self, loc_pt, goal_pt):
        """The roadmap distance from ``loc_pt`` to ``goal_pt``; the straight-line distance (a lower bound)
        when either is not a roadmap node or the goal cannot be reached."""
        u, goal = self._node_index(loc_pt), self._node_index(goal_pt)
        if u is not None and goal is not None:
            d = self._goal_distance_map(goal).get(u)
            if d is not None:
                return d
        return sum((a - b) ** 2 for a, b in zip(loc_pt, goal_pt)) ** 0.5

    def _turn_distance_map(self, goal_idx: int, rate: float) -> dict:
        """``{(node, heading): cost to the goal}`` for an agent at ``node`` about to leave along the edge it
        faces, when turning costs steps (``rate`` degrees per step): Dijkstra backwards over the directed
        roadmap edges, where going from one edge to the next costs the turn between their headings. Exact
        without constraints: the agent turns straight to the edge it will take and never waits. Cached
        per goal and rate."""
        key = (goal_idx, rate)
        dist = self._turn_dist_maps.get(key)
        if dist is None:
            _, pred = self._roadmap_links()
            dist = self._turn_dist_maps[key] = {}
            heap = [(length, u, facing) for u, facing, length in pred[goal_idx]]
            heapq.heapify(heap)
            while heap:
                g, u, facing = heapq.heappop(heap)
                if (u, facing) in dist:
                    continue
                dist[(u, facing)] = g
                for p, before, length in pred[u]:                 # arrive at u heading ``before``, turn, leave
                    if (p, before) not in dist:
                        heapq.heappush(heap, (g + length + _turn_steps(before, facing, rate), p, before))
        return dist

    def _turn_heuristic(self, loc_pt, ori, agent_name):
        """Exact cost to the goal from ``loc_pt`` with heading ``ori`` for an agent that has to turn, moves
        and turn steps included (``_turn_distance_map``). None when either point is off the roadmap or the
        goal cannot be reached."""
        u, goal = self._node_index(loc_pt), self._node_index(self.agent_dict[agent_name]["goal"].location.point)
        if u is None or goal is None:
            return None
        if u == goal:
            return 0.0
        rate = self.turn_rate(agent_name)
        dist = self._turn_distance_map(goal, rate)
        costs = [dist[(u, facing)] + (0 if ori == FREE else _turn_steps(ori, facing, rate))
                 for _, facing, _ in self._roadmap_links()[0][u] if (u, facing) in dist]
        return min(costs, default=None)

    def admissible_heuristic(self, state, agent_name):
        """A lower bound on the cost from ``state`` to the agent's goal. For an agent that has to turn it
        is the exact roadmap cost with its turn steps (whatever ``heuristic_type`` says); otherwise the
        distance given by ``heuristic_type``."""
        ori = getattr(state, "ori", None)                     # (states of other planners have none)
        key = (state.location.point, agent_name) if ori is None else (state.location.point, agent_name, ori)
        cached = self._heuristic_cache.get(key)
        if cached is not None:
            return cached
        if ori is not None and self.turn_rate(agent_name) is not None:
            result = self._turn_heuristic(state.location.point, ori, agent_name)
            if result is not None:
                self._heuristic_cache[key] = result
                return result
        goal_pt = self.agent_dict[agent_name]["goal"].location.point
        loc_pt = state.location.point
        if self.heuristic_type == HEURISTIC_TYPE["manhattan"]:
            result = sum(fabs(a - b) for a, b in zip(loc_pt, goal_pt))
        elif self.heuristic_type == HEURISTIC_TYPE["euclidean"]:
            result = sum((a - b) ** 2 for a, b in zip(loc_pt, goal_pt)) ** 0.5
        elif self.heuristic_type == HEURISTIC_TYPE["dijkstra"]:
            result = self._dijkstra_heuristic(loc_pt, goal_pt)
        else:
            raise ValueError(f"Invalid heuristic type: {self.heuristic_type}")
        self._heuristic_cache[key] = result
        return result

    def get_step_cost(self, state_1: State, state_2: State) -> float:
        """The cost of one step: the roadmap edge length for a move, 1 for a wait or a turn in place."""
        p1, p2 = state_1.location.point, state_2.location.point
        return 1.0 if p1 == p2 else float(self.graph_map.get_distance(p1, p2))
