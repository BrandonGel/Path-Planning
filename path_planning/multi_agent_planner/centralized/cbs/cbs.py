"""
Conflict-based search for multi-agent path planning
author: Brandon Ho 
original author: Ashwin Bose (@atb033)
description: This file implements the Conflict-based search algorithm for multi-agent path planning. Modified from the original implementation to work with the new common environment.
"""

from typing import Any
from path_planning.common.environment.node import Node
from path_planning.multi_agent_planner.centralized.cbs.a_star import AStar
import heapq
import time
from math import fabs
from itertools import count,combinations
import numpy as np
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.multi_agent_planner.data_type import HEURISTIC_TYPE

class Location(object):
    def __init__(self, point:tuple = None):
        self.point = tuple(point) if point is not None else None
    def __hash__(self):
        return hash(self.point)
    def __eq__(self, other):
        return self.point == other.point
    def __str__(self):
        return str(self.point)
    def __len__(self):
        return len(self.point)
    def __getitem__(self, index):
        return self.point[index]
    def __add__(self, other):
        return Location(tuple(self.point + other.point))

class State(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash((self.time, self.location.point))
    def is_equal_except_time(self, state):
        return self.location == state.location
    def __str__(self):
        return str((self.time, self.location))

class Conflict(object):
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')'

class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash((self.time, self.location))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')'

class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2
    def __hash__(self):
        return hash((self.time, self.location_1, self.location_2))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')'

class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])

class Environment(object):
    def __init__(self, graph_map:GraphSampler, agents, astar_max_iterations=10000, radius = 0.0, velocity = 0.0, use_constraint_sweep=True,heuristic_type: str = 'manhattan'):
        self.graph_map = graph_map
        if radius > 0:
            self.graph_map.set_constraint_sweep()
        self.agents = agents
        self.agent_dict = {}
        if heuristic_type not in HEURISTIC_TYPE or heuristic_type is None:
            self.heuristic_type = HEURISTIC_TYPE["manhattan"]
        else:
            self.heuristic_type = HEURISTIC_TYPE[heuristic_type]
        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.a_star = AStar(self, astar_max_iterations)
        self.radius = radius
        self.velocity = velocity
        self.use_constraint_sweep = use_constraint_sweep
        self._constraint_sweep_cache = {}  # (p1, p2, r) -> (nodes, edges, start_nodes)
        self._constraint_segment_cache = {}  # (p1a, p1b, p2a, p2b, v, r) -> bool
        self._dijkstra_cache = {}  # (start_idx, goal_idx) -> (states, cost)
        self._heuristic_cache = {}  # (location.point, agent_name) -> heuristic value

    def _static_shortest_path_states(self, agent_name: str):
        """Compute a static shortest path on the roadmap (no time expansion).

        Safe to use only when there are no time-indexed constraints for this agent.
        Returns (states, cost) in the same format as A*.
        """
        start_pt = tuple(float(x) for x in self.agent_dict[agent_name]["start"].location.point)
        goal_pt = tuple(float(x) for x in self.agent_dict[agent_name]["goal"].location.point)
        start_node = Node(start_pt, None, 0, 0)
        goal_node = Node(goal_pt, None, 0, 0)
        node_index_dict = getattr(self.graph_map, "node_index_dict", {})
        if start_node not in node_index_dict or goal_node not in node_index_dict:
            return False, float("inf")
        s = int(node_index_dict[start_node])
        g = int(node_index_dict[goal_node])
        if s == g:
            return [State(0, Location(start_pt))], 0.0

        cache_key = (s, g)
        if cache_key in self._dijkstra_cache:
            return self._dijkstra_cache[cache_key]

        road_map = getattr(self.graph_map, "road_map", None)
        if road_map is None or len(road_map) == 0:
            return False, float("inf")

        # Dijkstra over node indices.
        INF = float("inf")
        dist = {s: 0.0}
        prev = {}
        heap = [(0.0, s)]
        seen = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            if u == g:
                break
            u_node = self.graph_map.nodes[u]
            for v in road_map[u]:
                nd = d + float(self.graph_map.get_cost(u_node, self.graph_map.nodes[v]))
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if g not in dist:
            return False, float("inf")

        # Reconstruct path indices.
        path_idx = [g]
        cur = g
        while cur != s:
            cur = prev[cur]
            path_idx.append(cur)
        path_idx.reverse()
        states = []
        for t, idx in enumerate(path_idx):
            pt = tuple(float(x) for x in self.graph_map.nodes[idx].current)
            states.append(State(t, Location(pt)))
        result = states, float(dist[g])
        self._dijkstra_cache[cache_key] = result
        return result

    def get_neighbors(self, state):
        neighbors = []
        node = Node(tuple[Any, ...](state.location.point))
        nodes = self.graph_map.get_neighbors(node)
        # Wait action
        n = State(state.time + 1,  state.location)
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)

        # Move action
        for node in nodes:
            n = State(state.time + 1, Location(tuple(node.current)))
            if self.state_valid(n) and self.transition_valid(state, n):
                neighbors.append(n)
        return neighbors

    def get_conflicts(self, solution, get_first_conflict: bool = True):
        max_t = max(len(plan) for plan in solution.values())
        conflicts = []
        agent_pairs = list(combinations(solution.keys(), 2))

        # For sweep-based conflict detection, the swept-edge set for an agent
        # at time t depends only on that agent's motion, not on the other
        # agent in the pair. Precompute once per (agent, t) so the pair loop
        # is pure set-membership instead of N^2 cache lookups.
        if self.radius > 0 and self.use_constraint_sweep:
            agent_edges = {agent: [] for agent in solution}
            sweeps = {agent: [] for agent in solution}
            two_r = 2 * self.radius
            for agent in solution:
                for t in range(max_t):
                    sa = self.get_state(agent, solution, t)
                    sb = self.get_state(agent, solution, t + 1)
                    pa = sa.location.point
                    pb = sb.location.point
                    agent_edges[agent].append((pa, pb))
                    sweeps[agent].append(
                        self._get_constraint_sweep_cached(pa, pb, self.velocity, two_r)
                    )

        for t in range(max_t):
            if self.radius == 0:
                for agent_1, agent_2 in agent_pairs:
                    state_1 = self.get_state(agent_1, solution, t)
                    state_2 = self.get_state(agent_2, solution, t)

                    if state_1.is_equal_except_time(state_2):
                        c = Conflict()
                        c.time = t
                        c.type = Conflict.VERTEX
                        c.location_1 = state_1.location
                        c.agent_1 = agent_1
                        c.agent_2 = agent_2
                        if get_first_conflict:
                            return c
                        conflicts.append(c)
                        continue

                    state_1b = self.get_state(agent_1, solution, t + 1)
                    state_2b = self.get_state(agent_2, solution, t + 1)
                    if state_1.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2):
                        c = Conflict()
                        c.time = t
                        c.type = Conflict.EDGE
                        c.agent_1 = agent_1
                        c.agent_2 = agent_2
                        c.location_1 = state_1.location
                        c.location_2 = state_1b.location
                        if get_first_conflict:
                            return c
                        conflicts.append(c)
            else:
                for agent_1, agent_2 in agent_pairs:
                    if self.use_constraint_sweep:
                        edge_1 = agent_edges[agent_1][t]
                        edge_2 = agent_edges[agent_2][t]
                        edge_conflict = edge_1 in sweeps[agent_2][t] or edge_2 in sweeps[agent_1][t]
                        loc_1_pt, loc_2_pt = edge_1
                    else:
                        state_1a = self.get_state(agent_1, solution, t)
                        state_1b = self.get_state(agent_1, solution, t + 1)
                        state_2a = self.get_state(agent_2, solution, t)
                        state_2b = self.get_state(agent_2, solution, t + 1)
                        edge_conflict = self._get_constraint_segment_cached(
                            state_1a.location.point, state_1b.location.point,
                            state_2a.location.point, state_2b.location.point,
                            self.velocity, self.radius,
                        )
                        loc_1_pt = state_1a.location.point
                        loc_2_pt = state_1b.location.point

                    if edge_conflict:
                        c = Conflict()
                        c.time = t
                        c.type = Conflict.EDGE
                        c.agent_1 = agent_1
                        c.agent_2 = agent_2
                        c.location_1 = Location(loc_1_pt)
                        c.location_2 = Location(loc_2_pt)
                        if get_first_conflict:
                            return c
                        conflicts.append(c)
        return conflicts

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state):
        if self.graph_map.in_collision_point(state.location.point):
            return False
        
        return  VertexConstraint(state.time,state.location) not in self.constraints.vertex_constraints

    def transition_valid(self, state_1, state_2):
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in self.constraints.edge_constraints

    def is_solution(self, agent_name):
        pass

    def admissible_heuristic(self, state, agent_name):
        key = (state.location.point, agent_name)
        cached = self._heuristic_cache.get(key)
        if cached is not None:
            return cached
        goal_pt = self.agent_dict[agent_name]["goal"].location.point
        loc_pt = state.location.point
        if self.heuristic_type == HEURISTIC_TYPE["manhattan"]:
            result = sum(fabs(a - b) for a, b in zip(loc_pt, goal_pt))
        elif self.heuristic_type == HEURISTIC_TYPE["euclidean"]:
            result = sum((a - b) ** 2 for a, b in zip(loc_pt, goal_pt)) ** 0.5
        else:
            raise ValueError(f"Invalid heuristic type: {self.heuristic_type}")
        self._heuristic_cache[key] = result
        return result

    def get_step_cost(self, state_1: State, state_2: State) -> float:
        """Incremental cost between consecutive time-expanded states.

        Matches roadmap edge weights via ``GraphSampler.get_cost`` for moves,
        so low-level costs align with discrete Dijkstra in
        ``_static_shortest_path_states``. Waiting in place consumes one timestep
        with unit cost so idle steps are not free.
        """
        p1 = tuple(state_1.location.point)
        p2 = tuple(state_2.location.point)
        if p1 == p2:
            return 1.0
        node1 = Node(tuple[Any, ...](p1), None, 0, 0)
        node2 = Node(tuple[Any, ...](p2), None, 0, 0)
        return float(self.graph_map.get_cost(node1, node2))

    def is_at_goal(self, state, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        for agent in self.agents:
            start_state = State(0, Location(agent['start']))
            goal_state = State(0, Location(agent['goal']))
            
            self.agent_dict.update({agent['name']:{'start':start_state, 'goal':goal_state}})

    def _get_constraint_sweep_cached(self, p1, p2,v, r):
        """Cached wrapper for get_constraint_sweep to avoid duplicate queries."""
        key = (p1, p2, v, r)
        if key not in self._constraint_sweep_cache:
            self._constraint_sweep_cache[key] = self.graph_map.get_constraint_sweep(p1, p2,v, r)
        return self._constraint_sweep_cache[key]

    def _get_constraint_segment_cached(self, p1a, p1b, p2a, p2b, v, r):
        """Cached wrapper for get_constraint_sweep to avoid duplicate queries."""
        key = (p1a, p1b, p2a, p2b, v, r)
        if key not in self._constraint_segment_cache:
            self._constraint_segment_cache[key] = self.graph_map.get_constraint_segment(p1a, p1b, p2a, p2b, v,r)
        return self._constraint_segment_cache[key]

    def compute_solution(self):
        # Clear caches for fresh solution attempt
        solution = {}
        solution_cost = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            if (
                len(self.constraints.vertex_constraints) == 0
                and len(self.constraints.edge_constraints) == 0
            ):
                local_solution, local_cost = self._static_shortest_path_states(agent)
            else:
                local_solution, local_cost = self.a_star.search(agent)
            if not local_solution:
                return False, float('inf')
            solution.update({agent:local_solution})
            solution_cost[agent] = local_cost
        return solution, solution_cost

class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
        self.solution_cost = {}
        self.constraint_dict = {}
        self.cost = 0

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash(self.cost)

    def __lt__(self, other):
        return self.cost < other.cost

class CBS(object):
    def __init__(self, environment: Environment, time_limit: float | None = None, max_iterations: int | None = None,verbose: bool = False):
        """
        :param environment: Environment instance
        :param time_limit: Optional wall-clock time limit (in seconds) for the
                           high-level CBS search. If None, no time limit is
                           enforced. If exceeded, the search terminates early
                           and returns an empty solution.
        """
        self.env = environment
        self.verbose = verbose
        self.open_list =  []  # for fast membership checks
        self.closed_set = set()
        self.time_limit = time_limit if time_limit is not None and time_limit > 0 else float('inf')
        self.max_iterations = max_iterations if max_iterations is not None and max_iterations > 0 else float('inf')
        self.counter = count()
        self.total_time = 0
        self.total_iterations = 0
        
    def search(self):
        st = time.time()
        iterations = 1
        success = False
        start = HighLevelNode()
        start.constraint_dict = {}
        solution = {}
        solution_info = {}

        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()

        start.solution, start.solution_cost = self.env.compute_solution()
        if not start.solution:
            if self.verbose:
                print("No initial solution found")
            self.total_time = min(self.time_limit, time.time() - st) 
            self.total_iterations = min(self.max_iterations, iterations)
            solution_info["runtime"] = self.total_time
            solution_info["total_iterations"] = self.total_iterations
            solution_info["success"] = success
            return solution,solution_info

        start.cost = sum(start.solution_cost.values())

        # Add start node to heap
        heapq.heappush(self.open_list, (start.cost, next(self.counter), start))
        while self.open_list :
            iterations += 1
            if self.time_limit is not None and (time.time() - st) > self.time_limit:
                if self.verbose:
                    print(f"Search terminated: time limit of {self.time_limit} seconds exceeded.")
                break

            if self.max_iterations is not None and iterations >= self.max_iterations:
                if self.verbose:
                    print(f"Search terminated: max iterations of {self.max_iterations} reached.")                
                break

            _, _, P = heapq.heappop(self.open_list)

            
            if P is None:
                break
            state_key = self._get_state_key(P)
            if state_key in self.closed_set:
                continue
            self.closed_set.add(state_key)

            self.env.constraint_dict = P.constraint_dict
            conflict_dict = self.env.get_conflicts(P.solution)
            if not conflict_dict:
                if self.verbose:
                    print("solution found")
                success = True
                solution = self.generate_plan(P.solution)
                break
            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)
            for agent in constraint_dict.keys():
                new_node = HighLevelNode()
                new_node.solution = P.solution.copy()
                new_node.solution_cost = P.solution_cost.copy()

                # Selective deep copy only for affected agent's constraints
                new_node.constraint_dict = {}
                for a in self.env.agent_dict.keys():
                    if a == agent:
                        # Deep copy only the modified agent's constraints
                        new_constraints = Constraints()
                        new_constraints.vertex_constraints = P.constraint_dict[a].vertex_constraints.copy()
                        new_constraints.edge_constraints = P.constraint_dict[a].edge_constraints.copy()
                        new_constraints.add_constraint(constraint_dict[agent])
                        new_node.constraint_dict[a] = new_constraints
                    else:
                        # Share unchanged constraints
                        new_node.constraint_dict[a] = P.constraint_dict[a]

                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution, new_node.solution_cost = self.env.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = sum(new_node.solution_cost.values())
                heapq.heappush(self.open_list, (new_node.cost, next(self.counter), new_node))
        
        self.total_time = min(self.time_limit, time.time() - st) 
        self.total_iterations = min(self.max_iterations, iterations)
        solution_info["runtime"] = self.total_time
        solution_info["total_iterations"] = self.total_iterations
        solution_info["success"] = success
        return solution,solution_info

    def _get_state_key(self, node):
        """Generate a hashable state key for closed-set deduplication.

        Keyed on the *solution* (every agent's full path), NOT the constraint
        set. Constraints uniquely determine a solution, but the converse is
        false: an "inactive" constraint (one the agent's optimal path already
        satisfies) leaves the solution unchanged, so two CBS nodes reached via
        different constraints can share an identical solution. Those are pure
        redundancy — same solution -> same conflicts -> same children — and
        must be deduped by solution. A constraint-based key fails to catch
        them and the search tree blows up combinatorially at large radius.
        """
        return tuple(
            (agent, tuple((s.time, s.location.point) for s in path))
            for agent, path in sorted(node.solution.items())
        )
   
    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = []
            for state in path:
                if len(state.location) == 2:
                    path_dict_list.append({'t':state.time, 'x':state.location[0], 'y':state.location[1]})
                elif len(state.location) == 3:
                    path_dict_list.append({'t':state.time, 'x':state.location[0], 'y':state.location[1], 'z':state.location[2]})
                else:
                    raise ValueError(f"Invalid location dimension: {len(state.location)}")
            plan[agent] = path_dict_list
        return plan

    