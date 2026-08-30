"""

SIPP implementation  

author: Ashwin Bose (@atb033)

See the article: DOI: 10.1109/ICRA.2011.5980306

"""

from math import fabs
import heapq
from collections import deque
import random
import time
from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippGraph, State
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.node import Node
import math
from path_planning.multi_agent_planner.data_type import HEURISTIC_TYPE
from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippNode

class SippPlanner(SippGraph):
    def __init__(self, graph_map: GraphSampler,dynamic_obstacles:dict = {},agents:list = [],radius:float = 0.0,velocity:float = 0.0,use_constraint_sweep:bool = True, heuristic_type: str = 'manhattan',time_limit: float | None = None, max_iterations: int | None = None,verbose: bool = False,sipp_max_iterations: int = 10000, obstacle_horizon: float | dict | None = None, require_goal_safe_forever: bool = True, goal_safe_until=None):
        SippGraph.__init__(self,graph_map,dynamic_obstacles,radius,velocity,use_constraint_sweep,heuristic_type,time_limit,max_iterations,verbose,obstacle_horizon)
        self.agents = agents
        # A finished agent occupies its goal for all future time, so a goal
        # arrival is only truly collision-free if the accepted safe interval
        # extends to infinity - otherwise a later-planned agent may legally
        # transit through that spot once its own (bounded) view of the
        # interval considers it free, producing an undetected collision.
        # Callers that replan around finished agents themselves (e.g.
        # NeuralATTF) can disable this to accept a goal in any safe interval.
        self.require_goal_safe_forever = require_goal_safe_forever
        # Optional refinement of the bounded-goal case: ``goal_safe_until(t_arrive)``
        # returns the time the goal must stay safe until for an arrival at
        # ``t_arrive`` (e.g. the end of the executor's tick plus the hold other
        # agents assume). A goal reached inside an interval that ends earlier is
        # not accepted; the search keeps waiting/detouring for a later interval.
        self.goal_safe_until = goal_safe_until
        self._goal_dist_maps = {}  # goal_idx -> {node_idx: shortest-path cost to goal}
        self.agent_names = [agent["name"] for agent in agents]
        self.plan = {}
        self.plan_cost = {}
        self.action_cost = {}
        self._mtime_cache = {}
        self.max_permutations = math.factorial(len(agents))
        self.max_iterations = min(self.max_iterations , self.max_permutations)
        self.sipp_max_iterations = sipp_max_iterations if sipp_max_iterations > 0 or sipp_max_iterations is None else float("inf")
        if heuristic_type not in HEURISTIC_TYPE or heuristic_type is None:
            self.heuristic_type = HEURISTIC_TYPE["manhattan"]
        else:
            self.heuristic_type = HEURISTIC_TYPE[heuristic_type]

    def _interval_containing(self, position, t: float):
        """The safe interval of ``position`` that contains time ``t`` (None if blocked)."""
        for interval in self.sipp_graph[position].interval_list:
            if interval[0] <= t <= interval[1]:
                return interval
        return None

    def shuffle_agents(self):
        random.shuffle(self.agents)

    def get_mtime(self, position1, position2):
        if (position1, position2) in self._mtime_cache:
            return self._mtime_cache[(position1, position2)]
        m_cost = float(self.graph_map.get_cost(Node(tuple(position1)), Node(tuple(position2))))
        if self.velocity > 0:
            m_time = m_cost / self.velocity
        else:
            m_time = 1.0
        self._mtime_cache[(position1, position2)] = m_time
        return m_time

    def get_earliest_no_collision_arrival_time_point(self, start_t, interval, start_pos, neighbour):
        arrive_t = max(start_t, interval[0])

        # Check for edge conflicts for point agents
        if arrive_t == interval[0]:
            collision = False
            for _, obstacle in self.dyn_obstacles.items():
                for idx in range(len(obstacle) - 1):
                    obs_state = obstacle[idx]
                    obs_t, obs_position = obs_state.time, obs_state.position
                    obs_next_state = obstacle[idx + 1]
                    obs_next_t, obs_next_position = obs_next_state.time, obs_next_state.position
                    edge_conflict = (
                        obs_position[0] == neighbour[0] and obs_position[1] == neighbour[1]
                        and start_pos[0] == obs_next_position[0] and start_pos[1] == obs_next_position[1]
                        and obs_t == arrive_t - 1 and obs_next_t == arrive_t
                    )
                    if edge_conflict:
                        collision = True
                        break
                if collision:
                    arrive_t = None
                    break
        return arrive_t

    def get_earliest_no_collision_arrival_time_body(self,start_t, vertex_interval, edge_interval, start_pos, neighbour,m_time):
        arrive_t = max(start_t, vertex_interval[0],edge_interval[0] + m_time) 
        depart_t = arrive_t - m_time
        if not (depart_t >= edge_interval[0] - 1e-9 and depart_t <= edge_interval[1]):
            return None
        if not(arrive_t >= vertex_interval[0] and arrive_t <= vertex_interval[1]):
            return None
        return arrive_t

    def get_successors(self, state):
        successors = []
        costs = []
        time_taken = []
        neighbour_list = self.get_valid_neighbours(state.position)
        for neighbour_pos in neighbour_list:
            m_time = self.get_mtime(state.position, neighbour_pos)
            start_pos = state.position
            start_t = state.time + m_time  # Earliest possible arrival time
            end_t = state.interval[1] + m_time #Latest possible arrival time
            
            for i in self.sipp_graph[neighbour_pos].interval_list:
                # If the interval is outside the possible arrival time, skip the interval
                if i[0] > end_t or i[1] < start_t:
                    continue

                if self.radius == 0:
                    # Get the earliest no collision arrival time
                    t = self.get_earliest_no_collision_arrival_time_point( start_t, i, start_pos, neighbour_pos)
                    if t is None: # Any collision, skip the interval
                        continue
                    
                    #Get total cost & wait cost for the successor
                    cost = t - state.time 
                    w_cost = cost - m_time

                    # Create the successor state
                    s = State(neighbour_pos, t, i)
                    successors.append(s)
                    costs.append(cost)
                    time_taken.append((w_cost, m_time))
                else:
                    early_depart_t = state.time
                    late_depart_t = state.interval[1]
                    unsafe_interval_list = self.sipp_graph[(start_pos,neighbour_pos)].get_unsafe_intervals(early_depart_t, end_t)
                    safe_node = SippNode()
                    safe_node.interval_list = [(early_depart_t, late_depart_t)]
                    for i_unsafe in range(len(unsafe_interval_list)):
                        t1_unsafe, t2_unsafe = unsafe_interval_list[i_unsafe]
                        # Correct unsafe departure window: any departure in [t1_unsafe - m_time,
                        # t2_unsafe] risks agent B being on this edge during the collision window.
                        # Using m_time as the lower-bound offset is the tightest conservative
                        # bound: B must depart no earlier than m_time before t1_unsafe to reach
                        # any point on the edge by t1_unsafe, and no later than t2_unsafe to
                        # still be on the edge when the collision window closes.
                        t_min_unsafe = t1_unsafe - m_time
                        t_max_unsafe = t2_unsafe
                        safe_node.split_interval(t_min_unsafe, t_max_unsafe)

                    for i_edge in safe_node.interval_list:
                        if i_edge[0] > late_depart_t or i_edge[1] < early_depart_t:
                            continue

                        t = self.get_earliest_no_collision_arrival_time_body(start_t, i,i_edge, start_pos, neighbour_pos,m_time)
                        if t is None: # Any collision, skip the interval
                            continue
                        
                        #Get total cost & wait cost for the successor
                        cost = t - state.time 
                        w_cost = cost - m_time

                        # Create the successor state
                        s = State(neighbour_pos, t, i)
                        successors.append(s)
                        costs.append(cost)
                        time_taken.append((w_cost, m_time))                     
        return successors, costs, time_taken

    def get_heuristic(self, position,goal):
        if self.heuristic_type == HEURISTIC_TYPE["manhattan"]:
            dist =  fabs(position[0] - goal[0]) + fabs(position[1]-goal[1])
        elif self.heuristic_type == HEURISTIC_TYPE["euclidean"]:
            dist = math.sqrt((position[0] - goal[0])**2 + (position[1]-goal[1])**2)
        elif self.heuristic_type == HEURISTIC_TYPE["dijkstra"]:
            dist = self._dijkstra_heuristic(position, goal)
        else:
            raise ValueError(f"Invalid heuristic type: {self.heuristic_type}")
        return dist if self.velocity == 0 else dist / self.velocity

    def _goal_distance_map(self, goal_idx: int):
        """Single-source Dijkstra from the goal over the roadmap.

        Roadmap edges are bidirectional with symmetric costs, so distances from
        the goal equal shortest-path costs to the goal from every node. Computed
        once per goal per SippPlanner instance (i.e. once per solve_mapf call,
        reused across every shuffle-order retry and every agent sharing that
        goal) and cached; used as the exact (admissible, consistent) low-level
        A* heuristic.
        """
        cached = self._goal_dist_maps.get(goal_idx)
        if cached is not None:
            return cached
        road_map = getattr(self.graph_map, "road_map", None)
        dist = {goal_idx: 0.0}
        if road_map is None or len(road_map) == 0:
            self._goal_dist_maps[goal_idx] = dist
            return dist
        heap = [(0.0, goal_idx)]
        seen = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            u_node = self.graph_map.nodes[u]
            for v in road_map[u]:
                nd = d + float(self.graph_map.get_cost(u_node, self.graph_map.nodes[v]))
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._goal_dist_maps[goal_idx] = dist
        return dist

    def _dijkstra_heuristic(self, position, goal):
        """True roadmap shortest-path distance from position to goal.

        Falls back to euclidean distance (an admissible lower bound, since
        move costs are euclidean edge lengths) when either point is not a
        roadmap node or the goal is unreachable from position.
        """
        node_index_dict = getattr(self.graph_map, "node_index_dict", {})
        loc_node = Node(tuple(float(x) for x in position), None, 0, 0)
        goal_node = Node(tuple(float(x) for x in goal), None, 0, 0)
        if loc_node in node_index_dict and goal_node in node_index_dict:
            dist = self._goal_distance_map(int(node_index_dict[goal_node]))
            d = dist.get(int(node_index_dict[loc_node]))
            if d is not None:
                return d
        return math.sqrt((position[0] - goal[0]) ** 2 + (position[1] - goal[1]) ** 2)

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
        solution = {}
        st = time.perf_counter()
        best_solution = None
        best_solution_cost = float('inf')
        best_success = False
        iterations = 0
        total_low_level_iterations = 0
        self.max_iterations = len(self.agents)
        for _ in range(self.max_iterations):
            self.shuffle_agents()
            if not getattr(self, "_graph_fresh", False):
                self.reset_graph()  # __init__ already built a fresh graph for the first pass
            self.plan = {}
            self.plan_cost = {}
            self.action_cost = {}
            iterations += 1
            success = True
            total_cost = 0
            for ii, agent in enumerate(self.agents):
                start = tuple(agent["start"])
                goal = tuple(agent["goal"])

                if len(self.sipp_graph[start].interval_list) == 0 or len(self.sipp_graph[goal].interval_list) == 0:
                    success = False
                    break
                # The agent is at ``start`` at t=0, so its initial safe interval must be the
                # one containing t=0. Seeding with interval_list[0] regardless would let the
                # search "wait" at the start through a blocked window (another agent
                # sweeping over it) and emit a plan that collides before it even moves.
                start_interval = self._interval_containing(start, 0.0)
                if start_interval is None:
                    success = False
                    break
                # If start already equals goal, low-level search should terminate immediately.
                # Treat this as a zero-cost single-state plan and continue.
                if start == goal:
                    initial_state = State(start, 0, start_interval)
                    self.plan[agent["name"]] = [initial_state]
                    self.plan_cost[agent["name"]] = 0.0
                    self.action_cost[agent["name"]] = [(0.0, 0.0)]
                    self.update_intervals([[initial_state]], [[(0.0, 0.0)]], [agent["name"]])
                    continue
                initial_state = State(start, 0, start_interval)
                initial_state_key = (start, initial_state.interval)

                # Min-heap: (f, counter, state); counter ensures we never compare State objects
                open_heap = []
                counter = 0
                closed_set = set()  # (position, interval) already expanded
                g_score  = {initial_state_key:0.0}      # (position, interval) -> g
                came_from   = {}      # (position, interval) -> State
                action_cost = {}    # (position, interval) -> (wait_cost, move_cost)
                
                f_start = self.get_heuristic(start, goal)
                heapq.heappush(open_heap, (f_start, counter, initial_state))
                counter += 1

                goal_reached = False
                goal_state   = None
                goal_cost = float('inf')
                low_level_iterations = 0
                while open_heap and not goal_reached and low_level_iterations < self.sipp_max_iterations and time.perf_counter() - st < self.time_limit:
                    low_level_iterations += 1
                    _, _, current = heapq.heappop(open_heap)
                    # if round(current.position[0],2) == 11. and round(current.position[1],2) == 17. and round(current.time,2) == 2.19:
                    #     print(current)
                    current_state_key = (current.position, current.interval)
                    if current_state_key in closed_set:
                        continue
                    closed_set.add(current_state_key)

                    successors, costs, time_takens = self.get_successors(current)

                    for cost, time_taken, successor in zip(costs, time_takens, successors):
                        succ_key = (successor.position, successor.interval)
                        if succ_key in closed_set:
                            continue

                        tentative_g_score = g_score.get(current_state_key, float('inf')) + cost
                        if tentative_g_score < g_score.get(succ_key, float('inf')):
                            came_from[succ_key]  = current
                            g_score[succ_key] = tentative_g_score
                            action_cost[succ_key] = time_taken
                            
                            if successor.position == goal and self._goal_interval_ok(successor):
                                # Accept the goal only once its safe interval extends to
                                # infinity - the agent stays there forever afterward, so a
                                # bounded interval means some later-planned agent could
                                # still legally transit through this spot. When the current
                                # interval is bounded, fall through and push this state onto
                                # the open heap like any other successor: the vertex's other
                                # (possibly infinite) intervals are separate successor states
                                # reachable by waiting/detouring, so the search can still find
                                # a permanently safe arrival if one exists.
                                if self.verbose:
                                    print("Plan successfully calculated!!")
                                goal_reached = True
                                goal_state = successor
                                goal_cost = successor.time
                                total_cost += goal_cost
                                break

                            f_score = g_score[succ_key] + self.get_heuristic(successor.position, goal)
                            heapq.heappush(open_heap, (f_score, counter, successor))
                            counter += 1

                # Accumulate low-level (A*/SIPP) expansions across every high-level
                # iteration and agent so callers can report total search effort.
                total_low_level_iterations += low_level_iterations

                if not goal_reached or goal_state is None:
                    success = False
                    break
                
                # Backtrack using (position, interval) keys
                plan,plan_action_cost = self.reconstruct_path(came_from,action_cost,goal_state)
                self.plan[agent["name"]] = plan
                self.plan_cost[agent["name"]] = goal_cost
                self.action_cost[agent["name"]] = plan_action_cost
                self.update_intervals([plan],[plan_action_cost],[agent["name"]])
     
            if success and  total_cost < best_solution_cost:
                best_solution_cost = total_cost
                best_solution = self.get_plan()
                best_success = True
        self.total_time += time.perf_counter() - st
        self.total_iterations = min(self.max_iterations, iterations)
        solution =best_solution if best_success else {}
        solution_info["runtime"] = self.total_time
        solution_info["total_iterations"] = self.total_iterations
        solution_info["low_level_iterations"] = total_low_level_iterations
        solution_info["success"] = best_success
        return solution,solution_info
            
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
                
    def get_plan(self):
        solution = {}
        for agent in self.agent_names:
            if agent not in self.plan or agent not in self.action_cost:
                continue
            plan = self.plan[agent]
            action_cost = self.action_cost[agent]
            path_list = []
            if self.radius == 0:
                setpoint = plan[0]
                temp_dict = {"t":setpoint.time,"x":setpoint.position[0], "y":setpoint.position[1]}
                path_list.append(temp_dict)

                for i in range(len(plan)-1):
                    for j in range(int(plan[i+1].time - plan[i].time-1)):
                        t = plan[i].time
                        x = plan[i].position[0]
                        y = plan[i].position[1]
                        setpoint = plan[i]
                        temp_dict = {"t":t,"x":x, "y":y}
                        path_list.append(temp_dict)
                    setpoint = plan[i+1]
                    temp_dict = {"t":setpoint.time,"x":setpoint.position[0], "y":setpoint.position[1]}
                    path_list.append(temp_dict)
            else:
                for action,state in zip(action_cost,plan):
                    temp_dict = {"t":state.time,"x":state.position[0], "y":state.position[1]}
                    path_list.append(temp_dict)

                    action_wait_cost, action_move_cost = action
                    if action_wait_cost > 1e-10:
                        temp_dict = {"t":state.time+action_wait_cost,"x":state.position[0], "y":state.position[1]}
                        path_list.append(temp_dict)
            solution[agent] = path_list
        return solution
