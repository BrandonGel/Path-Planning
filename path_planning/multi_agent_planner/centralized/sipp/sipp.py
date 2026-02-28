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

class SippPlanner(SippGraph):
    def __init__(self, graph_map: GraphSampler,dynamic_obstacles:dict = {},agents:list = [],radius:float = 0.0,velocity:float = 0.0,use_constraint_sweep:bool = True, heuristic_type: str = 'manhattan',time_limit: float | None = None, max_iterations: int | None = None,verbose: bool = False):
        SippGraph.__init__(self,graph_map,dynamic_obstacles,radius,velocity,use_constraint_sweep,heuristic_type,time_limit,max_iterations,verbose)
        self.agents = agents
        self.agent_names = [agent["name"] for agent in agents]
        self.plan = {}
        self.plan_cost = {}
        self.action_cost = {}
        self._mtime_cache = {}
        self.max_permutations = math.factorial(len(agents))
        self.max_iterations = min(self.max_iterations , self.max_permutations)
        if heuristic_type not in HEURISTIC_TYPE or heuristic_type is None:
            self.heuristic_type = HEURISTIC_TYPE["manhattan"]
        else:
            self.heuristic_type = HEURISTIC_TYPE[heuristic_type]

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
        if arrive_t > vertex_interval[1]:
            return None
        if not self.sipp_graph[start_pos].is_in_safe_interval(depart_t):
            return None
        if not self.sipp_graph[neighbour].is_in_safe_interval(arrive_t):
            return None
        if not self.sipp_graph[(start_pos,neighbour)].is_in_safe_interval(depart_t):
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
                    for i_edge in self.sipp_graph[(start_pos,neighbour_pos)].interval_list:
                        if i_edge[0] > late_depart_t or i_edge[1] < early_depart_t:
                            continue
                        # Skip if edge (departure) interval does not overlap arrival window shifted to departure time
                        if i_edge[1] < i[0] - m_time or i_edge[0] > i[1] - m_time:
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
        else:
            raise ValueError(f"Invalid heuristic type: {self.heuristic_type}")
        return dist if self.velocity == 0 else dist / self.velocity

    def compute_plan(self):
        solution_info = {}
        solution = {}
        st = time.time()
        best_solution = None
        best_solution_cost = float('inf')
        best_success = False
        iterations = 0
        for _ in range(self.max_iterations):
            self.shuffle_agents()
            self.reset_graph()
            iterations += 1
            success = True
            total_cost = 0
            for ii, agent in enumerate(self.agents):
                start = tuple(agent["start"])
                goal = tuple(agent["goal"])

                initial_state = State(start, 0, self.sipp_graph[start].interval_list[0])
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
                while open_heap and not goal_reached:
                    _, _, current = heapq.heappop(open_heap)
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
                            if successor.position == goal:
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
        self.total_time += time.time() - st
        self.total_iterations = min(self.max_iterations, iterations)
        solution =best_solution if best_success else {}
        solution_info["runtime"] = self.total_time
        solution_info["total_iterations"] = self.total_iterations
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
