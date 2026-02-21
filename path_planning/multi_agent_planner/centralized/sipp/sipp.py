"""

SIPP implementation  

author: Ashwin Bose (@atb033)

See the article: DOI: 10.1109/ICRA.2011.5980306

"""

from math import fabs
import heapq
from collections import deque
import random

from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippGraph, State
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.node import Node

class SippPlanner(SippGraph):
    def __init__(self, graph_map: GraphSampler,dynamic_obstacles:dict = {},agents:list = [],radius:float = 0.0,velocity:float = 0.0,use_constraint_sweep:bool = True,verbose:bool = False):
        SippGraph.__init__(self, graph_map, dynamic_obstacles,radius,velocity,use_constraint_sweep)
        self.agents = agents
        self.agent_names = [agent["name"] for agent in agents]
        self.verbose = verbose
        self.plan = {}
        self.plan_cost = {}
        self.action_cost = {}
        self.cost = 0
        random.shuffle(self.agents)

    def get_mtime(self, position1, position2):
        m_cost = float(self.graph_map.get_cost(Node(tuple(position1)), Node(tuple(position2))))
        if self.velocity > 0:
            m_time = m_cost / self.velocity
        else:
            m_time = 1.0
        return m_time

    def get_earliest_no_collision_arrival_time(self, start_t, interval, start_pos, neighbour, m_time):
        t = max(start_t, interval[0]) 

        # Check for edge conflicts for point agents
        if self.radius == 0:
            if t == interval[0]:
                collision = False
                for _,obstacle in self.dyn_obstacles.items():
                    for idx in range(len(obstacle)-1):   
                        obs_state = obstacle[idx]
                        obs_t,obs_position = obs_state.time,obs_state.position
                        obs_next_state = obstacle[idx+1]
                        obs_next_t,obs_next_position = obs_next_state.time,obs_next_state.position
                        edge_conflict = (obs_position[0] == neighbour[0] and obs_position[1] == neighbour[1] and
                            start_pos[0] == obs_next_position[0] and start_pos[1] == obs_next_position[1] and
                            obs_t == t-1 and obs_next_t == t)
                        if edge_conflict:
                            collision = True
                            break
                    if collision:
                        t = None
                        break
            return t

        if t >= interval[0] and t <= interval[1]:
            # Departure time for edge traversal (agent may have waited at start_pos)
            depart_t = t - m_time
            overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(start_pos,neighbour,self.velocity,self.radius)
            vertex_indicies = [ index for index in overlapping_vertices.keys() if index != start_pos and index != neighbour]
            # Exclude only the direct edge from start_pos to neighbour; all other
            # overlapping edges must be safe during the traversal.
            edge_indicies = [ index for index in overlapping_edges.keys()
                              if not (index[0] == start_pos and index[1] == neighbour)]
            for vertex_index in vertex_indicies:
                if not self.sipp_graph[vertex_index].is_in_safe_interval(t):
                    return None
            for edge_index in edge_indicies:
                # Check safety at departure time; the traversal interval
                # [depart_t, t] must lie within a safe edge interval.
                if not self.sipp_graph[edge_index].is_in_safe_interval(depart_t, t):
                    return None
            return t
        return None

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

                # Get the earliest no collision arrival time
                t = self.get_earliest_no_collision_arrival_time(start_t, i, start_pos, neighbour_pos, m_time)
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
        dist =  fabs(position[0] - goal[0]) + fabs(position[1]-goal[1])
        return dist if self.velocity == 0 else dist / self.velocity

    def compute_plan(self):
        self.cost = 0
        for ii, agent in enumerate(self.agents):
            start = tuple(agent["start"])
            goal = tuple(agent["goal"])

            # Min-heap: (f, tie_break, state); tie_break ensures we never compare State objects
            OPEN = []
            tie_break = 0
            closed = set()  # (position, interval) already expanded
            g_values  = {}      # (position, interval) -> g
            parents   = {}      # (position, interval) -> State
            action_cost = {}    # (position, interval) -> (wait_cost, move_cost)
            
            s_start = State(start, 0, self.sipp_graph[start].interval_list[0])
            start_key = (start, self.sipp_graph[start].interval_list[0])
            g_values[start_key] = 0.0
            f_start = self.get_heuristic(start, goal)
            heapq.heappush(OPEN, (f_start, tie_break, s_start))
            tie_break += 1

            goal_reached = False
            goal_state   = None
            goal_cost = float('inf')
            while OPEN and not goal_reached:
                _, _, s = heapq.heappop(OPEN)
                state_key = (s.position, s.interval)
                if state_key in closed:
                    continue
                closed.add(state_key)

                successors, costs, time_takens = self.get_successors(s)

                for cost, time_taken, successor in zip(costs, time_takens, successors):
                    succ_key = (successor.position, successor.interval)
                    if succ_key in closed:
                        continue

                    new_g = g_values.get(state_key, float('inf')) + cost
                    if new_g < g_values.get(succ_key, float('inf')):
                        g_values[succ_key] = new_g
                        parents[succ_key]  = s
                        action_cost[succ_key] = time_taken
                        if successor.position == goal:
                            if self.verbose:
                                print("Plan successfully calculated!!")
                            goal_reached = True
                            goal_state = successor
                            goal_cost = successor.time
                            break

                        f = new_g + self.get_heuristic(successor.position, goal)
                        heapq.heappush(OPEN, (f, tie_break, successor))
                        tie_break += 1

            if not goal_reached or goal_state is None:
                return {}

            # Backtrack using (position, interval) keys
            plan = deque()
            plan_action_cost = deque()
            current = goal_state
            current_action_cost = (0,0)
            while True:
                plan.appendleft(current)
                plan_action_cost.appendleft(current_action_cost)
                if current.position == start and current.time == 0:
                    break
                key = (current.position, current.interval)
                
                current = parents[key]
                current_action_cost = action_cost[key]
            self.plan[agent["name"]] = plan
            self.plan_cost[agent["name"]] = goal_cost
            self.action_cost[agent["name"]] = plan_action_cost
            self.update_intervals([plan],[agent["name"]])
        self.cost = sum(self.plan_cost.values())
        return self.get_plan()
            
    def get_plan(self):
        solution = {}
        for agent in self.agent_names:
            plan = self.plan[agent]
            action_cost = self.action_cost[agent]
            path_list = []
            if self.radius == 0:
                setpoint = plan[0]
                temp_dict = {"x":setpoint.position[0], "y":setpoint.position[1], "t":setpoint.time}
                path_list.append(temp_dict)

                for i in range(len(plan)-1):
                    for j in range(int(plan[i+1].time - plan[i].time-1)):
                        x = plan[i].position[0]
                        y = plan[i].position[1]
                        t = plan[i].time
                        setpoint = plan[i]
                        temp_dict = {"x":x, "y":y, "t":t+j+1}
                        path_list.append(temp_dict)
                    setpoint = plan[i+1]
                    temp_dict = {"x":setpoint.position[0], "y":setpoint.position[1], "t":setpoint.time}
                    path_list.append(temp_dict)
            else:
                for action,state in zip(action_cost,plan):
                    temp_dict = {"x":state.position[0], "y":state.position[1], "t":state.time}
                    path_list.append(temp_dict)

                    action_wait_cost, action_move_cost = action
                    if action_wait_cost > 1e-10:
                        temp_dict = {"x":state.position[0], "y":state.position[1], "t":state.time+action_wait_cost}
                        path_list.append(temp_dict)
            solution[agent] = path_list
        return solution

    def compute_solution_cost(self, solution = None):
        if solution is None:
            return self.cost
        cost = 0
        for agent, path in solution.items():
            cost += float(path[-1]["t"]) - float(path[0]["t"])
        return cost
        