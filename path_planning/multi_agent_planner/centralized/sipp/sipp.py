"""

SIPP implementation  

author: Ashwin Bose (@atb033)

See the article: DOI: 10.1109/ICRA.2011.5980306

"""

from math import fabs
from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippGraph, State
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.node import Node
from collections import deque
import random

class SippPlanner(SippGraph):
    def __init__(self, graph_map: GraphSampler,dynamic_obstacles:dict = {},agents:list = [],radius:float = 0.0,velocity:float = 0.0,use_constraint_sweep:bool = True,verbose:bool = False):
        SippGraph.__init__(self, graph_map, dynamic_obstacles,radius,velocity,use_constraint_sweep)
        self.agents = agents
        self.verbose = verbose
        self.plan = {}
        self.plan_cost = {}
        self.cost = 0
        random.shuffle(self.agents)

    def get_mtime(self, position1, position2):
        m_cost = float(self.graph_map.get_cost(Node(tuple(position1)), Node(tuple(position2))))
        if self.velocity > 0:
            m_time = m_cost / self.velocity
        else:
            m_time = 1.0
        return m_time

    def get_earliest_no_collision_arrival_time(self, start_t, interval,start_pos,neighbour):
        t = max(start_t, interval[0]) 

        # Check for edge conflicts for point agents
        if self.radius == 0:
            if t == interval[0]:
                collision = False
                for _,obstacle in self.dyn_obstacles.items():
                    for location in obstacle:   
                        if location["x"] == neighbour[0] and location["y"] == neighbour[1] and location["t"] == t-1:
                            collision = True
                            break
                    if collision:
                        t = None
                        break
            return t

        if t >= interval[0] and t <= interval[1]:
            overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(start_pos,neighbour,self.velocity,self.radius)
            vertex_indicies = [ index for index in overlapping_vertices.keys() if index != start_pos and index != neighbour]
            edge_indicies = [ index for index in overlapping_edges.keys() if index[0] != start_pos and index[1] != neighbour]
            for vertex_index in vertex_indicies:
                if not self.sipp_graph[vertex_index].is_in_safe_interval(t):
                    return None
            for edge_index in edge_indicies:
                if not self.sipp_graph[edge_index].is_in_safe_interval(t):
                    return None
            return t
        return None

    def get_successors(self, state):
        successors = []
        costs = []
        neighbour_list = self.get_valid_neighbours(state.position)
        for neighbour_pos in neighbour_list:
            m_time = self.get_mtime(state.position, neighbour_pos)
            start_pos = state.position
            start_t = state.time + m_time  # Earliest possible arrival time
            end_t = state.interval[1] + m_time #Latest possible arrival time
            edge_position = (start_pos, neighbour_pos)
            self.sipp_graph[edge_position]
            
            
            for i in self.sipp_graph[neighbour_pos].interval_list:
                # If the interval is outside the possible arrival time, skip the interval
                if i[0] > end_t or i[1] < start_t:
                    continue

                # Get the earliest no collision arrival time
                t = self.get_earliest_no_collision_arrival_time(start_t, i,start_pos, neighbour_pos)
                if t is None: # Any collision, skip the interval
                    continue
                
                # Create the successor state
                s = State(neighbour_pos, t, i)
                successors.append(s)
                costs.append(m_time)
        return successors, costs

    def get_heuristic(self, position,goal):
        dist =  fabs(position[0] - goal[0]) + fabs(position[1]-goal[1])
        return dist if self.velocity == 0 else dist / self.velocity

    def compute_plan(self):
        self.cost = 0
        for ii,agent in enumerate(self.agents):
            start = tuple(agent["start"])
            goal = tuple(agent["goal"])
            OPEN = []
            goal_reached = False
            s_start = State(start, 0) 

            self.sipp_graph[start].g = 0.
            f_start = self.get_heuristic(start,goal)
            self.sipp_graph[start].f = f_start

            OPEN.append((f_start, s_start))

            while (not goal_reached):
                if not OPEN: 
                    # Plan not found
                    return {}
                s = OPEN.pop(0)[1]
                successors, costs = self.get_successors(s)
        
                for cost, successor in zip(costs, successors):
                    if self.sipp_graph[successor.position].g > self.sipp_graph[s.position].g + cost:
                        self.sipp_graph[successor.position].g = self.sipp_graph[s.position].g + cost
                        self.sipp_graph[successor.position].parent_state = s

                        if successor.position == goal:
                            if self.verbose:
                                print("Plan successfully calculated!!")
                            goal_reached = True
                            break

                        self.sipp_graph[successor.position].f = self.sipp_graph[successor.position].g + self.get_heuristic(successor.position,goal)
                        OPEN.append((self.sipp_graph[successor.position].f, successor))

            # Tracking back
            start_reached = False
            plan = deque()
            current = successor
            while not start_reached:
                plan.appendleft(current)
                if current.position == start:
                    start_reached = True
                current = self.sipp_graph[current.position].parent_state
            self.plan[agent["name"]] = plan
            self.plan_cost[agent["name"]] = self.sipp_graph[successor.position].g 
            self.clear_sipp_graph_values(no_clear_interval=True)
            self.update_intervals([plan])
        self.cost = sum(self.plan_cost.values())
        return self.get_plan()
            
    def get_plan(self):
        solution = {}
        for agent, plan in self.plan.items():
            path_list = []
            for state in plan:
                temp_dict = {"x":state.position[0], "y":state.position[1], "t":state.time}
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