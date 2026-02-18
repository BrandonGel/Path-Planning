"""

SIPP implementation  

author: Ashwin Bose (@atb033)

See the article: DOI: 10.1109/ICRA.2011.5980306

"""

import argparse
import yaml
from math import fabs
from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippGraph, State
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.node import Node

class SippPlanner(SippGraph):
    def __init__(self, graph_map: GraphSampler,dynamic_obstacles:dict = {},start:tuple = None,goal:tuple = None,name:str = None,radius:float = 0.0,velocity:float = 0.0,use_constraint_sweep:bool = True,verbose:bool = False):
        SippGraph.__init__(self, graph_map, dynamic_obstacles,radius,velocity,use_constraint_sweep)
        self.open = []
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.name = name
        self.verbose = verbose

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

    def get_heuristic(self, position):
        dist =  fabs(position[0] - self.goal[0]) + fabs(position[1]-self.goal[1])
        return dist if self.velocity == 0 else dist / self.velocity

    def compute_plan(self):
        self.open = []
        goal_reached = False

        s_start = State(self.start, 0) 

        self.sipp_graph[self.start].g = 0.
        f_start = self.get_heuristic(self.start)
        self.sipp_graph[self.start].f = f_start

        self.open.append((f_start, s_start))

        while (not goal_reached):
            if not self.open: 
                # Plan not found
                return 0
            s = self.open.pop(0)[1]
            successors, costs = self.get_successors(s)
    
            for cost, successor in zip(costs, successors):
                if self.sipp_graph[successor.position].g > self.sipp_graph[s.position].g + cost:
                    self.sipp_graph[successor.position].g = self.sipp_graph[s.position].g + cost
                    self.sipp_graph[successor.position].parent_state = s

                    if successor.position == self.goal:
                        if self.verbose:
                            print("Plan successfully calculated!!")
                        goal_reached = True
                        break

                    self.sipp_graph[successor.position].f = self.sipp_graph[successor.position].g + self.get_heuristic(successor.position)
                    self.open.append((self.sipp_graph[successor.position].f, successor))

        # Tracking back
        start_reached = False
        self.plan = []
        current = successor
        while not start_reached:
            self.plan.insert(0,current)
            if current.position == self.start:
                start_reached = True
            current = self.sipp_graph[current.position].parent_state
        return 1
            
    def get_plan(self):
        path_list = []

        for i in range(len(self.plan)):
            setpoint = self.plan[i]
            temp_dict = {"x":setpoint.position[0], "y":setpoint.position[1], "t":setpoint.time}
            path_list.append(temp_dict)

        data = {self.name:path_list}
        return data

    def get_cost(self):
        cost = 0
        for i in range(len(self.plan)-1):
            cost += self.get_mtime(self.plan[i].position, self.plan[i+1].position)
        return cost