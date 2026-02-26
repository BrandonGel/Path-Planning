from path_planning.multi_agent_planner.centralized.sipp.graph_generation import State
from path_planning.common.environment.node import Node
from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner as SippBaseEnvironment
import heapq
from collections import deque
import numpy as np

class SIPP():
    def __init__(self, env:SippBaseEnvironment, max_iterations=-1,verbose=False):
        self.env = env # Access to get_conflicts logic
        self.agent_dict = env.agent_dict
        self.get_heuristic = env.get_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_successors = env.get_successors
        self.get_initial_state = env.get_initial_state
        self.max_iterations = max_iterations if max_iterations > 0 or max_iterations is None else float("inf")
        self.verbose = verbose

    def reconstruct_path(self, came_from, came_from_action_cost, current):
        total_path = deque([current])
        total_path_action_cost = deque([(0, 0)])
        key = (current.position, current.interval)
        while key in came_from:
            current = came_from[key]
            current_action = came_from_action_cost[key]
            total_path.appendleft(current)
            total_path_action_cost.appendleft(current_action)
            key = (current.position, current.interval)
        return list(total_path), list(total_path_action_cost)

    def search(self, agent_name, solution=None,solution_action_cost=None):
        """
        Modified low level search with conflict-aware tie-breaking.
        """
        goal_state = self.agent_dict[agent_name]["goal"]
        initial_state = self.get_initial_state(self.agent_dict[agent_name]["start"].position)
        initial_state_key = (initial_state.position, initial_state.interval)

        # Min-heap: (f, tie_break, state); tie_break ensures we never compare State objects
        open_heap = []
        counter = 0
        closed_set = set()  # (position, interval) already expanded
        g_score  = {initial_state_key:0.0}      # (position, interval) -> g
        came_from   = {}      # (position, interval) -> State
        came_from_action_cost = {}    # (position, interval) -> (wait_cost, move_cost)
        conflicts_count = {initial_state_key:0}
        
        f_start = self.get_heuristic(initial_state.position, goal_state.position)
        heapq.heappush(open_heap, (f_start, conflicts_count[initial_state_key], counter, initial_state))
        counter += 1

        while open_heap and len(closed_set) < self.max_iterations:
            _, _, _, current = heapq.heappop(open_heap)
            current_state_key = (current.position, current.interval)
            if current_state_key in closed_set:
                continue

            if self.env.is_at_goal(current, goal_state):
                if self.verbose:
                    print("Plan successfully calculated!!")
                goal_cost = current.time
                plan,plan_action_cost = self.reconstruct_path(came_from,came_from_action_cost,current)
                return plan,plan_action_cost,goal_cost

            closed_set.add(current_state_key)
            successors, costs, time_takens = self.get_successors(current)

            for cost, time_taken, successor in zip(costs, time_takens, successors):
                succ_key = (successor.position, successor.interval)
                if succ_key in closed_set:
                    continue
                
                tentative_g_score = g_score[current_state_key] + cost

                num_conflicts= 0
                if solution and solution_action_cost and len(solution) > 0:
                    num_conflicts = self._count_conflicts(agent_name, successor, current, solution,solution_action_cost)

                if tentative_g_score < g_score.get(succ_key, float('inf')):
                    came_from[succ_key] = current
                    g_score[succ_key] = tentative_g_score
                    came_from_action_cost[succ_key] = time_taken
                    conflicts_count[succ_key] = conflicts_count[current_state_key] + num_conflicts
                    f_score = g_score[succ_key] + self.get_heuristic(successor.position, goal_state.position)
                    heapq.heappush(open_heap, (f_score, conflicts_count[succ_key], counter, successor))
                    counter += 1
        return False,False, float("inf")

    def _count_conflicts(self, agent_name, successor, current, solution,solution_action_cost):
        count = 0
        # for other_agent, path, action_cost in zip(solution.items(),solution_action_cost.items()):
        #     if other_agent == agent_name: continue
        #     position_1a = current.position
        #     position_1b = successor.position
        #     velocity = (np.array(position_1b) - np.array(position_1a))/
        #     velocity1 = current.time
        #     agent_2_position = successor.time
        #     agent_2_velocity = successor.position
        #     tdur = successor.time - current.time
        #     res = self.env.check_collision(self, position_1a,velocity1, position_2a,velocity2,tdur)
        #     if res:
        #         count += 1
        return count