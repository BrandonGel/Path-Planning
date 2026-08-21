"""
AStar search
author: Ashwin Bose (@atb033)
author: Giacomo Lodigiani (@Lodz97)
"""
import heapq
from itertools import count

class AStar:
    def __init__(self, env, max_iterations= -1):
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors
        self.get_step_cost = env.get_step_cost
        self.max_iterations = max_iterations if max_iterations > 0 or max_iterations is None else float("inf")
        self.cost_map = env.cost_map

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name):
        """
        low level search
        """
        initial_state = self.agent_dict[agent_name]["start"]

        closed_set = set()
        counter = count()
        open_heap = []
        open_set = set()

        came_from = {}

        g_score = {} 
        g_score[initial_state] = 0

        f_initial = self.admissible_heuristic(initial_state, agent_name)

        # Initialize open list
        heapq.heappush(open_heap, (f_initial, next(counter), initial_state))
        open_set.add(initial_state)

        iterations = 0
        while open_heap and iterations < self.max_iterations:
            iterations += 1
            # Extract minimum f-score state - O(log n)
            _, _, current = heapq.heappop(open_heap)

            # Skip if we've already processed this state
            # (can happen with duplicate entries in heap)
            if current not in open_set:
                continue

            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current), g_score[current]

            open_set.remove(current)
            closed_set.add(current)

            neighbor_list = self.get_neighbors(current)

            for neighbor in neighbor_list:
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + self.get_step_cost(current, neighbor)

                if neighbor not in open_set:
                    open_set |= {neighbor}
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g_score
                h_score = self.admissible_heuristic(neighbor, agent_name)

                if self.cost_map is None:
                    guidance_cost = 0
                elif neighbor.is_equal_except_time(current):
                    guidance_cost = 1
                elif callable(self.cost_map):
                    guidance_cost = float(self.cost_map(neighbor.location.point))
                else:
                    pt = neighbor.location.point
                    guidance_cost = float(self.cost_map[int(round(pt[0]))][int(round(pt[1]))])

                f_score_neighbor = g_score[neighbor] + h_score + guidance_cost
                heapq.heappush(open_heap, (f_score_neighbor,next(counter), neighbor))
        if iterations >= self.max_iterations:
            print('Low level A* - Maximum iteration reached')
       
        return False, float("inf")