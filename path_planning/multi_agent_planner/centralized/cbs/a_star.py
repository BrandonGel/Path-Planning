"""

AStar search

author: Ashwin Bose (@atb033)

"""
import heapq
class AStar():
    def __init__(self, env):
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors
        # Internal counter used as a deterministic tie-breaker in the
        # priority queue when f-scores are equal.
        self._counter = 0

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
        step_cost = 1
        

        closed_set = set()
        # Use a heap-based priority queue for deterministic selection of the
        # next node to expand instead of a raw set whose iteration order can
        # vary. Each heap entry is (f_score, counter, state).
        open_heap = []
        open_set = set()

        came_from = {}

        g_score = {} 
        g_score[initial_state] = 0

        f_score = {} 

        f_score[initial_state] = self.admissible_heuristic(initial_state, agent_name)

        # Initialize open list
        heapq.heappush(open_heap, (f_score[initial_state], self._counter, initial_state))
        self._counter += 1
        open_set.add(initial_state)

        while open_heap:
            # Pop the best node; skip any stale entries no longer in open_set.
            _, _, current = heapq.heappop(open_heap)
            if current not in open_set:
                continue

            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current)

            open_set -= {current}
            closed_set |= {current}

            neighbor_list = self.get_neighbors(current)

            for neighbor in neighbor_list:
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score.setdefault(current, float("inf")) + step_cost

                if neighbor not in open_set or tentative_g_score < g_score.setdefault(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(neighbor, agent_name)

                    if neighbor not in open_set:
                        open_set.add(neighbor)

                    # Push (or push updated) entry into the heap. Stale entries
                    # for this neighbor will simply be ignored when popped.
                    heapq.heappush(open_heap, (f_score[neighbor], self._counter, neighbor))
                    self._counter += 1
        return False

