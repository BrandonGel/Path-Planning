"""
The low-level A* search of CBS (and, through a subclass, ICBS).

author: Ashwin Bose (@atb033)
"""
import heapq
from itertools import count


class AStar:
    """A* over an agent's time-expanded states under the environment's current constraints.

    ``max_iterations``: the search budget (None or <= 0: none). It counts heap pops, stale ones included;
    with ``count_expanded`` (ICBS) it counts expanded states instead."""

    count_expanded = False

    def __init__(self, env, max_iterations=-1):
        self.env = env
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors
        self.get_step_cost = env.get_step_cost
        self.max_iterations = max_iterations if max_iterations is not None and max_iterations > 0 else float("inf")

    @staticmethod
    def reconstruct_path(came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name, solution=None, cost_limit=None):
        """The agent's cheapest path: ``(states, cost)``, or ``(False, inf)``.

        - ``solution``: the other agents' paths; among equally good states, the one whose path so far
          conflicts with fewer of them is expanded first (``_count_conflicts``; 0 here).
        - ``cost_limit``: only look for a path of at most this cost. A state whose cost so far plus the
          heuristic exceeds it is dropped (the heuristic never overestimates, so no such path goes through
          it), which makes a search that cannot succeed end early."""
        self.env.current_agent = agent_name                      # get_neighbors turns at this agent's rate
        initial_state = self.agent_dict[agent_name]["start"]
        closed_set, counter, came_from = set(), count(), {}
        g_score = {initial_state: 0}
        conflicts = {initial_state: 0}                           # conflicts along the path so far (tie-break)
        # Heap entries: (f, conflicts so far, push order, state).
        open_heap = [(self.admissible_heuristic(initial_state, agent_name), 0, next(counter), initial_state)]
        iterations = 0
        while open_heap and (len(closed_set) if self.count_expanded else iterations) < self.max_iterations:
            iterations += 1
            current = heapq.heappop(open_heap)[3]
            if current in closed_set:                            # a stale heap entry
                continue
            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current), g_score[current]
            closed_set.add(current)
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                tentative_g_score = g_score[current] + self.get_step_cost(current, neighbor)
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    f_score = tentative_g_score + self.admissible_heuristic(neighbor, agent_name)
                    if cost_limit is not None and f_score > cost_limit + 1e-6:
                        continue
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    conflicts[neighbor] = conflicts[current] + (
                        self._count_conflicts(agent_name, neighbor, current, solution) if solution else 0)
                    heapq.heappush(open_heap, (f_score, conflicts[neighbor], next(counter), neighbor))
        if not self.count_expanded and iterations >= self.max_iterations:
            print('Low level A* - Maximum iteration reached')
        return False, float("inf")

    def _count_conflicts(self, agent_name, neighbor, current, solution) -> int:
        """How many conflicts the step ``current -> neighbor`` has with the other agents' paths."""
        return 0
