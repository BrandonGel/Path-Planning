import heapq
from itertools import count

class AStar():
    def __init__(self, env, max_iterations=-1,radius=0.0):
        self.env = env # Access to get_conflicts logic
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors
        self.max_iterations = max_iterations if max_iterations > 0 or max_iterations is None else float("inf")
        self.radius = radius

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name, solution=None):
        """
        Modified low level search with conflict-aware tie-breaking.
        """
        initial_state = self.agent_dict[agent_name]["start"]
        step_cost = 1 # Standard step cost
        
        closed_set = set()
        counter = count()
        open_heap = []
        open_set = set()
        came_from = {}
        g_score = {initial_state: 0}
        
        # Track number of conflicts for tie-breaking
        conflicts_count = {initial_state: 0}

        f_initial = self.admissible_heuristic(initial_state, agent_name)

        # Heap now stores: (f_score, num_conflicts, counter, state)
        heapq.heappush(open_heap, (f_initial, 0, next(counter), initial_state))
        open_set.add(initial_state)

        while open_heap and len(closed_set) < self.max_iterations:
            # Pops the lowest f_score; if tied, pops the lowest num_conflicts
            _, _, _, current = heapq.heappop(open_heap)

            if current not in open_set: continue
            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current), g_score[current]

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set: continue
                
                tentative_g_score = g_score[current] + step_cost

                # Calculate conflicts if a solution/other paths are provided
                num_conflicts = 0
                if solution:
                    num_conflicts = self._count_conflicts(agent_name, neighbor, current, solution)

                if neighbor not in open_set or tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    conflicts_count[neighbor] = conflicts_count[current] + num_conflicts
                    
                    f_score = g_score[neighbor] + self.admissible_heuristic(neighbor, agent_name)
                    
                    if neighbor not in open_set:
                        open_set.add(neighbor)
                    
                    # Tie-break using the cumulative conflict count
                    heapq.heappush(open_heap, (f_score, conflicts_count[neighbor], next(counter), neighbor))
        return False, float("inf")

    def _count_conflicts(self, agent_name, neighbor, current, solution):
        """Helper to check if this move conflicts with any other agent's current path."""
        count = 0
        for other_agent, path in solution.items():
            if other_agent == agent_name: continue
            
            # Check vertex conflict at neighbor.time
            other_state = self.env.get_state(other_agent, solution, neighbor.time)
            if self.radius == 0:
                if neighbor.location == other_state.location:
                    count += 1
                
            # Check edge conflict
            prev_other = self.env.get_state(other_agent, solution, current.time)
            if self.radius == 0:
                if neighbor.location == prev_other.location and current.location == other_state.location:
                    count += 1
            else:
                if not self.env.transition_valid(neighbor, other_state):
                    count += 1
        return count