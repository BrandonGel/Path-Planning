"""ICBS's low-level search: CBS's A* with a conflict-aware tie-break and a budget on expanded states."""
from path_planning.multi_agent_planner.centralized.cbs.a_star import AStar as _AStar


class AStar(_AStar):
    count_expanded = True

    def _count_conflicts(self, agent_name, neighbor, current, solution) -> int:
        """Point agents: the vertex and swap conflicts of the step ``current -> neighbor`` with the other
        agents' paths. (With bodies every neighbour already obeys the constraints, so there is nothing to
        count.)"""
        if self.env.has_bodies:
            return 0
        count = 0
        get_state = self.env.get_state
        for other_agent in solution:
            if other_agent == agent_name:
                continue
            other_state = get_state(other_agent, solution, neighbor.time)
            if neighbor.location == other_state.location:        # the same vertex
                count += 1
            prev_other = get_state(other_agent, solution, current.time)
            if neighbor.location == prev_other.location and current.location == other_state.location:   # a swap
                count += 1
        return count
