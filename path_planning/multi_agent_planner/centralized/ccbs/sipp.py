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

    def _build_segments(self, plan, action_cost):
        """
        Build piecewise-constant-velocity segments from a SIPP plan and its
        per-step (wait_time, move_time) action costs.

        Returns a list of tuples:
        (t_start, t_end, pos_start (np.array), vel (np.array))
        """
        segments = []
        if not plan or not action_cost or len(plan) < 2:
            return segments

        # Actions are defined per step; we ignore any trailing action element
        # beyond len(plan) - 1 for safety.
        n_steps = min(len(plan) - 1, len(action_cost))
        for i in range(n_steps):
            state_i = plan[i]
            state_next = plan[i + 1]
            t0 = float(state_i.time)
            pos0 = np.asarray(state_i.position, dtype=float)
            wait_time, move_time = action_cost[i]
            wait_time = float(wait_time)
            move_time = float(move_time)

            # Wait segment: agent stays at state_i.position
            if wait_time > 0.0:
                t1 = t0 + wait_time
                if t1 > t0:
                    segments.append(
                        (t0, t1, pos0, np.zeros_like(pos0))
                    )
                t0 = t1  # advance start time for potential move

            # Move segment: constant velocity from state_i to state_next
            if move_time > 0.0:
                t1 = t0 + move_time
                if t1 > t0:
                    pos1 = np.asarray(state_next.position, dtype=float)
                    vel = (pos1 - pos0) / move_time
                    segments.append((t0, t1, pos0, vel))

        return segments

    def _count_conflicts(self, agent_name, successor, current, solution,solution_action_cost):
        """
        Count how many other agents' current paths would collide with the
        candidate transition current -> successor.

        Uses the continuous-time Environment.check_collision primitive by
        decomposing both agents' motions into piecewise-constant-velocity
        segments and checking overlapping time windows.
        """
        if not solution or not solution_action_cost:
            return 0

        # Build segments for the candidate agent's single transition.
        cur_state = current
        succ_state = successor
        t0 = float(cur_state.time)
        t1 = float(succ_state.time)
        dt = t1 - t0
        if dt <= 0.0:
            return 0

        # Derive wait/move decomposition from geometry and SIPP's travel time.
        move_time = float(self.env.get_mtime(cur_state.position, succ_state.position))
        if move_time < 0.0:
            move_time = 0.0
        if move_time > dt:
            # Clamp in case of small numerical inconsistencies
            move_time = dt
        wait_time = max(0.0, dt - move_time)

        candidate_plan = [
            State(cur_state.position, t0, cur_state.interval),
            State(succ_state.position, t1, succ_state.interval),
        ]
        candidate_actions = [(wait_time, move_time)]
        cand_segments = self._build_segments(candidate_plan, candidate_actions)
        if not cand_segments:
            return 0

        count = 0
        # Iterate over all other agents' piecewise-constant-velocity segments.
        for other_name, other_plan in solution.items():
            if other_name == agent_name:
                continue
            other_actions = solution_action_cost.get(other_name)
            if not other_plan or not other_actions:
                continue

            other_segments = self._build_segments(other_plan, other_actions)
            if not other_segments:
                continue

            found_for_this_agent = False
            for t0_cand, t1_cand, pos1_0, vel1 in cand_segments:
                for t0_other, t1_other, pos2_0, vel2 in other_segments:
                    # Overlapping global time window
                    t_start = max(t0_cand, t0_other)
                    t_end = min(t1_cand, t1_other)
                    tdur = t_end - t_start
                    if tdur <= 0.0:
                        continue

                    # Positions of both agents at t_start
                    dt1 = t_start - t0_cand
                    dt2 = t_start - t0_other
                    pos1_start = pos1_0 + vel1 * dt1
                    pos2_start = pos2_0 + vel2 * dt2

                    res = self.env.check_collision(
                        pos1_start, vel1, pos2_start, vel2, tdur
                    )
                    if res:
                        count += 1
                        found_for_this_agent = True
                        break
                if found_for_this_agent:
                    break

        return count

    def _count__conflicts(self, *args, **kwargs):
        """
        Backwards-compatible alias for _count_conflicts with the double-underscore
        name requested by the user.
        """
        return self._count_conflicts(*args, **kwargs)