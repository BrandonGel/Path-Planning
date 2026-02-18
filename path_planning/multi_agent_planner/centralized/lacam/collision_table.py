"""Collision table for tracking space-time conflicts between agent paths.

This is a coordinate-based Python adaptation of the original C++
`CollisionTable` used in LaCAM*/PIBT. It keeps track of which agents
occupy which vertices at which timesteps and provides:

* A global collision counter `collision_cnt`
* Incremental updates when paths are enrolled/cleared
* Local collision cost queries for candidate moves
"""

from __future__ import annotations

from typing import Dict, List

from path_planning.multi_agent_planner.centralized.lacam.utility import Coord


class CollisionTable:
    """Tracks space-time conflicts between agent paths.

    Attributes:
        collision_cnt: Global count of collisions over all registered paths.
        body: Mapping from vertex coordinate to a list over timesteps, where
            each entry is a list of agent indices occupying that vertex.
        body_last: Mapping from vertex coordinate to a list of final timesteps
            at which agents terminate at that vertex.
    """

    def __init__(self, graph_map, num_agents: int, max_timestep: int) -> None:
        # The graph_map and max_timestep are kept for API parity but are not
        # strictly required in this coordinate-based implementation.
        self.collision_cnt: int = 0
        self.N: int = num_agents
        self.body: Dict[Coord, List[List[int]]] = {}
        self.body_last: Dict[Coord, List[int]] = {}

    def getCollisionCost(self, v_from: Coord, v_to: Coord, t_from: int) -> int:
        """Return collision cost for moving from v_from to v_to at timestep t_from.

        This mirrors the C++ logic:

        - Vertex collision: number of agents already at v_to at t_to = t_from + 1
        - Edge collision: agents swapping positions along v_from <-> v_to
        - Goal collision: finished agents at v_to whose final timestep is < t_to
        """
        t_to = t_from + 1
        collision = 0

        # Vertex collision
        entry_to = self.body.get(v_to)
        if entry_to is not None and t_to < len(entry_to):
            collision += len(entry_to[t_to])

        # Edge collision
        entry_from = self.body.get(v_from)
        if (
            entry_from is not None
            and t_to < len(entry_from)
            and entry_to is not None
            and t_from < len(entry_to)
        ):
            for j in entry_from[t_to]:
                for k in entry_to[t_from]:
                    if j == k:
                        collision += 1

        # Goal collision
        last_list = self.body_last.get(v_to)
        if last_list is not None:
            for last_timestep in last_list:
                if t_to > last_timestep:
                    collision += 1

        return collision

    def enrollPath(self, agent_id: int, path: List[Coord]) -> None:
        """Register an agent's path and update the global collision count.

        Follows the C++ implementation:

        - For each timestep t, update `collision_cnt` using getCollisionCost
          on the edge (path[t-1] -> path[t]).
        - Register agent_id at (coord, t) in the body table.
        - Track final timestep in body_last for the goal vertex and account
          for collisions with agents arriving in later timesteps.
        """
        if not path:
            return

        T_i = len(path) - 1

        for t, v in enumerate(path):
            # Update collision count for the edge from t-1 -> t
            if t > 0:
                self.collision_cnt += self.getCollisionCost(path[t - 1], path[t], t - 1)

            # Register in body table
            entry = self.body.setdefault(v, [])
            while len(entry) <= t:
                entry.append([])
            entry[t].append(agent_id)

        # Goal handling
        goal_v = path[-1]
        last_list = self.body_last.setdefault(goal_v, [])
        last_list.append(T_i)

        # Account for collisions with agents that arrive at the same goal later
        entry_goal = self.body.get(goal_v)
        if entry_goal is not None:
            for t in range(T_i + 1, len(entry_goal)):
                self.collision_cnt += len(entry_goal[t])

    def clearPath(self, agent_id: int, path: List[Coord]) -> None:
        """Remove an agent's path from the collision table.

        This is the inverse of enrollPath():

        - Remove agent_id from each (coord, t) in the body table.
        - Decrease `collision_cnt` using getCollisionCost on each edge.
        - Remove the final timestep from body_last and update future goal
          collisions accordingly.
        """
        if not path:
            return

        T_i = len(path) - 1

        for t, v in enumerate(path):
            # Remove from body table
            entry = self.body.get(v)
            if entry is not None and t < len(entry):
                agents_here = entry[t]
                if agent_id in agents_here:
                    agents_here.remove(agent_id)

            # Update collision count for the edge from t-1 -> t
            if t > 0:
                self.collision_cnt -= self.getCollisionCost(path[t - 1], path[t], t - 1)

        # Goal handling
        goal_v = path[-1]
        last_list = self.body_last.get(goal_v)
        if last_list is not None and T_i in last_list:
            last_list.remove(T_i)

        # Remove future goal collisions
        entry_goal = self.body.get(goal_v)
        if entry_goal is not None:
            for t in range(T_i + 1, len(entry_goal)):
                self.collision_cnt -= len(entry_goal[t])

