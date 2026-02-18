"""Scatter construction for PIBT-based MAPF solvers.

This module provides a Python adaptation of the original C++ `Scatter`
builder used in LaCAM*. It performs iterative, collision-aware A* path
planning for each agent to generate *scatter hints* – preferred next
vertices – that guide PIBT toward low-collision motion.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.multi_agent_planner.centralized.lacam.dist_table import DistTable
from path_planning.multi_agent_planner.centralized.lacam.collision_table import (
    CollisionTable,
)
from path_planning.multi_agent_planner.centralized.lacam.utility import (
    Coord,
    Config,
    Deadline,
    get_makespan_lower_bound,
    get_neighbors,
)


@dataclass
class Scatter:
    """Scatter hint generator using collision-aware path planning.

    This class mirrors the behavior of the original C++ implementation
    (`scatter.cpp`). It iteratively plans single-agent paths with A*,
    using a global collision table to bias away from conflicts, and
    then extracts per-agent local hints:

        scatter_data[i][current_coord] -> prioritized_next_coord
    """

    graph_map: GraphSampler
    starts: Config
    goals: Config
    dist_tables: List[DistTable]
    deadline: Deadline
    seed: int = 0
    verbose: int = 0
    cost_margin: int = 0

    # Computed after construct()
    scatter_data: List[Dict[Coord, Coord]] = field(init=False)
    paths: List[List[Coord]] = field(init=False)
    collision_table: CollisionTable = field(init=False)

    def __post_init__(self) -> None:
        self.N = len(self.starts)
        self.rng = np.random.default_rng(self.seed)
        T = get_makespan_lower_bound(
            self.graph_map, self.starts, self.goals, self.dist_tables
        ) + self.cost_margin
        self.T = T
        self.paths = [[] for _ in range(self.N)]
        self.scatter_data = [{} for _ in range(self.N)]
        self.collision_table = CollisionTable(
            self.graph_map, self.N, self.T + 10
        )

    def construct(self) -> None:
        """Main scatter construction via iterative collision-aware A*."""
        collision_cnt_last = 0
        paths_prev: List[List[Coord]] = [[] for _ in range(self.N)]

        loop = 0
        while loop < 2 or self.collision_table.collision_cnt < collision_cnt_last:
            loop += 1
            collision_cnt_last = self.collision_table.collision_cnt

            # Randomize planning order
            order = list(range(self.N))
            self.rng.shuffle(order)

            # Plan for each agent
            for _i in range(self.N):
                if self.deadline.is_expired:
                    break

                i = order[_i]
                cost_ub = self.dist_tables[i].get(self.starts[i]) + self.cost_margin

                # Clear old path from collision table
                if self.paths[i]:
                    self.collision_table.clearPath(i, self.paths[i])

                # Run collision-aware A* for agent i
                self.paths[i] = self._plan_single_agent(i, cost_ub)

                # Register new path
                self.collision_table.enrollPath(i, self.paths[i])

            paths_prev = [p[:] for p in self.paths]

            if self.collision_table.collision_cnt == 0:
                break
            if self.deadline.is_expired:
                break

        # Use the last best set of paths
        self.paths = paths_prev

        # Build scatter_data from final paths
        for i in range(self.N):
            if not self.paths[i]:
                continue
            for t in range(len(self.paths[i]) - 1):
                self.scatter_data[i][self.paths[i][t]] = self.paths[i][t + 1]

    def _plan_single_agent(self, agent_id: int, cost_ub: int) -> List[Coord]:
        """Run collision-aware A* for a single agent.

        Returns:
            Path from start to goal as list of coordinates. Returns []
            if no path is found within the given cost bound or deadline.
        """
        s_i = self.starts[agent_id]
        g_i = self.goals[agent_id]

        # A* node: (collision_cost, f_score, g, coord, parent)
        # Priority: collision_cost asc, then f_score asc
        OPEN: list[tuple[int, int, int, Coord, Coord | None]] = []
        heapq.heappush(
            OPEN,
            (
                0,  # collision_cost
                self.dist_tables[agent_id].get(s_i),  # f = g + h
                0,  # g
                s_i,  # coord
                None,  # parent
            ),
        )

        CLOSED: Dict[Coord, Coord | None] = {}

        while OPEN and not self.deadline.is_expired:
            collision_cost, f_val, g, v, parent = heapq.heappop(OPEN)

            if v in CLOSED:
                continue
            CLOSED[v] = parent

            # Goal check
            if v == g_i:
                # Backtrack
                path: List[Coord] = []
                curr: Coord | None = v
                while curr is not None:
                    path.append(curr)
                    curr = CLOSED.get(curr)
                path.reverse()
                return path

            # Expand neighbors
            for u in get_neighbors(self.graph_map, v):
                d_u = self.dist_tables[agent_id].get(u)
                if u != s_i and u not in CLOSED and d_u + g + 1 <= cost_ub:
                    # Collision cost from moving v -> u at timestep g
                    c_cost = self.collision_table.getCollisionCost(v, u, g)
                    heapq.heappush(
                        OPEN,
                        (
                            collision_cost + c_cost,
                            g + 1 + d_u,
                            g + 1,
                            u,
                            v,
                        ),
                    )

        # No path found within constraints
        return []

