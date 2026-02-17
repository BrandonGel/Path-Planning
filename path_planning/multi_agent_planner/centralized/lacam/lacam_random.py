
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.multi_agent_planner.centralized.lacam.utility import Coord,is_valid_coord,get_neighbors,Config,Deadline,Configs
from path_planning.multi_agent_planner.centralized.lacam.dist_table import DistTable
from path_planning.multi_agent_planner.centralized.lacam.pibt import PIBT

NO_AGENT: int = np.iinfo(np.int32).max
"""Sentinel value indicating no agent occupies a location."""

NO_LOCATION: Coord = (np.iinfo(np.int32).max, np.iinfo(np.int32).max)
"""Sentinel coordinate indicating an unassigned location."""

@dataclass
class LowLevelNode:
    """Low-level search node representing partial agent assignments.

    In LaCAM*, low-level nodes represent constraints on which agents must move
    to which locations. The low-level tree explores different assignments to 
    generate diverse configurations.

    Attributes:
        who: List of agent IDs with assigned next locations.
        where: List of next locations for the corresponding agents.
        depth: Number of agents with assigned locations (len(who) == len(where)).
    """

    who: list[int] = field(default_factory=lambda: [])
    where: list[Coord] = field(default_factory=lambda: [])
    depth: int = 0

    def get_child(self, who: int, where: Coord) -> LowLevelNode:
        """Create a child node with one additional agent assignment.

        Args:
            who: Agent ID to assign.
            where: Location to assign to the agent.

        Returns:
            New LowLevelNode with the additional assignment.
        """
        return LowLevelNode(
            who=self.who + [who],
            where=self.where + [where],
            depth=self.depth + 1,
        )


@dataclass
class HighLevelNode:
    """High-level search node representing a complete configuration.

    High-level nodes form the main search space, where each node represents
    a configuration (positions of all agents). The search is performed in a 
    depth-first search manner to find solutions quickly.

    Attributes:
        Q: Current configuration (positions of all agents).
        order: Order in which agents are assigned locations in low-level search.
        parent: Parent node in the search tree (for solution reconstruction).
        tree: Low-level search tree for this node (list of constraint nodes).
        g: Actual cost from start to this configuration.
        h: Heuristic estimate of cost from this configuration to goal.
        f: Total estimated cost (g + h).
        neighbors: Set of neighboring configurations generated from this node.
    """

    Q: Config
    order: list[int]
    parent: HighLevelNode | None = None
    tree: deque[LowLevelNode] = field(default_factory=lambda: deque([LowLevelNode()]))
    g: int = 0
    h: int = 0
    f: int = field(init=False)
    neighbors: set[HighLevelNode] = field(default_factory=lambda: set())

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass initialization."""
        self.f = self.g + self.h

    def __eq__(self, other: object) -> bool:
        """Check equality based on configuration.

        Args:
            other: Object to compare with.

        Returns:
            True if other is a HighLevelNode with the same configuration.
        """
        if isinstance(other, HighLevelNode):
            return self.Q == other.Q
        return False

    def __hash__(self) -> int:
        """Compute hash based on configuration for use in sets/dicts.

        Returns:
            Hash value of the configuration.
        """
        return self.Q.__hash__()


class LaCAM:
    """LaCAM* solver for Multi-Agent Path Finding problems.

    LaCAM* is an anytime search-based algorithm that performs a two-level search
    in the configuration space to find collision-free paths for multiple agents.

    Algorithm Overview:
        **High-level search**: Conducts search over configurations (states of all 
        agents). Each configuration is evaluated using f = g + h, where g is the 
        actual cost and h is a heuristic lower bound.

        **Low-level search**: For each high-level configuration, explores movement 
        constraints to generate diverse successor configurations.

    Solution Modes:
        - **Anytime mode (flg_star=True)**: Continues refining the solution
          after finding an initial solution, eventually converging to optimal
          (given sufficient time). This is the default mode.
        - **First-solution mode (flg_star=False)**: Returns immediately after
          finding the first valid solution (suboptimal).

    Optimality Guarantee:
        When run in anytime mode with sufficient time, LaCAM* is **eventually
        optimal** for the sum-of-loss objective.

    """

    def __init__(self) -> None:
        """Initialize the LaCAM* solver."""
        pass

    def solve(
        self,
        graph_map: GraphSampler,
        starts: Config,
        goals: Config,
        time_limit_ms: int = 3000,
        deadline: Deadline | None = None,
        flg_star: bool = True,
        seed: int = 0,
        verbose: int = 1,
    ) -> Configs:
        """Solve a MAPF problem instance.

        Args:
            graph_map: The graph map.
            starts: Starting configuration (initial positions of all agents).
            goals: Goal configuration (target positions of all agents).
            time_limit_ms: Time limit in milliseconds (default: 3000).
            deadline: Optional Deadline object (if None, created from time_limit_ms).
            flg_star: If True, refine solution for optimality (default: True).
                     If False, return first found solution (suboptimal).
            seed: Random seed for tie-breaking and action ordering (default: 0).
            verbose: Verbosity level (0: silent, 1: basic, 2+: detailed) (default: 1).

        Returns:
            List of configurations representing the solution path from starts to goals.
            Returns empty list if no solution found within time limit.
        """
        # set problem
        self.num_agents: int = len(starts)
        self.graph_map: GraphSampler = graph_map
        self.starts: Config = starts
        self.goals: Config = goals
        self.deadline: Deadline = (
            deadline if deadline is not None else Deadline(time_limit_ms)
        )
        # set hyper parameters
        self.flg_star: bool = flg_star
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)
        self.verbose = verbose
        return self._solve()

    def _solve(self) -> Configs:
        """Internal method performing the main LaCAM* search algorithm.

        Returns:
            Solution path as a list of configurations.
        """
        self.info(1, "start solving MAPF")
        # set cache, used for collision check
        self.occupied_from: np.ndarray = np.full(self.graph_map.shape, NO_AGENT, dtype=int)
        self.occupied_to: np.ndarray = np.full(self.graph_map.shape, NO_AGENT, dtype=int)

        # set distance tables
        self.dist_tables: list[DistTable] = [
            DistTable(self.graph_map, goal) for goal in self.goals
        ]

        # set search scheme
        OPEN: deque[HighLevelNode] = deque([])
        EXPLORED: dict[Config, HighLevelNode] = {}
        N_goal: HighLevelNode | None = None

        # set initial node
        Q_init = self.starts
        N_init = HighLevelNode(
            Q=Q_init, order=self.get_order(Q_init), h=self.get_h_value(Q_init)
        )
        OPEN.appendleft(N_init)
        EXPLORED[N_init.Q] = N_init

        # main loop
        while len(OPEN) > 0 and not self.deadline.is_expired:
            N: HighLevelNode = OPEN[0]

            # goal check
            if N_goal is None and N.Q == self.goals:
                N_goal = N
                self.info(1, f"initial solution found, cost={N_goal.g}")
                # no refinement -> terminate
                if not self.flg_star:
                    break

            # lower bound check
            if N_goal is not None and N_goal.g <= N.f:
                OPEN.popleft()
                continue

            # low-level search end
            if len(N.tree) == 0:
                OPEN.popleft()
                continue

            # low-level search
            C: LowLevelNode = N.tree.popleft()  # constraints
            if C.depth < self.num_agents:
                i = N.order[C.depth]
                v = N.Q[i]
                cands = [v] + get_neighbors(self.graph_map, v)
                self.rng.shuffle(cands)
                for u in cands:
                    N.tree.append(C.get_child(i, u))

            # generate the next configuration
            Q_to = self.configuration_generator(N, C)
            if Q_to is None:
                # invalid configuration
                continue
            elif Q_to in EXPLORED.keys():
                # known configuration
                N_known = EXPLORED[Q_to]
                N.neighbors.add(N_known)
                OPEN.appendleft(N_known)  # typically helpful
                # rewrite, Dijkstra update
                D = deque([N])
                while len(D) > 0:
                    N_from = D.popleft()
                    for N_to in N_from.neighbors:
                        g = N_from.g + self.get_edge_cost(N_from.Q, N_to.Q)
                        if g < N_to.g:
                            if N_goal is not None and N_to is N_goal:
                                self.info(2, f"cost update: {N_goal.g:4d} -> {g:4d}")
                            N_to.g = g
                            N_to.f = N_to.g + N_to.h
                            N_to.parent = N_from
                            D.append(N_to)
                            if N_goal is not None and N_to.f < N_goal.g:
                                OPEN.appendleft(N_to)
            else:
                # new configuration
                N_new = HighLevelNode(
                    Q=Q_to,
                    parent=N,
                    order=self.get_order(Q_to),
                    g=N.g + self.get_edge_cost(N.Q, Q_to),
                    h=self.get_h_value(Q_to),
                )
                N.neighbors.add(N_new)
                OPEN.appendleft(N_new)
                EXPLORED[Q_to] = N_new

        # categorize result
        if N_goal is not None and len(OPEN) == 0:
            self.info(1, f"reach optimal solution, cost={N_goal.g}")
        elif N_goal is not None:
            self.info(1, f"suboptimal solution, cost={N_goal.g}")
        elif len(OPEN) == 0:
            self.info(1, "detected unsolvable instance")
        else:
            self.info(1, "failure due to timeout")
        return self.backtrack(N_goal)

    @staticmethod
    def backtrack(_N: HighLevelNode | None) -> Configs:
        """Reconstruct solution path by following parent pointers.

        Args:
            _N: Goal node (or None if no solution found).

        Returns:
            List of configurations from start to goal. Returns empty list if _N is None.
        """
        configs: Configs = []
        N = _N
        while N is not None:
            configs.append(N.Q)
            N = N.parent
        configs.reverse()
        return configs

    def get_edge_cost(self, Q_from: Config, Q_to: Config) -> int:
        """Calculate the cost of transitioning between two configurations.

        Cost is the number of agents that are not at their goal or moved from their goal.

        Args:
            Q_from: Source configuration.
            Q_to: Destination configuration.

        Returns:
            Transition cost (sum of agents not staying at goal).
        """
        # e.g., \sum_i | not (Q_from[i] == Q_to[k] == g_i) |
        cost = 0
        for i in range(self.num_agents):
            if not (self.goals[i] == Q_from[i] == Q_to[i]):
                cost += 1
        return cost

    def get_h_value(self, Q: Config) -> int:
        """Calculate heuristic value (lower bound on remaining cost).

        Uses sum of individual shortest path distances to goals.

        Args:
            Q: Configuration to evaluate.

        Returns:
            Heuristic value (sum of distances to goals for all agents).
            Returns maximum int value if any agent cannot reach its goal.
        """
        # e.g., \sum_i dist(Q[i], g_i)
        cost = 0
        for agent_idx, loc in enumerate(Q):
            c = self.dist_tables[agent_idx].get(loc)
            # Note: DistTable.get() always returns int, no None check needed
            if c >= self.graph_map.size:
                return np.iinfo(np.int32).max
            cost += c
        return cost

    def get_order(self, Q: Config) -> list[int]:
        """Determine the order in which agents are assigned in low-level search.

        Agents are ordered by descending distance to goal (with random tie-breaking).
        This heuristic prioritizes agents that are farther from their goals.

        Args:
            Q: Configuration to determine agent ordering for.

        Returns:
            List of agent indices in processing order.
        """
        # e.g., by descending order of dist(Q[i], g_i)
        order = list(range(self.num_agents))
        self.rng.shuffle(order)
        order.sort(key=lambda i: self.dist_tables[i].get(Q[i]), reverse=True)
        return order

    def configuration_generator(
        self, N: HighLevelNode, C: LowLevelNode
    ) -> Config | None:
        """Generate a successor configuration from constraints.

        Uses the low-level node constraints to assign positions, then randomly
        assigns remaining agents. Checks for collisions during generation.

        Args:
            N: Current high-level node.
            C: Low-level constraint node specifying partial assignments.

        Returns:
            A valid successor configuration, or None if generation fails due to collisions.
        """
        Q_to = Config([NO_LOCATION for _ in range(self.num_agents)])

        # set constraints to Q_to
        for k in range(C.depth):
            Q_to[C.who[k]] = C.where[k]

        # generate configuration
        flg_success = True
        for i in range(self.num_agents):
            v_i_from = N.Q[i]
            self.occupied_from[v_i_from] = i

            # set next position by random choice when without constraint
            if Q_to[i] == NO_LOCATION:
                cands = [v_i_from] + get_neighbors(self.graph_map, v_i_from)
                v =tuple(self.rng.choice(cands).tolist() )
                if is_valid_coord(self.graph_map, v):
                    Q_to[i] = v
                else:
                    flg_success = False
                    break

            v_i_to: Coord = Q_to[i]
            # check vertex collision
            if self.occupied_to[v_i_to] != NO_AGENT:
                flg_success = False
                break
            # check edge collision
            j = self.occupied_from[v_i_to]
            if j != NO_AGENT and j != i and Q_to[j] == v_i_from:
                flg_success = False
                break
            self.occupied_to[v_i_to] = i

        # cleanup cache used for collision checking
        for i in range(self.num_agents):
            v_i_from = N.Q[i]
            self.occupied_from[v_i_from] = NO_AGENT
            v_i_next = Q_to[i]
            if v_i_next != NO_LOCATION:
                self.occupied_to[v_i_next] = NO_AGENT

        return Q_to if flg_success else None

    def info(self, level: int, msg: str) -> None:
        """Log an informational message if verbosity level is sufficient.

        Args:
            level: Minimum verbosity level required to display this message.
            msg: Message to log.
        """
        if self.verbose < level:
            return
        logger.debug(f"{int(self.deadline.elapsed):4d}ms  {msg}")

    def get_solution_dict(self, solution: Configs) -> dict:
        solution_dict = {}
        for i in range(self.num_agents):
            solution_dict.update({f'agent_{i}': []})
            for j in range(len(solution)):
                if self.graph_map.dim == 2:
                    solution_dict[f'agent_{i}'].append({
                        't': j,
                        'x': solution[j][i][0],
                        'y': solution[j][i][1]
                    })
                elif self.graph_map.dim == 3:
                    solution_dict[f'agent_{i}'].append({
                        't': j,
                        'x': solution[j][i][0],
                        'y': solution[j][i][1],
                        'z': solution[j][i][2]
                    })
                else:
                    raise ValueError(f"Invalid dimension: {self.graph_map.dim}")
        return solution_dict
    
    def compute_solution_cost(self, solution: dict) -> int:
        cost = 0
        for agent, path in solution.items():
            path_array = np.array([[point['x'], point['y']] for point in path])
            dist_travel = np.linalg.norm(path_array[1:] - path_array[:-1], axis=1)
            travel_cost = dist_travel.sum()
            wait_cost = (dist_travel == 0).sum()
            cost += travel_cost + wait_cost
        return cost