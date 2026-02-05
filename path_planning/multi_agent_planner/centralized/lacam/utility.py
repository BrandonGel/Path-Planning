"""Utility functions and data structures for Multi-Agent Path Finding (MAPF).

This module provides core data structures (Grid, Coord, Config, Deadline) and
utility functions for loading MAPF problem instances, validating solutions, and
computing solution costs.
"""

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias
from path_planning.common.environment.map.graph_sampler import GraphSampler
import numpy as np
from numpy.typing import NDArray
from typing import Any
from path_planning.common.environment.node import Node


Coord: TypeAlias = tuple[int, int]
"""Coordinate tuple (x, y) representing a position in the grid."""


@dataclass
class Config:
    """Configuration representing positions of all agents at a specific timestep.

    A configuration is essentially a list of coordinates, one for each agent.
    It supports list-like access patterns and hashing for use in search algorithms.

    Attributes:
        positions: List of agent positions as (y, x) coordinates.
    """

    positions: list[Coord] = field(default_factory=lambda: [])

    def __getitem__(self, k: int) -> Coord:
        """Get the position of agent k.

        Args:
            k: Agent index.

        Returns:
            The (y, x) coordinate of agent k.
        """
        return self.positions[k]

    def __setitem__(self, k: int, coord: Coord) -> None:
        """Set the position of agent k.

        Args:
            k: Agent index.
            coord: New (y, x) coordinate for agent k.
        """
        self.positions[k] = coord

    def __len__(self) -> int:
        """Get the number of agents in this configuration.

        Returns:
            Number of agents.
        """
        return len(self.positions)

    def __hash__(self) -> int:
        """Compute hash for use in sets and dictionaries.

        Returns:
            Hash value based on agent positions.
        """
        return hash(tuple(self.positions))

    def __eq__(self, other: object) -> bool:
        """Check equality with another configuration.

        Args:
            other: Object to compare with.

        Returns:
            True if other is a Config with identical positions.
        """
        if not isinstance(other, Config):
            return NotImplemented
        return self.positions == other.positions

    def append(self, coord: Coord) -> None:
        """Add a new agent position to this configuration.

        Args:
            coord: The (y, x) coordinate to add.
        """
        self.positions.append(coord)

    def __iter__(self) -> Iterator[Coord]:
        """Iterate over agent positions.

        Returns:
            Iterator over agent positions.
        """
        return iter(self.positions)


Configs: TypeAlias = list[Config]
"""List of configurations representing a solution path over time."""


@dataclass
class Deadline:
    """Time limit manager for search algorithms.

    Tracks elapsed time and checks whether a time limit has been exceeded.

    Attributes:
        time_limit_ms: Maximum allowed time in milliseconds.
    """

    time_limit_ms: int

    def __post_init__(self) -> None:
        """Initialize the start time when the deadline is created."""
        self.start_time = time.time()

    @property
    def elapsed(self) -> float:
        """Get elapsed time since deadline creation.

        Returns:
            Elapsed time in milliseconds.
        """
        return (time.time() - self.start_time) * 1000

    @property
    def is_expired(self) -> bool:
        """Check if the time limit has been exceeded.

        Returns:
            True if elapsed time exceeds the time limit.
        """
        return self.elapsed > self.time_limit_ms


def set_starts_goals_config(starts_list: list = None, goals_list:list = None) -> tuple[Config, Config]:
    """Load start and goal configurations.
    Args:
        starts: List of starting positions
        goals: List of goal positions

    Returns:
        A tuple (starts, goals) where:
        - starts: Configuration of starting positions
        - goals: Configuration of goal positions
    """
    starts,goals = Config(),Config()
    if starts_list is not None:
        for start in starts_list:
            starts.append(tuple(start))  # align with grid
    if goals_list is not None:
        for goal in goals_list:
            goals.append(tuple(goal))
    return starts, goals

def is_valid_coord(graph_map: GraphSampler, coord: Coord) -> bool:
    """Check if a coordinate is valid and passable in the grid.

    Args:
        grid: The grid map.
        coord: The (x, y) coordinate to check.

    Returns:
        True if the coordinate is within bounds and represents a passable cell,
        False otherwise.
    """
    return not graph_map.in_collision_point(coord)

def get_neighbors(graph_map: GraphSampler, coord: Coord) -> list[Coord]:
    """Get all valid neighboring coordinates (4-connected).

    Args:
        grid: The grid map.
        coord: The (x, y) coordinate whose neighbors to find.

    Returns:
        List of valid neighboring coordinates. Returns empty list if coord
        itself is invalid.
    """
    # coord: y, x
    neigh: list[Coord] = []

    # check valid input
    if not is_valid_coord(graph_map, coord):
        return neigh

    node = Node(tuple[Any, ...](coord))
    nodes = graph_map.get_neighbors(node)

    # Move action
    for node in nodes:
        if is_valid_coord(graph_map,node.current):
            pos = tuple(int(pt) for pt in node.current)
            neigh.append(pos)
    return neigh


def save_configs_for_visualizer(configs: Configs, filename: str | Path) -> None:
    """Save solution configurations to a file for visualization tools.

    The output format is compatible with mapf-visualizer tools.

    Args:
        configs: List of configurations representing the solution path.
        filename: Output file path. Parent directories will be created if needed.
    """
    output_dirname = Path(filename).parent
    if not output_dirname.exists():
        output_dirname.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        for t, config in enumerate(configs):
            row = f"{t}:" + "".join([f"({x},{y})," for (y, x) in config]) + "\n"
            f.write(row)


def validate_mapf_solution(
    graph_map: GraphSampler,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> None:
    """Validate a MAPF solution for correctness.

    Checks that:
    - Solution starts with the start configuration
    - Solution ends with the goal configuration
    - All transitions are valid (agents move to adjacent cells or stay)
    - No vertex collisions (two agents at same location)
    - No edge collisions (two agents swap positions)

    Args:
        grid: The grid map.
        starts: Starting configuration.
        goals: Goal configuration.
        solution: List of configurations representing the solution path.

    Raises:
        AssertionError: If any validation check fails.
    """
    # starts
    assert all(
        [u == v for (u, v) in zip(starts, solution[0])]
    ), "invalid solution, check starts"

    # goals
    assert all(
        [u == v for (u, v) in zip(goals, solution[-1])]
    ), "invalid solution, check goals"

    T = len(solution)
    N = len(starts)

    for t in range(T):
        for i in range(N):
            v_i_now = tuple(int(pt) for pt in np.array(solution[t][i]).tolist())
            v_i_pre = tuple(int(pt) for pt in np.array(solution[max(t - 1, 0)][i]).tolist())

            # check continuity
            if v_i_now not in [v_i_pre] + get_neighbors(graph_map, v_i_pre):
                pass
            assert v_i_now in [v_i_pre] + get_neighbors(
                graph_map, v_i_pre
            ), "invalid solution, check connectivity"

            # check collision
            for j in range(i + 1, N):
                v_j_now = solution[t][j]
                v_j_pre = solution[max(t - 1, 0)][j]
                assert not (v_i_now == v_j_now), "invalid solution, vertex collision"
                assert not (
                    v_i_now == v_j_pre and v_i_pre == v_j_now
                ), "invalid solution, edge collision"


def is_valid_mapf_solution(
    graph_map: GraphSampler,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> bool:
    """Check if a MAPF solution is valid without raising exceptions.

    Args:
        grid: The grid map.
        starts: Starting configuration.
        goals: Goal configuration.
        solution: List of configurations representing the solution path.

    Returns:
        True if the solution is valid, False otherwise.
    """
    try:
        validate_mapf_solution(graph_map, starts, goals, solution)
        return True
    except Exception as e:
        print(e)
        return False


def get_sum_of_loss(configs: Configs) -> int:
    """Calculate the sum of loss (total number of non-goal moves) in a solution.

    For each timestep and agent, counts 1 if the agent is not at its goal or
    moved from its goal. This is a common MAPF solution quality metric.

    Args:
        configs: List of configurations representing the solution path.
        The last configuration is assumed to be the goal configuration.

    Returns:
        Total sum of loss across all timesteps and agents.
    """
    cost = 0
    for t in range(1, len(configs)):
        cost += sum(
            [
                not (v_from == v_to == goal)
                for (v_from, v_to, goal) in zip(configs[t - 1], configs[t], configs[-1])
            ]
        )
    return cost


def get_makespan_lower_bound(
    graph_map: GraphSampler,
    starts: Config,
    goals: Config,
    dist_tables,
) -> int:
    """Compute a lower bound on the makespan (max individual path length).

    This mirrors the behavior of the C++ helper used by Scatter, which
    takes the maximum, over all agents, of the shortest-path distance
    from their start to their goal according to the corresponding
    distance table.

    Args:
        graph_map: Underlying graph (unused here, kept for API parity).
        starts: Configuration of start positions.
        goals: Configuration of goal positions.
        dist_tables: Sequence of DistTable-like objects, one per agent.

    Returns:
        Integer lower bound on the makespan.
    """
    if len(starts) == 0:
        return 0
    return max(dist_tables[i].get(starts[i]) for i in range(len(starts)))
