"""Utility functions and data structures for Multi-Agent Path Finding (MAPF).

This module provides core data structures (Grid, Coord, Config, Deadline) and
utility functions for loading MAPF problem instances, validating solutions, and
computing solution costs.
"""

import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias
from path_planning.common.environment.map.graph_sampler import GraphSampler
import numpy as np
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


# LaCAM works on the grid's cells (integer indices; its tables are arrays of ``graph_map.shape``), one
# roadmap node per cell, at the cell centre. These convert at its boundary.
def cell_of(graph_map: GraphSampler, point) -> Coord:
    """The cell of a roadmap point (a node at its cell centre)."""
    return graph_map.world_to_map(point, discrete=True)


def centre_of(graph_map: GraphSampler, cell: Coord) -> tuple:
    """The world position of a cell: its centre, where the grid node is."""
    return graph_map.map_to_world(cell)


def set_starts_goals_config(starts_list: list, goals_list: list, graph_map: GraphSampler) -> tuple[Config, Config]:
    """The start and goal configurations: the cells of the agents' start / goal roadmap nodes on
    ``graph_map``, which must be a grid roadmap (one node per cell)."""
    if not getattr(graph_map, "use_discrete_space", False):
        raise ValueError("LaCAM plans on a grid roadmap (use_discrete_space=True)")
    to_config = lambda points: Config([cell_of(graph_map, p) for p in points])
    return to_config(starts_list), to_config(goals_list)


# graph_map -> (roadmap signature, {cell: passable}, {cell: neighbour cells}): what LaCAM's inner loops
# (PIBT, the distance tables) ask for again and again, worked out once per cell. ``reset_cells`` empties it
# (each solve does, so a changed map is never served stale results).
_CELLS = weakref.WeakKeyDictionary()


def _cells(graph_map: GraphSampler):
    sig = graph_map._roadmap_signature()
    entry = _CELLS.get(graph_map)
    if entry is None or entry[0] != sig:
        entry = _CELLS[graph_map] = (sig, {}, {})
    return entry


def reset_cells(graph_map: GraphSampler) -> None:
    """Forget the cached cell validity and neighbours of ``graph_map``."""
    _CELLS.pop(graph_map, None)

def is_valid_coord(graph_map: GraphSampler, coord: Coord) -> bool:
    """Check if a coordinate is valid and passable in the grid.

    Args:
        grid: The grid map.
        coord: The (x, y) coordinate to check.

    Returns:
        True if the coordinate is within bounds and represents a passable cell,
        False otherwise.
    """
    valid = _cells(graph_map)[1]
    ok = valid.get(coord)
    if ok is None:
        ok = valid[coord] = not graph_map.in_collision(centre_of(graph_map, coord))
    return ok

def get_neighbors(graph_map: GraphSampler, coord: Coord) -> list[Coord]:
    """Get all valid neighboring coordinates (4-connected).

    Args:
        grid: The grid map.
        coord: The (x, y) coordinate whose neighbors to find.

    Returns:
        List of valid neighboring coordinates. Returns empty list if coord
        itself is invalid.
    """
    table = _cells(graph_map)[2]
    neigh = table.get(coord)
    if neigh is None:
        neigh = ()
        if is_valid_coord(graph_map, coord):
            nodes = graph_map.get_neighbors(Node(centre_of(graph_map, coord)))   # that cell's roadmap node
            neigh = tuple(pos for pos in (cell_of(graph_map, n.current) for n in nodes) if is_valid_coord(graph_map, pos))
        table[coord] = neigh
    return list(neigh)                                    # a copy: callers extend it


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
