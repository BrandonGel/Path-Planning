"""
Decentralized Neural-ATTF Environment.

Wraps the GraphSampler-based centralized CBS Environment with:
  * dynamic point-keyed obstacles (moving agents) and static point-keyed
    obstacles (idle/parked agents),
  * an optional ``cost_map`` (callable or 2D array) used as A* guidance,
  * an ``update_env`` hook so the same Environment instance can be reused
    across token-passing iterations.

Bounds and static collision are inherited from the base via
``graph_map.in_collision_point``; we do not re-check rectangular bounds.
"""

from typing import Callable, Iterable, Optional, Union

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.multi_agent_planner.centralized.cbs.cbs import (
    Environment as EnvironmentBase,
    Constraints,
    EdgeConstraint,
    Location,
    State,
    VertexConstraint,
)
from path_planning.multi_agent_planner.decentralized.neural_attf.a_star import AStar

CostMap = Optional[Union[Callable, "object"]]


class Environment(EnvironmentBase):
    def __init__(
        self,
        graph_map: GraphSampler,
        agents,
        obstacles: Optional[Iterable] = None,
        moving_obstacles: Optional[dict] = None,
        t: int = 0,
        astar_max_iterations: int = 10000,
        radius: float = 0.0,
        velocity: float = 0.0,
        use_constraint_sweep: bool = True,
        heuristic_type: str = "euclidean",
        cost_map: CostMap = None,
        alpha: float = 0.001,
    ):
        # Initialize point-keyed obstacle bookkeeping BEFORE super().__init__()
        # because the base constructor invokes self.make_agent_dict(), which
        # is overridden here to call self.state_valid() — and state_valid()
        # reads point_obstacles / moving_obstacles / constraints.
        self.point_obstacles = {tuple(o) for o in (obstacles or [])}
        self.moving_obstacles = moving_obstacles or {}
        self.cost_map = cost_map
        self.alpha = alpha
        self.ignore_agent_dict = {agent["name"]: False for agent in agents}
        self.constraints = Constraints()

        super().__init__(
            graph_map,
            agents,
            astar_max_iterations=astar_max_iterations,
            radius=radius,
            velocity=velocity,
            use_constraint_sweep=use_constraint_sweep,
            heuristic_type=heuristic_type,
        )
        # Re-bind A* so its cached env references include our cost_map.
        self.a_star = AStar(self, astar_max_iterations)

    def update_env(self, agents, obstacles=None, moving_obstacles=None, t: int = 0):
        if moving_obstacles is not None:
            self.moving_obstacles = moving_obstacles
        if obstacles is not None:
            self.point_obstacles = {tuple(o) for o in obstacles}
        self.agents = agents
        self.agent_dict = {}
        self.ignore_agent_dict = {agent["name"]: False for agent in agents}
        self.make_agent_dict(t)

    def state_valid(self, state):
        pt = tuple(state.location.point)
        if pt in self.point_obstacles:
            return False
        if (pt[0], pt[1], state.time) in self.moving_obstacles:
            return False
        if self.graph_map.in_collision_point(pt):
            return False
        return VertexConstraint(state.time, state.location) not in self.constraints.vertex_constraints

    def transition_valid(self, state_1, state_2):
        p1 = tuple(state_1.location.point)
        p2 = tuple(state_2.location.point)
        tup_1 = (p1[0], p1[1], state_2.time)
        tup_2 = (p2[0], p2[1], state_1.time)
        if (
            tup_1 in self.moving_obstacles
            and tup_2 in self.moving_obstacles
            and self.moving_obstacles[tup_1] == self.moving_obstacles[tup_2]
        ):
            return False
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in self.constraints.edge_constraints

    def compute_solution(self):
        """Always use A* (never the Dijkstra fallback in the base class).

        The base ``Environment.compute_solution`` skips A* and runs static
        Dijkstra when no CBS-style ``constraint_dict`` entries exist for an
        agent. That fallback is unaware of our point-keyed ``moving_obstacles``
        and ``point_obstacles`` (idle agents), so it would happily route an
        agent through another agent. We rely on A* here to honor those.
        """
        solution = {}
        solution_cost = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            local_solution, local_cost = self.a_star.search(agent)
            if not local_solution:
                return False, float("inf")
            solution[agent] = local_solution
            solution_cost[agent] = local_cost
        return solution, solution_cost

    def make_agent_dict(self, t: int = 0):
        for agent in self.agents:
            start_state = State(t, Location(tuple(agent["start"])))
            goal_state = State(t, Location(tuple(agent["goal"])))
            if not self.state_valid(goal_state) or not self.state_valid(start_state):
                self.ignore_agent_dict[agent["name"]] = True
            else:
                self.ignore_agent_dict[agent["name"]] = False
            self.agent_dict[agent["name"]] = {"start": start_state, "goal": goal_state}
