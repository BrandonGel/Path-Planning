"""
A PIBT implementation adapted from
https://github.com/Kei18/pypibt and the original LaCAM* C++ code.

Compared to the original toy Python version, this implementation
adds the missing features from the C++ PIBT:

* Distance-based candidate ordering with persistent tie-breaking
* Optional scatter data to bias moves
* Swap detection and execution (push & swap style)
"""

import numpy as np

from path_planning.multi_agent_planner.centralized.lacam.utility import (
    Coord,
    Config,
    get_neighbors,
)
from path_planning.multi_agent_planner.centralized.lacam.dist_table import DistTable
from path_planning.multi_agent_planner.centralized.lacam.scatter import Scatter


class PIBT:
    def __init__(
        self,
        dist_tables: list[DistTable],
        goals: Config,
        seed: int = 0,
        flg_swap: bool = True,
        scatter: Scatter | None = None,
    ) -> None:
        """Priority Inheritance with Backtracking and Tie-breaking (PIBT).

        Args:
            dist_tables: Per‑agent distance tables to their goals.
            goals: Goal configuration for all agents.
            seed: RNG seed used for tie‑breaking.
            flg_swap: Whether to enable push & swap style operations.
            scatter: Optional scatter hints to bias candidate ordering.
        """
        self.N = len(dist_tables)
        assert self.N > 0
        self.dist_tables = dist_tables
        self.graph_map = self.dist_tables[0].graph_map
        self.goals = goals

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.graph_map.shape  # meaning \bot
        self.occupied_now = np.full(self.graph_map.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(self.graph_map.shape, self.NIL, dtype=int)

        # candidate buffer and tie‑breakers
        self.C_next: list[list[Coord]] = [[] for _ in range(self.N)]
        self.tie_breakers: dict[Coord, float] = {}

        # flags / helpers
        self.flg_swap = flg_swap
        self.scatter = scatter

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int) -> bool:
        """Core PIBT procedure for a single agent.

        Returns:
            True if a valid next location is assigned to agent ``i``,
            False otherwise.
        """
        v_from = Q_from[i]

        # ------------------------------------------------------------------
        # Build candidate list C_next[i] with optional scatter bias
        # ------------------------------------------------------------------
        prioritized_vertex: Coord | None = None
        if self.scatter is not None and i < len(self.scatter.scatter_data):
            prioritized_vertex = self.scatter.scatter_data[i].get(v_from)

        # candidate next vertices: neighbors + stay
        C = get_neighbors(self.graph_map, v_from)
        C.append(v_from)
        self.C_next[i] = C

        # fresh tie‑breakers for this call
        self.tie_breakers.clear()
        for u in self.C_next[i]:
            self.tie_breakers[u] = float(self.rng.random())

        # sort by distance + tie‑breaker, then move prioritized vertex to front
        self.C_next[i].sort(
            key=lambda u: self.dist_tables[i].get(u) + self.tie_breakers[u]
        )
        if prioritized_vertex is not None and prioritized_vertex in self.C_next[i]:
            self.C_next[i].remove(prioritized_vertex)
            self.C_next[i].insert(0, prioritized_vertex)

        # ------------------------------------------------------------------
        # Optional swap emulation (push & swap style)
        # ------------------------------------------------------------------
        swap_agent = self.NIL
        if self.flg_swap:
            swap_agent = self.is_swap_required_and_possible(i, Q_from, Q_to)
            if swap_agent != self.NIL:
                # reverse vertex scoring as in the C++ implementation
                self.C_next[i].reverse()

        def swap_operation() -> None:
            # Pull swap_agent into v_from if possible
            if (
                swap_agent != self.NIL
                and Q_to[swap_agent] == self.NIL_COORD
                and self.occupied_nxt[v_from] == self.NIL
            ):
                self.occupied_nxt[v_from] = swap_agent
                Q_to[swap_agent] = v_from

        # ------------------------------------------------------------------
        # Main PIBT loop over ordered candidates
        # ------------------------------------------------------------------
        for k, v in enumerate(self.C_next[i]):
            # avoid vertex conflicts
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid swap conflicts with constraints
            if j != self.NIL and Q_to[j] == v_from:
                continue

            # reserve next location
            self.occupied_nxt[v] = i
            Q_to[i] = v

            # priority inheritance
            if (
                j != self.NIL
                and v != v_from
                and Q_to[j] == self.NIL_COORD
                and not self.funcPIBT(Q_from, Q_to, j)
            ):
                # reservation failed downstream, try next candidate
                continue

            # success to plan next one step
            if self.flg_swap and k == 0:
                swap_operation()
            return True

        # failed to secure node
        self.occupied_nxt[v_from] = i
        Q_to[i] = v_from
        return False

    def step(
        self,
        Q_from: Config,
        Q_to: Config,
        order: list[int],
    ) -> bool:
        """Plan one timestep for all agents given a priority order."""
        flg_success = True

        # ------------------------------------------------------------------
        # Setup cache & check pre‑assigned moves (constraints)
        # ------------------------------------------------------------------
        for i, (v_i_from, v_i_to) in enumerate(zip(Q_from, Q_to)):
            # mark current occupancy
            self.occupied_now[v_i_from] = i

            # handle pre‑assigned moves (constraints in Q_to)
            if v_i_to != self.NIL_COORD:
                # vertex collision
                if self.occupied_nxt[v_i_to] != self.NIL:
                    flg_success = False
                    break
                # swap / edge collision
                j = self.occupied_now[v_i_to]
                if j != self.NIL and j != i and Q_to[j] == v_i_from:
                    flg_success = False
                    break
                self.occupied_nxt[v_i_to] = i

        # ------------------------------------------------------------------
        # Perform PIBT for agents without pre‑assigned moves
        # ------------------------------------------------------------------
        if flg_success:
            for i in order:
                if Q_to[i] == self.NIL_COORD:
                    flg_success = self.funcPIBT(Q_from, Q_to, i)
                    if not flg_success:
                        break

        # ------------------------------------------------------------------
        # Cleanup occupancy caches
        # ------------------------------------------------------------------
        for v_from, v_to in zip(Q_from, Q_to):
            self.occupied_now[v_from] = self.NIL
            if v_to != self.NIL_COORD:
                self.occupied_nxt[v_to] = self.NIL

        return flg_success

    # ----------------------------------------------------------------------
    # Swap detection helpers (adapted from C++ implementation)
    # ----------------------------------------------------------------------

    def is_swap_required_and_possible(
        self, i: int, Q_from: Config, Q_to: Config
    ) -> int:
        """Check if agent ``i`` should perform a swap with another agent.

        Returns:
            Index of the swap partner, or ``self.NIL`` if no swap is
            required or possible.
        """
        if not self.C_next[i]:
            return self.NIL

        best = self.C_next[i][0]
        j = self.occupied_now[best]

        # direct swap candidate
        if (
            j != self.NIL
            and j != i
            and Q_to[j] == self.NIL_COORD
            and self.is_swap_required(i, j, Q_from[i], Q_from[j])
            and self.is_swap_possible(Q_from[j], Q_from[i])
        ):
            return j

        # clear operation: emulate push & swap from one step ahead
        if best != Q_from[i]:
            for u in get_neighbors(self.graph_map, Q_from[i]):
                k = self.occupied_now[u]
                if (
                    k != self.NIL
                    and best != Q_from[k]
                    and self.is_swap_required(k, i, Q_from[i], best)
                    and self.is_swap_possible(best, Q_from[i])
                ):
                    return k

        return self.NIL

    def is_swap_required(
        self,
        pusher: int,
        puller: int,
        v_pusher_origin: Coord,
        v_puller_origin: Coord,
    ) -> bool:
        """Determine if a swap is required for the pusher to make progress."""
        v_pusher = v_pusher_origin
        v_puller = v_puller_origin
        tmp: Coord | None = None

        D_pusher = self.dist_tables[pusher]
        D_puller = self.dist_tables[puller]

        while D_pusher.get(v_puller) < D_pusher.get(v_pusher):
            neighbors = get_neighbors(self.graph_map, v_puller)
            n = len(neighbors)
            tmp = None

            # remove agents who need not move
            for u in neighbors:
                i = self.occupied_now[u]
                deg_u = len(get_neighbors(self.graph_map, u))
                if u == v_pusher or (
                    deg_u == 1 and i != self.NIL and self.goals[i] == u
                ):
                    n -= 1
                else:
                    tmp = u

            if n >= 2:
                # able to swap at v_l
                return False
            if n <= 0 or tmp is None:
                break

            v_pusher = v_puller
            v_puller = tmp

        return (D_puller.get(v_pusher) < D_puller.get(v_puller)) and (
            D_pusher.get(v_pusher) == 0
            or D_pusher.get(v_puller) < D_pusher.get(v_pusher)
        )

    def is_swap_possible(self, v_pusher_origin: Coord, v_puller_origin: Coord) -> bool:
        """Check whether a physical swap between two locations is feasible."""
        v_pusher = v_pusher_origin
        v_puller = v_puller_origin
        tmp: Coord | None = None

        # simulate pull operation
        while v_puller != v_pusher_origin:
            neighbors = get_neighbors(self.graph_map, v_puller)
            n = len(neighbors)
            tmp = None

            for u in neighbors:
                i = self.occupied_now[u]
                deg_u = len(get_neighbors(self.graph_map, u))
                if u == v_pusher or (
                    deg_u == 1 and i != self.NIL and self.goals[i] == u
                ):
                    n -= 1
                else:
                    tmp = u

            if n >= 2:
                # able to swap at v_next
                return True
            if n <= 0 or tmp is None:
                return False

            v_pusher = v_puller
            v_puller = tmp

        return False