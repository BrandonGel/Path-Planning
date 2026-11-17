"""SIPP's safe-interval tables: when one kind of agent may be at each roadmap vertex and on each edge,
given the timed paths of everything else (dynamic obstacles and the agents planned before it)."""
import bisect
import math
import weakref

import numpy as np

from path_planning.agent import Agent
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.multi_agent_planner.centralized.cbs.data_types import _heading_of, _make_ori, _ori_between
from path_planning.multi_agent_planner.sweep_cache import shared_sweep_cache

_STATIC = weakref.WeakKeyDictionary()     # graph_map -> (roadmap signature, its move lists and edge ids)

class State(object):
    def __init__(self, position=(-1,-1), t=0, interval=(0,float('inf'))):
        self.position = tuple(position)
        self.time = t
        self.interval = interval # safe interval (start_time, end_time)
    
    def __eq__(self, other):
        return self.position == other.position and self.time == other.time and self.interval == other.interval
    def __hash__(self):
        return hash((self.position, self.time, self.interval))
    def __str__(self):
        return str((self.position, self.time, self.interval))
    def __repr__(self):
        return str((self.position, self.time, self.interval))

class SippNode(object):
    def __init__(self):
        self.interval_list = [(0, float('inf'))]

    # Split the safety interval with the agent depature time, and agent arrival time
    def split_interval(self, t1, t2, t_buffer=1e-10):
        """
        Function to generate safe-intervals
        """
        interval_list = []
        # Apply buffer to the blocked window to ensure safety margins
        b_start = t1 - t_buffer
        b_end = t2 + t_buffer

        for s_start, s_end in self.interval_list:
            # Case 1: Blocked window is entirely after this interval
            if b_start >= s_end:
                if s_end - s_start > 1e-6:
                    interval_list.append((s_start, s_end))

            # Case 2: Blocked window is entirely before this interval
            elif b_end <= s_start:
                if s_end - s_start > 1e-6:
                    interval_list.append((s_start, s_end))

            # Case 3: Overlap occurs: keep the left and right remnants (slivers of 1e-6 or less go)
            else:
                if b_start - s_start > 1e-6:
                    interval_list.append((s_start, b_start))
                if s_end - b_end > 1e-6:
                    interval_list.append((b_end, s_end))
        interval_list.sort()
        self.interval_list = interval_list

# Used for only when the agent is moving on an edge and not on a vertex
class UnsafeIntervalList(object):
    """Sorted list of pairwise-disjoint closed unsafe intervals.

    Stored as two parallel lists (`starts`, `ends`) so bisect can locate the
    overlap range in O(log n). Intervals are treated as closed, so touching
    intervals ([1,2] and [2,3]) are merged as well. Insertion finds the run of
    existing intervals overlapping [t1, t2], collapses it into one merged
    interval, and splices it in place. Iteration yields (t1, t2) tuples in
    ascending order.
    """

    __slots__ = ("starts", "ends")

    def __init__(self):
        # Lists are allocated on first add(): init_graph builds one record per
        # directed edge on every reset_graph and most never receive an interval.
        self.starts = None
        self.ends = None

    def __len__(self):
        return 0 if self.starts is None else len(self.starts)

    def __bool__(self):
        return bool(self.starts)

    def __iter__(self):
        return iter(()) if self.starts is None else iter(zip(self.starts, self.ends))

    def __repr__(self):
        return f"UnsafeIntervalList({list(self)})"

    def _overlap_range(self, t1, t2):
        """Index range [lo, hi) of stored intervals overlapping the closed window [t1, t2]."""
        lo = bisect.bisect_left(self.ends, t1)     # first interval with end >= t1
        hi = bisect.bisect_right(self.starts, t2)  # first interval with start > t2
        return lo, hi

    def add(self, t1, t2):
        """Insert [t1, t2], merging with every existing interval it overlaps."""
        if t2 < t1:
            t1, t2 = t2, t1
        if self.starts is None:
            self.starts, self.ends = [t1], [t2]
            return (t1, t2)
        lo, hi = self._overlap_range(t1, t2)
        if lo < hi:
            if self.starts[lo] < t1:
                t1 = self.starts[lo]
            if self.ends[hi - 1] > t2:
                t2 = self.ends[hi - 1]
        self.starts[lo:hi] = [t1]
        self.ends[lo:hi] = [t2]
        return (t1, t2)

    def overlapping(self, t1, t2):
        """Stored intervals overlapping the closed window [t1, t2], in order."""
        if self.starts is None:
            return []
        lo, hi = self._overlap_range(t1, t2)
        return list(zip(self.starts[lo:hi], self.ends[lo:hi]))

    def intersects(self, t1, t2):
        if self.starts is None:
            return False
        lo = bisect.bisect_left(self.ends, t1)
        return lo < len(self.starts) and self.starts[lo] <= t2


class SippEdge(UnsafeIntervalList):
    """Edge record for the SIPP graph: a merged, sorted list of unsafe intervals.

    Subclasses UnsafeIntervalList directly (no wrapper object) because
    init_graph builds one SippEdge per directed edge on every
    reset_graph, so construction cost matters.
    """

    __slots__ = ()

    def add_unsafe_interval(self, t1, t2):
        self.add(t1, t2)

    def get_unsafe_intervals(self, t1, t2):
        return self.overlapping(t1, t2)

    def is_in_unsafe_interval(self, t1, t2):
        return self.intersects(t1, t2)


def cached_collision_sweep(graph_map, p1, p2, v, r):
    """``graph_map.get_collision_sweep(p1, p2, v, r, use_interval=True, get_time_interval=True)``, memoized:
    ``(vertex -> (t_start, t_end), (u, v) edge -> (t_start, t_end))`` for a disk of radius ``r`` moving
    ``p1 -> p2`` at speed ``v``, times relative to the start of that move. The memo is the planners'
    ``shared_sweep_cache(graph_map)``, so it survives across SippGraph instances (NeuralATTF builds one per
    low-level call) and is renewed when the map's sweep is rebound. The returned dicts are shared: do not
    modify them."""
    return shared_sweep_cache(graph_map).get_or_compute(
        (p1, p2, v, r), lambda: graph_map.get_collision_sweep(p1, p2, v, r, use_interval=True, get_time_interval=True))


class SippGraph(object):
    """The safe intervals of every vertex and the unsafe intervals of every directed edge, for agents with
    the body ``agent`` (an ``Agent``; None: point particles), given the timed paths of other bodies.

    - ``dynamic_obstacles``: ``{name: schedule}``, a schedule being ``[{t, x, y[, z][, yaw[, pitch]]}, ...]``
      (a plain list: an obstacle with the same body as ``agent``) or ``{"agent": Agent, "schedule": [...]}``
      (its own body). A waiting obstacle that is not a ball keeps the heading of its last move unless the
      entry gives ``yaw``; before its first move it faces 0 (``(0, 0)`` in 3D) unless given.
    - ``obstacle_horizon``: how long an obstacle's last point blocks its footprint: one value for all or
      ``{name: horizon}``; None or <= 0 means forever (names missing from the dict also rest forever).

    Bodies use the map's exact sweeps (``GraphSampler.sweep_paths``): an obstacle's body grown by this
    table's agent's bounding radius, or, when both are polygons / polyhedra without clearance, the exact
    Minkowski rule (a moving agent facing along its edge). At a vertex an agent that is not a ball is
    taken as its bounding ball (``sipp_graph[position]``, a superset); with ``exact_vertices``,
    ``vertex_node(position, ori)``
    gives the exact safe intervals for it standing there with heading ``ori`` (degrees; ``(yaw, pitch)`` in
    3D), worked out on demand from the obstacles that block the bounding ball there. Point particles keep
    the unit-step rules: a vertex is taken for one step, and swaps along an edge are checked against the
    obstacles' states.
    """

    def __init__(self, graph_map: GraphSampler, dynamic_obstacles: dict | None = None, agent: Agent | None = None,
                 obstacle_horizon: float | dict | None = None, exact_vertices: bool = False):
        self.graph_map = graph_map
        self.agent = agent
        if isinstance(obstacle_horizon, dict):
            self.obstacle_horizon = float('inf')
            self.obstacle_horizons = {name: self._norm_horizon(h) for name, h in obstacle_horizon.items()}
        else:
            self.obstacle_horizon = self._norm_horizon(obstacle_horizon)
            self.obstacle_horizons = {}
        self.dynamic_obstacles = dynamic_obstacles or {}
        bodies = [self._obstacle(spec)[0] for spec in self.dynamic_obstacles.values()]
        # Point rules only when nothing has a body; otherwise every element is swept.
        self.points = agent is None and all(b is None for b in bodies)
        if not self.points:
            self.graph_map.set_collision_sweep()
        if agent is not None and not agent.is_ball and not agent.clearance:
            self._rule = "minkowski"
        else:
            self._rule = "radius"
        self._cache = shared_sweep_cache(graph_map, "sipp")
        # Vertices per heading (vertex_node) for a body that is not a ball, unless switched off.
        self.oriented = exact_vertices and agent is not None and not agent.is_ball
        if self.oriented:
            self._standing = Agent(agent.shape, None, None, agent.type, agent.clearance)
        self.reset_graph()

    # ------------------------------------------------------------------ the table
    def init_graph(self):
        self.sipp_graph = {}
        for node in self.graph_map.nodes:
            self.sipp_graph[node.current] = SippNode()
        # Node ids for the compiled search: the vertices first (id = vertex index), per-heading lists after.
        self._nobj = [self.sipp_graph[node.current] for node in self.graph_map.nodes]
        # Directed edge records keyed by their end positions (graph_map.edges usually lists both ways).
        for src_idx, tgt_idx in self.graph_map.edges:
            src_pos = self.graph_map.nodes[src_idx].current
            tgt_pos = self.graph_map.nodes[tgt_idx].current
            if (src_pos, tgt_pos) not in self.sipp_graph:
                self.sipp_graph[(src_pos, tgt_pos)] = SippEdge()
            if (tgt_pos, src_pos) not in self.sipp_graph:
                self.sipp_graph[(tgt_pos, src_pos)] = SippEdge()

    def reset_graph(self):
        """Back to the dynamic obstacles alone."""
        self.swaps = set()                   # (from, to, t_from, t_to) of every obstacle step, for that check
        self._pieces = []                    # (body, path, t0, hold, heading): every blocking piece
        self._blockers = {}                  # position -> indices of the pieces that block it (as a ball)
        self._exact = {}                     # position -> {ori: [SippNode, pieces applied]}
        self._current = {}                   # position -> blockers applied to all its headings
        # For the compiled search (arrays()): what changed since the arrays were last brought up to date.
        self._arr = None
        self._static = None
        self._dirty_nodes, self._dirty_edges, self._dirty_into = set(), set(), set()
        self._nid = {}                       # (position, ori) -> node id of a per-heading interval list
        self._nobj = []                      # node id -> its SippNode (ids below the vertex count: the vertices)
        self.init_graph()
        self.init_intervals(self.dynamic_obstacles)

    @staticmethod
    def _norm_horizon(horizon) -> float:
        """``None`` / non-positive -> rest forever (inf); otherwise the horizon in seconds."""
        return float(horizon) if horizon is not None and horizon > 0 else float('inf')

    def _horizon_for(self, dyn_name) -> float:
        """Resting horizon of one dynamic obstacle (per-name override, else the global value)."""
        return self.obstacle_horizons.get(dyn_name, self.obstacle_horizon)

    def _obstacle(self, spec):
        """``(body, schedule)`` of a ``dynamic_obstacles`` entry (a plain schedule has this table's body)."""
        if isinstance(spec, dict):
            return spec.get("agent"), spec["schedule"]
        return self.agent, spec

    @staticmethod
    def _position(entry) -> tuple:
        return (entry["x"], entry["y"]) if "z" not in entry else (entry["x"], entry["y"], entry["z"])

    # ------------------------------------------------------------------ blocking
    def init_intervals(self, dyn_obstacles: dict | None = None):
        """Block the table with timed schedules ``{name: schedule}`` (see the class docstring).

        - The last point blocks its footprint (vertices AND edges) for ``[t, t + horizon]``.
        - A wait (next point at the same position) blocks it for exactly ``[t, next t]``.
        - A move blocks what the body sweeps, at the speed the schedule implies when that differs from the
          body's own (a time-sampled schedule can cover less than a full move per tick).
        """
        for name, spec in (dyn_obstacles or {}).items():
            body, schedule = self._obstacle(spec)
            horizon = self._horizon_for(name)
            points = [tuple(self._position(entry)) for entry in schedule]
            times = [max(0.0, float(entry["t"])) for entry in schedule]
            self._index_swaps([State(position=p, t=entry["t"]) for p, entry in zip(points, schedule)])
            if self.points:
                for i, (p, t) in enumerate(zip(points, times)):
                    last = i == len(points) - 1
                    self.sipp_graph[p].split_interval(t, t + 1 if not last else t + horizon, 1)
                continue
            heads = self._wait_headings(body, points, [entry.get("yaw") for entry in schedule],
                                        [entry.get("pitch") for entry in schedule])
            for i, (p, t) in enumerate(zip(points, times)):
                if i == len(points) - 1:
                    self._block_wait(body, p, heads[i], t, t + horizon)
                    continue
                q, next_t = points[i + 1], times[i + 1]
                if q == p:
                    if next_t - t > 1e-9:
                        self._block_wait(body, p, heads[i], t, next_t)
                    continue
                self._block_move(body, p, q, t, self._implied_speed(body, p, q, next_t - t))

    def update_intervals(self, plans, action_costs, dyn_names, bodies=None, starts=None):
        """Block the table with planned paths: ``plans`` (lists of ``State``), their ``(wait, move)`` action
        costs, the agents' names, bodies (one ``Agent`` or None each; default: this table's body) and start
        headings (``State.ori`` values; default 0)."""
        if not plans:
            return
        bodies = [self.agent] * len(plans) if bodies is None else bodies
        starts = [None] * len(plans) if starts is None else starts
        for plan, action_cost, name, body, start in zip(plans, action_costs, dyn_names, bodies, starts):
            if self.points:
                for i, state in enumerate(plan):
                    last = i == len(plan) - 1
                    self.sipp_graph[state.position].split_interval(state.time, state.time + 1 if not last else float('inf'), 1)
                # A planned point is an obstacle for the swap check too.
                self._index_swaps(self._unit_states(plan))
                continue
            points = [state.position for state in plan]
            heads = self._wait_headings(body, points, start=start)
            for i, state in enumerate(plan):
                p, t = state.position, state.time
                if i == len(plan) - 1:
                    self._block_wait(body, p, heads[i], t, float('inf'))
                    continue
                wait, _ = action_cost[i]
                if wait > 1e-10:
                    self._block_wait(body, p, heads[i], t, t + wait)
                self._block_move(body, p, plan[i + 1].position, t + (wait if wait > 1e-10 else 0.0),
                                 None if body is None else body.max_speed)
        self.refresh()

    def _index_swaps(self, states):
        for a, b in zip(states, states[1:]):
            self.swaps.add((a.position, b.position, a.time, b.time))

    @staticmethod
    def _unit_states(plan):
        """A unit-step point plan as one state per time step (waits written out)."""
        states = [plan[0]]
        for a, b in zip(plan, plan[1:]):
            for k in range(1, int(round(b.time - a.time))):
                states.append(State(a.position, a.time + k))
            states.append(b)
        return states

    def _implied_speed(self, body, p, q, duration):
        """The speed of a schedule's move ``p -> q`` taking ``duration``: the body's own speed (None: one
        time unit per move) unless the schedule says otherwise."""
        nominal = None if body is None else body.max_speed
        if duration <= 1e-9:
            return nominal
        if nominal is None:
            return None if abs(duration - 1.0) <= 1e-6 else math.dist(p, q) / duration
        implied = math.dist(p, q) / duration
        return nominal if abs(implied - nominal) <= 1e-6 else implied

    def _wait_headings(self, body, points, yaws=None, pitches=None, start=None):
        """The attitude a body holds while waiting at each point of a path (None for a ball): the given
        ``yaw`` (+ ``pitch``) if any, else the direction of its last move (before the first: ``start``, a
        ``State.ori``; default 0)."""
        if body is None or body.is_ball:
            return [None] * len(points)
        moves = [_ori_between(a, b) if a != b else None for a, b in zip(points, points[1:])]
        last = start if start is not None else (0.0 if len(points[0]) == 2 else (0.0, 0.0))
        heads = []
        for k in range(len(points)):
            if k and moves[k - 1] is not None:
                last = moves[k - 1]
            yaw = yaws[k] if yaws else None
            if yaw is not None:
                last = _make_ori(yaw, (pitches[k] or 0.0) if len(points[k]) == 3 else None)
            heads.append(_heading_of(last))
        return heads

    # ------------------------------------------------------------------ sweeps
    def _sweep(self, body, path, hold, heading, speed):
        """What ``body`` covers along ``path`` (from t = 0) for this table's agent: ``(vertex windows
        [(position, t1, t2)], edge windows [((p, q), t1, t2)])``. Cached for every table on the map."""
        other = self.agent
        mover = body if body is not None else Agent.disc(1e-9)   # a point obstacle: covered by the agent's reach
        key = (mover.key, None if other is None else other.key, self._rule, tuple(map(tuple, path)), hold,
               None if heading is None else np.asarray(heading, dtype=float).tobytes(), speed)

        def compute():
            (vid, t1, t2), (src, tgt, e1, e2) = self.graph_map.sweep_paths(
                mover, [np.array(path, dtype=float)], goal_hold=hold, other=other, clearance=self._rule,
                heading=None if heading is None else [heading], speed=speed)[0]
            nt = self.graph_map._node_current_tuples
            return ([(nt[v], a, b) for v, a, b in zip(vid.tolist(), t1.tolist(), t2.tolist())],
                    [((nt[u], nt[w]), a, b) for u, w, a, b in zip(src.tolist(), tgt.tolist(), e1.tolist(), e2.tolist()) if u != w])

        return self._cache.get_or_compute(key, compute)

    def _block_wait(self, body, p, heading, t1, t2):
        """``body`` standing at ``p`` over ``[t1, t2]`` blocks its footprint then."""
        vertices, edges = self._sweep(body, [p], float('inf'), heading, None)
        for v, _, _ in vertices:
            self.sipp_graph[v].split_interval(t1, t2)
        for e, _, _ in edges:
            self.sipp_graph[e].add_unsafe_interval(t1, t2)
        self._mark(vertices, edges)
        if self.oriented and vertices:
            self._remember((body, (p,), t1, t2 - t1, heading), [v for v, _, _ in vertices])

    def _block_move(self, body, p, q, t0, speed):
        """``body`` moving ``p -> q`` from ``t0`` (at ``speed``; None: in one time unit) blocks what it sweeps."""
        vertices, edges = self._sweep(body, [p, q], 0.0, None, speed)
        for v, a, b in vertices:
            self.sipp_graph[v].split_interval(t0 + a, t0 + b)
        for e, a, b in edges:
            self.sipp_graph[e].add_unsafe_interval(t0 + a, t0 + b)
        self._mark(vertices, edges)
        if self.oriented and vertices:
            if body is not None and speed != body.max_speed:     # (the exact test reads the speed from the body)
                body = Agent(body.shape, speed, body.max_rotation, body.type, body.clearance)
            self._remember((body, (p, q), t0, 0.0, None), [v for v, _, _ in vertices])

    # ------------------------------------------------------------------ vertices per heading
    def _remember(self, piece, vertices):
        k = len(self._pieces)
        self._pieces.append(piece)
        for v in vertices:
            if v not in self._blockers:              # its arrivals now need per-heading intervals
                self._dirty_into.add(v)
            self._blockers.setdefault(v, []).append(k)

    def vertex_node(self, position, ori):
        """The safe intervals of ``position`` for this table's agent standing there with heading ``ori``
        (None, or an agent that is a ball: its bounding-ball intervals). Built from the pieces that block
        the bounding ball there, each tested exactly, and kept up to date as pieces are added. The first
        request for a vertex works out every heading it can be entered with at once."""
        if not self.oriented or ori is None:
            return self.sipp_graph[position]
        nodes = self._exact.get(position)
        if nodes is None or ori not in nodes or self._current.get(position) != len(self._blockers.get(position, ())):
            self.prepare([(position, ori)])
            nodes = self._exact[position]
        return nodes[ori][0]

    def refresh(self):
        """Bring every per-heading vertex already worked out up to date with the pieces added since, in one
        exact test (called once a plan is committed, instead of vertex by vertex during the next search)."""
        if self.oriented and self._exact:
            self.prepare([(position, next(iter(nodes))) for position, nodes in self._exact.items()
                          if nodes and self._current.get(position) != len(self._blockers.get(position, ()))])

    def prepare(self, requests):
        """Bring the per-heading safe intervals of every ``(position, ori)`` in ``requests`` up to date, with
        one exact test for all of them (the search asks for all of a state's neighbours at once)."""
        todo, touched = [], []
        for position, ori in requests:
            blockers = self._blockers.get(position, ())
            nodes = self._exact.get(position)
            if nodes is not None and ori in nodes and self._current.get(position) == len(blockers):
                continue
            if nodes is None:
                nodes = self._exact[position] = {h: self._new_node(position, h) for h in self._arrival_headings(position)}
                self._dirty_into.add(position)
            if ori not in nodes:
                nodes[ori] = self._new_node(position, ori)
            for h, entry in nodes.items():
                if entry[1] < len(blockers):
                    todo.extend((position, h, k) for k in blockers[entry[1]:])
                    entry[1] = len(blockers)
            self._current[position] = len(blockers)
            touched.append(position)
        if todo:
            for (position, h, k), windows in zip(todo, self._exact_windows(todo)):
                t0 = self._pieces[k][2]
                node = self._exact[position][h][0]
                for w1, w2 in windows:
                    node.split_interval(t0 + w1, t0 + w2)
                if windows:
                    self._dirty_nodes.add(self._nid[(position, h)])

    def _arrival_headings(self, position) -> list:
        """The headings an agent has on arriving at ``position`` along a roadmap edge (worked out once per
        roadmap, for every table on it)."""
        S = self._static_arrays()
        heads = S["arrival"].get(position)
        if heads is None:
            if S["pred"] is None:
                pred = S["pred"] = [[] for _ in self.graph_map.nodes]
                for u, out in enumerate(self.graph_map.road_map):
                    for v in out:
                        pred[int(v)].append(u)
            v = int(self.graph_map.pos_index_dict[tuple(position)])
            nodes = self.graph_map.nodes
            heads = S["arrival"][position] = list(dict.fromkeys(_ori_between(nodes[u].current, position) for u in S["pred"][v]))
        return heads

    def _exact_windows(self, todo) -> list:
        """For each ``(position, heading, piece index)``: when the piece overlaps this table's agent standing
        at ``position`` with that heading, as windows from the piece's start. One exact test for all of
        them; cached per (agent, vertex, heading, piece)."""
        results, missing, standing, pieces = [None] * len(todo), [], {}, {}
        for n, (position, ori, k) in enumerate(todo):
            body, path, _, hold, head = self._pieces[k]
            mover = body if body is not None else Agent.disc(1e-9)
            key = ("exact", self._standing.key, position, ori, mover.key, path, hold,
                   None if head is None else np.asarray(head, dtype=float).tobytes())
            hit = self._cache.get(key)
            if hit is None:
                missing.append((n, key, standing.setdefault((position, ori), len(standing)), pieces.setdefault(k, len(pieces))))
            else:
                results[n] = hit
        if missing:
            # The agent stands at each (vertex, heading) for good from t = 0; each piece is started at
            # t = 0 too, so the windows come out relative to the piece's start.
            stands, ks = list(standing), list(pieces)
            body_of = lambda k: self._pieces[k][0] if self._pieces[k][0] is not None else Agent.disc(1e-9)
            agents = [self._standing] * len(stands) + [body_of(k) for k in ks]
            paths = [np.array([pos], dtype=float) for pos, _ in stands] + [np.array(self._pieces[k][1], dtype=float) for k in ks]
            heads = [_heading_of(o) for _, o in stands] + [self._pieces[k][4] for k in ks]
            holds = [float('inf')] * len(stands) + [self._pieces[k][3] for k in ks]
            pairs = [(si, len(stands) + pi) for _, _, si, pi in missing]
            i, j, t1, t2 = self.graph_map.agent_conflicts(agents, paths, t0=[0.0] * len(paths), goal_hold=holds,
                                                          heading=heads, pairs=sorted(set(pairs)), as_arrays=True)
            found = {pair: [] for pair in pairs}
            for a, b, w1, w2 in zip(np.asarray(i).tolist(), np.asarray(j).tolist(), np.asarray(t1).tolist(), np.asarray(t2).tolist()):
                found[(a, b)].append((w1, w2))
            for (n, key, _, _), pair in zip(missing, pairs):
                results[n] = found[pair]
                self._cache.get_or_compute(key, lambda pair=pair: found[pair])
        return results

    # ------------------------------------------------------------------ array copies (compiled search)
    def _new_node(self, position, ori):
        """A fresh per-heading interval list for ``(position, ori)``, with its node id."""
        node = SippNode()
        self._nid[(position, ori)] = len(self._nobj)
        self._dirty_nodes.add(len(self._nobj))
        self._nobj.append(node)
        return [node, 0]

    def _mark(self, vertices, edges):
        """Remember what changed for the next ``arrays()`` (from the table's creation on: a new table's arrays
        start out all safe and only what was blocked is copied in)."""
        S = self._static_arrays()
        vid, eid = S["vid"], S["eid"]
        self._dirty_nodes.update(vid[v] for v, _, _ in vertices)
        self._dirty_edges.update(eid[e] for e, _, _ in edges)

    def _static_arrays(self):
        """What the arrays share for every table on this roadmap (move lists, edge ids), built once per map."""
        if self._static is None:
            sig = self.graph_map._roadmap_signature()
            entry = _STATIC.get(self.graph_map)
            if entry is None or entry[0] != sig:
                entry = _STATIC[self.graph_map] = (sig, self._build_static())
            self._static = entry[1]
        return self._static

    def arrays(self, speed, duration):
        """The table as arrays for ``fast_search.search_chunk``, brought up to date: ``(mv_ptr, mv_to,
        mv_dur, mv_edge, mv_node, nd_lo, nd_hi, nd_n, ed_lo, ed_hi, ed_n)``. ``duration(length)``: move times
        for an agent of ``speed``. A move whose next vertex still needs its per-heading intervals has
        ``mv_node = -1``."""
        A = self._arr
        if A is None:
            A = self._arr = self._build_arrays()
            self._dirty_into.clear()
        else:
            for k in self._dirty_nodes:
                self._write_node(A, k)
            for k in self._dirty_edges:
                self._write_edge(A, k)
            if self.oriented:
                for position in self._dirty_into:
                    self._write_into(A, position)
        self._dirty_nodes.clear(); self._dirty_edges.clear(); self._dirty_into.clear()
        dur = A["dur"].get(speed)                          # (shared by every table on the map)
        if dur is None:
            dur = A["dur"][speed] = np.array([duration(L) for L in A["mv_len"]], dtype=np.float64)
        return (A["mv_ptr"], A["mv_to"], dur, A["mv_edge"], A["mv_node"],
                A["nd_lo"], A["nd_hi"], A["nd_n"], A["ed_lo"], A["ed_hi"], A["ed_n"])

    def _build_static(self):
        gm = self.graph_map
        nodes = gm.nodes
        n = len(nodes)
        vid = {tuple(node.current): i for i, node in enumerate(nodes)}
        edge_keys = [k for k in self.sipp_graph if isinstance(k[0], tuple)]     # (the order init_graph adds them)
        eid = {k: i for i, k in enumerate(edge_keys)}
        ptr, to, edge, length, heads = [0], [], [], [], []
        for u in range(n):                                   # the moves in the order the Python search takes them
            p = tuple(nodes[u].current)
            for nb in gm.get_neighbors(u):
                q = tuple(nb.current)
                to.append(vid[q]); edge.append(eid.get((p, q), -1))
                length.append(float(gm.get_distance(p, q))); heads.append(_ori_between(p, q))
            ptr.append(len(to))
        into = [[] for _ in range(n)]
        for k, v in enumerate(to):
            into[v].append(k)
        return {"vid": vid, "eid": eid, "edge_keys": edge_keys, "dur": {}, "mv_ptr": np.array(ptr, dtype=np.int64),
                "mv_to": np.array(to, dtype=np.int64), "mv_edge": np.array(edge, dtype=np.int64), "mv_len": length,
                "mv_head": heads, "into": into, "pred": None, "arrival": {}}

    def _build_arrays(self):
        """This table's arrays: every vertex safe for good and every edge clear, then what was blocked."""
        S = self._static_arrays()
        A = dict(S)
        A["edges"] = [self.sipp_graph[k] for k in S["edge_keys"]]
        A["mv_node"] = S["mv_to"].copy()
        rows = max(16, 2 * len(self._nobj))
        A["nd_lo"] = np.zeros((rows, 4)); A["nd_hi"] = np.zeros((rows, 4)); A["nd_hi"][:, 0] = np.inf
        A["nd_n"] = np.ones(rows, dtype=np.int64)
        A["ed_lo"] = np.zeros((max(1, len(S["edge_keys"])), 4)); A["ed_hi"] = np.zeros_like(A["ed_lo"])
        A["ed_n"] = np.zeros(A["ed_lo"].shape[0], dtype=np.int64)
        for k in self._dirty_nodes:
            self._write_node(A, k)
        for k in self._dirty_edges:
            self._write_edge(A, k)
        if self.oriented:
            for position in self._blockers:
                self._write_into(A, position)
        return A

    def _write_node(self, A, k):
        lst = self._nobj[k].interval_list
        if k >= A["nd_lo"].shape[0] or len(lst) > A["nd_lo"].shape[1]:
            rows = max(A["nd_lo"].shape[0], 2 * (k + 1))
            cols = max(A["nd_lo"].shape[1], 2 * len(lst))
            for name in ("nd_lo", "nd_hi"):
                grown = np.zeros((rows, cols)); grown[:A[name].shape[0], :A[name].shape[1]] = A[name]; A[name] = grown
            grown = np.zeros(rows, dtype=np.int64); grown[:len(A["nd_n"])] = A["nd_n"]; A["nd_n"] = grown
        A["nd_n"][k] = len(lst)
        for j, (lo, hi) in enumerate(lst):
            A["nd_lo"][k, j], A["nd_hi"][k, j] = lo, hi

    def _write_edge(self, A, k):
        rec = A["edges"][k]
        m = len(rec)
        if m > A["ed_lo"].shape[1]:
            cols = 2 * m
            for name in ("ed_lo", "ed_hi"):
                grown = np.zeros((A[name].shape[0], cols)); grown[:, :A[name].shape[1]] = A[name]; A[name] = grown
        A["ed_n"][k] = m
        if m:
            A["ed_lo"][k, :m] = rec.starts
            A["ed_hi"][k, :m] = rec.ends

    def _write_into(self, A, position):
        """The interval list each move into ``position`` arrives in: the vertex's own while nothing blocks
        it, else the one for the move's heading (-1 until it is worked out)."""
        v = A["vid"][position]
        blocked = position in self._blockers
        for k in A["into"][v]:
            A["mv_node"][k] = self._nid.get((position, A["mv_head"][k]), -1) if blocked else v
