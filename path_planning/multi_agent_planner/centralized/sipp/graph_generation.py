from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.node import Node
from typing import Any, List, Tuple
import bisect
import math
import numpy as np

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
    def is_equal_location(self, other):
        return self.position == other.position

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
                interval_list.append((s_start, s_end))
                
            # Case 2: Blocked window is entirely before this interval
            elif b_end <= s_start:
                interval_list.append((s_start, s_end))
                
            # Case 3: Overlap occurs
            else:
                # Check for a left remnant
                if b_start > s_start:
                    interval_list.append((s_start, b_start))
                
                # Check for a right remnant
                if b_end < s_end:
                    interval_list.append((b_end, s_end))
        self.interval_list = sorted([(start, end) for start, end in interval_list if end-start > 1e-6])

    def is_in_safe_interval(self, arrive_t):
        lo, hi = 0, len(self.interval_list) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end = self.interval_list[mid]
            if start <= arrive_t and arrive_t <= end:
                return True
            elif arrive_t < start:
                hi = mid - 1
            else:
                lo = mid + 1
        return False

    def copy(self):
        """Return a new SippNode with the same safe intervals."""
        n = SippNode()
        n.interval_list = list(self.interval_list)
        return n

    def merge_safe_intervals(self, other):
        """
        Set self.interval_list to the intersection of self's and other's safe
        intervals (so that a time is safe only if it is safe in both).
        """
        a = self.interval_list
        b = other.interval_list
        result = []
        i, j = 0, 0
        while i < len(a) and j < len(b):
            a_lo, a_hi = a[i]
            b_lo, b_hi = b[j]
            lo = max(a_lo, b_lo)
            hi = min(a_hi, b_hi)
            if lo <= hi:
                result.append((lo, hi))
            if a_hi <= b_hi:
                i += 1
            else:
                j += 1
        self.interval_list = result

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

    def __getitem__(self, idx):
        return (self.starts[idx], self.ends[idx])

    def __repr__(self):
        return f"UnsafeIntervalList({list(self)})"

    def to_list(self):
        return list(self)

    def clear(self):
        self.starts = None
        self.ends = None

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
    init_graph builds one SippEdge per directed edge and self-loop on every
    reset_graph, so construction cost matters.
    """

    __slots__ = ()

    @property
    def unsafe_interval_list(self):
        return self

    def add_unsafe_interval(self, t1, t2):
        self.add(t1, t2)

    def get_unsafe_intervals(self, t1, t2):
        return self.overlapping(t1, t2)

    def is_in_unsafe_interval(self, t1, t2):
        return self.intersects(t1, t2)


# Upper bound on the roadmap-level sweep memo (entries); cleared when exceeded so
# very long runs with ever-new time-sampled query points stay bounded in memory.
SWEEP_MEMO_MAX_ENTRIES = 250_000


def cached_constraint_sweep(graph_map, p1, p2, v, r):
    """Memoized ``graph_map.get_constraint_sweep(p1, p2, v, r, use_interval=True,
    get_time_interval=True)``: ``(vertex -> (t_start, t_end), (u, v) edge ->
    (t_start, t_end))`` for a disk of radius ``r`` moving ``p1 -> p2`` at speed
    ``v``, times relative to the start of that move. The memo lives on the roadmap
    (``graph_map._sipp_sweep_memo``) so it survives across SippGraph instances:
    NeuralATTF builds one planner per low-level call and re-sweeps the other
    agents' still-committed paths every time, which is why the same queries recur
    call after call. ``GraphSampler.set_constraint_sweep`` drops it whenever the
    roadmap itself changes."""
    key = (p1, p2, v, r)
    memo = getattr(graph_map, "_sipp_sweep_memo", None)
    if memo is None:
        memo = graph_map._sipp_sweep_memo = {}
    hit = memo.get(key)
    if hit is None:
        if len(memo) >= SWEEP_MEMO_MAX_ENTRIES:
            memo.clear()
        hit = memo[key] = graph_map.get_constraint_sweep(p1, p2, v, r, use_interval=True, get_time_interval=True)
    return hit


class SippGraph(object):
    def __init__(self, graph_map: GraphSampler,dynamic_obstacles:dict = {},radius:float = 0.0,velocity:float = 0.0,use_constraint_sweep:bool = True, heuristic_type: str = 'manhattan',time_limit: float | None = None, max_iterations: int | None = None,verbose: bool = False, obstacle_horizon: float | dict | None = None):
        self.graph_map = graph_map
        self.dyn_obstacles = {}
        # How long a dynamic obstacle's FINAL (resting) position blocks its footprint.
        # Unbounded (float('inf')) reproduces the original "rest forever" behaviour; a finite
        # value lets the planner route through a spot another agent currently rests on for
        # arrivals beyond the horizon. ``obstacle_horizon`` is either one scalar for every
        # obstacle, or a dict ``{obstacle_name: horizon}`` (``None``/``<= 0`` -> inf; names
        # missing from the dict rest forever). NeuralATTF uses the dict form: a moving agent's
        # final waypoint is held for one timestep (it is reassigned next tick), while an idle,
        # unassigned agent blocks its spot until it is told to move.
        if isinstance(obstacle_horizon, dict):
            self.obstacle_horizon = float('inf')
            self.obstacle_horizons = {name: self._norm_horizon(h) for name, h in obstacle_horizon.items()}
        else:
            self.obstacle_horizon = self._norm_horizon(obstacle_horizon)
            self.obstacle_horizons = {}
        self.sipp_graph = {}
        if radius > 0:
            # Radius-based SIPP always relies on constraint sweep queries.
            # Force-enable the graph flag to prevent get_constraint_sweep returning None.
            if hasattr(self.graph_map, "use_constraint_sweep") and not self.graph_map.use_constraint_sweep:
                self.graph_map.use_constraint_sweep = True
            self.graph_map.set_constraint_sweep()
        self.radius = radius
        self.velocity = velocity
        self.use_constraint_sweep = use_constraint_sweep
        self.heuristic_type = heuristic_type
        self.dynamic_obstacles = dynamic_obstacles
        self.reset_graph()
        self._valid_neighbours_cache = {}
        self.time_limit = time_limit if time_limit is not None and time_limit > 0 else float('inf')
        self.max_iterations = max_iterations if max_iterations is not None and max_iterations > 0 else 1
        self.verbose = verbose
        self.total_time = 0
        self.total_iterations = 0

    def init_graph(self):
        for node in self.graph_map.nodes:
            node_sipp_dict = {node.current:SippNode()}
            self.sipp_graph.update(node_sipp_dict)
            # Self-loop (wait) edge: the constraint sweep emits (p, p) edge keys.
            self.sipp_graph[(node.current, node.current)] = SippEdge()

        # Initialize SIPP edges keyed by endpoint positions (p1, p2),
        # to match the keys returned by GraphSampler.get_constraint_sweep.
        for edge in self.graph_map.edges:
            src_idx, tgt_idx = edge
            src_pos = self.graph_map.nodes[src_idx].current
            tgt_pos = self.graph_map.nodes[tgt_idx].current
            # graph_map.edges usually lists both directions; setdefault avoids
            # constructing (and discarding) a second SippEdge per undirected edge.
            # graph_map.edges usually lists both directions: construct each record
            # once (setdefault would build and discard a SippEdge per duplicate).
            if (src_pos, tgt_pos) not in self.sipp_graph:
                self.sipp_graph[(src_pos, tgt_pos)] = SippEdge()
            if (tgt_pos, src_pos) not in self.sipp_graph:
                self.sipp_graph[(tgt_pos, src_pos)] = SippEdge()

    @staticmethod
    def _norm_horizon(horizon) -> float:
        """``None`` / non-positive -> rest forever (inf); otherwise the horizon in seconds."""
        return float(horizon) if horizon is not None and horizon > 0 else float('inf')

    def _horizon_for(self, dyn_name) -> float:
        """Resting horizon of one dynamic obstacle (per-name override, else the global value)."""
        return self.obstacle_horizons.get(dyn_name, self.obstacle_horizon)

    def init_intervals(self,dyn_obstacles:dict = {}):
        """Block the SIPP graph with other agents' timed schedules ``[{x, y, t}, ...]``.

        Per schedule entry (radius > 0):
        - final point: the resting footprint (vertices AND edges) is blocked for
          ``[t, t + horizon]`` where ``horizon`` comes from :meth:`_horizon_for`;
        - a wait (next point at the same position): the footprint is blocked for
          exactly ``[t, next_t]`` from the schedule's own timestamps. The stationary
          sweep itself reports ``(0, inf)``, so using it directly (as this method
          once did) blocked every wait forever, regardless of the horizon;
        - a move: the sweep's relative windows, computed at the speed the schedule
          actually implies (``dist / (next_t - t)``). A time-sampled schedule whose
          tick straddles a wait-then-move covers less than ``velocity * dt``; sweeping
          it at the nominal velocity would compress the window and leave the segment
          end unblocked while the obstacle is still moving through it.
        """
        if not dyn_obstacles or len(dyn_obstacles) == 0: return
        for dyn_name, schedule in dyn_obstacles.items():
            self.dyn_obstacles[dyn_name] = np.array([State(position=(location["x"],location["y"]), t=location["t"]) for location in schedule])
            horizon = self._horizon_for(dyn_name)
            n_points = len(schedule)
            for i in range(n_points):
                location = schedule[i]
                position = (location["x"],location["y"])
                t = max(0.0, float(location["t"]))

                last_t = i == n_points-1

                if self.radius > 0:
                    if last_t:
                        overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, position,self.velocity, 2*self.radius)
                        next_t = t + horizon  # inf -> blocks forever
                        for vertex_pos in overlapping_vertices:
                            self.sipp_graph[vertex_pos].split_interval(t, next_t)
                        for edge_pos in overlapping_edges:
                            self.sipp_graph[edge_pos].add_unsafe_interval(t, next_t)
                        continue

                    next_location = schedule[i + 1]
                    next_position = (next_location["x"], next_location["y"])
                    next_t = max(0.0, float(next_location["t"]))
                    if next_position == position:
                        # Wait: block the footprint for the schedule's own window only.
                        if next_t - t > 1e-9:
                            overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, position,self.velocity, 2*self.radius)
                            for vertex_pos in overlapping_vertices:
                                self.sipp_graph[vertex_pos].split_interval(t, next_t)
                            for edge_pos in overlapping_edges:
                                self.sipp_graph[edge_pos].add_unsafe_interval(t, next_t)
                        continue

                    # Move: sweep at the speed the schedule implies (falls back to the
                    # nominal velocity when they agree, to keep the sweep cache warm).
                    sweep_velocity = self.velocity
                    duration = next_t - t
                    if self.velocity > 0 and duration > 1e-9:
                        implied = math.dist(position, next_position) / duration
                        if abs(implied - self.velocity) > 1e-6:
                            sweep_velocity = implied
                    overlapping_vertices, overlapping_edges = self._get_constraint_sweep_cached(position, next_position, sweep_velocity, 2 * self.radius)
                    for vertex_pos, vertex_interval in overlapping_vertices.items():
                        t_start,t_end = vertex_interval
                        t1 = t+t_start
                        t2 = t+t_end
                        self.sipp_graph[vertex_pos].split_interval(t1, t2)
                    for edge_pos, edge_interval in overlapping_edges.items():
                        # edge_interval is a 2-tuple (t_start, t_end) from
                        # get_constraint_sweep(get_time_interval=True); mirror the
                        # (correct) edge handling in update_intervals.
                        t_start, t_end = edge_interval
                        self.sipp_graph[edge_pos].add_unsafe_interval(t + t_start, t + t_end)
                else:
                    t1 = t
                    t2 = t1 + 1 if not last_t else (t1 + horizon)
                    self.sipp_graph[position].split_interval(t1, t2,1)

        # Update the intervals of the SIPP graph based on the agent's plan (treated as dynamic obstacles)
   
    def update_intervals(self,plans: List[List[State]] | List[State] | State,action_costs: List[List[Tuple[float,float]]] | List[Tuple[float,float]] | Tuple[float,float],dyn_names: List[str]):
        if not plans or len(plans) == 0: return
        self._graph_fresh = False
        for plan,action_cost,dyn_name in zip(plans,action_costs,dyn_names):
            for i in range(len(plan)):
                location = plan[i]
                position = location.position
                t = location.time
                last_t = i == len(plan)-1

                if self.radius > 0:
                    # Last time step
                    if last_t:
                        overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, position,self.velocity, 2*self.radius)
                        next_t = float('inf')
                        for vertex_pos, vertex_interval in overlapping_vertices.items():
                            self.sipp_graph[vertex_pos].split_interval(t, next_t)
                        for edge_pos, edge_interval in overlapping_edges.items():
                            self.sipp_graph[edge_pos].add_unsafe_interval(t, float('inf'))
                        continue
                    
                    # Intermediate time step between two locations
                    next_location = plan[i+1]
                    next_position = next_location.position
                    next_t = next_location.time
                    wait_time, move_time = action_cost[i]
                    if wait_time > 1e-10:
                        t0 = t+wait_time
                        overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, position,self.velocity, 2*self.radius)
                        for vertex_pos, vertex_interval in overlapping_vertices.items():
                            self.sipp_graph[vertex_pos].split_interval(t, t0)
                        for edge_pos, edge_interval in overlapping_edges.items():
                            self.sipp_graph[edge_pos].add_unsafe_interval(t, t0)
                    else:
                        t0 = t
                    overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, next_position,self.velocity, 2*self.radius)
                    for vertex_pos, vertex_interval in overlapping_vertices.items():
                        t_start,t_end = vertex_interval
                        t1 = t0+t_start
                        t2 = t0+t_end 
                        self.sipp_graph[vertex_pos].split_interval(t1, t2)
                    for edge_pos, edge_interval in overlapping_edges.items():
                        t_start,t_end = edge_interval
                        self.sipp_graph[edge_pos].add_unsafe_interval(t0+t_start, t0+t_end)

                    
                else:
                    t1 = t
                    if self.velocity > 0:
                        t2 = t1 + 1/self.velocity
                    else:
                        t2 = t1 + 1 if not last_t else float('inf')
                    self.sipp_graph[position].split_interval(t1, t2,1)

    def is_valid_position(self, position):
        return not self.graph_map.in_collision_point(position)

    def get_valid_neighbours(self, position):
        neighbors = []
        node = Node(tuple(position))
        nodes = self.graph_map.get_neighbors(node)

        # Move action
        for node in nodes:
            # if self.is_valid_position(node.current):
            neighbors.append(node.current)
        return neighbors

    def _get_constraint_sweep_cached(self, p1, p2,v, r):
        """Memoized ``get_constraint_sweep`` (interval form); see
        :func:`cached_constraint_sweep`."""
        return cached_constraint_sweep(self.graph_map, p1, p2, v, r)

    def reset_graph(self):
        self.sipp_graph = {}
        self.init_graph()
        self.init_intervals(self.dynamic_obstacles)
        # Fresh: only dynamic obstacles are in it. compute_plan skips its own
        # reset while this holds (a planner built and used once pays one build).
        self._graph_fresh = True