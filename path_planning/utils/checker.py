import numpy as np
from typing import Mapping, Union

from path_planning.utils.agent_shape import (
    AlignedShape,
    ConvexPolygon,
    Disc,
    heading_key,
    moving_point_interval,
    oriented_pair_clearance,
    schedule_headings,
)

# Agent footprint spec: one scalar radius for every agent, or ``{agent_name:
# radius | ConvexPolygon | AlignedShape}`` for heterogeneous agents. Two discs
# collide when their centre distance is ``<= r_i + r_j`` (``2r`` for a scalar);
# when either agent is a polygon the pair collides when ``c_j - c_i`` lies inside
# the combined footprint ``P_i (+) (-P_j)`` (``agent_shape.pair_clearance``). An
# ``AlignedShape`` is rotated to the heading the agent holds on each segment
# (``heading`` keys of the schedule entries, else ``agent_shape.path_headings``).
RadiusSpec = Union[float, Mapping[str, Union[float, ConvexPolygon, AlignedShape, Disc]]]
VelocitySpec = Union[float, Mapping[str, float], None]


def _footprint_for(r: RadiusSpec, name: str):
    """Footprint of ``name`` under spec ``r``: a float radius, a ``ConvexPolygon``
    or an ``AlignedShape``."""
    if isinstance(r, Mapping):
        if name not in r:
            raise ValueError(f"no radius given for agent {name!r}")
        val = r[name]
    else:
        val = r
    if isinstance(val, (ConvexPolygon, AlignedShape)):
        return val
    if isinstance(val, Disc):
        return float(val.radius)
    return float(val)


def _radius_for(r: RadiusSpec, name: str) -> float:
    """Radius of ``name`` under spec ``r`` (bounding radius for polygon agents)."""
    fp = _footprint_for(r, name)
    return float(fp.bounding_radius) if isinstance(fp, (ConvexPolygon, AlignedShape)) else fp


def _pair_footprint(r: RadiusSpec, name_i: str, name_j: str, heading_i=None, heading_j=None):
    """``(collision_radius, polygon)``: the float ``r_i + r_j`` with ``polygon=None``
    for a disc pair, else ``(None, P_i (+) (-P_j))`` with each aligned footprint
    rotated to its heading."""
    fi = _footprint_for(r, name_i)
    fj = _footprint_for(r, name_j)
    if isinstance(fi, (ConvexPolygon, AlignedShape)) or isinstance(fj, (ConvexPolygon, AlignedShape)):
        si = fi if isinstance(fi, (ConvexPolygon, AlignedShape)) else Disc(fi)
        sj = fj if isinstance(fj, (ConvexPolygon, AlignedShape)) else Disc(fj)
        combo = oriented_pair_clearance(si, heading_i, sj, heading_j)
        if isinstance(combo, Disc):
            # both resolved to discs (e.g. an aligned shape rotating in place -> envelope)
            return combo.radius, None
        return None, combo
    return fi + fj, None


def _velocity_for(v: VelocitySpec, name: str):
    if v is None:
        return None
    if isinstance(v, Mapping):
        return float(v[name]) if name in v else None
    return float(v)

def check_time_anomaly(solution,verbose: bool = True):
    agents = {agent:np.array([[p['t'],p['x'], p['y']] for p in path] ) for agent,path in solution.items()}
    no_anomaly = True
    for agent, path in agents.items():
        t_diff = np.diff(path[:,0], axis=0)
        if np.any(t_diff < 0):
            t_ind = np.where(t_diff < 0)[0][0]
            print(f"Agent {agent} has negative time difference at t={path[t_ind]['t']:.3f}")
            no_anomaly = False
    if no_anomaly and verbose:
        print("No time anomaly detected")
    return no_anomaly


def check_velocity_anomaly(solution,is_using_constant_speed: bool = False,verbose: bool = True, velocity: VelocitySpec = None, tol: float = 1e-6, speed_zones=None):
    """Flag agents whose motion speed is inconsistent.

    ``velocity`` (scalar or ``{agent_name: speed}``): when given and > 0 for an
    agent, every moving segment of that agent must run at that nominal speed
    (relative tolerance ``tol``); 0 / None keeps the historical median-based
    constant-speed check driven by ``is_using_constant_speed``. With
    ``speed_zones`` (``SpeedZone`` list, see ``speed_zones.segment_speed_limit``)
    the expected speed of a segment is ``min(nominal, lowest limit of the zones the
    segment touches)``.
    """
    from path_planning.common.environment.map.speed_zones import segment_speed_limit

    agents = {agent:np.array([[p['t'],p['x'], p['y']] for p in path] ) for agent,path in solution.items()}
    no_anomaly = True
    for agent, path in agents.items():
        diff = np.diff(path[:,1:], axis=0)
        t_diff = np.diff(path[:,0], axis=0)
        keep = t_diff > 1e-10
        v_diff = diff[keep] / t_diff[keep].reshape(-1,1)
        speed = np.linalg.norm(v_diff, axis=1)
        if speed.size == 0:
            # No motion segments for this agent (e.g., start==goal), so nothing to validate.
            continue

        v_nom = _velocity_for(velocity, agent)
        if v_nom is not None and v_nom > 0:
            moving_mask = speed > 1e-9
            moving = speed[moving_mask]
            expected = np.full(moving.shape, float(v_nom))
            if speed_zones:
                seg_pts = path[:-1, 1:][keep][moving_mask]
                seg_nxt = path[1:, 1:][keep][moving_mask]
                for k, (a, b) in enumerate(zip(seg_pts, seg_nxt)):
                    lim = segment_speed_limit(speed_zones, a, b)
                    if lim is not None:
                        expected[k] = min(expected[k], lim)
            if moving.size and np.any(np.abs(moving - expected) > tol * np.maximum(1.0, expected)):
                print(f"Agent {agent} moves at {moving.min():.4f}..{moving.max():.4f}, nominal speed {v_nom:.4f}"
                      + (" (zone limits applied)" if speed_zones else ""))
                no_anomaly = False
            continue

        median_speed = np.median(speed)
        if is_using_constant_speed and np.any(speed-median_speed> 1e-9) :
            non_zero_speed = speed[speed > 1e-9]
            if np.any(non_zero_speed> 0) :
                continue
            print(f"Agent {agent} has non-constant velocity")
            no_anomaly = False
    if no_anomaly and verbose:
        print("No velocity anomaly detected")
    return no_anomaly


def check_direction_anomaly(solution, direction_zones, verbose: bool = True):
    """Moves that break a direction zone (``DirectionZone`` list or YAML-style
    dicts, see ``direction_zones.py``): driving a one-way region against its
    heading, or an angle region at a forbidden angle.

    Every moving segment of every agent's schedule is tested with the zones it
    touches. Returns the list of violations
    ``{"agent", "t", "from", "to", "heading_deg", "zone"}`` (empty = clean).
    """
    import math
    from path_planning.common.environment.map.direction_zones import parse_direction_zones, violated_zone

    zones = parse_direction_zones(direction_zones)
    violations = []
    if not zones:
        return violations
    for agent, path in solution.items():
        for a, b in zip(path, path[1:]):
            pa = (float(a["x"]), float(a["y"]))
            pb = (float(b["x"]), float(b["y"]))
            if math.dist(pa, pb) <= 1e-9:
                continue
            z = violated_zone(zones, pa, pb)
            if z is None:
                continue
            item = {"agent": agent, "t": float(a["t"]), "from": pa, "to": pb,
                    "heading_deg": math.degrees(math.atan2(pb[1] - pa[1], pb[0] - pa[0])) % 360.0,
                    "zone": z.name if z.name is not None else z.to_spec()}
            violations.append(item)
            if verbose:
                print(f"Direction violation: {agent} at t={item['t']:.3f} moves {pa} -> {pb} "
                      f"(heading {item['heading_deg']:.1f} deg) in zone {item['zone']}")
    if verbose and not violations:
        print("No direction violation detected")
    return violations


def check_collision(solution, r: RadiusSpec, verbose: bool = True):
    """
    Check for collisions between all pairs of agents in a solution, and
    compute the time intervals during which they collide.

    `solution` is a dict: agent_name -> list of {"t", "x", "y"}.
    Agents are modeled as discs; ``r`` is either one radius for every agent or a
    ``{agent_name: radius}`` mapping (heterogeneous agents). Two agents collide
    when their centre distance is <= r_i + r_j (2r for a scalar) at any time.
    Polygon footprints (``ConvexPolygon`` / heading-aligned ``AlignedShape``
    values, see ``RadiusSpec``) collide when the relative position enters the
    combined polygon; aligned ones are rotated to the heading held on each
    segment (``heading`` keys in the schedule, else carried from the moves).

    Returns
    -------
    collisions : dict
        Mapping (agent_i, agent_j) -> list of (t_start, t_end) intervals
        (in global time) over which the two agents are in collision.
    """
    # Keep trajectories as lists of dicts for clarity
    agents = {name: path for name, path in solution.items()}
    agent_names = list(agents.keys())
    no_collision = True
    collisions = {}
    # Per-point headings (the orientation held from that point to the next).
    aligned_names = {
        name for name in agent_names if isinstance(_footprint_for(r, name), AlignedShape)
    }
    headings = {name: schedule_headings(agents[name]) for name in aligned_names}
    pair_cache = {}

    def get_heading(name, traj, t):
        """Heading of ``name`` on the segment that starts at or before ``t``;
        ``None`` (envelope footprint) while the agent rotates in place
        (``turn_to`` key on the segment's start point)."""
        if name not in headings:
            return None
        hs = headings[name]
        n = len(traj)
        k = 0
        while k + 1 < n and traj[k + 1]["t"] <= t:
            k += 1
        if k + 1 < n and traj[k].get("turn_to") is not None:
            return None
        return hs[k]

    def pair_footprint(name_i, name_j, h_i, h_j):
        key = (name_i, name_j, h_i, h_j)
        fp = pair_cache.get(key)
        if fp is None:
            fp = _pair_footprint(r, name_i, name_j, h_i, h_j)
            pair_cache[key] = fp
        return fp

    def get_pos(traj, t):
        """Linear interpolation for position at time t along a trajectory."""
        n = len(traj)
        if n == 0:
            return np.array([0.0, 0.0])
        if n == 1:
            return np.array([traj[0]["x"], traj[0]["y"]])
        for i in range(n - 1):
            t0 = traj[i]["t"]
            t1 = traj[i + 1]["t"]
            if t0 <= t <= t1:
                dt = t1 - t0
                if dt == 0:
                    return np.array([traj[i]["x"], traj[i]["y"]])
                ratio = (t - t0) / dt
                x = traj[i]["x"] + ratio * (traj[i + 1]["x"] - traj[i]["x"])
                y = traj[i]["y"] + ratio * (traj[i + 1]["y"] - traj[i]["y"])
                return np.array([x, y])
        # If t is beyond the last timestamp, hold last position
        return np.array([traj[-1]["x"], traj[-1]["y"]])

    # Check all unordered pairs of agents
    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            name_i = agent_names[i]
            name_j = agent_names[j]
            traj_i = agents[name_i]
            traj_j = agents[name_j]

            pair_key = (name_i, name_j)
            pair_intervals = []
            oriented = name_i in aligned_names or name_j in aligned_names
            collision_radius, pair_polygon = pair_footprint(name_i, name_j, None, None)
            collision_radius_sq = None if pair_polygon is not None else collision_radius ** 2

            # Collect all unique time breakpoints from both trajectories
            times_i = [p["t"] for p in traj_i]
            times_j = [p["t"] for p in traj_j]
            all_t = sorted(set(times_i + times_j))
            if len(all_t) < 2:
                continue

            # Sweep through each time interval and check minimum distance
            for k in range(len(all_t) - 1):
                t_start, t_end = all_t[k], all_t[k + 1]
                duration = t_end - t_start
                if duration <= 0:
                    continue

                if oriented:
                    # Footprints rotated to the headings held on this slice.
                    collision_radius, pair_polygon = pair_footprint(
                        name_i, name_j, get_heading(name_i, traj_i, t_start), get_heading(name_j, traj_j, t_start)
                    )
                    collision_radius_sq = None if pair_polygon is not None else collision_radius ** 2

                # Positions and velocities over this slice
                p_i_start = get_pos(traj_i, t_start)
                p_j_start = get_pos(traj_j, t_start)
                v_i = (get_pos(traj_i, t_end) - p_i_start) / duration
                v_j = (get_pos(traj_j, t_end) - p_j_start) / duration

                # Relative motion: agent j relative to agent i
                P_rel = p_j_start - p_i_start
                V_rel = v_j - v_i
                # Quadratic distance^2(t) = a t^2 + b t + c on local t in [0, duration]
                a = np.dot(V_rel, V_rel)
                b = 2 * np.dot(P_rel, V_rel)
                c = np.dot(P_rel, P_rel)

                # Solve for times in [0, duration] where distance^2(t) <= collision_radius_sq
                # i.e. a t^2 + b t + (c - collision_radius_sq) <= 0
                eps = 1e-9
                if pair_polygon is not None:
                    # Polygon pair: relative position strictly inside P_i (+) (-P_j).
                    t_lo, t_hi, hit = moving_point_interval(
                        P_rel.reshape(1, -1), V_rel, pair_polygon.normals, pair_polygon.offsets,
                        duration, exit_slack=0.0,
                    )
                    if not hit[0]:
                        continue
                    local_start, local_end = float(t_lo[0]), float(t_hi[0])
                    if local_start >= local_end - eps:
                        continue
                elif a < eps:
                    # Relative velocity ~ 0: distance approximately constant over this slice.
                    if c + eps < collision_radius_sq:
                        # Entire local interval is colliding.
                        local_start, local_end = 0.0, duration
                    else:
                        continue
                else:
                    A = a
                    B = b
                    C = c - collision_radius_sq + eps
                    disc = B * B - 4 * A * C
                    if disc < -eps:
                        # No real roots -> always outside or always inside; but since
                        # a > 0 and we subtracted the threshold, this means no collision.
                        continue
                    disc = max(disc, 0.0)
                    sqrt_disc = np.sqrt(disc)
                    t1 = (-B - sqrt_disc) / (2 * A)
                    t2 = (-B + sqrt_disc) / (2 * A)
                    if t1 > t2:
                        t1, t2 = t2, t1
                    # Intersection with [0, duration]
                    local_start = max(0.0, t1)
                    local_end = min(duration, t2)
                    if local_start >= local_end - eps:
                        continue

                # Convert to global time interval
                g_start = t_start + local_start
                g_end = t_start + local_end
                pair_intervals.append((g_start, g_end,t_start, t_end))

            # Optionally merge overlapping/adjacent intervals for this pair
            if pair_intervals:
                pair_intervals.sort(key=lambda iv: iv[0])
                merged = [pair_intervals[0]]
                for s, e, t_start_local, t_end_local in pair_intervals[1:]:
                    last_s, last_e, last_t_start_local, last_t_end_local = merged[-1]
                    if s <= last_e + 1e-9:
                        merged[-1] = (last_s, max(last_e, e), last_t_start_local, max(last_t_end_local, t_end_local))
                    else:
                        merged.append((s, e, t_start_local, t_end_local))
                collisions[pair_key] = merged

    if verbose:
        if collisions:
            for (name_i, name_j), ivs in collisions.items():
                for g_start, g_end, t_start, t_end in ivs:
                    print(
                        f"Agents {name_i} and {name_j} collide over "
                        f"[t={g_start:.3f}, t={g_end:.3f}] in local time [t={t_start:.3f}, t={t_end:.3f}]"
                    )
        else:
            print("No collision detected")
    return collisions


def check_solution(solution):
    check_time_anomaly(solution)
    check_velocity_anomaly(solution)
    check_collision(solution, 1.0)


def is_conflict_free(solution, r: RadiusSpec, verbose: bool = False):
    """
    Return True iff no pair of agents in `solution` collides.

    Parameters
    ----------
    solution : dict
        Mapping agent_name -> list of waypoints {"t", "x", "y"}.
    r : float | Mapping[str, float]
        Agent radius, or per-agent radii (collision if distance <= r_i + r_j).
    verbose : bool, optional
        Whether `check_collision` prints per-collision details.

    Returns
    -------
    bool
        True if the solution is non-empty and collision-free, False otherwise.
    """
    if not solution:
        return False
    return len(check_collision(solution, r, verbose=verbose)) == 0


def check_solution_full(
    solution,
    r: RadiusSpec,
    is_using_constant_speed: bool = False,
    verbose: bool = True,
    velocity: VelocitySpec = None,
    speed_zones=None,
    direction_zones=None,
):
    """
    Run all anomaly checks (time, velocity, collision, and direction zones when
    ``direction_zones`` is given: key ``"direction_violations"``) in one call.

    Parameters
    ----------
    solution : dict
        Mapping agent_name -> list of waypoints {"t", "x", "y"}.
    r : float
        Agent radius.
    is_using_constant_speed : bool, optional
        Whether to enforce constant speed in `check_velocity_anomaly`.
    verbose : bool, optional
        Whether to print detailed messages from the underlying checks.

    Returns
    -------
    result : dict
        {
            "no_time_anomaly": bool,
            "no_velocity_anomaly": bool,
            "collisions": dict,  # as returned by check_collision
        }
    """
    no_time_anomaly = check_time_anomaly(solution, verbose=verbose)
    no_velocity_anomaly = check_velocity_anomaly(
        solution,
        is_using_constant_speed=is_using_constant_speed,
        verbose=verbose,
        velocity=velocity,
        speed_zones=speed_zones,
    )
    collisions = check_collision(solution, r, verbose=verbose)

    result = {
        "no_time_anomaly": no_time_anomaly,
        "no_velocity_anomaly": no_velocity_anomaly,
        "collisions": collisions,
    }
    if direction_zones:
        result["direction_violations"] = check_direction_anomaly(solution, direction_zones, verbose=verbose)
    return result
