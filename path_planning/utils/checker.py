import numpy as np


def _xyz(p) -> list:
    """Position of one schedule entry: ``[x, y]``, or ``[x, y, z]`` in 3D."""
    return [p["x"], p["y"], p["z"]] if "z" in p else [p["x"], p["y"]]


def check_time_anomaly(solution,verbose: bool = True):
    agents = {agent:np.array([[p['t']] + _xyz(p) for p in path] ) for agent,path in solution.items()}
    no_anomaly = True
    for agent, path in agents.items():
        t_diff = np.diff(path[:,0], axis=0)
        if np.any(t_diff < 0):
            t_ind = np.where(t_diff < 0)[0][0]
            print(f"Agent {agent} has negative time difference at t={path[t_ind, 0]:.3f}")
            no_anomaly = False
    if no_anomaly and verbose:
        print("No time anomaly detected")
    return no_anomaly


def check_velocity_anomaly(solution,is_using_constant_speed: bool = False,verbose: bool = True  ):
    agents = {agent:np.array([[p['t']] + _xyz(p) for p in path] ) for agent,path in solution.items()}
    no_anomaly = True
    for agent, path in agents.items():
        diff = np.diff(path[:,1:], axis=0)
        t_diff = np.diff(path[:,0], axis=0)
        v_diff = diff[t_diff > 1e-10] / t_diff[t_diff > 1e-10].reshape(-1,1)
        speed = np.linalg.norm(v_diff, axis=1)
        if speed.size == 0:
            # No motion segments for this agent (e.g., start==goal), so nothing to validate.
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


def check_collision(solution, r, verbose: bool = True):
    """
    Check for collisions between all pairs of agents in a solution, and
    compute the time intervals during which they collide.

    `solution` is a dict: agent_name -> list of {"t", "x", "y"} (plus "z" in 3D).
    Agents are modeled as discs (spheres in 3D) of radius `r`; a collision occurs if
    the distance between two agents is <= 2r at any time.

    Returns
    -------
    collisions : dict
        Mapping (agent_i, agent_j) -> list of (t_start, t_end) intervals
        (in global time) over which the two agents are in collision.
    """
    # Keep trajectories as lists of dicts for clarity
    agents = {name: path for name, path in solution.items()}
    agent_names = list(agents.keys())
    collision_radius_sq = (2 * r) ** 2
    no_collision = True
    collisions = {}

    def get_pos(traj, t):
        """Linear interpolation for position at time t along a trajectory."""
        n = len(traj)
        if n == 0:
            return np.array([0.0, 0.0])
        if n == 1:
            return np.array(_xyz(traj[0]))
        for i in range(n - 1):
            t0 = traj[i]["t"]
            t1 = traj[i + 1]["t"]
            if t0 <= t <= t1:
                dt = t1 - t0
                if dt == 0:
                    return np.array(_xyz(traj[i]))
                ratio = (t - t0) / dt
                a, b = np.array(_xyz(traj[i])), np.array(_xyz(traj[i + 1]))
                return a + ratio * (b - a)
        # If t is beyond the last timestamp, hold last position
        return np.array(_xyz(traj[-1]))

    # Check all unordered pairs of agents
    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            name_i = agent_names[i]
            name_j = agent_names[j]
            traj_i = agents[name_i]
            traj_j = agents[name_j]

            pair_key = (name_i, name_j)
            pair_intervals = []

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
                if a < eps:
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


def check_collision_bodies(solution, bodies, graph_map=None, verbose: bool = True):
    """Exact collision check for agents with bodies (any 2D / 3D shape, with clearance), as every planner
    with ``Agent`` bodies outputs them (CBS / ICBS, CCBS, SIPP, Neural-ATTF).

    ``solution``: agent -> ``[{"t", "x", "y"[, "z"][, "yaw"[, "pitch"]]}, ...]``, piecewise linear in time.
    ``bodies``: agent -> ``Agent`` (or its ``to_dict()``, as solution summaries store it). ``graph_map``: a map
    whose collision sweep to use (optional: the test needs no roadmap; default backend otherwise).
    A moving body faces along its move; a waiting one faces its entry's ``yaw`` (+ ``pitch`` in 3D,
    degrees), or counts as its bounding ball when the entry has none or when it turns in place (the next
    entry, at the same position, has another heading). After its last entry an agent stays where it is
    until the latest entry of any agent.

    Returns ``{(agent_i, agent_j): first contact time}`` (names sorted), empty when collision-free.
    """
    from path_planning.agent import Agent
    from path_planning.multi_agent_planner.centralized.cbs.data_types import _bounding_ball, _heading_of, _make_ori

    bodies = {name: b if isinstance(b, Agent) else Agent.from_dict(b) for name, b in bodies.items()}
    names = sorted(solution)
    t_end = max((float(path[-1]["t"]) for path in solution.values() if path), default=0.0)
    owner, agents, paths, t0s, holds, heads = [], [], [], [], [], []
    movers = {}                                     # (body key, speed) -> the body moving at that speed
    for name in names:
        body, path = bodies[name], solution[name]
        for k, p in enumerate(path):
            a, t = np.asarray(_xyz(p), dtype=float), float(p["t"])
            q = path[k + 1] if k + 1 < len(path) else None
            b = None if q is None else np.asarray(_xyz(q), dtype=float)
            if b is not None and np.any(np.abs(b - a) > 1e-12):           # a move
                dur = float(q["t"]) - t
                if dur <= 1e-12:
                    continue
                speed = float(np.linalg.norm(b - a)) / dur
                key = (body.key, speed)
                if key not in movers:
                    movers[key] = Agent(body.shape, speed, None, body.type, body.clearance)
                agents.append(movers[key])
                paths.append(np.array([a, b]))
                holds.append(0.0)
                heads.append(None)
            else:                                                       # a wait (or the final stay)
                hold = (t_end if q is None else float(q["t"])) - t
                if hold <= 1e-12:
                    continue
                yaw = p.get("yaw")
                turning = q is not None and (q.get("yaw"), q.get("pitch")) != (yaw, p.get("pitch"))
                if body.is_ball or yaw is None or turning:
                    agents.append(_bounding_ball(body))
                    heads.append(None)
                else:
                    agents.append(body)
                    heads.append(_heading_of(_make_ori(yaw, (p.get("pitch") or 0.0) if a.size == 3 else None)))
                paths.append(a[None])
                holds.append(hold)
            owner.append(name)
            t0s.append(t)
    found = {}
    if not paths:
        return found
    heading = None if all(h is None for h in heads) else heads
    if graph_map is not None:
        i, j, t1, _ = graph_map.agent_conflicts(agents, paths, t0=t0s, goal_hold=holds, as_arrays=True,
                                                heading=heading)
    else:
        from path_planning.common.world.collision_checking.collision_sweep import CollisionSweep

        backend = CollisionSweep(dim=paths[0].shape[1])
        i, j, t1, _ = backend.agent_conflicts(
            paths, [a.shape for a in agents], [a.max_speed for a in agents], t0s,
            [a.max_rotation for a in agents], holds, heading, None, False, True, [a.clearance for a in agents])
    for a, b, w in zip(i.tolist(), j.tolist(), t1.tolist()):
        if owner[a] != owner[b]:
            key = tuple(sorted((owner[a], owner[b])))
            found[key] = min(w, found.get(key, float("inf")))
    if verbose:
        for (a, b), w in sorted(found.items(), key=lambda kv: kv[1]):
            print(f"Collision between {a} and {b} from t={w:.3f}")
        if not found:
            print("No collision detected")
    return found


def check_solution(solution):
    check_time_anomaly(solution)
    check_velocity_anomaly(solution)
    check_collision(solution, 1.0)


def is_conflict_free(solution, r, verbose: bool = False):
    """
    Return True iff no pair of agents in `solution` collides.

    Parameters
    ----------
    solution : dict
        Mapping agent_name -> list of waypoints {"t", "x", "y"}.
    r : float
        Agent radius (agents are modeled as discs; collision if distance <= 2r).
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
    r: float = 0.0,
    is_using_constant_speed: bool = False,
    verbose: bool = True,
    bodies: dict | None = None,
    graph_map=None,
):
    """
    Run all anomaly checks (time, velocity, collision) in one call.

    Parameters
    ----------
    solution : dict
        Mapping agent_name -> list of waypoints {"t", "x", "y"}.
    r : float
        Agent radius (discs), used when ``bodies`` is not given.
    bodies : dict, optional
        agent -> ``Agent`` (or its ``to_dict()``): the exact check (:func:`check_collision_bodies`), with
        ``graph_map``'s sweep when given.
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
            "collisions": dict,  # as returned by check_collision / check_collision_bodies
        }
    """
    no_time_anomaly = check_time_anomaly(solution, verbose=verbose)
    no_velocity_anomaly = check_velocity_anomaly(
        solution,
        is_using_constant_speed=is_using_constant_speed,
        verbose=verbose,
    )
    if bodies is not None:
        collisions = check_collision_bodies(solution, bodies, graph_map, verbose=verbose)
    else:
        collisions = check_collision(solution, r, verbose=verbose)

    return {
        "no_time_anomaly": no_time_anomaly,
        "no_velocity_anomaly": no_velocity_anomaly,
        "collisions": collisions,
    }
