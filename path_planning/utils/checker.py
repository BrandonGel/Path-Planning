import numpy as np

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


def check_velocity_anomaly(solution,is_using_constant_speed: bool = False,verbose: bool = True  ):
    agents = {agent:np.array([[p['t'],p['x'], p['y']] for p in path] ) for agent,path in solution.items()}
    no_anomaly = True
    for agent, path in agents.items():
        diff = np.diff(path[:,1:], axis=0)
        t_diff = np.diff(path[:,0], axis=0)
        v_diff = diff[t_diff > 1e-10] / t_diff[t_diff > 1e-10].reshape(-1,1)
        speed = np.linalg.norm(v_diff, axis=1)

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

    `solution` is a dict: agent_name -> list of {"t", "x", "y"}.
    Agents are modeled as discs of radius `r`; a collision occurs if
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
                    if c <= collision_radius_sq + eps:
                        # Entire local interval is colliding.
                        local_start, local_end = 0.0, duration
                    else:
                        continue
                else:
                    A = a
                    B = b
                    C = c - collision_radius_sq
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
                no_collision = False

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
