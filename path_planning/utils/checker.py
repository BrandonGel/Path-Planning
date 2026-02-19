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
        


def check_velocity_anomaly(solution,is_using_constant_speed: bool = False,verbose: bool = True  ):
    agents = {agent:np.array([[p['t'],p['x'], p['y']] for p in path] ) for agent,path in solution.items()}
    no_anomaly = True
    for agent, path in agents.items():
        diff = np.diff(path[:,1:], axis=0)
        t_diff = np.diff(path[:,0], axis=0)
        v_diff = diff / t_diff.reshape(-1,1)
        speed = np.linalg.norm(v_diff, axis=1)

        if is_using_constant_speed and np.any(speed-speed[0]> 1e-9) :
            print(f"Agent {agent} has non-constant velocity")
            no_anomaly = False
    if no_anomaly and verbose:
        print("No velocity anomaly detected")


def check_collision(solution, r, verbose: bool = True):
    """
    Check for collisions between all pairs of agents in a solution.

    `solution` is a dict: agent_name -> list of {"t", "x", "y"}.
    Agents are modeled as discs of radius `r`; a collision occurs if
    the distance between two agents is <= 2r at any time.
    """
    # Keep trajectories as lists of dicts for clarity
    agents = {name: path for name, path in solution.items()}
    agent_names = list(agents.keys())
    collision_radius_sq = (2 * r) ** 2
    no_collision = True
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

                # Quadratic distance^2(t) = a t^2 + b t + c on [0, duration]
                a = np.dot(V_rel, V_rel)
                b = 2 * np.dot(P_rel, V_rel)
                c = np.dot(P_rel, P_rel)

                # Time of minimum distance within this interval
                if a > 1e-9:
                    t_min = -b / (2 * a)
                    t_check = max(0.0, min(duration, t_min))
                else:
                    # Relative velocity ~ 0: distance is ~constant over slice
                    t_check = 0.0

                min_dist_sq = a * (t_check ** 2) + b * t_check + c

                if min_dist_sq < collision_radius_sq:
                    t_collide = t_start + t_check
                    print(
                        f"Agents {name_i} and {name_j} collided at "
                        f"t={t_collide:.3f} (distance={np.sqrt(min_dist_sq):.3f})"
                    )
                    no_collision = False
    if no_collision and verbose:
        print("No collision detected")


def check_solution(solution):
    check_time_anomaly(solution)
    check_velocity_anomaly(solution)
    check_collision(solution, 1.0)
