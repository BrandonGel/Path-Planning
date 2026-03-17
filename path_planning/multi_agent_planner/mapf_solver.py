from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, CBS
from path_planning.multi_agent_planner.centralized.icbs.icbs import IEnvironment, ICBS
from path_planning.multi_agent_planner.centralized.lacam.lacam import LaCAM
from path_planning.multi_agent_planner.centralized.lacam.lacam_random import LaCAM as LaCAM_random
from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner
from path_planning.multi_agent_planner.centralized.ccbs.ccbs import CCBS
from path_planning.multi_agent_planner.centralized.ccbs.graph_generation import Environment as CCBS_Environment
from path_planning.multi_agent_planner.centralized.lacam.utility import set_starts_goals_config, is_valid_mapf_solution
from typing import Tuple
import numpy as np
import time
from path_planning.utils.checker import check_collision

def solve_mapf(map_, agents,mapf_solver_config:dict) -> Tuple[dict, float]:
    mapf_solver_name = mapf_solver_config['mapf_solver_name'].lower()
    agent_radius = mapf_solver_config['agent_radius']
    agent_velocity = mapf_solver_config['agent_velocity']
    time_limit = mapf_solver_config['time_limit']
    max_iterations = mapf_solver_config['max_iterations']
    heuristic_type = mapf_solver_config['heuristic_type']
    if mapf_solver_name == 'cbs':
        env = Environment(map_, agents, radius=agent_radius, use_constraint_sweep=True,heuristic_type=heuristic_type)
        cbs = CBS(env,time_limit=time_limit,max_iterations=max_iterations,verbose=False)
        solution,solution_info = cbs.search()
    elif mapf_solver_name == 'icbs':
        env = IEnvironment(map_, agents, radius=agent_radius, use_constraint_sweep=True,heuristic_type=heuristic_type)
        icbs = ICBS(env,time_limit=time_limit,max_iterations=max_iterations,verbose=False)
        solution, solution_info = icbs.search()
    elif mapf_solver_name == 'lacam':
        starts,goals = set_starts_goals_config(map_.start,map_.goal)
        planner = LaCAM()
        solution_config = planner.solve(map_, starts, goals,seed=0,time_limit_ms=1000*time_limit,max_iterations=max_iterations,verbose=0)
        solution, solution_info = planner.get_solution_dict(solution_config)
    elif mapf_solver_name == 'lacam_random':
        starts,goals = set_starts_goals_config(map_.start,map_.goal)
        planner = LaCAM_random()
        solution_config = planner.solve(map_, starts, goals,seed=0,time_limit_ms=1000*time_limit,max_iterations=max_iterations,verbose=0)
        solution, solution_info = planner.get_solution_dict(solution_config)
    elif mapf_solver_name == 'sipp':
        dynamic_obstacles = dict()
        sipp_planner = SippPlanner(map_,dynamic_obstacles,agents,agent_radius,agent_velocity,heuristic_type=heuristic_type,time_limit=time_limit,max_iterations=max_iterations)
        solution,solution_info = sipp_planner.compute_plan()
    elif mapf_solver_name == 'ccbs':
        env = CCBS_Environment(map_,{}, agents, radius=agent_radius, velocity=agent_velocity, use_constraint_sweep=True,heuristic_type=heuristic_type)
        ccbs = CCBS(env,time_limit=time_limit,max_iterations=max_iterations,verbose=False)
        solution, solution_info = ccbs.search()
    else:
        raise ValueError(f"Invalid algorithm: {mapf_solver_name}")
    solution_info["collision"] = False
    if agent_radius != 0 and solution_info["success"]:
        collisions = check_collision(solution, agent_radius, verbose=False)
        if collisions:
            solution_info["success"] = False
            solution_info["collision"] = True
    summary = summarize_solution(solution,solution_info,mapf_solver_config,map_)
    return summary


def get_mapf_solver(mapf_solver_name:str) -> Tuple[CBS,Environment]:
    mapf_solver_name = mapf_solver_name.lower()
    if mapf_solver_name == "cbs":
        return CBS,Environment
    elif mapf_solver_name == "icbs":
        return ICBS,IEnvironment
    elif mapf_solver_name == "lacam":
        return LaCAM
    elif mapf_solver_name == "lacam_random":
        return LaCAM_random
    elif mapf_solver_name == "sipp":
        return SippPlanner
    elif mapf_solver_name == "ccbs":
        return CCBS,CCBS_Environment
    else:
        raise ValueError(f"Invalid algorithm: {mapf_solver_name}")

def computer_solution_cost(solution:dict):
    cost_dict = {}
    flowtime = 0
    total_travel_time = 0
    total_wait_time = 0
    makespan = 0
    success = True if solution is not None and solution != {} and solution != False else False
    if success:
        for agent, path in solution.items():
            path_array = np.array([list(p.values())[1:] for p in path]) # (N, 2)
            dist_travel = np.linalg.norm(path_array[1:] - path_array[:-1], axis=1) # (N-1,)
            time_stamp = np.array([list(p.values())[0] for p in path]) # (N,)
            delta_time = np.diff(time_stamp) # (N-1,)

            wait_time = delta_time[dist_travel == 0]
            travel_time = delta_time[dist_travel > 0]

            flowtime += dist_travel.sum().item()
            total_travel_time += travel_time.sum().item()
            total_wait_time += wait_time.sum().item()
            makespan = max(makespan, path[-1]["t"] - path[0]["t"])

            cost_dict[agent] = {}
            cost_dict[agent]["path_length"] = dist_travel.sum().item()
            cost_dict[agent]["flowtime"] = travel_time.sum().item()
            cost_dict[agent]["wait_time"] = wait_time.sum().item()
            cost_dict[agent]["makespan"] = path[-1]["t"] - path[0]["t"]
        solution_temp = {}
        for agent, path in solution.items():
            solution_temp[agent] = []
            for state in path:
                for key, value in state.items():
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        state[key] = float(value)
                solution_temp[agent].append(state)
        solution = solution_temp
    all_cost_dict = {}
    all_cost_dict["path_length"] = flowtime
    all_cost_dict["flowtime"] = total_travel_time
    all_cost_dict["wait_time"] = total_wait_time
    all_cost_dict["makespan"] = makespan
    all_cost_dict["success"] = success
    return cost_dict, all_cost_dict

def summarize_solution(solution,solution_info,mapf_solver_config,map_):
    agent_cost, total_cost = computer_solution_cost(solution)
    summary = {}
    summary["schedule"] = solution
    summary["schedule_cost"] = agent_cost
    summary.update(total_cost)
    summary.update(solution_info)
    summary["mapf_solver_name"] = mapf_solver_config["mapf_solver_name"]
    summary["agent_radius"] = mapf_solver_config["agent_radius"]
    summary["agent_velocity"] = mapf_solver_config["agent_velocity"]
    summary["time_limit"] = mapf_solver_config["time_limit"]
    summary["max_iterations"] = mapf_solver_config["max_iterations"]
    summary["heuristic_type"] = mapf_solver_config["heuristic_type"]
    summary["num_nodes"] = len(map_.nodes)
    summary["num_edges"] = len(map_.edges)
    return summary