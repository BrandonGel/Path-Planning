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

def solve_mapf(map_, agents, mapf_solver_name:str, timeout:float=60.0, max_attempts:int=10000, agent_radius:float=0.0, agent_velocity:float=0.0) -> Tuple[dict, float]:
    mapf_solver_name = mapf_solver_name.lower()
    if 'cbs' in mapf_solver_name:
        CBS,Environment = get_mapf_solver(mapf_solver_name)
        env = Environment(map_, agents,radius=agent_radius,velocity=agent_velocity)
        cbs = CBS(env, time_limit=timeout, max_iterations=max_attempts)
        start_time = time.time()
        solution = cbs.search()
        runtime = time.time() - start_time
        cost = env.compute_solution_cost(solution)
        return solution, runtime, cost
    elif 'lacam' in mapf_solver_name:
        starts,goals = set_starts_goals_config(map_.start,map_.goal)
        if agent_radius > 0 or agent_velocity > 0:
            assert False, "Agent radius and velocity are not supported for LaCAM"
        planner = LaCAM()
        st = time.time()
        solution_config = planner.solve(map_, starts, goals,seed=0,time_limit_ms=timeout*1000,verbose=0)
        if not is_valid_mapf_solution(map_, starts, goals, solution_config):
            print("Solution is not valid")
        ft = time.time()
        solution = planner.get_solution_dict(solution_config)
        cost = planner.compute_solution_cost(solution)
        runtime = ft - st
        return solution, runtime, cost
    elif 'sipp' in mapf_solver_name:
        st = time.time()
        sipp_planner = SippPlanner(map_,{},agents,agent_radius,agent_velocity)
        if sipp_planner.compute_plan():
            solution = sipp_planner.get_plan()
            cost = sipp_planner.compute_solution_cost()
            runtime = time.time() - st
            return solution, runtime, cost
        else:
            return None, None, None
    else:
        raise ValueError(f"Invalid algorithm: {mapf_solver_name}")


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
            makespan = max(makespan, path[-1]["t"] - path[0]["t"],makespan)

            cost_dict[agent] = {}
            cost_dict[agent]["path_length"] = dist_travel.sum().item()
            cost_dict[agent]["flowtime"] = travel_time.sum().item()
            cost_dict[agent]["wait_time"] = wait_time.sum().item()
            cost_dict[agent]["makespan"] = path[-1]["t"] - path[0]["t"]
    all_cost_dict = {}
    all_cost_dict["path_length"] = flowtime
    all_cost_dict["flowtime"] = total_travel_time
    all_cost_dict["wait_time"] = total_wait_time
    all_cost_dict["makespan"] = makespan
    all_cost_dict["success"] = success
    return cost_dict, all_cost_dict

def summarize_solution(solution,solution_info):
    agent_cost, total_cost = computer_solution_cost(solution)
    summary = {}
    summary["schedule"] = solution
    summary["schedule_cost"] = agent_cost
    summary.update(total_cost)
    summary.update(solution_info)
    return summary