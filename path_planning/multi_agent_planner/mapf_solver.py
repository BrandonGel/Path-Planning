from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, CBS
from path_planning.multi_agent_planner.centralized.icbs.icbs import IEnvironment, ICBS
from path_planning.multi_agent_planner.centralized.lacam.lacam import LaCAM
from path_planning.multi_agent_planner.centralized.lacam.lacam_random import LaCAM as LaCAM_random
from path_planning.multi_agent_planner.centralized.lacam.utility import set_starts_goals_config, is_valid_mapf_solution
from typing import Tuple
import time

def solve_mapf(map_, agents, mapf_solver_name:str, timeout:float=60.0, max_attempts:int=10000) -> Tuple[dict, float]:
    mapf_solver_name = mapf_solver_name.lower()
    if 'cbs' in mapf_solver_name:
        CBS,Environment = get_mapf_solver(mapf_solver_name)
        env = Environment(map_, agents,radius=0,velocity=0)
        cbs = CBS(env, time_limit=timeout, max_iterations=max_attempts)
        start_time = time.time()
        solution = cbs.search()
        runtime = time.time() - start_time
        cost = env.compute_solution_cost(solution)
        return solution, runtime, cost
    elif 'lacam' in mapf_solver_name:
        starts,goals = set_starts_goals_config(map_.start,map_.goal)
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
    else:
        raise ValueError(f"Invalid algorithm: {mapf_solver_name}")