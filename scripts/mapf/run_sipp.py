"""
Run CBS for MAPF algorithms with different agent radii (0.0, 1.0, 2.0).
python scripts/mapf/run_cbs.py 
"""

import argparse
from path_planning.utils.util import set_global_seed
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.common.visualizer.visualizer_3d import Visualizer3D
from path_planning.utils.util import write_to_yaml
from path_planning.utils.util import read_graph_sampler_from_yaml, read_agents_from_yaml
from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, CBS
from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner
import time
import numpy as np
import os
from copy import deepcopy

if __name__ == "__main__":
    os.makedirs("figs/sipp", exist_ok=True)
    os.makedirs("path_planning/maps/2d/sipp", exist_ok=True)

    set_global_seed(42)
    num_perm = 5
    for agent_radius in [0.0, 1.0, 2.0, 3.0]:
        for _ in range(num_perm):
            map_ = read_graph_sampler_from_yaml("path_planning/maps/2d/2d.yaml")
            agents = read_agents_from_yaml("path_planning/maps/2d/2d.yaml")
            map_.inflate_obstacles(radius=agent_radius+np.sqrt(2)/2)
            map_.set_parameters(
                sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1
            )

            start = [agent["start"] for agent in agents]
            goal = [agent["goal"] for agent in agents]
            map_.set_start(start)
            map_.set_goal(goal)
            nodes = map_.generateRandomNodes(generate_grid_nodes=True)
            road_map = map_.generate_roadmap(nodes)

            start = [s.current for s in map_.get_start_nodes()]
            goal = [g.current for g in map_.get_goal_nodes()]
            agents = [
                {"start": start[i], "name": agent["name"], "goal": goal[i]}
                for i, agent in enumerate(agents)
            ]

            map_.set_constraint_sweep()
            
            # Output file
            output = dict()
            solution = dict()
            dynamic_obstacles = dict()
            cost = 0

            #Searching
            st = time.time()
            sipp_planner = SippPlanner(map_,dynamic_obstacles,agents,agent_radius,0.0)
            plan_found = False
            if sipp_planner.compute_plan():
                solution = sipp_planner.get_plan()
                cost = sipp_planner.compute_solution_cost()
                plan_found = True
            else:
                continue
            ft = time.time()
            print(f"Time taken to search with radius {agent_radius}: {ft - st} seconds")
            # Write to output file
            output = dict()
            output["schedule"] = solution
            output["cost"] = cost
            output["runtime"] = ft - st
            write_to_yaml(output, f"path_planning/maps/2d/sipp/solution_radius{agent_radius}.yaml")

            vis = Visualizer2D()
            vis.plot_grid_map(map_)
            vis.plot_road_map(map_, nodes, road_map)

            # Plot each agent's path
            for agent_name, trajectory in solution.items():
                path = np.array([([point["x"], point["y"]]) for point in trajectory])
                vis.plot_path(path)
            vis.savefig(f"figs/sipp/sipp_2d_radius{agent_radius}.png")
            vis.show()
            vis.close()

            # Create animation
            schedule = {"schedule": deepcopy(solution)}
            gif_filename = f"figs/sipp/sipp_2d_radius{agent_radius}.gif"
            vis = Visualizer2D()
            vis.animate(
                gif_filename,
                map_,
                schedule,
                road_map=road_map,
                skip_frames=1,
                intermediate_frames=3,
                speed=3,
                radius=agent_radius,
                map_frame=True,
            )
            print(f"Animation saved to: {gif_filename}")
            vis.show()
            vis.close()

            if plan_found:
                break
        if not plan_found:
            print(f"Plan not found for radius {agent_radius}")

    agent_radius = 1.0
    use_discrete_space = False
    for agent_velocity in [0.0, 1.0, 2.0]:
        for _ in range(num_perm):
            map_ = read_graph_sampler_from_yaml("path_planning/maps/2d/2d.yaml",use_discrete_space=use_discrete_space)
            agents = read_agents_from_yaml("path_planning/maps/2d/2d.yaml")
            map_.inflate_obstacles(radius=agent_radius+np.sqrt(2)/2)
            map_.set_parameters(
                sample_num=1000, num_neighbors=13.0, min_edge_len=0.0, max_edge_len=5.1
            )

            start = [agent["start"] for agent in agents]
            goal = [agent["goal"] for agent in agents]
            map_.set_start(start)
            map_.set_goal(goal)
            nodes = map_.generateRandomNodes(generate_grid_nodes=use_discrete_space)
            road_map = map_.generate_roadmap(nodes)

            start = [s.current for s in map_.get_start_nodes()]
            goal = [g.current for g in map_.get_goal_nodes()]
            agents = [
                {"start": start[i], "name": agent["name"], "goal": goal[i]}
                for i, agent in enumerate(agents)
            ]

            map_.set_constraint_sweep()
            
            # Output file
            output = dict()
            solution = dict()
            dynamic_obstacles = dict()
            cost = 0

            #Searching
            st = time.time()
            sipp_planner = SippPlanner(map_,dynamic_obstacles,agents,agent_radius,agent_velocity)
            plan_found = False
            if sipp_planner.compute_plan():
                solution = sipp_planner.get_plan()
                cost = sipp_planner.compute_solution_cost()
                plan_found = True
            else:
                continue
            ft = time.time()
            print(f"Time taken to search with velocity {agent_velocity}: {ft - st} seconds")
            # Write to output file
            output = dict()
            output["schedule"] = solution
            output["cost"] = cost
            output["runtime"] = ft - st
            write_to_yaml(output, f"path_planning/maps/2d/sipp/solution_velocity{agent_velocity}.yaml")

            vis = Visualizer2D()
            vis.plot_grid_map(map_)
            vis.plot_road_map(map_, nodes, road_map,map_frame=use_discrete_space)

            # Plot each agent's path
            for agent_name, trajectory in solution.items():
                path = np.array([([point["x"], point["y"]]) for point in trajectory])
                vis.plot_path(path,map_frame=use_discrete_space)
            vis.savefig(f"figs/sipp/sipp_2d_velocity{agent_velocity}.png")
            vis.show()
            vis.close()

            # Create animation
            schedule = {"schedule": deepcopy(solution)}
            gif_filename = f"figs/sipp/sipp_2d_velocity{agent_velocity}.gif"
            vis = Visualizer2D()
            vis.animate(
                gif_filename,
                map_,
                schedule,
                road_map=road_map,
                skip_frames=1,
                intermediate_frames=1,
                speed=3,
                radius=1,            
                map_frame=use_discrete_space,
            )
            print(f"Animation saved to: {gif_filename}")
            vis.show()
            vis.close()

            if plan_found:
                break
        if not plan_found:
            print(f"Plan not found for velocity {agent_velocity}")