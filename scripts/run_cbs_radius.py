

from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, CBS
from path_planning.utils.util import read_agents_from_yaml, read_graph_sampler_from_yaml
import time
import numpy as np
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from copy import deepcopy

if __name__ == "__main__":
    discrete_space = True
    map_ =read_graph_sampler_from_yaml('path_planning/maps/2d/2d.yaml',discrete_space)
    agents = read_agents_from_yaml('path_planning/maps/2d/2d.yaml')
    map_.inflate_obstacles(radius=np.sqrt(2))
    # map_.inflate_obstacles(radius=0)
    if discrete_space:
        map_.set_parameters(sample_num=000, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
    else:
        map_.set_parameters(sample_num=2000, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=5.1)

    start = [agent['start'] for agent in agents]
    goal = [agent['goal'] for agent in agents]
    map_.set_start(start)
    map_.set_goal(goal)
    nodes = map_.generateRandomNodes(generate_grid_nodes = discrete_space)
    # road_map = map_.generate_roadmap(nodes)
    road_map = map_.generate_planar_map(nodes)

    start = [s.current for s in map_.get_start_nodes()]
    goal = [g.current for g in map_.get_goal_nodes()]
    agents =[
            {
            "start": start[i],
            "name": agent['name'],
            "goal": goal[i]
        }
        for i, agent in enumerate(agents)
    ]

    agent_radius = 2.0
    map_.set_constraint_sweep()

    # Searching
    env = Environment(map_, agents,radius=0)
    cbs = CBS(env)
    st = time.time()
    solution = cbs.search()
    print(f"Time taken to search: {time.time() - st} seconds")
    if not solution:
        print(" Solution not found" )

    # Searching
    env = Environment(map_, agents,radius=agent_radius, use_constraint_sweep=False)
    cbs = CBS(env)
    st = time.time()
    solution = cbs.search()
    print(f"Time taken to search: {time.time() - st} seconds")
    if not solution:
        print(" Solution not found" )

    # Searching
    env = Environment(map_, agents,radius=agent_radius, use_constraint_sweep=True)
    cbs = CBS(env)
    st = time.time()
    solution = cbs.search()
    print(f"Time taken to search: {time.time() - st} seconds")
    if not solution:
        print(" Solution not found" )


    # schedule = {"schedule":deepcopy(solution)}

    # # Create animation
    # temp_filename = f"temp3.gif"
    # vis = Visualizer2D()
    # st = time.time()
    # vis.animate(temp_filename, map_, schedule, road_map=road_map,skip_frames=1, intermediate_frames=1,speed=3,radius=agent_radius)
    # print(f"Time taken to animate: {time.time() - st} seconds")
    # print(f"Animation saved to: {temp_filename}")
