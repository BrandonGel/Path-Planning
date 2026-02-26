'''
Run the checker for the MAPF solutions with different constant radii and constant velocities (except for zero velocity).
If velocity is 0.0, then the agent is reaching the next waypoint in 1 second
Or the time to travel between two waypoints is 1 second.

For LaCAM, LaCAM_random,
For radii 0.5 (no yet implemented with different radii)

For CBS, ICBS, & SIPP,
For radii 0.0, 1.0, 2.0.

For SIPP,
For velocities 0.0, 1.0, 2.0.

python scripts/mapf/run_mapf_checker.py
'''

import yaml
from path_planning.utils.checker import check_time_anomaly, check_velocity_anomaly, check_collision

def check_mapf_solution(file,radius: float = 0.0,velocity: float = 0.0,verbose: bool = False):
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
    solution = data['schedule']
    print(file)
    check_time_anomaly(solution,verbose)
    check_velocity_anomaly(solution,is_using_constant_speed=(velocity != 0),verbose=verbose)
    check_collision(solution, radius,verbose)

if __name__ == "__main__":
    for ii in [0.0,1.0,2.0]:
        file = f'path_planning/maps/2d/cbs/solution_radius{float(ii)}.yaml'
        check_mapf_solution(file,radius=float(ii),velocity=0.0)

    for ii in [0.0,1.0,2.0]:
        file = f'path_planning/maps/2d/icbs/solution_radius{float(ii)}.yaml'
        check_mapf_solution(file,radius=float(ii),velocity=0.0)
    
    file = f'path_planning/maps/2d/lacam/solution.yaml'
    check_mapf_solution(file,radius=0.5,velocity=0.0)

    file = f'path_planning/maps/2d/lacam_random/solution.yaml'
    check_mapf_solution(file,radius=0.5,velocity=0.0)

    for ii in [0.0,1.0,2.0]:
        file = f'path_planning/maps/2d/sipp/solution_radius{float(ii)}.yaml'
        check_mapf_solution(file,radius=float(ii),velocity=0.0)

    for ii in [0.0,1.0,2.0]:
        file = f'path_planning/maps/2d/sipp/solution_velocity{float(ii)}.yaml'
        check_mapf_solution(file,radius=1.0,velocity=float(ii))

    