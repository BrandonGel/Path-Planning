import python_motion_planning as pmp
from path_planning.utils.util import convert_grid_to_yaml,convert_map_to_yaml

def make_grid_environment():
    # Create environment with custom obstacles
    env = pmp.Grid(51, 31)
    obstacles = env.obstacles
    for i in range(10, 21):
        obstacles.add((i, 15))
    for i in range(15):
        obstacles.add((20, i))
    for i in range(15, 30):
        obstacles.add((30, i))
    for i in range(16):
        obstacles.add((40, i))
    env.update(obstacles)
    convert_grid_to_yaml(env,"path_planning/environment/grid/grid.yaml")

def make_map_environment():
    # Create environment with custom obstacles
    env = pmp.Map(51, 31)
    obs_rect = [
        [14, 12, 8, 2],
        [18, 22, 8, 3],
        [26, 7, 2, 12],
        [32, 14, 10, 2]
    ]
    obs_circ = [
        [7, 12, 3],
        [46, 20, 2],
        [15, 5, 2],
        [37, 7, 3],
        [37, 23, 3]
    ]
    env.update(obs_rect=obs_rect, obs_circ=obs_circ)
    convert_map_to_yaml(env,"path_planning/environment/map/map.yaml")

if __name__ == "__main__":
    make_grid_environment()
    make_map_environment()