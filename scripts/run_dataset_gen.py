from path_planning.data_generation.dataset_gen import create_solutions,create_path_parameter_directory
import argparse
from pathlib import Path
import yaml

if __name__ == "__main__":
    """Main entry point for dataset generation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--path",type=str, default='benchmark/train', help="input file containing map and obstacles")
    parser.add_argument("-b","--bounds",type=list, nargs=2, default=[[0,32],[0,32]], help="bounds of the map")
    parser.add_argument("-n","--nb_agents",type=int, default=8, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.1, help="number of obstacles or obstacle density")
    parser.add_argument("-p","--nb_permutations",type=int, default=16, help="number of permutations")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-c","--num_cases",type=int, default=1, help="number of cases to generate")
    parser.add_argument("-y","--config",type=str, default='', help="config file to use")
    args = parser.parse_args("")
    
    base_path = Path(args.path)
    if args.config != '':
        with open(args.config, 'r') as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    else:
        config = {
            "bounds": args.bounds,
            "resolution": args.resolution,
            "nb_agents": args.nb_agents,
            "nb_obstacles": args.nb_obstacles,
            "nb_permutations": args.nb_permutations,
        }
    path = create_path_parameter_directory(base_path, config)
    create_solutions(path, args.nb_cases, config)

