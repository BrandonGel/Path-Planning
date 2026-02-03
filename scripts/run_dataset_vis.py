from path_planning.data_generation.dataset_gen import create_path_parameter_directory
from path_planning.data_generation.dataset_visualize import load_and_visualize_case
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
    parser.add_argument("-y","--config",type=str, default='', help="config file to use")
    parser.add_argument("-cm","--case_mode",type=str, default='first_n', choices=['all', 'first_n', 'range', 'specific'], help="mode to visualize the dataset")
    parser.add_argument("-cn","--num_cases",type=int, default=3, help="number of cases to visualize")
    parser.add_argument("-cr","--case_range",type=list, nargs=2, default=[0, 16], help="range of cases to visualize")
    parser.add_argument("-cs","--specific_cases",type=list, nargs='+', default=[0, 5, 10], help="specific cases to visualize")
    parser.add_argument("-pv","--permutation_mode",type=str, default='first_n', choices=['all', 'first_n', 'range', 'specific'], help="mode to visualize the dataset")
    parser.add_argument("-pn","--num_permutations",type=int, default=4, help="number of permutations to visualize")
    parser.add_argument("-pr","--permutations_range",type=list, nargs=2, default=[0, 16], help="range of permutations to visualize")
    parser.add_argument("-ps","--specific_permutations",type=list, nargs='+', default=[0, 5, 10], help="specific permutations to visualize")
    parser.add_argument("-ss","--show_static",type=bool, default=True, help="show static paths")
    parser.add_argument("-sa","--show_animation",type=bool, default=False, help="show animation")
    args = parser.parse_args()
    


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


    # Get all case directories
    case_dirs = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_")],
                    key=lambda x: int(x.name.split("_")[1]))

    print(f"Found {len(case_dirs)} cases to visualize")

    # User can modify this to visualize specific cases
    # Options: "all", "first_n", "range", "specific"
    viz_mode = args.case_mode  # Change this to control which cases to visualize
    n_cases = args.num_cases  # Number of cases if using "first_n"
    case_range = args.case_range  # Range if using "range"
    specific_cases = args.specific_cases  # Specific case numbers if using "specific"

    # Select cases to visualize
    if viz_mode == "all":
        cases_to_viz = case_dirs
    elif viz_mode == "first_n":
        cases_to_viz = case_dirs[:n_cases]
    elif viz_mode == "range":
        cases_to_viz = case_dirs[case_range[0]:case_range[1]]
    elif viz_mode == "specific":
        cases_to_viz = [case_dirs[i] for i in specific_cases if i < len(case_dirs)]
    else:
        cases_to_viz = case_dirs[:3]  # Default to first 3

    perm_mode = args.permutation_mode  # Change this to control which permutations to visualize
    n_permutations = args.num_permutations
    permutations_range = args.permutations_range
    specific_permutations = args.specific_permutations
    print(f"\nVisualizing {len(cases_to_viz)} cases... ")
    # Visualize selected cases
    for case_path in cases_to_viz:
        root = case_path / "ground_truth" if (case_path / "ground_truth").exists() else case_path
        perm_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("perm_")],
                           key=lambda x: int(x.name.split("_")[1]))
        if perm_mode == "all":
            permutations_to_viz = perm_dirs
        elif perm_mode == "first_n":
            permutations_to_viz = perm_dirs[:n_permutations]
        elif perm_mode == "range":
            permutations_to_viz = perm_dirs[permutations_range[0]:permutations_range[1]]
        elif perm_mode == "specific":
            permutations_to_viz = [case_dirs[i] for i in specific_cases if i < len(case_dirs)]
        else:
            permutations_to_viz = case_dirs[:3]  # Default to first 3
        print(f"\nGoing through {len(permutations_to_viz)} Permutations for case {case_path.name}")
        for perm_dir in permutations_to_viz:
            try:
                load_and_visualize_case(perm_dir, show_static=args.show_static, show_animation=args.show_animation)
            except Exception as e:
                print(f"Error visualizing {case_path.name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
