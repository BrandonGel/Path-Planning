"""
Cluster-distilled roadmaps for the dataset pipeline.

For a case's saved roadmap (``.../case_i/maps/<road_map_type>/graph_map.pkl``)
this module runs one of the clustering methods in ``path_planning/cluster/``
(CTopPRM wavefront, graph k-means, graph EM), distills the resulting cluster
roadmap (``CTopPRM.get_roadmap``) into a NEW ``GraphSampler`` and saves it as
``maps/<road_map_type>/cluster/graph_map_<method>.pkl`` with a
``graph_map_<method>_runtime.yaml`` sidecar, so the MAPF solvers can run on
the distilled maps via ``create_solutions(..., graph_files=[...])``.

Driver: ``scripts/test/run_all_cluster_solvers.py``.
"""

import math
import time
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from path_planning.cluster.CTopPRMpy.ctopprm import CTopPRM
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.data_generation.dataset_ground_truth_map import create_map
from path_planning.data_generation.dataset_util import (
    generate_base_case_path,
    generate_cluster_path,
    get_graph_file_path,
    get_graph_runtime_file_path,
    get_input_file_path,
    write_runtime_yaml,
)
from path_planning.utils.util import agents_yaml_to_roadmap_frame, set_global_seed

# method name (folder / file suffix) -> CTopPRM clustering mode
CLUSTER_METHODS = {"ctopprm": "wavefront", "kmeans": "kmeans", "em": "em"}


def get_cluster_graph_name(method: str, source_graph_name: str = "graph_map.pkl") -> str:
    """``graph_map.pkl`` + ``kmeans`` -> ``graph_map_kmeans.pkl``."""
    if method not in CLUSTER_METHODS:
        raise ValueError(f"method must be one of {sorted(CLUSTER_METHODS)}, got {method!r}")
    return f"{Path(source_graph_name).stem}_{method}.pkl"


def _build_setup_sampler(source_map: GraphSampler) -> GraphSampler:
    """Fresh, node-free GraphSampler sharing the source map's world/obstacle
    setup, ready for generate_custom_nodes/generate_custom_roadmap.

    Built from the loaded source map (not input.yaml) so use_discrete_space
    and the coordinate frame always match the roadmap being clustered."""
    fresh = GraphSampler(
        bounds=[list(map(float, b)) for b in np.asarray(source_map.bounds, dtype=float)],
        resolution=float(source_map.resolution),
        start=[],
        goal=[],
        use_discrete_space=source_map.use_discrete_space,
        sample_num=0,
        min_edge_len=source_map.min_edge_length,
        max_edge_len=source_map.max_edge_length,
        num_neighbors=source_map.num_neighbors,
        sampling_dist_dict=source_map.sampling_dist_dict,
    )
    fresh.obs_size = source_map.obs_size
    if len(source_map.obstacles):
        fresh.set_obstacles(np.asarray(source_map.obstacles, dtype=np.int64))
    fresh.set_inflation_radius(source_map.inflation_radius)
    return fresh


def _snap_points(sampler: GraphSampler, points: np.ndarray) -> np.ndarray:
    """Node coordinates generate_custom_nodes will store for ``points``
    (identity in continuous space, corner-lattice snap in discrete space)."""
    pts = np.asarray(points, dtype=float)
    if not sampler.use_discrete_space:
        return pts
    b = np.asarray(sampler.bounds, dtype=float)[:, 0]
    return b + sampler.resolution * np.round((pts - b) / sampler.resolution)


def build_cluster_map(
    source_map: GraphSampler,
    agents: List[dict],
    method: str,
    cluster_fraction: float = 0.05,
) -> Tuple[GraphSampler, dict]:
    """Cluster ``source_map`` with ``method`` and distill the cluster roadmap
    into a new GraphSampler (endpoints stay on-node for the solvers).

    K = max(num_endpoints + 2, ceil(cluster_fraction * num_nodes)).
    Returns (distilled sampler, stats dict with timings and counts)."""
    agents_rt = agents_yaml_to_roadmap_frame(source_map, agents)
    starts = [tuple(a["start"]) for a in agents_rt]
    goals = [tuple(a["goal"]) for a in agents_rt]
    num_endpoints = len(dict.fromkeys(starts + goals))
    k = max(num_endpoints + 2, math.ceil(cluster_fraction * len(source_map.nodes)))

    t0 = time.perf_counter()
    planner = CTopPRM(source_map, clustering=CLUSTER_METHODS[method], min_clusters=k)
    planner.set_up_distinct_paths(list(zip(starts, goals)))
    t_cluster = time.perf_counter()
    points, edges = planner.get_roadmap()
    t_distill = time.perf_counter()

    fresh = _build_setup_sampler(source_map)
    fresh.set_start(starts)  # before generate_custom_nodes: it appends/dedupes
    fresh.set_goal(goals)    # endpoint nodes against start/goal itself
    # filter_points=False: the points come from the source map's validated
    # roadmap, which may legally hold nodes point_expandable rejects (e.g.
    # RRG nodes slightly outside the bounds); dropping any would shift every
    # later index and corrupt the edge list.
    nodes = fresh.generate_custom_nodes(points, filter_points=False)
    # Safety net: verify the positional alignment anyway.
    expected = _snap_points(fresh, points)
    got = np.asarray([n.current for n in nodes[: len(points)]], dtype=float)
    if len(nodes) < len(points) or not np.allclose(got, expected, atol=1e-9):
        raise RuntimeError(
            f"{method}: distilled points rejected by point_expandable "
            f"({len(nodes)} nodes for {len(points)} points); refusing to save "
            f"a roadmap with misaligned edge indices"
        )
    fresh.generate_custom_roadmap(edges)
    t_build = time.perf_counter()

    stats = {
        "method": method,
        "clustering_mode": CLUSTER_METHODS[method],
        "min_clusters": k,
        "num_clusters": len(planner.seed_indices),
        "num_nodes": len(fresh.nodes),
        "num_edges": len(fresh.edges),
        "source_num_nodes": len(source_map.nodes),
        "source_num_edges": len(source_map.edges),
        "runtime_breakdown": {
            "cluster": t_cluster - t0,
            "distill": t_distill - t_cluster,
            "build": t_build - t_distill,
        },
    }
    return fresh, stats


def save_cluster_map(cluster_map: GraphSampler, graph_file: Path) -> None:
    """Save a distilled sampler so load_graph_sampler round-trips cleanly."""
    # grid_points must be a subset of nodes on load; the distilled roadmap
    # keeps none of the source grid nodes as such.
    cluster_map.grid_points = []
    # _load_from_dict recomputes num_total_nodes = sample_num + #start + #goal.
    cluster_map.sample_num = (
        len(cluster_map.nodes) - len(cluster_map.start) - len(cluster_map.goal)
    )
    cluster_map.save_graph_sampler(graph_file)


def process_single_case_cluster(args: Tuple) -> List[Tuple[int, str, str]]:
    """Cluster one case's source roadmap with every requested method.

    Returns a list of (case_id, method, error) failures (empty on success).
    """
    case_id, path, config, methods, cluster_fraction, overwrite, verbose = args
    set_global_seed(config.get("seed", 42) + case_id)
    road_map_type = config["road_map_type"]
    case_path, map_path = generate_base_case_path(path, case_id, road_map_type)
    cluster_dir = generate_cluster_path(map_path)

    with open(get_input_file_path(case_path), "r") as f:
        inpt = yaml.safe_load(f)

    source_graph_file = get_graph_file_path(map_path)
    if not source_graph_file.exists():
        # Without this guard create_map would silently generate a fresh
        # roadmap of whatever type the case's input.yaml last recorded.
        return [(case_id, m, f"source roadmap missing: {source_graph_file}") for m in methods]

    todo = [
        m for m in methods
        if overwrite or not get_graph_file_path(cluster_dir, get_cluster_graph_name(m)).exists()
    ]
    if not todo:
        return []

    t0 = time.perf_counter()
    source_map = create_map(
        inpt, graph_file=source_graph_file, verbose=False,
        args={"use_constraint_sweep": False},
    )
    t_load = time.perf_counter() - t0

    failures = []
    for method in todo:
        graph_name = get_cluster_graph_name(method)
        graph_file = get_graph_file_path(cluster_dir, graph_name)
        try:
            cluster_map, stats = build_cluster_map(
                source_map, inpt["agents"], method, cluster_fraction
            )
            t_save0 = time.perf_counter()
            save_cluster_map(cluster_map, graph_file)
            t_save = time.perf_counter() - t_save0
            stats["source_graph"] = str(source_graph_file)
            stats["cluster_fraction"] = cluster_fraction
            stats["runtime_breakdown"]["load"] = t_load
            stats["runtime_breakdown"]["save"] = t_save
            stats["runtime"] = sum(stats["runtime_breakdown"].values())
            write_runtime_yaml(get_graph_runtime_file_path(cluster_dir, graph_name), stats)
            if verbose:
                print(
                    f"case_{case_id} {road_map_type}/{method}: "
                    f"{stats['source_num_nodes']} -> {stats['num_nodes']} nodes, "
                    f"{stats['num_edges']} edges ({stats['runtime']:.2f}s)"
                )
        except Exception as exc:  # noqa: BLE001 - reported to the driver
            failures.append((case_id, method, f"{exc}\n{traceback.format_exc()}"))
    return failures


def create_cluster_maps(
    path: Path,
    num_cases: int,
    config: Dict,
    methods: List[str],
    cluster_fraction: float = 0.05,
    num_workers: Optional[int] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Cluster + save distilled roadmaps for all cases (no solving).

    Raises RuntimeError listing every (case, method) failure: skipping would
    leave holes in the per-case graph_files list create_solutions indexes.
    """
    path = Path(path)
    unknown = [m for m in methods if m not in CLUSTER_METHODS]
    if unknown:
        raise ValueError(f"unknown cluster methods {unknown}; choose from {sorted(CLUSTER_METHODS)}")
    if num_workers is None:
        num_workers = cpu_count()
    case_tasks = [
        (i, path, config, list(methods), cluster_fraction, overwrite, verbose)
        for i in range(num_cases)
    ]
    if verbose:
        print(
            f"Clustering {num_cases} cases with methods {list(methods)} "
            f"(fraction {cluster_fraction}) using {num_workers} workers"
        )
    failures: List[Tuple[int, str, str]] = []
    if num_workers > 1 and len(case_tasks) > 1:
        with Pool(processes=num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_case_cluster, case_tasks),
                total=len(case_tasks),
                desc="Clustering maps",
            ):
                failures.extend(result)
    else:
        for task in tqdm(case_tasks, desc="Clustering maps"):
            failures.extend(process_single_case_cluster(task))
    if failures:
        detail = "\n".join(
            f"  case_{cid} [{method}]: {err.splitlines()[0]}" for cid, method, err in failures
        )
        raise RuntimeError(
            f"{len(failures)} cluster map(s) failed for road_map_type="
            f"{config.get('road_map_type')}:\n{detail}\n"
            f"First full traceback:\n{failures[0][2]}"
        )
