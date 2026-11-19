"""cluster_map._build_setup_sampler copies the source's grid without its start / goal marks: it must match the
grid rebuilt from the obstacles (the old behaviour), also for a loaded version 3 map that keeps no obstacle list."""
import numpy as np

from path_planning.cluster.cluster_map import _build_setup_sampler
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.utils.util import set_global_seed

BOUNDS = [[0.0, 20.0], [0.0, 20.0]]
RES = 0.5
BLOCK = [(x, y) for x in range(16, 24) for y in range(12, 28)]      # cells: world x in [8, 12], y in [6, 14]
# Endpoints (cell centres): on the inflation band's edge (diagonal, distance == the expanded radius, sqrt(2)),
# inside the band, just outside it, and in free space.
STARTS = [(7.25, 5.25), (7.75, 10.25)]
GOALS = [(13.25, 10.25), (1.25, 1.25)]


def source_map():
    set_global_seed(0)
    m = GraphSampler(bounds=BOUNDS, resolution=RES, start=[], goal=[], use_discrete_space=False)
    m.set_start(STARTS)
    m.set_goal(GOALS)
    m.set_obstacles(np.array(BLOCK))
    m.inflate_obstacles(1.0)
    m.set_roadmap_config(sample_num=200, num_neighbors=8, min_edge_length=1e-10, max_edge_length=4.0)
    m.generateRandomNodes()
    m.generate_roadmap("prm")
    return m


def rebuilt(source):
    """The setup sampler's grid as it was built before: obstacles re-marked and inflated, no endpoints."""
    m = GraphSampler(bounds=BOUNDS, resolution=RES, start=[], goal=[], use_discrete_space=False)
    m.set_obstacles(np.array(BLOCK))
    m.inflate_obstacles(source.inflation_radius)
    return m


def grid(m):
    return m.type_map.data.tobytes(), np.asarray(m.esdf).tobytes(), m.inflation_radius, sorted(m.obstacles)


def test_endpoint_cells_are_marked_on_band_and_free():
    m = source_map()
    ref = rebuilt(m).type_map.data
    cells = [tuple(m.world_to_map(p, discrete=True)) for p in STARTS + GOALS]
    from python_motion_planning.common import TYPES
    assert [ref[c] for c in cells] == [TYPES.INFLATION, TYPES.INFLATION, TYPES.FREE, TYPES.FREE]


def test_setup_sampler_matches_rebuild():
    m = source_map()
    fresh = _build_setup_sampler(m)
    assert grid(fresh) == grid(rebuilt(m))
    assert len(fresh.nodes) == 0


def test_setup_sampler_of_loaded_map(tmp_path):
    m = source_map()
    path = tmp_path / "graph_map.pkl"
    m.save_graph_sampler(str(path))
    loaded = GraphSampler(bounds=BOUNDS, resolution=RES, start=[], goal=[])
    loaded.load_graph_sampler(str(path), args={"use_collision_sweep": False})
    fresh = _build_setup_sampler(loaded)
    assert loaded._obstacles is None and fresh._obstacles is None     # nothing derived the obstacle list
    assert grid(fresh) == grid(rebuilt(m))
    fresh.set_start(STARTS)                                          # as build_cluster_map does next
    fresh.set_goal(GOALS)
    assert fresh.type_map.data.tobytes() == m.type_map.data.tobytes()
