"""GraphSampler.pruned: in place (default) vs return_copy=True."""
import numpy as np
import pytest

from tests.helpers import GOAL, START, build_block_map
from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner
from path_planning.utils.util import set_global_seed


def state(m):
    """Everything pruning may change, plus the grid it must not."""
    return {
        "nodes": [tuple(n.current) for n in m.nodes],
        "edges": [tuple(e) for e in m.edges],
        "weights": np.round(np.asarray(m.edge_weights, dtype=float), 12).tolist(),
        "starts": dict(m.start_nodes_index), "goals": dict(m.goal_nodes_index),
        "boundary": sorted(m.boundary_nodes_index.values()),
        "pos_index": dict(m.pos_index_dict),
        "type_map": m.type_map.data.tobytes(), "esdf": np.asarray(m.esdf).tobytes(),
        "inflation": m.inflation_radius,
    }


@pytest.fixture
def block_map():
    set_global_seed(0)
    return build_block_map([START], [GOAL])


def kept_of(m, seed=0):
    rng = np.random.default_rng(seed)
    return np.flatnonzero(rng.random(len(m.nodes)) < 0.5)


def test_copy_leaves_source_alone(block_map):
    before = state(block_map)
    copy = block_map.pruned(kept_of(block_map), return_copy=True)
    assert copy is not block_map
    assert state(block_map) == before
    assert len(copy.nodes) < len(block_map.nodes)


def test_in_place_matches_copy(block_map):
    kept = kept_of(block_map)
    copy = block_map.pruned(kept, return_copy=True)
    out = block_map.pruned(kept)
    assert out is block_map
    assert state(block_map) == state(copy)


def test_keeps_start_and_goal_and_renumbers(block_map):
    start_pos = block_map.nodes[block_map.start_nodes_index[0] if 0 in block_map.start_nodes_index
                                else next(iter(block_map.start_nodes_index.values()))].current
    block_map.pruned(np.array([0, 1, 2]))
    assert len(block_map.nodes) <= 3 + 2
    assert tuple(block_map.nodes[next(iter(block_map.start_nodes_index.values()))].current) == tuple(start_pos)
    assert sorted(block_map.pos_index_dict.values()) == list(range(len(block_map.nodes)))
    n = len(block_map.nodes)
    assert all(0 <= u < n and 0 <= v < n for u, v in block_map.edges)


def test_in_place_rebinds_sweep_and_plans_like_copy(block_map):
    kept = kept_of(block_map, seed=1)
    copy = block_map.pruned(kept, return_copy=True)
    version = block_map.sweep_version
    block_map.get_start_nodes_with_all_edges()          # fill an index-keyed cache before pruning
    block_map.pruned(kept)
    assert block_map.sweep_version > version or not block_map.use_collision_sweep
    assert block_map.get_start_nodes_with_all_edges() == copy.get_start_nodes_with_all_edges()
    agents = [{"name": "a0", "start": START, "goal": GOAL}]
    plans = [SippPlanner(m, agents, time_limit=10).compute_plan()[0] for m in (block_map, copy)]
    assert plans[0] == plans[1]


def test_empty_kept_raises(block_map):
    with pytest.raises(ValueError):
        block_map.pruned(np.array([], dtype=int))
