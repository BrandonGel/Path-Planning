"""
Tests for the Python CTopPRM (path_planning/cluster/).
python -m unittest tests.test_ctopprm -v
"""

import unittest

import numpy as np

try:
    from path_planning.common.environment.map.graph_sampler import GraphSampler
    from path_planning.cluster.ctopprm import CTopPRM
    from path_planning.cluster.shortening import path_length, polyline_free_sampled
    from path_planning.utils.util import obstacles_world_to_grid, set_global_seed

    _IMPORTS_OK = True
    _IMPORT_ERROR = ""
except ModuleNotFoundError as exc:  # pragma: no cover
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(exc)


def _build_block_map(starts, goals, sample_num=400):
    """20x20 continuous map with a central 4x8 vertical block.

    The block guarantees exactly two homotopy classes between endpoints
    placed left and right of it at its mid-height.
    """
    obstacles_world = np.array(
        [(x + 0.5, y + 0.5) for x in range(8, 12) for y in range(6, 14)]
    )
    map_ = GraphSampler(
        bounds=[[0.0, 20.0], [0.0, 20.0]],
        resolution=1.0,
        start=list(starts),
        goal=list(goals),
        use_discrete_space=False,
    )
    map_.set_obstacles(obstacles_world_to_grid(map_, obstacles_world, 1.0))
    map_.update_esdf()
    map_.set_start(list(starts))
    map_.set_goal(list(goals))
    map_.set_parameters(
        sample_num=sample_num, num_neighbors=13, min_edge_len=1e-10, max_edge_len=5.0
    )
    nodes = map_.generateRandomNodes()
    map_.generate_roadmap(nodes)
    return map_


BLOCK_CENTER_Y = 10.0
START = (2.0, 10.0)
GOAL = (18.0, 10.0)


@unittest.skipUnless(_IMPORTS_OK, f"imports failed: {_IMPORT_ERROR}")
class TestCTopPRM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_global_seed(42)
        cls.map_ = _build_block_map([START], [GOAL])
        cls.planner = CTopPRM(cls.map_)
        cls.results = cls.planner.find_distinct_paths([(START, GOAL)])
        cls.pair_key = next(iter(cls.results))
        cls.paths = cls.results[cls.pair_key]

    def test_two_routes_around_obstacle(self):
        self.assertGreaterEqual(len(self.paths), 2)
        mean_ys = sorted(p[:, 1].mean() for p in self.paths[:2])
        self.assertLess(mean_ys[0], BLOCK_CENTER_Y)
        self.assertGreater(mean_ys[1], BLOCK_CENTER_Y)

    def test_paths_collision_free(self):
        bounds = np.asarray(self.map_.bounds, dtype=float)
        for path in self.paths:
            self.assertTrue(
                polyline_free_sampled(self.map_, path, self.planner.geometry)
            )
            np.testing.assert_allclose(path[0], START, atol=1e-6)
            np.testing.assert_allclose(path[-1], GOAL, atol=1e-6)
            self.assertTrue(np.all(path >= bounds[:, 0]))
            self.assertTrue(np.all(path <= bounds[:, 1]))
            self.assertTrue(np.issubdtype(path.dtype, np.floating))

    def test_seeds_grow_on_nondeformable(self):
        # The block makes the pair's min/max connections non-deformable, so
        # the centroid loop must have added seeds beyond the 2 endpoints.
        self.assertGreater(len(self.planner.seed_indices), 2)
        self.assertLessEqual(len(self.planner.seed_indices), self.planner.max_clusters)

    def test_path_length_budget(self):
        lengths = [path_length(p) for p in self.paths]
        self.assertEqual(lengths, sorted(lengths))
        cutoff = self.planner.cutoff_distance_ratio_to_shortest
        self.assertLessEqual(lengths[-1], cutoff * lengths[0] + 1e-6)
        straight = float(np.linalg.norm(np.array(GOAL) - np.array(START)))
        self.assertGreaterEqual(lengths[0], straight - 1e-6)

    def test_multi_pair_shared_clustering(self):
        set_global_seed(7)
        starts = [(2.0, 10.0), (2.0, 3.0)]
        goals = [(18.0, 10.0), (18.0, 17.0)]
        map_ = _build_block_map(starts, goals)
        planner = CTopPRM(map_)
        pairs = list(zip(starts, goals))
        results = planner.find_distinct_paths(pairs)
        self.assertEqual(len(results), 2)
        # Every seed labels its own cluster and keeps distance 0.
        num_endpoint_seeds = 4
        for k, seed in enumerate(planner.seed_indices[:num_endpoint_seeds]):
            self.assertEqual(int(planner.cluster_labels[seed]), k)
            self.assertEqual(planner.dist[seed], 0.0)
        for (s, g), paths in results.items():
            self.assertGreaterEqual(len(paths), 1)
        # On-demand cross pair (agent0 start -> agent1 goal) also plans.
        cross = planner.find_distinct_paths([(starts[0], goals[1])])
        self.assertGreaterEqual(len(next(iter(cross.values()))), 1)

    def test_greedy_fallback(self):
        set_global_seed(42)
        map_ = _build_block_map([START], [GOAL])
        planner = CTopPRM(map_, shortening_mode="greedy")
        results = planner.find_distinct_paths([(START, GOAL)])
        paths = next(iter(results.values()))
        self.assertGreaterEqual(len(paths), 1)
        for path in paths:
            self.assertTrue(polyline_free_sampled(map_, path, planner.geometry))

    def test_endpoint_resolution(self):
        # Exact node coordinate and index forms resolve identically; a nearby
        # off-node coordinate snaps to the nearest node.
        s_idx, g_idx = self.pair_key
        self.assertEqual(self.planner.resolve_endpoint(s_idx), s_idx)
        self.assertEqual(self.planner.resolve_endpoint(START), s_idx)
        snapped = self.planner.resolve_endpoint((START[0] + 0.05, START[1] - 0.05))
        self.assertEqual(snapped, s_idx)
        with self.assertRaises(ValueError):
            self.planner.find_distinct_paths([(s_idx, s_idx)])


if __name__ == "__main__":
    unittest.main()
