"""
Tests for the graph-aware k-means (path_planning/cluster/kmeans/).
python -m unittest tests.test_kmeans -v
"""

import unittest

import numpy as np

try:
    from path_planning.common.environment.map.graph_sampler import GraphSampler
    from path_planning.cluster.CTopPRMpy.ctopprm import CTopPRM
    from path_planning.cluster.kmeans.graph_kmeans import (
        GraphKMeans,
        build_symmetric_adjacency,
        multi_source_dijkstra,
    )
    from path_planning.utils.util import obstacles_world_to_grid, set_global_seed

    _IMPORTS_OK = True
    _IMPORT_ERROR = ""
except ModuleNotFoundError as exc:  # pragma: no cover
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(exc)


def _build_block_map(starts, goals, sample_num=400):
    """20x20 continuous map with a central 4x8 vertical block."""
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


START = (2.0, 10.0)
GOAL = (18.0, 10.0)
SEEDS = [START, GOAL, (2.0, 3.0), (18.0, 17.0)]
K = 8


@unittest.skipUnless(_IMPORTS_OK, f"imports failed: {_IMPORT_ERROR}")
class TestGraphKMeans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_global_seed(42)
        cls.map_ = _build_block_map([START], [GOAL])
        cls.km = GraphKMeans(cls.map_, K).fit(SEEDS)

    def test_fixed_seeds_stay_fixed(self):
        km = self.km
        for k, seed in enumerate(SEEDS):
            s = km.resolve_endpoint(seed)
            self.assertEqual(km.center_node_indices[k], s)
            self.assertEqual(int(km.cluster_labels[s]), k)
            self.assertEqual(km.dist[s], 0.0)
            self.assertTrue(km.center_is_fixed[k])
            np.testing.assert_allclose(km.centers[k], km._points[s])

    def test_cluster_count_and_coverage(self):
        km = self.km
        self.assertEqual(len(km.center_node_indices), km.n_clusters)
        self.assertEqual(km.n_clusters, K)
        finite = np.isfinite(km.dist)
        # label -1 exactly on unreachable nodes
        self.assertTrue(np.all(km.cluster_labels[finite] >= 0))
        self.assertTrue(np.all(km.cluster_labels[~finite] == -1))
        self.assertTrue(np.all(km.cluster_labels[finite] < km.n_clusters))
        # prev forest of a member walks back to its own cluster's anchor
        for k in range(km.n_clusters):
            members = np.flatnonzero(km.cluster_labels == k)
            self.assertGreater(members.size, 0)
            node = int(members[np.argmax(km.dist[members])])
            while km.prev[node] >= 0:
                node = int(km.prev[node])
            self.assertEqual(node, km.center_node_indices[k])

    def test_centroids_free_or_snapped(self):
        km = self.km
        for k in range(km.n_clusters):
            if km.center_is_fixed[k]:
                continue
            if km.center_snapped[k]:
                np.testing.assert_allclose(
                    km.centers[k], km._points[km.center_node_indices[k]]
                )
            else:
                self.assertTrue(self.map_.point_expandable(tuple(km.centers[k])))

    def test_convergence(self):
        km = self.km
        self.assertLessEqual(km.n_iter_, km.max_iters)
        self.assertEqual(len(km.inertia_history), km.n_iter_)
        self.assertTrue(all(np.isfinite(v) for v in km.inertia_history))

    def test_k_equals_seeds_matches_wavefront(self):
        # With no free centers there is nothing to move: fit converges in
        # one iteration to the plain multi-source Dijkstra partition.
        km = GraphKMeans(self.map_, len(SEEDS)).fit(SEEDS)
        self.assertEqual(km.n_iter_, 1)
        self.assertEqual(
            km.center_node_indices, [km.resolve_endpoint(s) for s in SEEDS]
        )
        points = np.asarray([n.current for n in self.map_.nodes], dtype=float)
        adj = build_symmetric_adjacency(
            points,
            self.map_.road_map,
            getattr(self.map_, "road_map_edge_weights", None),
        )
        labels, dist, prev = multi_source_dijkstra(adj, km.center_node_indices)
        np.testing.assert_array_equal(km.cluster_labels, labels)
        np.testing.assert_array_equal(km.prev, prev)
        np.testing.assert_allclose(km.dist, dist)
        # ... and to CTopPRM's own wavefront fill (cross-implementation check).
        planner = CTopPRM(self.map_)
        planner._wavefront_fill(km.center_node_indices)
        np.testing.assert_array_equal(km.cluster_labels, planner.cluster_labels)
        np.testing.assert_array_equal(km.prev, planner.prev)
        np.testing.assert_allclose(km.dist, planner.dist)

    def test_reproducibility(self):
        set_global_seed(7)
        a = GraphKMeans(self.map_, K).fit(SEEDS)
        set_global_seed(7)
        b = GraphKMeans(self.map_, K).fit(SEEDS)
        self.assertEqual(a.center_node_indices, b.center_node_indices)
        np.testing.assert_array_equal(a.cluster_labels, b.cluster_labels)
        np.testing.assert_allclose(a.centers, b.centers)

    def test_validation(self):
        with self.assertRaises(ValueError):
            GraphKMeans(self.map_, 1).fit(SEEDS)  # k < #fixed
        with self.assertRaises(ValueError):
            GraphKMeans(self.map_, K).fit([START, START])  # duplicate seeds
        with self.assertRaises(ValueError):
            GraphKMeans(self.map_, K, init="bogus")

    def test_clusters_respect_obstacle(self):
        # Graph-aware assignment: no cluster may straddle the central block
        # by pairing a member with a center on the opposite side at the
        # block's mid-height band (Euclidean k-means would).
        km = self.km
        block_lo, block_hi = 8.0, 12.0
        for k in range(km.n_clusters):
            members = np.flatnonzero(km.cluster_labels == k)
            pts = km._points[members]
            cx = km.centers[k][0]
            band = pts[np.abs(pts[:, 1] - 10.0) < 2.0]
            if cx <= block_lo:
                self.assertTrue(np.all(band[:, 0] < block_hi))
            elif cx >= block_hi:
                self.assertTrue(np.all(band[:, 0] > block_lo))

    def test_ctopprm_kmeans_clustering(self):
        set_global_seed(42)
        map_ = _build_block_map([START], [GOAL])
        planner = CTopPRM(map_, clustering="kmeans", min_clusters=6)
        results = planner.find_distinct_paths([(START, GOAL)])
        paths = next(iter(results.values()))
        self.assertGreaterEqual(len(paths), 2)
        # endpoint seeds keep the positional contract after k-means seeding
        self.assertGreaterEqual(len(planner.seed_indices), 6)
        for k, seed in enumerate(planner.seed_indices):
            self.assertEqual(int(planner.cluster_labels[seed]), k)
            self.assertEqual(planner.dist[seed], 0.0)


if __name__ == "__main__":
    unittest.main()
