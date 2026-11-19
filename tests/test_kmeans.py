"""
Tests for the graph-aware k-means (path_planning/cluster/kmeans/).
python -m unittest tests.test_kmeans -v
"""

import unittest

import numpy as np

from path_planning.cluster.CTopPRMpy.ctopprm import CTopPRM
from path_planning.cluster.kmeans.graph_kmeans import (
    GraphKMeans,
    build_symmetric_adjacency,
    multi_source_dijkstra,
)
from path_planning.utils.util import set_global_seed
from tests.helpers import GOAL, K, SEEDS, START, ClusteringContract, build_block_map




class TestGraphKMeans(ClusteringContract, unittest.TestCase):
    CLUSTERING = "kmeans"

    @classmethod
    def setUpClass(cls):
        set_global_seed(42)
        cls.map_ = build_block_map([START], [GOAL])
        cls.model = cls.km = GraphKMeans(cls.map_, K).fit(SEEDS)

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
            self.map_.road_map_edge_weights,
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
