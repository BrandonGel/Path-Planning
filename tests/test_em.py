"""
Tests for the graph-distance EM clustering (path_planning/cluster/em/).
python -m unittest tests.test_em -v
"""

import unittest

import numpy as np

from path_planning.cluster.em.graph_em import GraphEM
from path_planning.cluster.kmeans.graph_kmeans import (
    build_symmetric_adjacency,
    multi_source_dijkstra,
)
from path_planning.utils.util import set_global_seed
from tests.helpers import GOAL, K, SEEDS, START, ClusteringContract, build_block_map




class TestGraphEM(ClusteringContract, unittest.TestCase):
    CLUSTERING = "em"

    @classmethod
    def setUpClass(cls):
        set_global_seed(42)
        cls.map_ = build_block_map([START], [GOAL])
        cls.model = cls.em = GraphEM(cls.map_, K).fit(SEEDS)

    def test_responsibilities_stochastic(self):
        em = self.em
        self.assertEqual(em.responsibilities.shape, (len(em._points), em.n_clusters))
        finite = np.isfinite(em.dist)
        np.testing.assert_allclose(
            em.responsibilities[finite].sum(axis=1), 1.0, atol=1e-9
        )
        np.testing.assert_array_equal(em.responsibilities[~finite], 0.0)
        self.assertTrue(np.all(em.responsibilities >= 0.0))

    def test_mixture_parameters(self):
        em = self.em
        self.assertAlmostEqual(float(em.weights_.sum()), 1.0, places=9)
        self.assertTrue(np.all(em.weights_ > 0.0))
        self.assertTrue(np.all(em.sigmas_ >= em.min_sigma))
        # full covariances: symmetric, eigenvalue-floored, and sigmas_ is
        # the effective width sqrt(trace/dim)
        covs = em.covariances_
        dim = em._points.shape[1]
        self.assertEqual(covs.shape, (em.n_clusters, dim, dim))
        for k in range(em.n_clusters):
            np.testing.assert_allclose(covs[k], covs[k].T)
            self.assertGreaterEqual(
                float(np.linalg.eigvalsh(covs[k]).min()),
                em.min_sigma ** 2 - 1e-9,
            )
        np.testing.assert_allclose(
            em.sigmas_, np.sqrt(np.trace(covs, axis1=1, axis2=2) / dim)
        )

    def test_hard_state_matches_wavefront(self):
        # Finalization contract: cluster_labels/dist/prev are the wavefront
        # fill from the final anchors.
        em = self.em
        points = np.asarray([n.current for n in self.map_.nodes], dtype=float)
        adj = build_symmetric_adjacency(
            points,
            self.map_.road_map,
            self.map_.road_map_edge_weights,
        )
        labels, dist, prev = multi_source_dijkstra(adj, em.center_node_indices)
        np.testing.assert_array_equal(em.cluster_labels, labels)
        np.testing.assert_array_equal(em.prev, prev)
        np.testing.assert_allclose(em.dist, dist)

    def test_convergence(self):
        em = self.em
        self.assertLessEqual(em.n_iter_, em.max_iters)
        self.assertEqual(len(em.log_likelihood_history), em.n_iter_)
        self.assertTrue(all(np.isfinite(v) for v in em.log_likelihood_history))

    def test_free_anchor_spacing(self):
        # Duplicate guard: EM never separates co-located components, so
        # anchors must keep at least dup_radius of Euclidean spacing even
        # at high K (without the guard most free anchors collapse into
        # near-duplicate pairs).
        set_global_seed(3)
        em = GraphEM(self.map_, len(SEEDS) + 20).fit(SEEDS)
        anchors = em._points[em.center_node_indices]
        d = np.linalg.norm(anchors[:, None, :] - anchors[None, :, :], axis=2)
        d[np.diag_indices_from(d)] = np.inf
        self.assertGreaterEqual(float(d.min()), em.dup_radius)

    def test_reproducibility(self):
        set_global_seed(7)
        a = GraphEM(self.map_, K).fit(SEEDS)
        set_global_seed(7)
        b = GraphEM(self.map_, K).fit(SEEDS)
        self.assertEqual(a.center_node_indices, b.center_node_indices)
        np.testing.assert_array_equal(a.cluster_labels, b.cluster_labels)
        np.testing.assert_allclose(a.centers, b.centers)
        np.testing.assert_allclose(a.responsibilities, b.responsibilities)
        np.testing.assert_allclose(a.covariances_, b.covariances_)

    def test_validation(self):
        with self.assertRaises(ValueError):
            GraphEM(self.map_, 1).fit(SEEDS)  # k < #fixed
        with self.assertRaises(ValueError):
            GraphEM(self.map_, K).fit([START, START])  # duplicate seeds
        with self.assertRaises(ValueError):
            GraphEM(self.map_, K, init="bogus")
        with self.assertRaises(ValueError):
            GraphEM(self.map_, K, min_sigma=0.0)
        with self.assertRaises(ValueError):
            GraphEM(self.map_, K, dup_radius=-1.0)
