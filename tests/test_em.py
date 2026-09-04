"""
Tests for the graph-distance EM clustering (path_planning/cluster/em/).
python -m unittest tests.test_em -v
"""

import unittest

import numpy as np

try:
    from path_planning.common.environment.map.graph_sampler import GraphSampler
    from path_planning.cluster.CTopPRMpy.ctopprm import CTopPRM
    from path_planning.cluster.em.graph_em import GraphEM
    from path_planning.cluster.kmeans.graph_kmeans import (
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
class TestGraphEM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_global_seed(42)
        cls.map_ = _build_block_map([START], [GOAL])
        cls.em = GraphEM(cls.map_, K).fit(SEEDS)

    def test_fixed_seeds_stay_fixed(self):
        em = self.em
        for k, seed in enumerate(SEEDS):
            s = em.resolve_endpoint(seed)
            self.assertEqual(em.center_node_indices[k], s)
            self.assertEqual(int(em.cluster_labels[s]), k)
            self.assertEqual(em.dist[s], 0.0)
            self.assertTrue(em.center_is_fixed[k])
            np.testing.assert_allclose(em.centers[k], em._points[s])

    def test_cluster_count_and_coverage(self):
        em = self.em
        self.assertEqual(len(em.center_node_indices), em.n_clusters)
        self.assertEqual(em.n_clusters, K)
        finite = np.isfinite(em.dist)
        # label -1 exactly on unreachable nodes
        self.assertTrue(np.all(em.cluster_labels[finite] >= 0))
        self.assertTrue(np.all(em.cluster_labels[~finite] == -1))
        self.assertTrue(np.all(em.cluster_labels[finite] < em.n_clusters))
        # prev forest of a member walks back to its own cluster's anchor
        for k in range(em.n_clusters):
            members = np.flatnonzero(em.cluster_labels == k)
            self.assertGreater(members.size, 0)
            node = int(members[np.argmax(em.dist[members])])
            while em.prev[node] >= 0:
                node = int(em.prev[node])
            self.assertEqual(node, em.center_node_indices[k])

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
            getattr(self.map_, "road_map_edge_weights", None),
        )
        labels, dist, prev = multi_source_dijkstra(adj, em.center_node_indices)
        np.testing.assert_array_equal(em.cluster_labels, labels)
        np.testing.assert_array_equal(em.prev, prev)
        np.testing.assert_allclose(em.dist, dist)

    def test_centroids_free_or_snapped(self):
        em = self.em
        for k in range(em.n_clusters):
            if em.center_is_fixed[k]:
                continue
            if em.center_snapped[k]:
                np.testing.assert_allclose(
                    em.centers[k], em._points[em.center_node_indices[k]]
                )
            else:
                self.assertTrue(self.map_.point_expandable(tuple(em.centers[k])))

    def test_convergence(self):
        em = self.em
        self.assertLessEqual(em.n_iter_, em.max_iters)
        self.assertEqual(len(em.log_likelihood_history), em.n_iter_)
        self.assertTrue(all(np.isfinite(v) for v in em.log_likelihood_history))

    def test_clusters_respect_obstacle(self):
        # Graph-aware assignment: no cluster may straddle the central block
        # by pairing a member with a center on the opposite side at the
        # block's mid-height band (Euclidean EM would).
        em = self.em
        block_lo, block_hi = 8.0, 12.0
        for k in range(em.n_clusters):
            members = np.flatnonzero(em.cluster_labels == k)
            pts = em._points[members]
            cx = em.centers[k][0]
            band = pts[np.abs(pts[:, 1] - 10.0) < 2.0]
            if cx <= block_lo:
                self.assertTrue(np.all(band[:, 0] < block_hi))
            elif cx >= block_hi:
                self.assertTrue(np.all(band[:, 0] > block_lo))

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

    def test_ctopprm_em_clustering(self):
        set_global_seed(42)
        map_ = _build_block_map([START], [GOAL])
        planner = CTopPRM(map_, clustering="em", em_clusters=6)
        results = planner.find_distinct_paths([(START, GOAL)])
        paths = next(iter(results.values()))
        self.assertGreaterEqual(len(paths), 2)
        # endpoint seeds keep the positional contract after EM seeding
        self.assertGreaterEqual(len(planner.seed_indices), 6)
        for k, seed in enumerate(planner.seed_indices):
            self.assertEqual(int(planner.cluster_labels[seed]), k)
            self.assertEqual(planner.dist[seed], 0.0)


if __name__ == "__main__":
    unittest.main()
