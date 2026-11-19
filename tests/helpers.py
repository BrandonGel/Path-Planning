"""Shared fixtures of the test suite (imported as ``tests.helpers``)."""
import os

import numpy as np

from path_planning.cluster.CTopPRMpy.ctopprm import CTopPRM
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.utils.util import obstacles_world_to_grid, set_global_seed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_2D = os.path.join(REPO, "path_planning", "maps", "2d", "2d.yaml")

# ---- the block map of the cluster tests
START = (2.0, 10.0)
GOAL = (18.0, 10.0)
SEEDS = [START, GOAL, (2.0, 3.0), (18.0, 17.0)]
K = 8


def build_block_map(starts, goals, sample_num=400):
    """20x20 continuous map with a central 4x8 vertical block (x in [8, 12], y in [6, 14]) and a PRM."""
    obstacles_world = np.array([(x + 0.5, y + 0.5) for x in range(8, 12) for y in range(6, 14)])
    map_ = GraphSampler(bounds=[[0.0, 20.0], [0.0, 20.0]], resolution=1.0, start=list(starts), goal=list(goals),
                        use_discrete_space=False)
    map_.set_obstacles(obstacles_world_to_grid(map_, obstacles_world, 1.0))
    map_.update_esdf()
    map_.set_start(list(starts))
    map_.set_goal(list(goals))
    map_.set_roadmap_config(sample_num=sample_num, num_neighbors=13, min_edge_length=1e-10, max_edge_length=5.0)
    map_.generateRandomNodes()
    map_.generate_roadmap("prm")
    return map_


class ClusteringContract:
    """Tests every graph clustering (k-means, EM) must pass. A subclass (with ``unittest.TestCase``) sets
    ``map_`` (``build_block_map([START], [GOAL])``), ``model`` (fitted on ``SEEDS`` with ``K`` clusters) and
    ``CLUSTERING`` (the name ``CTopPRM(clustering=...)`` takes)."""

    def test_fixed_seeds_stay_fixed(self):
        model = self.model
        for k, seed in enumerate(SEEDS):
            s = model.resolve_endpoint(seed)
            self.assertEqual(model.center_node_indices[k], s)
            self.assertEqual(int(model.cluster_labels[s]), k)
            self.assertEqual(model.dist[s], 0.0)
            self.assertTrue(model.center_is_fixed[k])
            np.testing.assert_allclose(model.centers[k], model._points[s])

    def test_cluster_count_and_coverage(self):
        model = self.model
        self.assertEqual(len(model.center_node_indices), model.n_clusters)
        self.assertEqual(model.n_clusters, K)
        finite = np.isfinite(model.dist)
        # label -1 exactly on unreachable nodes
        self.assertTrue(np.all(model.cluster_labels[finite] >= 0))
        self.assertTrue(np.all(model.cluster_labels[~finite] == -1))
        self.assertTrue(np.all(model.cluster_labels[finite] < model.n_clusters))
        # prev forest of a member walks back to its own cluster's anchor
        for k in range(model.n_clusters):
            members = np.flatnonzero(model.cluster_labels == k)
            self.assertGreater(members.size, 0)
            node = int(members[np.argmax(model.dist[members])])
            while model.prev[node] >= 0:
                node = int(model.prev[node])
            self.assertEqual(node, model.center_node_indices[k])

    def test_centroids_free_or_snapped(self):
        model = self.model
        for k in range(model.n_clusters):
            if model.center_is_fixed[k]:
                continue
            if model.center_snapped[k]:
                np.testing.assert_allclose(model.centers[k], model._points[model.center_node_indices[k]])
            else:
                self.assertFalse(self.map_.in_collision(tuple(model.centers[k])))

    def test_clusters_respect_obstacle(self):
        # Graph-aware assignment: no cluster may straddle the central block by pairing a member with a
        # center on the opposite side at the block's mid-height band (a Euclidean clustering would).
        model = self.model
        block_lo, block_hi = 8.0, 12.0
        for k in range(model.n_clusters):
            members = np.flatnonzero(model.cluster_labels == k)
            pts = model._points[members]
            cx = model.centers[k][0]
            band = pts[np.abs(pts[:, 1] - 10.0) < 2.0]
            if cx <= block_lo:
                self.assertTrue(np.all(band[:, 0] < block_hi))
            elif cx >= block_hi:
                self.assertTrue(np.all(band[:, 0] > block_lo))

    def test_ctopprm_clustering(self):
        set_global_seed(42)
        map_ = build_block_map([START], [GOAL])
        planner = CTopPRM(map_, clustering=self.CLUSTERING, min_clusters=6)
        results = planner.find_distinct_paths([(START, GOAL)])
        paths = next(iter(results.values()))
        self.assertGreaterEqual(len(paths), 2)
        # endpoint seeds keep the positional contract after the clustering's seeding
        self.assertGreaterEqual(len(planner.seed_indices), 6)
        for k, seed in enumerate(planner.seed_indices):
            self.assertEqual(int(planner.cluster_labels[seed]), k)
            self.assertEqual(planner.dist[seed], 0.0)
