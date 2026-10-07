"""
Tests for the roadmap embedding controls (path_planning/cluster/embeddings.py)
and their routing through gnn_cluster_map.gnn_seed_indices.
python -m unittest tests.test_embeddings -v

The gnn cases need the gnn6 checkpoint (GNN6_RUN below) and are skipped when
it is absent. Everything runs on CPU: the CUDA forward is not bit-reproducible
(~2e-7 run to run), the CPU forward is.
"""

import os
import unittest
from pathlib import Path

import numpy as np

try:
    from path_planning.cluster.embeddings import (
        EMBEDDING_METHODS,
        compute_embedding,
        default_method_name,
        explained_variance,
        largest_component,
        symmetric_roadmap_csr,
    )
    from path_planning.cluster.gnn_cluster_map import (
        _kmeans_medoids,
        gnn_seed_indices,
        load_cluster_encoder,
        read_run_config,
        ablation_flags_from_config,
        sampler_to_cluster_heterodata,
    )
    from path_planning.utils.util import set_global_seed
    from tests.test_ctopprm import _build_block_map, START, GOAL

    _IMPORTS_OK = True
    _IMPORT_ERROR = ""
except ModuleNotFoundError as exc:  # pragma: no cover
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(exc)

GNN6_RUN = Path(os.environ.get(
    "GNN6_RUN",
    Path(__file__).resolve().parents[1] / "logs/cluster/gatv2_compile/wandb/"
    "offline-run-20260915_070451-miszrtsa"))
GNN6_EPOCH = 100


def _detach_vertex(map_, i: int) -> None:
    """Remove every roadmap edge touching node i (road_map + aligned weights)."""
    for u in range(len(map_.road_map)):
        keep = [k for k, v in enumerate(map_.road_map[u]) if v != i]
        map_.road_map[u] = [map_.road_map[u][k] for k in keep]
        map_.road_map_edge_weights[u] = [map_.road_map_edge_weights[u][k] for k in keep]
    map_.road_map[i] = []
    map_.road_map_edge_weights[i] = []


@unittest.skipUnless(_IMPORTS_OK, f"imports failed: {_IMPORT_ERROR}")
class TestEmbeddingControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_global_seed(42)
        cls.map = _build_block_map([START], [GOAL], sample_num=300)
        cls.n = len(cls.map.nodes)
        cls.sg = set(cls.map.start_nodes_index.values()) | set(cls.map.goal_nodes_index.values())
        cls.K = 24

    # -- shape / vertex-set contract ------------------------------------------------
    def test_every_method_returns_one_row_per_node(self):
        for method, d in (("euclid", 32), ("isomap", 8), ("spectral", 8)):
            z, info = compute_embedding(self.map, method, d)
            self.assertEqual(z.shape[0], self.n, method)
            self.assertEqual(info["n_embedded"], self.n)
            self.assertEqual(z.dtype, np.float64)

    def test_euclid_is_raw_coordinates(self):
        z, info = compute_embedding(self.map, "euclid", 32)
        pos = np.array([n.current for n in self.map.nodes], dtype=float)
        self.assertTrue(np.array_equal(z, pos))
        self.assertEqual(info["dim"], self.map.dim)
        self.assertEqual(len(info["dropped"]), 0)

    def test_default_method_names(self):
        self.assertEqual(default_method_name("gnn", 32), "gnn")
        self.assertEqual(default_method_name("euclid", 2), "euclid")
        self.assertEqual(default_method_name("isomap", 8), "isomap_8")
        self.assertEqual(default_method_name("spectral", 32), "spectral_32")

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            compute_embedding(self.map, "pca", 8)

    # -- isomap --------------------------------------------------------------------
    def test_isomap_prefix_property_and_explained_variance(self):
        z8, i8 = compute_embedding(self.map, "isomap", 8, full_spectrum=True)
        z32, i32 = compute_embedding(self.map, "isomap", 32, full_spectrum=True)
        ok = ~np.isnan(z8[:, 0])
        self.assertTrue(np.allclose(z8[ok], z32[ok][:, :8]))
        self.assertGreater(i8["explained_variance"], 0.5)
        self.assertLessEqual(i8["explained_variance"], i32["explained_variance"] + 1e-12)
        self.assertLessEqual(i32["explained_variance"], 1.0 + 1e-12)
        self.assertGreaterEqual(i32["negative_mass"], 0.0)
        self.assertEqual(i8["dim"], 8)
        self.assertIsNotNone(i32["spectrum"])
        # eigenvalues descending and positive
        ev = i32["eigenvalues"]
        self.assertTrue(np.all(np.diff(ev) <= 1e-9))
        self.assertTrue(np.all(ev > 0))
        self.assertEqual(explained_variance(ev, 32, i32["spectrum"]), i32["explained_variance"])

    def test_isomap_geodesic_consistency_on_lcc(self):
        """Embedding distances approximate geodesics on the largest component."""
        from scipy.sparse.csgraph import shortest_path
        z, info = compute_embedding(self.map, "isomap", 32)
        csr, _ = symmetric_roadmap_csr(self.map)
        lcc, _, _ = largest_component(csr)
        geo = shortest_path(csr[lcc][:, lcc], method="D", directed=False)
        emb = np.linalg.norm(z[lcc][:, None, :] - z[lcc][None, :, :], axis=-1)
        iu = np.triu_indices(len(lcc), 1)
        rho = np.corrcoef(geo[iu], emb[iu])[0, 1]
        self.assertGreater(rho, 0.9)

    def test_detached_vertex_is_dropped_and_kmeans_still_runs(self):
        set_global_seed(42)
        m = _build_block_map([START], [GOAL], sample_num=300)
        sg = set(m.start_nodes_index.values()) | set(m.goal_nodes_index.values())
        victim = next(i for i in range(len(m.nodes)) if i not in sg)
        _detach_vertex(m, victim)
        csr, _ = symmetric_roadmap_csr(m)
        _, n_comp, _ = largest_component(csr)
        for method in ("isomap", "spectral"):
            z, info = compute_embedding(m, method, 8)
            self.assertIn(victim, info["dropped"].tolist(), method)
            self.assertTrue(np.all(np.isnan(z[victim])), method)
            self.assertEqual(info["n_components"], n_comp)
            nan_rows = np.flatnonzero(np.isnan(z[:, 0]))
            self.assertTrue(np.array_equal(np.sort(nan_rows), np.sort(info["dropped"])))
            seeds, times, latent = gnn_seed_indices(None, None, m, self.K, None,
                                                    embedding_method=method, embedding_dim=8)
            self.assertNotIn(victim, seeds)
            self.assertTrue(all(s not in sg for s in seeds))
            e = times["embedding"]
            self.assertEqual(e["n_candidates"], len(m.nodes) - len(sg) - e["n_dropped"])
            self.assertEqual(len(seeds), min(self.K - len(sg), e["n_candidates"] - 1))
            self.assertTrue(np.all(np.isfinite(latent["centroids"])))

    def test_start_outside_component_raises(self):
        set_global_seed(42)
        m = _build_block_map([START], [GOAL], sample_num=300)
        s_idx = next(iter(m.start_nodes_index.values()))
        _detach_vertex(m, s_idx)
        for method in ("isomap", "spectral"):
            with self.assertRaises(RuntimeError):
                compute_embedding(m, method, 8)

    # -- spectral ------------------------------------------------------------------
    def test_spectral_eigenvalues(self):
        z, info = compute_embedding(self.map, "spectral", 8)
        ev = info["eigenvalues"]
        self.assertEqual(len(ev), 8)
        self.assertTrue(np.all(ev > 1e-8))            # trivial (zero) eigenvalue dropped
        self.assertTrue(np.all(np.diff(ev) >= -1e-9))  # ascending
        self.assertTrue(np.all(ev <= 2.0 + 1e-9))      # L_sym spectrum lies in [0, 2]
        self.assertGreater(info["sigma"], 0.0)
        ok = ~np.isnan(z[:, 0])
        # eigenvectors of a symmetric operator are orthonormal
        gram = z[ok].T @ z[ok]
        self.assertTrue(np.allclose(gram, np.eye(8), atol=1e-6))

    # -- cache ---------------------------------------------------------------------
    def test_cache_roundtrip_reports_stored_time(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            z1, i1 = compute_embedding(self.map, "isomap", 8, cache_dir=tmp)
            z2, i2 = compute_embedding(self.map, "isomap", 8, cache_dir=tmp)
            self.assertFalse(i1["cache_hit"])
            self.assertTrue(i2["cache_hit"])
            self.assertTrue(np.array_equal(np.nan_to_num(z1), np.nan_to_num(z2)))
            self.assertEqual(i1["compute_time"], i2["compute_time"])
            self.assertGreater(i1["compute_time"], 0.0)
            # full spectrum requested after a spectrum-less entry -> recomputed, then cached
            z3, i3 = compute_embedding(self.map, "isomap", 8, cache_dir=tmp, full_spectrum=True)
            self.assertFalse(i3["cache_hit"])
            self.assertIsNotNone(i3["spectrum"])
            z4, i4 = compute_embedding(self.map, "isomap", 8, cache_dir=tmp, full_spectrum=True)
            self.assertTrue(i4["cache_hit"])

    # -- K-means over a 2-dim control through the shared path -------------------------
    def test_euclid_kmeans_path(self):
        seeds, times, latent = gnn_seed_indices(None, None, self.map, self.K, None,
                                                embedding_method="euclid", embedding_dim=32)
        self.assertEqual(len(seeds), self.K - len(self.sg))
        self.assertEqual(latent["embeddings"].shape, (self.n, 2))
        self.assertEqual(latent["centroids"].shape, (self.K - len(self.sg), 2))
        self.assertEqual(times["embedding"]["method"], "euclid")
        self.assertEqual(times["embedding"]["n_anchors"], len(self.sg))
        self.assertTrue(all(s not in self.sg for s in seeds))

    def test_drop_disconnected_equalises_candidates_for_euclid(self):
        set_global_seed(42)
        m = _build_block_map([START], [GOAL], sample_num=300)
        sg = set(m.start_nodes_index.values()) | set(m.goal_nodes_index.values())
        victim = next(i for i in range(len(m.nodes)) if i not in sg)
        _detach_vertex(m, victim)
        _, t_keep, _ = gnn_seed_indices(None, None, m, self.K, None, embedding_method="euclid")
        _, t_drop, _ = gnn_seed_indices(None, None, m, self.K, None, embedding_method="euclid",
                                        drop_disconnected=True)
        _, t_iso, _ = gnn_seed_indices(None, None, m, self.K, None, embedding_method="isomap")
        self.assertEqual(t_keep["embedding"]["n_dropped"], 0)
        self.assertEqual(t_drop["embedding"]["n_candidates"], t_iso["embedding"]["n_candidates"])
        self.assertEqual(t_drop["embedding"]["dropped"], t_iso["embedding"]["dropped"])

    # -- gnn path: bit-identical to the direct forward -----------------------------------
    @unittest.skipUnless((GNN6_RUN / "files" / "model" / f"epoch_{GNN6_EPOCH}.pth").exists(),
                         f"gnn6 checkpoint not found under {GNN6_RUN}")
    def test_gnn_compute_embedding_is_bit_identical_to_direct_forward(self):
        import torch
        device = torch.device("cpu")
        flags = ablation_flags_from_config(read_run_config(GNN6_RUN))
        data = sampler_to_cluster_heterodata(self.map, *flags)
        model, _ = load_cluster_encoder(GNN6_RUN, data, epoch=GNN6_EPOCH, device=device)
        with torch.no_grad():
            ds = data.to(device)
            z_direct = model(dict(ds.x_dict), ds.edge_index_dict, ds.edge_attr_dict)['node'].cpu().numpy()
        z_new, info = compute_embedding(self.map, "gnn", model=model, data=data, device=device)
        self.assertTrue(np.array_equal(z_direct, z_new))
        self.assertEqual(z_new.dtype, np.float32)
        self.assertEqual(info["dim"], z_direct.shape[1])
        # default kwargs of gnn_seed_indices reproduce the original inline path exactly
        seeds, times, latent = gnn_seed_indices(model, data, self.map, self.K, device)
        sg = self.sg
        free = np.array([i for i in range(self.n) if i not in sg], dtype=np.int64)
        k_free = min(max(self.K - len(sg), 1), len(free) - 1)
        ref_seeds, ref_centroids = _kmeans_medoids(z_direct, free, k_free)
        self.assertEqual(seeds, ref_seeds)
        self.assertTrue(np.array_equal(latent["centroids"], ref_centroids))
        self.assertTrue(np.array_equal(latent["embeddings"], z_direct))
        self.assertEqual(times["embedding"]["n_dropped"], 0)
        self.assertEqual(times["embedding"]["n_candidates"], len(free))


if __name__ == "__main__":
    unittest.main()
