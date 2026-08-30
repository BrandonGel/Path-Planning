"""Start/goal placement modes of InputFile.gen_input (config/gen.yaml): uniform must stay bit-for-bit
identical to the historical generator; gaussian must cluster starts/goals around recorded means."""

import hashlib
import json
import math
import tempfile
import unittest
from itertools import product, combinations
from pathlib import Path

import numpy as np

try:
    from path_planning.data_generation.dataset_ground_truth_map import InputFile
    from path_planning.data_generation.dataset_util import (
        normalize_gen_config,
        read_gen_config_from_yaml,
    )
    from path_planning.utils.util import set_global_seed, obstacles_world_to_grid
    from path_planning.common.environment.map.graph_sampler import GraphSampler

    _HAS_SAMPLER = True
except ModuleNotFoundError:
    _HAS_SAMPLER = False


BASE = dict(
    bounds=[[0, 32.0], [0, 32.0]], resolution=1.0, nb_agents=8, nb_obstacles=0.1, agent_radius=0.5,
    obs_size=0.5, time_limit=1, max_iterations=10, road_map_type="grid", use_discrete_space=True,
    sample_num=0, num_neighbors=4.0, min_edge_len=1e-10, max_edge_len=1.1,
)
# sha256 of {agents, obstacles} produced by the generator BEFORE the gaussian mode was added.
GOLDEN = {
    42: "cfe4663b05a648aea7add90eaf86b749a26cc2916eedb49c3ba4691194815ece",
    7: "8c3bbc016cdb2647e8a2f5e7386627319bd501efb0542ebf075aed382e70d301",
}


def _gen(seed, gen=None, **overrides):
    set_global_seed(seed)
    cfg = dict(BASE, **overrides)
    if gen is not None:
        cfg["gen"] = gen
    return InputFile(Path("/nonexistent/input.yaml"), 0).gen_input(**cfg)


def _digest(inpt):
    payload = {
        "agents": [{"start": [float(v) for v in a["start"]], "goal": [float(v) for v in a["goal"]]} for a in inpt["agents"]],
        "obstacles": [[float(v) for v in o] for o in inpt["map"]["obstacles"]],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cells(inpt):
    m = GraphSampler(bounds=BASE["bounds"], resolution=BASE["resolution"], start=[], goal=[], use_discrete_space=True)
    conv = lambda p: tuple(int(v) for v in np.asarray(m.world_to_map(tuple(float(x) for x in p), discrete=True)).reshape(-1))
    return m, [(conv(a["start"]), conv(a["goal"])) for a in inpt["agents"]]


GAUSS = {"type": "gaussian", "gaussian": {"std_scale": 0.05, "max_attempts": 100, "separate_means": True}}


@unittest.skipUnless(_HAS_SAMPLER, "GraphSampler not available in this environment")
class TestUniformRegression(unittest.TestCase):
    def test_bitwise_identical_to_historical_generator(self):
        for seed, golden in GOLDEN.items():
            for gen in (None, {"type": "uniform"}, {}):
                inpt = _gen(seed, gen)
                self.assertEqual(_digest(inpt), golden, f"seed={seed} gen={gen}")
                self.assertNotIn("gen", inpt)


@unittest.skipUnless(_HAS_SAMPLER, "GraphSampler not available in this environment")
class TestGaussian(unittest.TestCase):
    def test_obstacles_identical_to_uniform_for_same_seed(self):
        for seed in range(1, 6):
            self.assertEqual(_gen(seed)["map"]["obstacles"], _gen(seed, GAUSS)["map"]["obstacles"])

    def test_record_and_clustering(self):
        std_world = GAUSS["gaussian"]["std_scale"] * 32.0
        d_start, d_goal = [], []
        for seed in range(1, 11):
            inpt = _gen(seed, GAUSS, agent_radius=0.0)
            rec = inpt["gen"]
            self.assertEqual(rec["type"], "gaussian")
            self.assertAlmostEqual(rec["std_world"][0], std_world)
            self.assertNotEqual(rec["start_mean_world"], rec["goal_mean_world"])
            for a in inpt["agents"]:
                for k in ("start", "goal"):
                    self.assertTrue(all(0.0 <= float(v) <= 32.0 for v in a[k]))
                d_start.append(math.dist(a["start"], rec["start_mean_world"]))
                d_goal.append(math.dist(a["goal"], rec["goal_mean_world"]))
        self.assertLess(np.mean(d_start), 2.5 * std_world)
        self.assertLess(np.mean(d_goal), 2.5 * std_world)

    def test_shared_mean(self):
        g = {"type": "gaussian", "gaussian": {"std_scale": 0.05, "separate_means": False}}
        rec = _gen(3, g)["gen"]
        self.assertEqual(rec["start_mean_world"], rec["goal_mean_world"])

    def test_respects_occupancy_and_separation(self):
        inpt = _gen(5, GAUSS)
        m, cells = _cells(inpt)
        n_inflate = math.ceil(2.0 * BASE["agent_radius"] / BASE["resolution"])
        occupied = set()
        for row in obstacles_world_to_grid(m, np.asarray(inpt["map"]["obstacles"], dtype=float), BASE["obs_size"]):
            base = tuple(int(v) for v in row)
            for off in product(range(-n_inflate, n_inflate + 1), repeat=2):
                occupied.add((base[0] + off[0], base[1] + off[1]))
        pts = [c for pair in cells for c in pair]
        for c in pts:
            self.assertNotIn(c, occupied)
        for a, b in combinations(pts, 2):
            self.assertGreater(max(abs(a[0] - b[0]), abs(a[1] - b[1])), n_inflate)

    def test_fallback_to_uniform(self):
        g = {"type": "gaussian", "gaussian": {"std_scale": 1e-6, "max_attempts": 3}}
        inpt = _gen(11, g)
        self.assertEqual(len(inpt["agents"]), BASE["nb_agents"])
        fb = inpt["gen"]["fallback"]
        self.assertGreater(fb["n_fallback_start"] + fb["n_fallback_goal"], 0)

    def test_deterministic(self):
        self.assertEqual(_digest(_gen(7, GAUSS)), _digest(_gen(7, GAUSS)))
        self.assertEqual(_gen(7, GAUSS)["gen"], _gen(7, GAUSS)["gen"])


class TestGenConfig(unittest.TestCase):
    def test_normalize_defaults(self):
        self.assertEqual(normalize_gen_config(None)["type"], "uniform")
        cfg = normalize_gen_config({"type": "gaussian"})
        self.assertEqual((cfg["std_scale"], cfg["max_attempts"], cfg["separate_means"]), (0.1, 100, True))
        flat = normalize_gen_config({"type": "gaussian", "std_scale": 0.3, "separate_means": False})
        self.assertEqual((flat["std_scale"], flat["separate_means"]), (0.3, False))
        with self.assertRaises(AssertionError):
            normalize_gen_config({"type": "banana"})

    def test_read_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "gen.yaml"
            p.write_text("type: gaussian\ngaussian:\n  std_scale: 0.2\n")
            cfg = read_gen_config_from_yaml(p)
            self.assertEqual((cfg["type"], cfg["std_scale"], cfg["max_attempts"]), ("gaussian", 0.2, 100))
            p.write_text("")
            self.assertEqual(read_gen_config_from_yaml(p)["type"], "uniform")


if __name__ == "__main__":
    unittest.main()
