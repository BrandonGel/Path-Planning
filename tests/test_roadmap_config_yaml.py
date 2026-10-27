"""RoadmapConfig YAML loading and its path into GraphSampler / Roadmap."""
import math
import os
import tempfile
import unittest

import yaml

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.roadmap import RoadmapConfig
from path_planning.utils.util import read_graph_sampler_from_yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(REPO, "config", "roadmap.yaml")
MAP_YAML = os.path.join(REPO, "path_planning", "maps", "2d", "2d_double.yaml")


class TestRoadmapConfigYaml(unittest.TestCase):
    def test_template_loads(self):
        cfg = RoadmapConfig.from_yaml(TEMPLATE)
        self.assertEqual(cfg.num_neighbors, 13)
        self.assertEqual(cfg.sample_num, 1000)
        self.assertAlmostEqual(cfg.max_edge_length, 30.0)
        self.assertIsNone(cfg.k_query)

    def test_round_trip_and_inf(self):
        cfg = RoadmapConfig({"num_neighbors": 6})          # max_edge_length = inf
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.yaml")
            cfg.to_yaml(path)
            self.assertEqual(RoadmapConfig.from_yaml(path), cfg)
            self.assertTrue(math.isinf(RoadmapConfig.from_yaml(path).max_edge_length))

    def test_flat_document_and_unknown_key(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.yaml")
            with open(path, "w") as f:
                yaml.safe_dump({"num_neighbors": 4, "max_edge_length": 1.1}, f)
            self.assertEqual(RoadmapConfig.from_yaml(path).num_neighbors, 4)
            with open(path, "w") as f:
                yaml.safe_dump({"roadmap_config": {"num_neighbour": 4}}, f)
            with self.assertRaises(TypeError):
                RoadmapConfig.from_yaml(path)

    def test_sampler_accepts_path(self):
        m = GraphSampler(bounds=[[0, 10.0], [0, 10.0]], resolution=1.0, start=[], goal=[],
                         roadmap_config=TEMPLATE)
        self.assertEqual(m.roadmap_config, RoadmapConfig.from_yaml(TEMPLATE))
        self.assertEqual(m.roadmap_config.sample_num, 1000)   # num_total_nodes is the live node count now
        m.set_roadmap_config(TEMPLATE)
        m.set_roadmap_config(sample_num=5)
        self.assertEqual(m.roadmap_config.sample_num, 5)
        self.assertEqual(m.roadmap_config.num_neighbors, 13)

    def test_read_graph_sampler_from_yaml(self):
        m = read_graph_sampler_from_yaml(MAP_YAML, use_discrete_space=False, roadmap_config=TEMPLATE)
        self.assertEqual(m.roadmap_config, RoadmapConfig.from_yaml(TEMPLATE))
        # a roadmap_config block inside the map yaml is picked up by itself
        with open(MAP_YAML) as f:
            data = yaml.safe_load(f)
        data["roadmap_config"] = {"sample_num": 40, "num_neighbors": 5, "max_edge_length": 12.0}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "map.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(data, f)
            m = read_graph_sampler_from_yaml(path, use_discrete_space=False)
        self.assertEqual(m.roadmap_config.sample_num, 40)
        m.start, m.goal = [(2.5, 2.5)], [(80.5, 40.5)]
        m.generateRandomNodes()
        m.generate_roadmap("prm")
        self.assertIs(m.roadmap.config, m.roadmap_config)        # same object reaches the Roadmap
        self.assertEqual(len(m.nodes), 42)
        self.assertLessEqual(max(len(r) for r in m.road_map), 5)
        self.assertLessEqual(max((max(w) for w in m.road_map_edge_weights if w), default=0), 12.0)


if __name__ == "__main__":
    unittest.main()
