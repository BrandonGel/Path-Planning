"""Direction zones (one-way edges) and edge zones (allowed line angles) on roadmaps."""
import math
import os
import tempfile
import unittest

import numpy as np
from scipy.sparse import csr_array

from path_planning.common.environment.roadmap import RoadmapConfig
from path_planning.common.environment.roadmap.edge_zones import constrain_adjacency
from path_planning.utils.util import read_graph_sampler_from_yaml, set_global_seed
from tests.helpers import MAP_2D

UNIT_SQUARE = [[0, 0], [0, 1], [1, 1], [1, 0]]


def _two_way(nodes, pairs):
    """Symmetric CSR adjacency over ``nodes`` with the undirected ``pairs``."""
    pairs = np.asarray(pairs)
    u = np.concatenate([pairs[:, 0], pairs[:, 1]])
    v = np.concatenate([pairs[:, 1], pairs[:, 0]])
    w = np.linalg.norm(np.asarray(nodes)[u] - np.asarray(nodes)[v], axis=1)
    return csr_array((w, (u, v)), shape=(len(nodes), len(nodes)))


def _edge_set(adj):
    rows = np.repeat(np.arange(adj.shape[0]), np.diff(adj.indptr))
    return set(zip(rows.tolist(), adj.indices.tolist()))


class TestConstrainAdjacency(unittest.TestCase):
    def test_todo_example_keeps_heading_pi_4_only(self):
        # Edge s -> t heads pi/4 (kept); t -> s heads 5*pi/4, outside (0, pi/4) and (pi/2, pi): dropped.
        nodes = np.array([[0.2, 0.2], [0.8, 0.8]])
        zone = {"points": UNIT_SQUARE, "directions": [[0, math.pi / 4], [math.pi / 2, math.pi]]}
        out = constrain_adjacency(nodes, _two_way(nodes, [[0, 1]]), direction_zones=[zone])
        self.assertEqual(_edge_set(out), {(0, 1)})

    def test_heading_in_second_range_is_kept(self):
        nodes = np.array([[0.8, 0.2], [0.2, 0.8]])            # 0 -> 1 heads 3*pi/4, in (pi/2, pi)
        zone = {"points": UNIT_SQUARE, "directions": [[0, math.pi / 4], [math.pi / 2, math.pi]]}
        out = constrain_adjacency(nodes, _two_way(nodes, [[0, 1]]), direction_zones=[zone])
        self.assertEqual(_edge_set(out), {(0, 1)})

    def test_edges_outside_the_zone_are_untouched(self):
        nodes = np.array([[0.2, 0.5], [0.8, 0.5], [5.0, 5.0], [6.0, 5.0]])
        zone = {"points": UNIT_SQUARE, "directions": [[0, 0.1]]}
        out = constrain_adjacency(nodes, _two_way(nodes, [[0, 1], [2, 3]]), direction_zones=[zone])
        self.assertEqual(_edge_set(out), {(0, 1), (2, 3), (3, 2)})

    def test_wrapping_range_through_zero(self):
        nodes = np.array([[0.5, 0.5], [0.9, 0.45]])           # 0 -> 1 heads just below 2*pi
        zone = {"points": UNIT_SQUARE, "directions": [[7 * math.pi / 4, math.pi / 4]]}
        out = constrain_adjacency(nodes, _two_way(nodes, [[0, 1]]), direction_zones=[zone])
        self.assertEqual(_edge_set(out), {(0, 1)})

    def test_edge_zone_keeps_allowed_lines_in_both_directions(self):
        # 3x3 grid with diagonals; only horizontal / vertical lines may be edges in the zone.
        g = np.array([(x, y) for y in range(3) for x in range(3)], dtype=float)
        pairs = [(i, j) for i in range(9) for j in range(i + 1, 9) if np.linalg.norm(g[i] - g[j]) < 1.5]
        zone = {"points": [[-1, -1], [-1, 3], [3, 3], [3, -1]], "directions": [[0, 5], [85, 95], [175, 180]],
                "degrees": True}
        out = _edge_set(constrain_adjacency(g, _two_way(g, pairs), edge_zones=[zone]))
        for a, b in out:
            d = g[b] - g[a]
            self.assertTrue(d[0] == 0 or d[1] == 0, f"diagonal edge {(a, b)} survived")
            self.assertIn((b, a), out)
        self.assertEqual(len(out), 2 * 12)                      # the 12 axis-aligned grid edges

    def test_rules(self):
        nodes = np.array([[0.5, 0.5], [2.0, 0.5]])            # starts inside, ends outside
        for rule, kept in (("intersects", {(0, 1)}), ("within", {(0, 1), (1, 0)}), ("midpoint", {(0, 1), (1, 0)})):
            zone = {"points": UNIT_SQUARE, "directions": [[0, 0.1]], "rule": rule}
            out = constrain_adjacency(nodes, _two_way(nodes, [[0, 1]]), direction_zones=[zone])
            self.assertEqual(_edge_set(out), kept, rule)

    def test_overlapping_zones_intersect(self):
        nodes = np.array([[0.2, 0.5], [0.8, 0.5]])
        east = {"points": UNIT_SQUARE, "directions": [[0, 0.1]]}
        west = {"points": UNIT_SQUARE, "directions": [[math.pi - 0.1, math.pi + 0.1]]}
        out = constrain_adjacency(nodes, _two_way(nodes, [[0, 1]]), direction_zones=[east, west])
        self.assertEqual(_edge_set(out), set())

    def test_bad_zones_raise(self):
        with self.assertRaises(ValueError):
            RoadmapConfig({"direction_zones": [{"points": [[0, 0], [1, 1]], "directions": [[0, 1]]}]})
        with self.assertRaises(ValueError):
            RoadmapConfig({"edge_zones": [{"points": UNIT_SQUARE, "directions": [[0, 4.0]]}]})   # > pi
        with self.assertRaises(ValueError):
            RoadmapConfig({"direction_zones": [{"points": UNIT_SQUARE, "directions": [[0, 1]], "rule": "near"}]})
        with self.assertRaises(TypeError):
            RoadmapConfig({"direction_zones": [{"points": UNIT_SQUARE, "directions": [[0, 1]], "speed": 2}]})


class TestGraphSamplerZones(unittest.TestCase):
    ZONE = {"points": [[4, 4], [4, 8], [12, 8], [12, 4]], "directions": [[0, 10]], "degrees": True}   # eastbound lane

    def _grid_map(self, **config):
        set_global_seed(42)
        m = read_graph_sampler_from_yaml(MAP_2D, use_discrete_space=True)
        m.set_roadmap_config(sample_num=0, num_neighbors=4.0, min_edge_length=0.0, max_edge_length=1.1, **config)
        nodes = m.generateRandomNodes(generate_grid_nodes=True)
        m.generate_roadmap("prm")                              # 4-neighbour PRM over the grid nodes
        return m

    def test_grid_lane_is_one_way(self):
        plain = self._grid_map()
        zoned = self._grid_map(direction_zones=[self.ZONE])
        pos = np.asarray([n.current for n in zoned.nodes])
        e_plain = {tuple(e) for e in np.asarray(plain.edges).tolist()}
        e_zoned = {tuple(e) for e in np.asarray(zoned.edges).tolist()}
        self.assertTrue(e_zoned < e_plain)
        lane = constrain_adjacency(pos, plain.roadmap.adj, direction_zones=[self.ZONE])
        self.assertEqual(e_zoned, _edge_set(lane))
        for a, b in e_zoned:                                   # no edge touching the lane heads anywhere but east
            p, q = pos[a], pos[b]
            touches = max(p[0], q[0]) >= 4 and min(p[0], q[0]) <= 12 and max(p[1], q[1]) >= 4 and min(p[1], q[1]) <= 8
            if touches:
                self.assertGreater(q[0], p[0])
                self.assertEqual(q[1], p[1])
        self.assertTrue(any((b, a) not in e_zoned for a, b in e_zoned))

    def test_zones_round_trip_through_yaml(self):
        cfg = RoadmapConfig({"direction_zones": [self.ZONE], "edge_zones": [dict(self.ZONE, directions=[[0, 180]])]})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "roadmap.yaml")
            cfg.to_yaml(path)
            back = RoadmapConfig.from_yaml(path)
        self.assertEqual(back.direction_zones, cfg.direction_zones)
        self.assertEqual(back.edge_zones, cfg.edge_zones)
