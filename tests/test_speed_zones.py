"""Speed zones (per-edge speed limits) and finite turning speeds in SIPP."""

from __future__ import annotations

import math
import os
import pickle
import sys
import tempfile
import unittest

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.map.speed_zones import (
    SpeedZone,
    parse_speed_zones,
    segment_speed_limit,
    zones_to_specs,
)
from path_planning.multi_agent_planner.mapf_solver import solve_mapf
from path_planning.utils.agent_shape import AlignedShape, heading_key, heading_of, schedule_headings, turn_angle, turn_time
from path_planning.utils.checker import check_collision, check_velocity_anomaly
from path_planning.utils.util import (
    agent_headings_dict,
    agent_turning_speeds_dict,
    max_agent_radius,
    normalize_agent_specs,
    read_graph_sampler_from_yaml,
    set_global_seed,
)

HALF_PI = heading_key(math.pi / 2)


# ----------------------------------------------------------------------------- unit


class TestSpeedZoneGeometry(unittest.TestCase):
    def test_segment_intersection(self):
        z = SpeedZone(3.0, 6.0, 0.0, 13.0, 0.5)
        self.assertTrue(z.intersects_segment((4, 4), (5, 5)))      # inside
        self.assertTrue(z.intersects_segment((0, 1), (10, 1)))     # crossing
        self.assertTrue(z.intersects_segment((0, 1), (3.0, 1)))    # touching the edge
        self.assertTrue(z.intersects_segment((0, 0), (3.0, 0.0)))  # touching the corner
        self.assertTrue(z.intersects_segment((4, 4), (4, 4)))      # point inside
        self.assertFalse(z.intersects_segment((0, 1), (2.9, 1)))
        self.assertFalse(z.intersects_segment((7, 0), (7, 20)))
        self.assertFalse(z.intersects_segment((0, 14), (10, 14)))
        self.assertFalse(z.intersects_segment((7, 7), (7, 7)))

    def test_lowest_limit_wins(self):
        zones = parse_speed_zones([
            {"x": [0, 10], "y": [0, 10], "speed_limit": 1.0},
            {"bounds": [[4, 6], [4, 6]], "limit": 0.25, "name": "core"},
        ])
        self.assertEqual(segment_speed_limit(zones, (1, 1), (2, 1)), 1.0)
        self.assertEqual(segment_speed_limit(zones, (1, 5), (9, 5)), 0.25)
        self.assertIsNone(segment_speed_limit(zones, (11, 11), (12, 12)))
        self.assertEqual(zones[1].name, "core")
        self.assertEqual(parse_speed_zones(zones_to_specs(zones)), zones)
        with self.assertRaises(ValueError):
            SpeedZone(0, 0, 0, 1, 1.0)
        with self.assertRaises(ValueError):
            SpeedZone(0, 1, 0, 1, 0.0)

    def test_graph_sampler_edge_speed(self):
        m = GraphSampler(bounds=[[0.0, 13.0], [0.0, 13.0]], resolution=1.0, start=[], goal=[], use_discrete_space=True)
        self.assertEqual(m.edge_speed((0.5, 0.5), (1.5, 0.5), 2.0), 2.0)
        m.set_speed_zones([{"x": [3.0, 6.0], "y": [0.0, 13.0], "speed_limit": 0.5}])
        v0 = m.speed_zones_version
        self.assertEqual(m.edge_speed((2.5, 6.5), (3.5, 6.5), 2.0), 0.5)
        self.assertEqual(m.edge_speed((2.5, 6.5), (3.5, 6.5), 0.25), 0.25)  # slower agent keeps its own speed
        self.assertEqual(m.edge_speed((7.5, 6.5), (8.5, 6.5), 2.0), 2.0)
        self.assertEqual(m.edge_speed((2.5, 6.5), (3.5, 6.5), 0.0), 0.0)   # unit-time mode untouched
        self.assertEqual(m.edge_speed_limit((7.5, 6.5), (8.5, 6.5)), float("inf"))
        m.set_speed_zones([])
        self.assertGreater(m.speed_zones_version, v0)
        self.assertEqual(m.edge_speed((2.5, 6.5), (3.5, 6.5), 2.0), 2.0)

    def test_yaml_and_pickle_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = os.path.join(tmp, "map.yaml")
            with open(yaml_path, "w") as f:
                f.write(
                    "map:\n  dimensions: [6, 6]\n  obstacles: []\n"
                    "  speed_zones:\n    - {x: [1.0, 3.0], y: [0.0, 6.0], speed_limit: 0.5, name: slow}\n"
                    "agents: []\n"
                )
            m = read_graph_sampler_from_yaml(yaml_path, use_discrete_space=True)
            self.assertEqual(len(m.speed_zones), 1)
            self.assertEqual(m.speed_zones[0].speed_limit, 0.5)
            m.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
            m.set_start([[0.5, 0.5]])
            m.set_goal([[5.5, 5.5]])
            m.generate_roadmap(m.generateRandomNodes(generate_grid_nodes=True))
            pkl = os.path.join(tmp, "graph.pkl")
            m.save_graph_sampler(pkl)
            with open(pkl, "rb") as f:
                self.assertEqual(pickle.load(f)["speed_zones"], zones_to_specs(m.speed_zones))
            m2 = read_graph_sampler_from_yaml(yaml_path, use_discrete_space=True, graph_file=pkl)
            self.assertEqual(m2.speed_zones, m.speed_zones)
            self.assertEqual(m2.edge_speed((0.5, 0.5), (1.5, 0.5), 2.0), 0.5)


class TestTurnHelpers(unittest.TestCase):
    def test_turn_angle_shortest_arc(self):
        self.assertAlmostEqual(turn_angle(0.0, 3 * math.pi / 2), -math.pi / 2)
        self.assertAlmostEqual(turn_angle(3 * math.pi / 2, 0.0), math.pi / 2)
        self.assertAlmostEqual(turn_angle(0.0, math.pi), math.pi)
        self.assertAlmostEqual(turn_angle(0.3, 0.3), 0.0)
        self.assertAlmostEqual(turn_time(0.0, math.pi / 2, math.pi), 0.5)
        self.assertEqual(turn_time(0.0, math.pi / 2, None), 0.0)
        self.assertEqual(turn_time(0.0, math.pi / 2, float("inf")), 0.0)

    def test_agent_spec_parsing(self):
        agents = normalize_agent_specs([
            {"name": "a", "start": [0, 0], "goal": [1, 1], "turning_speed": 2, "heading_deg": 90},
            {"name": "b", "start": [0, 0], "goal": [1, 1], "heading": -math.pi / 2},
            {"name": "c", "start": [0, 0], "goal": [1, 1], "turning_speed": None},
        ])
        self.assertEqual(agents[0]["turning_speed"], 2.0)
        self.assertEqual(agents[0]["heading"], HALF_PI)
        self.assertNotIn("heading_deg", agents[0])
        self.assertEqual(agents[1]["heading"], heading_key(-math.pi / 2))
        self.assertNotIn("turning_speed", agents[2])
        self.assertEqual(agent_turning_speeds_dict(agents), {"a": 2.0})
        self.assertEqual(agent_turning_speeds_dict(agents, 1.5), {"a": 2.0, "b": 1.5, "c": 1.5})
        self.assertEqual(agent_headings_dict(agents), {"a": HALF_PI, "b": heading_key(-math.pi / 2), "c": 0.0})
        with self.assertRaises(ValueError):
            normalize_agent_specs([{"name": "x", "start": [0, 0], "goal": [1, 1], "turning_speed": -1}])

    def test_velocity_check_with_zones(self):
        zones = parse_speed_zones([{"x": [3, 6], "y": [0, 13], "speed_limit": 0.5}])
        sched = {"a": [{"t": 0.0, "x": 0.5, "y": 6.5}, {"t": 1.0, "x": 2.5, "y": 6.5},
                       {"t": 3.0, "x": 3.5, "y": 6.5}, {"t": 3.5, "x": 4.5, "y": 6.5}]}
        # last segment runs at 2.0 inside the zone -> anomaly; without it, fine
        self.assertFalse(check_velocity_anomaly(sched, velocity=2.0, verbose=False, speed_zones=zones))
        self.assertTrue(check_velocity_anomaly({"a": sched["a"][:3]}, velocity=2.0, verbose=False, speed_zones=zones))
        self.assertFalse(check_velocity_anomaly({"a": sched["a"][:3]}, velocity=2.0, verbose=False))


# ----------------------------------------------------------------------------- SIPP


def _empty_map(backend, agents, default_radius=0.3, zones=None):
    set_global_seed(1)
    m = GraphSampler(bounds=[[0.0, 13.0], [0.0, 13.0]], resolution=1.0, start=[], goal=[],
                     use_discrete_space=True, sweep_backend=backend)
    m.inflate_obstacles(radius=max_agent_radius(agents, default_radius) + np.sqrt(2) / 2)
    m.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
    m.set_start([a["start"] for a in agents])
    m.set_goal([a["goal"] for a in agents])
    m.generate_roadmap(m.generateRandomNodes(generate_grid_nodes=True))
    m.set_constraint_sweep()
    if zones:
        m.set_speed_zones(zones)
    return m


def _solve(m, agents, **cfg):
    config = {"mapf_solver_name": "sipp", "time_limit": 60, "max_iterations": 50,
              "agent_radius": 0.3, "agent_velocity": 1.0, "heuristic_type": "euclidean"}
    config.update(cfg)
    return solve_mapf(m, agents, config)


def _moves(path):
    return [(a, b) for a, b in zip(path, path[1:]) if (a["x"], a["y"]) != (b["x"], b["y"])]


class TestSippSpeedZones(unittest.TestCase):
    ZONES = [{"x": [3.0, 6.0], "y": [0.0, 13.0], "speed_limit": 0.5, "name": "slow"}]

    def test_edge_times_follow_the_limit(self):
        agents = normalize_agent_specs([
            {"name": "a", "start": [0.5, 6.5], "goal": [12.5, 6.5], "velocity": 2.0, "radius": 0.3},
        ])
        for backend in ("cgal", "shapely"):
            m = _empty_map(backend, agents, zones=self.ZONES)
            s = _solve(m, agents)
            self.assertTrue(s["success"], s.get("failure_reason"))
            path = s["schedule"]["a"]
            expect = 0.0
            for a, b in _moves(path):
                v = min(2.0, m.edge_speed_limit((a["x"], a["y"]), (b["x"], b["y"])))
                self.assertAlmostEqual(b["t"] - a["t"], math.dist((a["x"], a["y"]), (b["x"], b["y"])) / v, places=9)
                expect += math.dist((a["x"], a["y"]), (b["x"], b["y"])) / v
            self.assertAlmostEqual(path[-1]["t"], expect, places=9)
            self.assertGreater(path[-1]["t"], 12.0 / 2.0 + 1e-6)
            self.assertTrue(check_velocity_anomaly(s["schedule"], velocity=s["agent_velocities"], verbose=False,
                                                   speed_zones=m.speed_zones))
            self.assertEqual(s["speed_zones"], zones_to_specs(m.speed_zones))
            # without the zone the straight run takes 6 s
            s0 = _solve(_empty_map(backend, agents), agents)
            self.assertAlmostEqual(s0["schedule"]["a"][-1]["t"], 6.0, places=9)

    def test_slowed_leader_is_swept_at_its_real_speed(self):
        """A follower on the same lane must respect the leader's slowed-down
        traversal of the zone (regression: committed plans used to be swept at
        the nominal speed)."""
        agents = normalize_agent_specs([
            {"name": "lead", "start": [1.5, 6.5], "goal": [12.5, 6.5], "velocity": 2.0, "radius": 0.45},
            {"name": "tail", "start": [0.5, 6.5], "goal": [11.5, 6.5], "velocity": 2.0, "radius": 0.45},
        ])
        for backend in ("cgal", "shapely"):
            m = _empty_map(backend, agents, zones=self.ZONES)
            s = _solve(m, agents)
            self.assertTrue(s["success"], s.get("failure_reason"))
            self.assertFalse(s["collision"])
            self.assertEqual(check_collision(s["schedule"], s["agent_footprints"], verbose=False), {})


class TestSippTurning(unittest.TestCase):
    def test_disc_turns_in_place(self):
        agents = normalize_agent_specs([
            {"name": "d", "start": [1.5, 1.5], "goal": [1.5, 11.5], "velocity": 1.0, "radius": 0.3,
             "turning_speed": math.pi, "heading_deg": 0.0},
        ])
        for backend in ("cgal", "shapely"):
            s = _solve(_empty_map(backend, agents), agents)
            self.assertTrue(s["success"], s.get("failure_reason"))
            path = s["schedule"]["d"]
            # faces +x, must rotate 90 deg at pi rad/s before the first (northbound) move
            self.assertEqual(path[0]["heading"], 0.0)
            self.assertEqual(path[0]["turn_to"], HALF_PI)
            self.assertAlmostEqual(path[1]["t"], 0.5, places=9)
            self.assertEqual(path[1]["heading"], HALF_PI)
            self.assertAlmostEqual(path[-1]["t"], 10.5, places=9)
            for a, b in _moves(path):
                self.assertEqual(a["heading"], heading_of((a["x"], a["y"]), (b["x"], b["y"])))
            self.assertEqual(schedule_headings(path), [p["heading"] for p in path])
            self.assertTrue(check_velocity_anomaly(s["schedule"], velocity=s["agent_velocities"], verbose=False))

    def test_l_shaped_path_pays_one_turn(self):
        agents = normalize_agent_specs([
            {"name": "d", "start": [1.5, 1.5], "goal": [5.5, 5.5], "velocity": 1.0, "radius": 0.3,
             "turning_speed": math.pi / 2, "heading_deg": 0.0},
        ])
        s = _solve(_empty_map("shapely", agents), agents)
        self.assertTrue(s["success"], s.get("failure_reason"))
        path = s["schedule"]["d"]
        # 8 unit moves + exactly one 90 deg turn (1 s): 4-connected grid, any monotone
        # staircase would need more turns
        turns = [p for p in path if p.get("turn_to") is not None]
        self.assertEqual(len(turns), 1)
        self.assertAlmostEqual(path[-1]["t"], 8.0 + 1.0, places=9)

    def test_instantaneous_agents_unchanged(self):
        agents = normalize_agent_specs([
            {"name": "d", "start": [1.5, 1.5], "goal": [5.5, 5.5], "velocity": 1.0, "radius": 0.3},
        ])
        s = _solve(_empty_map("shapely", agents), agents)
        path = s["schedule"]["d"]
        self.assertTrue(all("heading" not in p and "turn_to" not in p for p in path))
        self.assertAlmostEqual(path[-1]["t"], 8.0, places=9)

    def test_aligned_rectangle_turning_with_non_incident_start_heading(self):
        agents = normalize_agent_specs([
            {"name": "rect", "start": [1.5, 6.5], "goal": [11.5, 6.5], "velocity": 2.0,
             "turning_speed": math.pi / 2, "heading_deg": 45.0,
             "shape": {"type": "rectangle", "width": 1.6, "height": 0.6}},
            {"name": "cross", "start": [6.5, 0.5], "goal": [6.5, 12.5], "velocity": 1.0,
             "turning_speed": math.pi, "heading_deg": 90.0,
             "shape": {"type": "regular_polygon", "n_sides": 3, "circumradius": 0.6}},
            {"name": "disc", "start": [12.5, 2.5], "goal": [0.5, 10.5], "velocity": 1.0, "radius": 0.3},
        ])
        for backend in ("cgal", "shapely"):
            s = _solve(_empty_map(backend, agents), agents)
            self.assertTrue(s["success"], s.get("failure_reason"))
            self.assertFalse(s["collision"])
            self.assertEqual(check_collision(s["schedule"], s["agent_footprints"], verbose=False), {})
            rect = s["schedule"]["rect"]
            self.assertIsInstance(s["agent_footprints"]["rect"], AlignedShape)
            self.assertEqual(rect[0]["heading"], heading_key(math.pi / 4))
            self.assertEqual(rect[0]["turn_to"], 0.0)
            self.assertAlmostEqual(rect[1]["t"], (math.pi / 4) / (math.pi / 2), places=9)
            for name, path in s["schedule"].items():
                if "heading" in path[0]:
                    self.assertEqual(schedule_headings(path), [p["heading"] for p in path])

    def test_rotation_reserves_the_envelope(self):
        """A long rectangle rotating next to a lane keeps a disc agent out of the
        lane while it turns (its bounding disc reaches into the lane; its aligned
        footprint before / after the turn does not)."""
        rect = {"name": "rect", "start": [5.5, 5.5], "goal": [5.5, 11.5], "velocity": 1.0,
                "turning_speed": math.pi / 8, "heading_deg": 0.0,
                "shape": {"type": "rectangle", "width": 2.4, "height": 0.6}}
        runner = {"name": "run", "start": [6.5, 0.5], "goal": [6.5, 12.5], "velocity": 1.0, "radius": 0.3}
        for backend in ("cgal", "shapely"):
            both = normalize_agent_specs([rect, runner])
            s = _solve(_empty_map(backend, both), both)
            self.assertTrue(s["success"], s.get("failure_reason"))
            self.assertEqual(check_collision(s["schedule"], s["agent_footprints"], verbose=False), {})
            # the rectangle turns for 4 s at its start; the runner alone needs 12 s
            self.assertAlmostEqual(s["schedule"]["rect"][1]["t"], 4.0, places=9)
            alone = normalize_agent_specs([runner])
            s_alone = _solve(_empty_map(backend, alone), alone)
            self.assertAlmostEqual(s_alone["schedule"]["run"][-1]["t"], 12.0, places=9)
            self.assertGreater(s["schedule"]["run"][-1]["t"], 12.0 + 1e-6)

    def test_ccbs_rejects(self):
        from path_planning.multi_agent_planner.centralized.ccbs.graph_generation import Environment as CCBSEnv

        agents = normalize_agent_specs([
            {"name": "d", "start": [1.5, 1.5], "goal": [5.5, 5.5], "velocity": 1.0, "radius": 0.3, "turning_speed": 1.0},
        ])
        m = _empty_map("shapely", agents)
        with self.assertRaises(NotImplementedError):
            CCBSEnv(m, {}, agents, radius=0.3, velocity=1.0)
        plain = normalize_agent_specs([{"name": "d", "start": [1.5, 1.5], "goal": [5.5, 5.5], "velocity": 1.0, "radius": 0.3}])
        m = _empty_map("shapely", plain, zones=[{"x": [0, 2], "y": [0, 2], "speed_limit": 0.5}])
        with self.assertRaises(NotImplementedError):
            CCBSEnv(m, {}, plain, radius=0.3, velocity=1.0)


if __name__ == "__main__":
    unittest.main()
