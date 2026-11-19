"""Neural-ATTF with agent bodies: headings on the committed waypoints and in the schedules other agents
plan against, the exact simulation backstop, the body-aware collision checker, and end-to-end runs with
rectangles (and a mixed fleet) that must execute without any body overlap."""

import random
import unittest

import numpy as np

from path_planning.agent import Agent
from path_planning.common.world.collision_checking.sweep.agent_shape import Rectangle
from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import (
    NeuralATTF,
    _Waypoint,
    _with_headings,
)
from path_planning.multi_agent_planner.decentralized.neural_attf.simulation import Simulation
from path_planning.utils.checker import check_collision_bodies
from path_planning.utils.util import read_graph_sampler_from_yaml, set_global_seed
from tests.helpers import MAP_2D

RECT = Agent(Rectangle(0.8, 0.3), 1.0)       # bounding radius ~0.43: two bounding balls need 0.85 apart
DISC = Agent.disc(0.3, 1.0)


def _room_map(reach=RECT.reach):
    """4-neighbour grid on 2d.yaml; its lower-left room is free for x in [2.5, 18.5], y in [2.5, 12.5]."""
    set_global_seed(42)
    map_ = read_graph_sampler_from_yaml(MAP_2D, use_discrete_space=True)
    map_.inflate_obstacles(radius=reach + np.sqrt(2) / 2)
    map_.set_roadmap_config(sample_num=0, num_neighbors=4.0, min_edge_length=0.0, max_edge_length=1.1)
    map_.generateRandomNodes(generate_grid_nodes=True)
    map_.generate_roadmap("prm")
    map_.set_collision_sweep()
    return map_


class TestHeadings(unittest.TestCase):
    def test_waypoints_face_their_last_move_through_vias_and_keep_it_while_waiting(self):
        pts = [(0.0, 0.0), _Waypoint((1.0, 1.0), via=[(1.0, 0.0)], via_t=[0.5]), (1.0, 1.0), (0.0, 1.0)]
        out = _with_headings(pts, 45.0)
        self.assertEqual([w.yaw for w in out], [45.0, 90.0, 90.0, 180.0])
        self.assertEqual(out[1].via, ((1.0, 0.0),))     # vias and their times kept
        self.assertEqual(out[1].via_t, (0.5,))

    def test_schedules_of_polygons_start_with_their_heading_discs_without(self):
        obj = NeuralATTF.__new__(NeuralATTF)
        obj.timestep_duration = 1.0
        obj.bodies = {"me": RECT, "idle": RECT, "mover": RECT, "disc": DISC}
        obj.token = {"agents": {
            "me": [_Waypoint((9.0, 9.0))],
            "idle": [_Waypoint((5.0, 5.0), yaw=90.0)],
            "mover": _with_headings([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)], 0.0),
            "disc": [_Waypoint((3.0, 3.0), yaw=90.0)],
        }}
        sched, _ = obj._other_agent_schedules("me")
        self.assertEqual(sched["idle"][0]["yaw"], 90.0)
        self.assertEqual(sched["mover"][0]["yaw"], 90.0)         # at t = 0 it is setting off up (+y)
        self.assertNotIn("yaw", sched["mover"][1])               # the rest follows from the moves
        obj.token["agents"]["mover"] = [_Waypoint((0.0, 0.0), yaw=0.0), _Waypoint((0.0, 0.0)), _Waypoint((0.0, 1.0))]
        sched, _ = obj._other_agent_schedules("me")
        self.assertEqual(sched["mover"][0]["yaw"], 0.0)          # waits first: keeps its heading
        self.assertNotIn("yaw", sched["disc"][0])                # a ball's heading does not matter
        # Planned from tick 1 on: it is at (0, 1) having moved up, so it faces +y (90).
        sched, _ = obj._other_agent_schedules("me", offset_steps=1)
        self.assertEqual(sched["mover"][0]["yaw"], 90.0)

    def test_bodies_must_be_given_for_all_or_none(self):
        map_ = _room_map()
        agents = [{"name": "a", "start": (3.5, 3.5), "agent": RECT}, {"name": "b", "start": (5.5, 3.5)}]
        with self.assertRaises(ValueError):
            NeuralATTF(map_, agents, [(3.5, 3.5), (5.5, 3.5)])


class TestBodyChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_ = _room_map()

    def _two(self, a, b, body=RECT):
        return check_collision_bodies({"a": a, "b": b}, {"a": body, "b": body}, self.map_, verbose=False)

    def test_crossing_rectangles_collide(self):
        a = [{"t": 0.0, "x": 0.0, "y": 0.0}, {"t": 4.0, "x": 4.0, "y": 0.0}]
        b = [{"t": 0.0, "x": 2.0, "y": -2.0}, {"t": 4.0, "x": 2.0, "y": 2.0}]
        self.assertIn(("a", "b"), self._two(a, b))

    def test_side_by_side_rectangles_closer_than_their_bounding_balls_do_not(self):
        # 0.5 apart sideways: the 0.3-wide bodies leave a 0.2 gap, the bounding balls overlap.
        a = [{"t": 0.0, "x": 0.0, "y": 0.0}, {"t": 4.0, "x": 4.0, "y": 0.0}]
        b = [{"t": 0.0, "x": 0.0, "y": 0.5}, {"t": 4.0, "x": 4.0, "y": 0.5}]
        self.assertEqual(self._two(a, b), {})

    def test_a_waiting_rectangle_uses_its_yaw(self):
        mover = [{"t": 0.0, "x": -2.0, "y": 0.5}, {"t": 4.0, "x": 2.0, "y": 0.5}]
        lying = [{"t": 0.0, "x": 0.0, "y": 0.0, "yaw": 0.0}, {"t": 4.0, "x": 0.0, "y": 0.0, "yaw": 0.0}]
        standing = [dict(p, yaw=90.0) for p in lying]               # now 0.4 tall: reaches y = 0.4 + 0.15
        self.assertEqual(self._two(mover, lying), {})
        self.assertIn(("a", "b"), self._two(mover, standing))


class TestExactBackstop(unittest.TestCase):
    def _run(self, mover, resting, yaw):
        map_ = _room_map()
        agents = [{"name": "a", "start": mover[0], "agent": RECT},
                  {"name": "b", "start": resting, "agent": RECT, "yaw": yaw}]
        sim = Simulation(tasks=[], agents=agents, rng=random.Random(0))

        class _Algo:
            graph_map = map_

            def __init__(self):
                self.token = {"agents": {"a": [_Waypoint(q) for q in mover], "b": [_Waypoint(resting)]}}

            def time_forward(self, *a, **k):
                pass

            def get_token(self):
                return self.token

        sim.time_forward(_Algo())
        return sim.actual_paths

    def test_passing_close_to_a_resting_rectangle_is_not_held(self):
        # The bounding balls would meet (0.5 < 0.85), the bodies do not: the old ball test held it.
        paths = self._run([(4.5, 5.0), (5.5, 5.0)], (5.0, 5.5), 0.0)
        self.assertEqual(paths["a"][-1]["x"], 5.5)
        self.assertEqual(paths["a"][-1]["yaw"], 0.0)

    def test_driving_into_a_resting_rectangle_is_held(self):
        paths = self._run([(4.5, 5.0), (5.5, 5.0)], (5.0, 5.5), 90.0)   # standing across the lane
        self.assertEqual(paths["a"][-1]["x"], 4.5)


class TestEndToEnd(unittest.TestCase):
    """Four bodies cross the room on SIPP; the executed paths are checked exactly."""

    STARTS = [(3.5, 3.5), (17.5, 3.5), (3.5, 11.5), (17.5, 11.5)]
    PARKING = STARTS + [(10.5, 2.5), (10.5, 12.5)]
    TASKS = [
        {"task_name": "t0", "start_time": 0, "waypoints": [(4.5, 7.5), (16.5, 7.5)]},
        {"task_name": "t1", "start_time": 0, "waypoints": [(16.5, 8.5), (4.5, 8.5)]},
        {"task_name": "t2", "start_time": 1, "waypoints": [(10.5, 4.5), (10.5, 10.5)]},
        {"task_name": "t3", "start_time": 2, "waypoints": [(11.5, 10.5), (11.5, 4.5)]},
    ]

    def _run(self, bodies, low_level="sipp", delay=0.0, dt=1.0):
        map_ = _room_map(max(b.reach for b in bodies))
        agents = [{"name": f"a{i}", "start": s, "agent": bodies[i % len(bodies)]} for i, s in enumerate(self.STARTS)]
        planner = NeuralATTF(map_, agents, self.PARKING, heuristic_type="manhattan", low_level=low_level,
                             timestep_duration=dt)
        sim = Simulation(tasks=[], agents=agents, delay_probability=delay, rng=random.Random(3),
                         timestep_duration=dt)
        while planner.get_completed_tasks() < len(self.TASKS) and sim.get_time() < 150:
            new = [t for t in self.TASKS if t["start_time"] == sim.get_time()]
            if new:
                planner.update_tasks(sim.get_time(), new)
            sim.time_forward(planner)
        paths = sim.get_actual_paths()
        hits = check_collision_bodies(paths, {a["name"]: a["agent"] for a in agents}, map_, verbose=False)
        return planner.get_completed_tasks(), hits, paths

    def test_rectangles_complete_all_tasks_without_overlap(self):
        done, hits, paths = self._run([RECT])
        self.assertEqual(done, len(self.TASKS))
        self.assertEqual(hits, {})
        self.assertTrue(all("yaw" in p for path in paths.values() for p in path))

    def test_mixed_fleet_with_delays_and_long_ticks(self):
        done, hits, _ = self._run([RECT, DISC], delay=0.1, dt=2.0)
        self.assertEqual(hits, {})
        self.assertGreater(done, 0)

    def test_grid_low_level_with_rectangles(self):
        done, hits, _ = self._run([Agent(Rectangle(0.8, 0.3))], low_level="grid")
        self.assertEqual(done, len(self.TASKS))
        self.assertEqual(hits, {})
