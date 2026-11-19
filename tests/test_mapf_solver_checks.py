"""mapf_solver's body handling and final checks, the exact checker without a map, and solve_mapd."""

import unittest

from path_planning.agent import Agent
from path_planning.common.world.collision_checking.sweep.agent_shape import Rectangle
from path_planning.multi_agent_planner.mapf_solver import (
    check_solution_collision,
    computer_solution_cost,
    config_agent,
    solve_mapd,
    with_bodies,
)
from path_planning.utils.checker import check_collision_bodies, check_solution_full

import tests.test_neural_attf_shapes as shapes
from tests.test_neural_attf_shapes import RECT, _room_map

CROSS = {
    "a": [{"t": 0.0, "x": 0.0, "y": 0.0, "yaw": 0.0}, {"t": 4.0, "x": 4.0, "y": 0.0, "yaw": 0.0}],
    "b": [{"t": 0.0, "x": 2.0, "y": -2.0, "yaw": 90.0}, {"t": 4.0, "x": 2.0, "y": 2.0, "yaw": 90.0}],
}


class TestBodies(unittest.TestCase):
    def test_config_body_and_speed(self):
        cfg = {"agent_radius": 0.4, "agent_velocity": 2.0, "agent_clearance": 0.1}
        self.assertIsNone(config_agent(cfg, "cbs").max_speed)            # unit steps
        self.assertEqual(config_agent(cfg, "sipp").max_speed, 2.0)       # continuous time
        self.assertEqual(config_agent(cfg, "neural_attf").clearance, 0.1)
        self.assertIsNone(config_agent({"agent_radius": 0.0}, "sipp"))   # point particles
        own = Agent(Rectangle(1.0, 0.5))
        agents = with_bodies([{"name": "a"}, {"name": "b", "agent": own}], cfg, "icbs")
        self.assertIs(agents[1]["agent"], own)                           # a dict's own body wins
        self.assertEqual(with_bodies([{"name": "a"}], cfg, "lacam"), [{"name": "a"}])

    def test_cost_reads_positions_not_headings(self):
        _, total = computer_solution_cost({k: [dict(p) for p in v] for k, v in CROSS.items()})
        self.assertAlmostEqual(total["path_length"], 8.0)


class TestFinalCheck(unittest.TestCase):
    def test_bodies_are_checked_exactly_without_a_map(self):
        info = check_solution_collision(CROSS, 0.0, {"success": True}, {"a": RECT, "b": RECT})
        self.assertTrue(info["collision"])
        self.assertFalse(info["success"])
        # Side by side, closer than the bounding balls but clear: fine (the old disc check flagged it).
        apart = {"a": CROSS["a"], "b": [dict(p, y=0.5) for p in CROSS["a"]]}
        self.assertFalse(check_solution_collision(apart, 0.0, {"success": True}, {"a": RECT, "b": RECT})["collision"])
        self.assertTrue(check_solution_collision(apart, RECT.radius, {"success": True})["collision"])

    def test_map_free_and_map_checks_agree_and_bodies_may_be_dicts(self):
        bodies = {"a": RECT.to_dict(), "b": RECT.to_dict()}
        self.assertEqual(check_collision_bodies(CROSS, bodies, verbose=False).keys(),
                         check_collision_bodies(CROSS, bodies, _room_map(), verbose=False).keys())
        self.assertTrue(check_solution_full(CROSS, bodies=bodies, verbose=False)["collisions"])

    def test_a_turn_in_place_counts_as_the_bounding_ball(self):
        # "b" turns at the origin from lying (clear of "a" passing at y = 0.5) to standing.
        a = [{"t": 0.0, "x": -2.0, "y": 0.5}, {"t": 4.0, "x": 2.0, "y": 0.5}]
        still = [{"t": 0.0, "x": 0.0, "y": 0.0, "yaw": 0.0}, {"t": 4.0, "x": 0.0, "y": 0.0, "yaw": 0.0}]
        turning = [{"t": 0.0, "x": 0.0, "y": 0.0, "yaw": 0.0}, {"t": 4.0, "x": 0.0, "y": 0.0, "yaw": 90.0}]
        self.assertEqual(check_collision_bodies({"a": a, "b": still}, {"a": RECT, "b": RECT}, verbose=False), {})
        self.assertTrue(check_collision_bodies({"a": a, "b": turning}, {"a": RECT, "b": RECT}, verbose=False))


class TestLacamPointsOnly(unittest.TestCase):
    def test_bodies_are_refused_and_points_are_validated(self):
        from path_planning.multi_agent_planner.mapf_solver import solve_mapf

        map_ = _room_map(0.0)
        agents = [{"name": "a", "start": (3.5, 3.5), "goal": (8.5, 3.5)}, {"name": "b", "start": (8.5, 3.5), "goal": (3.5, 3.5)}]
        cfg = {"mapf_solver_name": "lacam", "agent_radius": 0.0, "agent_velocity": 0.0, "time_limit": 10,
               "max_iterations": 100000, "heuristic_type": "euclidean"}
        summary = solve_mapf(map_, [dict(a) for a in agents], cfg)
        self.assertTrue(summary["success"])
        self.assertFalse(summary["collision"])
        self.assertNotIn("agents", summary)
        for bad in ({"agent_radius": 0.3}, {"agent": RECT}):
            with self.assertRaises(ValueError):
                solve_mapf(map_, [dict(a) for a in agents], dict(cfg, **bad))
        with self.assertRaises(ValueError):
            solve_mapf(map_, [dict(agents[0], agent=RECT), dict(agents[1])], dict(cfg, mapf_solver_name="lacam_random"))


class TestSolveMapd(unittest.TestCase):
    def test_rectangles_from_the_config(self):
        map_ = _room_map()
        agents = [{"name": f"a{i}", "start": s} for i, s in enumerate(shapes.TestEndToEnd.STARTS)]
        summary = solve_mapd(map_, agents, shapes.TestEndToEnd.TASKS, shapes.TestEndToEnd.PARKING,
                             {"agent": RECT, "heuristic_type": "manhattan", "horizon": 150})
        self.assertTrue(summary["success"])
        self.assertFalse(summary["collision"])
        self.assertEqual(summary["completed_tasks"], len(shapes.TestEndToEnd.TASKS))
        self.assertEqual(summary["agents"]["a0"], RECT.to_dict())
        self.assertGreater(summary["path_length"], 0.0)
        self.assertEqual(summary["mapf_solver_name"], "neural_attf")
