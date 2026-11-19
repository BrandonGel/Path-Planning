"""Agent types (config/agents.yaml) and the task generator (config/tasks.yaml): the fleet, ready-made
assignments, the four finish modes, the release frequency, type-limited tasks, and runs through
``solve_mapf`` / ``solve_mapd``."""
import math
import os
import random
import unittest

from path_planning.agent import Agent
from path_planning.agent.agent_types import load_agent_types
from path_planning.common.world.collision_checking.sweep.agent_shape import (CustomPolygon, Rectangle, RegularPolygon,
                                                                             Triangle)
from path_planning.multi_agent_planner.mapf_solver import solve_mapd, solve_mapf
from path_planning.tasks import TaskGenerator
from path_planning.utils.checker import check_collision_bodies
from tests.helpers import REPO
from tests.test_cbs_shapes import grid_map

TYPES = {"types": {"amr": {"shape": "disc", "size": 0.3, "max_speed": 1.0},
                   "cart": {"shape": "rectangle", "size": [0.7, 0.4], "max_speed": 0.5, "turning_speed": 90,
                            "clearance": 0.02}}}
POOL = {"a": [1.5, 1.5], "b": [[2.5, 2.5], [3.5, 3.5]], "c": {"goal": [4.5, 4.5]}}


def stream(config, steps=30, **more):
    """``(generator, [(release time, task name)])`` over ``steps`` time steps."""
    generator = TaskGenerator({"agents": 2, "tasks": POOL, **config}, **more)
    return generator, [(t, task["task_name"]) for t in range(steps) for task in generator.release(t)]


class TestAgentTypes(unittest.TestCase):
    def test_every_shape(self):
        types = load_agent_types({"types": {
            "disc": {"shape": "disc", "size": 0.35, "max_speed": 1.5},
            "rect": {"shape": "rectangle", "size": [1.6, 0.8], "turning_speed": 90, "clearance": 0.05},
            "tri": {"shape": "triangle", "size": [1, 1, 1]},
            "hex": {"shape": "regular_polygon", "size": {"sides": 6, "side": 0.4}},
            "poly": {"shape": "polygon", "size": [[0.9, 0], [-0.6, 0.5], [-0.6, -0.5]], "max_speed": 0.8}}})
        self.assertEqual(types["disc"], Agent(0.35, max_speed=1.5, type="disc"))
        self.assertTrue(types["disc"].is_ball)
        for name, kind in (("rect", Rectangle), ("tri", Triangle), ("hex", RegularPolygon), ("poly", CustomPolygon)):
            self.assertIsInstance(types[name].shape, kind)
            self.assertEqual(types[name].type, name)
        self.assertAlmostEqual(types["rect"].max_rotation, math.pi / 2)        # degrees in the file
        self.assertEqual(types["rect"].clearance, 0.05)
        self.assertIsNone(types["tri"].max_speed)                               # one time unit per move
        self.assertIsNone(types["tri"].max_rotation)                            # turns instantly

    def test_bad_entries(self):
        for spec in ({"shape": "blob", "size": 1}, {"shape": "disc", "size": -1}, {"shape": "disc"},
                     {"shape": "rectangle", "size": 1.0}, {"shape": "disc", "size": 1, "max_speed": 0},
                     {"shape": "disc", "size": 1, "speed": 2}, {"shape": "regular_polygon", "size": [6, 0.4]}):
            with self.assertRaises(ValueError, msg=spec):
                load_agent_types({"types": {"x": spec}})
        with self.assertRaises(ValueError):
            load_agent_types({"types": {}})

    def test_the_config_files_load(self):
        types = load_agent_types(os.path.join(REPO, "config", "agents.yaml"))
        for tasks in ("tasks.yaml", "tasks_assigned.yaml"):
            generator = TaskGenerator.from_yaml(os.path.join(REPO, "config", tasks), os.path.join(REPO, "config", "agents.yaml"))
            self.assertTrue({agent["type"] for agent in generator.fleet} <= set(types))


class TestFleet(unittest.TestCase):
    def setUp(self):
        self.types = load_agent_types(TYPES)
        self.map = grid_map()

    def test_counts_per_type(self):
        generator = TaskGenerator({"agents": {"amr": 3, "cart": 1}, "tasks": POOL}, self.types, random.Random(1))
        agents = generator.agents(self.map)
        self.assertEqual([a["name"] for a in agents], ["agent0", "agent1", "agent2", "agent3"])
        self.assertEqual([a["type"] for a in agents], ["amr", "amr", "amr", "cart"])
        self.assertTrue(all(a["agent"] is self.types[a["type"]] for a in agents))
        # Drawn starts: on free roadmap nodes, apart by the bodies' reach, and off every goal.
        nodes = {tuple(n.current) for n in self.map.nodes}
        goals = generator.goal_points()
        for i, a in enumerate(agents):
            self.assertIn(a["start"], nodes)
            self.assertNotIn(a["start"], goals)
            for b in agents[i + 1:]:
                self.assertGreater(math.dist(a["start"], b["start"]), a["agent"].reach + b["agent"].reach)
        self.assertEqual(generator.agents(self.map), agents)                    # the same fleet when asked again
        self.assertAlmostEqual(generator.reach(), self.types["cart"].reach)

    def test_point_particles_without_agent_types(self):
        agents = TaskGenerator({"agents": 3, "tasks": POOL}).agents(self.map)
        self.assertEqual(len(agents), 3)
        self.assertTrue(all("agent" not in a and a["type"] is None for a in agents))

    def test_type_names_have_to_match(self):
        for config, types in (({"agents": {"tug": 1}}, self.types),                                   # unknown type
                              ({"agents": {"amr": 1}}, None),                                         # a type, no agents.yaml
                              ({"agents": 2}, self.types),                                            # no count per type
                              ({"agents": [{"goal": [1.5, 1.5]}]}, self.types),                       # an agent without a type
                              ({"agents": {"amr": 1}, "tasks": {"t": {"goal": [1.5, 1.5], "types": ["tug"]}}}, self.types),
                              ({"agents": {"amr": 1}, "tasks": {"t": {"goal": [1.5, 1.5], "types": ["cart"]}}}, self.types),
                              ({"agents": 1, "tasks": {"t": {"goal": [1.5, 1.5], "types": ["amr"]}}}, None)):
            with self.assertRaises(ValueError, msg=config):
                TaskGenerator(config, types)

    def test_assignments_for_a_one_shot_solver(self):
        config = {"agents": [{"type": "amr", "start": [1.5, 1.5], "goal": [6.5, 1.5], "yaw": 90},
                             {"type": "cart", "name": "tug", "goals": [[1.5, 6.5]]}]}
        agents = TaskGenerator(config, self.types).mapf_agents(self.map)
        self.assertEqual([a["name"] for a in agents], ["agent0", "tug"])
        self.assertEqual((agents[0]["start"], agents[0]["goal"], agents[0]["yaw"]), ((1.5, 1.5), (6.5, 1.5), 90))
        self.assertEqual(agents[1]["goal"], (1.5, 6.5))
        self.assertIn("start", agents[1])                                       # drawn
        # A sequence of goals, a pool of tasks, another mode or a frequency need the lifelong planner.
        for bad in ({"agents": [{"type": "amr", "goals": [[1.5, 1.5], [2.5, 2.5]]}]},
                    {"agents": [{"type": "amr"}]},
                    {"agents": [{"type": "amr", "goal": [1.5, 1.5]}], "tasks": POOL},
                    {"agents": [{"type": "amr", "goal": [1.5, 1.5]}], "mode": "repeated", "count": 2},
                    {"agents": [{"type": "amr", "goal": [1.5, 1.5]}], "frequency": 0.5}):
            with self.assertRaises(ValueError, msg=bad):
                TaskGenerator(bad, self.types).mapf_agents(self.map)

    def test_a_point_off_the_map_or_in_an_obstacle_is_rejected(self):
        for point in ([20.5, 1.5], [-1.0, 1.5]):                                # the map is 9 x 9
            with self.assertRaises(ValueError, msg=point):
                TaskGenerator({"agents": [{"start": [1.5, 1.5], "goal": point}]}).agents(self.map)
        blocked = TaskGenerator.from_yaml(os.path.join(REPO, "config", "tasks_assigned.yaml"),
                                          os.path.join(REPO, "config", "agents.yaml"))
        blocked.fleet[0]["start"] = (20.5, 10.5)                                # a wall of 2d.yaml
        from path_planning.utils.util import read_graph_sampler_from_yaml
        from tests.helpers import MAP_2D
        with self.assertRaises(ValueError):
            blocked.agents(read_graph_sampler_from_yaml(MAP_2D, use_discrete_space=True))


class TestStream(unittest.TestCase):
    def test_singular(self):
        generator, released = stream({})
        self.assertEqual(released, [(0, "a"), (0, "b"), (0, "c")])
        self.assertEqual(generator.target, 3)
        self.assertEqual([generator.finished(n) for n in (2, 3)], [False, True])
        self.assertEqual(generator.created[1]["waypoints"], [(2.5, 2.5), (3.5, 3.5)])

    def test_repeated(self):
        generator, released = stream({"mode": "repeated", "count": 2})
        self.assertEqual([name for _, name in released], ["a#1", "b#1", "c#1", "a#2", "b#2", "c#2"])
        self.assertTrue(all(t == 0 for t, _ in released))
        self.assertEqual(generator.target, 6)

    def test_achieved(self):
        generator, released = stream({"mode": "achieved", "count": 4})
        self.assertEqual(released, [(0, "a#1"), (0, "b#1"), (0, "c#1"), (0, "a#2")])
        self.assertTrue(generator.finished(4))
        # With a frequency tasks keep coming until enough are done (a stuck task does not end the run).
        generator, released = stream({"mode": "achieved", "count": 2, "frequency": 0.25}, steps=13)
        self.assertEqual(released, [(0, "a#1"), (4, "b#1"), (8, "c#1"), (12, "a#2")])
        self.assertEqual(generator.target, 2)

    def test_endless(self):
        generator, released = stream({"mode": "endless", "frequency": 0.1}, steps=45)
        self.assertEqual(released, [(0, "a#1"), (10, "b#1"), (20, "c#1"), (30, "a#2"), (40, "b#2")])
        self.assertIsNone(generator.target)
        self.assertFalse(generator.finished(10 ** 6))

    def test_frequency_and_per_release(self):
        _, released = stream({"mode": "repeated", "count": 2, "frequency": 0.5, "per_release": 2})
        self.assertEqual(released, [(0, "a#1"), (0, "b#1"), (2, "c#1"), (2, "a#2"), (4, "b#2"), (4, "c#2")])

    def test_random_order_is_seeded_and_covers_the_pool(self):
        runs = [stream({"mode": "repeated", "count": 3, "order": "random"}, rng=random.Random(5))[1] for _ in range(2)]
        self.assertEqual(runs[0], runs[1])
        names = [name for _, name in runs[0]]
        self.assertEqual(sorted(names[:3]), ["a#1", "b#1", "c#1"])
        self.assertNotEqual(names, sorted(names, key=lambda n: (n[-1], n)))      # (not the sequential order)

    def test_bad_settings(self):
        for config in ({"mode": "sometimes"}, {"mode": "repeated"}, {"mode": "achieved", "count": 0},
                       {"mode": "endless"}, {"count": 2}, {"frequency": 0}, {"order": "sorted"}, {"per_release": 0},
                       {"speed": 1}):
            with self.assertRaises(ValueError, msg=config):
                TaskGenerator({"agents": 2, "tasks": POOL, **config})
        with self.assertRaises(ValueError):
            TaskGenerator({"agents": 2, "mode": "endless", "frequency": 1})      # nothing to create

    def test_an_agents_own_goals_are_a_task_only_it_takes(self):
        generator = TaskGenerator({"agents": [{"name": "x", "goals": [[1.5, 1.5], [2.5, 2.5]]}, {"name": "y"}]})
        self.assertEqual(generator.release(0), [{"task_name": "x_task", "start_time": 0, "agent": "x",
                                                 "waypoints": [(1.5, 1.5), (2.5, 2.5)]}])

    def test_a_list_of_tasks(self):
        tasks = [{"task_name": "late", "start_time": 5, "waypoints": [(1.5, 1.5)]},
                 {"task_name": "early", "start_time": 1, "waypoints": [(2.5, 2.5)]}]
        generator = TaskGenerator.from_list(tasks)
        self.assertEqual([[task["task_name"] for task in generator.release(t)] for t in range(7)],
                         [[], ["early"], [], [], [], ["late"], []])
        self.assertTrue(generator.finished(2))


class TestRuns(unittest.TestCase):
    PARKING = [(0.5, 8.5), (4.5, 8.5), (8.5, 8.5)]

    def setUp(self):
        self.types = load_agent_types(TYPES)
        self.map = grid_map()
        self.map.set_collision_sweep()

    def check(self, summary, agents):
        self.assertTrue(summary["success"])
        self.assertFalse(summary["collision"])
        self.assertEqual(check_collision_bodies(summary["schedule"], {a["name"]: a["agent"] for a in agents},
                                                self.map, verbose=False), {})

    def test_sipp_plans_a_mixed_fleet(self):
        config = {"agents": [{"type": "amr", "start": [0.5, 4.5], "goal": [8.5, 4.5]},
                             {"type": "cart", "start": [8.5, 4.5], "goal": [0.5, 4.5]},
                             {"type": "amr", "start": [4.5, 0.5], "goal": [4.5, 8.5]}]}
        agents = TaskGenerator(config, self.types).mapf_agents(self.map)
        summary = solve_mapf(self.map, agents, {"mapf_solver_name": "sipp", "time_limit": 30, "agent_radius": 0.0,
                                                "agent_velocity": 0.0, "max_iterations": 10000, "heuristic_type": "euclidean"})
        self.check(summary, agents)
        self.assertEqual({name: body["type"] for name, body in summary["agents"].items()},
                         {"agent0": "amr", "agent1": "cart", "agent2": "amr"})
        # The cart moves at half the speed of the robots.
        self.assertGreaterEqual(summary["schedule"]["agent1"][-1]["t"], 16.0)

    def lifelong(self, config, **options):
        generator = TaskGenerator({"agents": {"amr": 2, "cart": 1}, "tasks": {
            "haul": {"goals": [[1.5, 1.5], [7.5, 1.5]], "types": ["cart"]},
            "fetch": [[7.5, 6.5], [1.5, 6.5]], "drop": [4.5, 4.5]}, **config}, self.types, random.Random(3))
        agents = generator.agents(self.map)
        summary = solve_mapd(self.map, agents, generator, self.PARKING,
                             {"heuristic_type": "manhattan", "horizon": 400, **options})
        return generator, agents, summary

    def test_lifelong_singular_with_a_type_limit(self):
        generator, agents, summary = self.lifelong({})
        self.check(summary, agents)
        self.assertEqual((summary["completed_tasks"], summary["n_tasks"], summary["target_tasks"]), (3, 3, 3))
        self.assertEqual(set(summary["completed_tasks_times"]), {"haul", "fetch", "drop"})

    def test_only_an_allowed_type_takes_a_limited_task(self):
        from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import NeuralATTF
        agents = TaskGenerator({"agents": {"amr": 2, "cart": 1}}, self.types, random.Random(3)).agents(self.map)
        planner = NeuralATTF(self.map, agents, self.PARKING, low_level="sipp")
        planner.update_tasks(0, [{"task_name": "haul", "start_time": 0, "waypoints": [(1.5, 1.5)], "types": ["cart"]},
                                 {"task_name": "mine", "start_time": 0, "waypoints": [(2.5, 2.5)], "agent": "agent1"}])
        idle = {a["name"]: [a["start"]] for a in agents}
        pairs, _ = planner.find_closest_agent(planner.token["tasks"], idle, planner.token)
        taken = {task: agent for agent, task, _, _ in pairs if task}
        self.assertEqual(taken, {"haul": "agent2", "mine": "agent1"})           # agent2 is the cart

    def test_lifelong_achieved_stops_at_the_count(self):
        generator, agents, summary = self.lifelong({"mode": "achieved", "count": 4, "frequency": 0.1})
        self.check(summary, agents)
        self.assertGreaterEqual(summary["completed_tasks"], 4)
        self.assertEqual(summary["target_tasks"], 4)
        self.assertLess(summary["steps"], 400)

    def test_lifelong_endless_runs_to_the_horizon(self):
        generator, agents, summary = self.lifelong({"mode": "endless", "frequency": 0.05}, horizon=60)
        self.check(summary, agents)
        self.assertEqual(summary["steps"], 60)
        self.assertIsNone(summary["target_tasks"])
        self.assertEqual(summary["n_tasks"], 3)                                 # released at t = 0, 20, 40
