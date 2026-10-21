"""Heterogeneous agents: per-agent radius / speed across the package.

Covers the agent-spec helpers, the obstacle-inflation rounding fix, the checker,
the simulation, SIPP / CCBS / CBS pairwise clearance (r_i + r_j) and the
homogeneous-equivalence guarantee (dict values all equal == scalar run).
"""
import math
import random
import unittest

import numpy as np

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.utils.util import (
    agent_radii_dict,
    agent_radius_of,
    agent_velocities_dict,
    agent_velocity_of,
    agents_yaml_to_roadmap_frame,
    is_heterogeneous,
    mapd_agents_to_roadmap_frame,
    max_agent_radius,
    normalize_agent_specs,
    read_graph_sampler_from_yaml,
    set_global_seed,
)
from path_planning.utils.checker import check_collision, check_velocity_anomaly

MAP_YAML = "path_planning/maps/2d/2d.yaml"
RADIUS = 1.0
VELOCITY = 1.0
# Two free grid cells 3 apart in the lower-left room of 2d.yaml (cell centres).
P = (5.5, 5.5)
Q = (8.5, 5.5)


def _build_grid_map():
    """Discrete 4-neighbour grid roadmap on 2d.yaml, inflated for the largest agent."""
    set_global_seed(42)
    map_ = read_graph_sampler_from_yaml(MAP_YAML, use_discrete_space=True)
    map_.inflate_obstacles(radius=RADIUS + np.sqrt(2) / 2)
    map_.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
    map_.set_start([P])
    map_.set_goal([Q])
    nodes = map_.generateRandomNodes(generate_grid_nodes=True)
    map_.generate_roadmap(nodes)
    return map_


def _point(p, t):
    return {"x": float(p[0]), "y": float(p[1]), "t": float(t)}


def _interp(path, t):
    """Position of a ``[{t, x, y}, ...]`` schedule at time ``t`` (hold at the ends)."""
    if t <= path[0]["t"]:
        return np.array([path[0]["x"], path[0]["y"]])
    for a, b in zip(path, path[1:]):
        if a["t"] <= t <= b["t"]:
            dt = b["t"] - a["t"]
            if dt <= 0:
                return np.array([a["x"], a["y"]])
            u = (t - a["t"]) / dt
            return np.array([a["x"] + u * (b["x"] - a["x"]), a["y"] + u * (b["y"] - a["y"])])
    return np.array([path[-1]["x"], path[-1]["y"]])


def _min_pair_distance(schedule, dt=0.01):
    names = list(schedule)
    t_end = max(p[-1]["t"] for p in schedule.values())
    best = float("inf")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for t in np.arange(0.0, t_end + dt, dt):
                d = float(np.linalg.norm(_interp(schedule[names[i]], t) - _interp(schedule[names[j]], t)))
                best = min(best, d)
    return best


def _segment_speeds(path):
    out = []
    for a, b in zip(path, path[1:]):
        d = math.dist((a["x"], a["y"]), (b["x"], b["y"]))
        dt = b["t"] - a["t"]
        if d > 1e-9 and dt > 1e-9:
            out.append(d / dt)
    return out


class TestAgentSpecHelpers(unittest.TestCase):
    def test_defaults_and_coercion(self):
        self.assertEqual(agent_radius_of({"name": "a"}, 0.3), 0.3)
        self.assertEqual(agent_radius_of({"name": "a", "radius": 1}, 0.3), 1.0)
        self.assertEqual(agent_radius_of({"name": "a", "radius": None}, 0.3), 0.3)
        self.assertEqual(agent_velocity_of({"velocity": "2.5"}, 0.0), 2.5)

    def test_max_radius_and_dicts(self):
        agents = [{"name": "a", "radius": 0.2}, {"name": "b"}, {"radius": 1.5}]
        self.assertEqual(max_agent_radius([], 0.7), 0.7)
        self.assertEqual(max_agent_radius(agents, 0.5), 1.5)
        self.assertEqual(max_agent_radius(agents, 2.0), 2.0)
        self.assertEqual(agent_radii_dict(agents, 0.5), {"a": 0.2, "b": 0.5, "agent2": 1.5})
        self.assertEqual(agent_velocities_dict(agents, 1.0), {"a": 1.0, "b": 1.0, "agent2": 1.0})
        self.assertTrue(is_heterogeneous(agent_radii_dict(agents, 0.5)))
        self.assertFalse(is_heterogeneous({"a": 1.0, "b": 1.0}))
        self.assertFalse(is_heterogeneous({}))

    def test_normalize(self):
        out = normalize_agent_specs([{"name": "a", "radius": 1, "velocity": None, "start": [0, 0]}])
        self.assertEqual(out, [{"name": "a", "radius": 1.0, "start": [0, 0]}])
        with self.assertRaises(ValueError):
            normalize_agent_specs([{"name": "a", "radius": -1}])


class TestFrameConversionKeepsSpec(unittest.TestCase):
    def setUp(self):
        self.map_ = GraphSampler(bounds=[[0, 8], [0, 8]], resolution=2.0, start=[], goal=[], use_discrete_space=True)

    def test_mapf_agents(self):
        agents = [
            {"name": "a", "start": [1.0, 1.0], "goal": [5.0, 5.0], "radius": 0.5, "velocity": 2.0},
            {"name": "b", "start": [3.0, 1.0], "goal": [7.0, 5.0]},
        ]
        out = agents_yaml_to_roadmap_frame(self.map_, agents)
        self.assertEqual(out[0]["radius"], 0.5)
        self.assertEqual(out[0]["velocity"], 2.0)
        self.assertEqual(set(out[1]), {"name", "start", "goal"})
        self.assertEqual(out[0]["start"], (1.0, 1.0))

    def test_mapd_agents(self):
        out = mapd_agents_to_roadmap_frame(self.map_, [{"name": "a", "start": [1.0, 1.0], "radius": 0.25}, {"start": [3.0, 1.0]}])
        self.assertEqual(out[0]["radius"], 0.25)
        self.assertEqual(set(out[0]), {"name", "start", "radius"})
        self.assertEqual(set(out[1]), {"name", "start"})
        self.assertEqual(out[1]["name"], "agent1")


class TestInflateRounding(unittest.TestCase):
    def _map(self):
        m = GraphSampler(bounds=[[0, 9], [0, 9]], resolution=1.0, start=[], goal=[], use_discrete_space=True)
        m.set_obstacles(np.array([[4, 4]], dtype=np.int64))
        return m

    def test_requested_radius_rounds_up_to_cell_diagonal_multiples(self):
        # Inflation only ever adds INFLATION cells (parent behaviour), so use a
        # fresh map per radius.
        m = self._map()
        step = math.sqrt(2.0)
        m.set_inflation_radius(2.0)
        self.assertAlmostEqual(m.inflation_radius, 2 * step)
        self.assertEqual(m.requested_inflation_radius, 2.0)
        inflated = {tuple(int(c) for c in p) for p in m.get_inflated_obstacles()}
        self.assertIn((3, 2), inflated)      # distance sqrt(5) = 2.24 <= 2.83
        self.assertIn((2, 3), inflated)
        self.assertIn((2, 4), inflated)      # distance 2 <= 2.83 (was NOT inflated by the old formula)
        self.assertNotIn((1, 4), inflated)   # distance 3 > 2.83
        m = self._map()
        m.set_inflation_radius(1.0)
        self.assertAlmostEqual(m.inflation_radius, step)
        inflated = {tuple(int(c) for c in p) for p in m.get_inflated_obstacles()}
        self.assertIn((3, 4), inflated)      # 4-neighbour
        self.assertIn((3, 3), inflated)      # diagonal neighbour (the point of the rounding)
        self.assertNotIn((2, 4), inflated)   # distance 2 > 1.41
        self.assertEqual(m.expanded_inflation_radius(0.5), m.expanded_inflation_radius(1.0))
        self.assertGreater(m.expanded_inflation_radius(1.5), m.expanded_inflation_radius(1.0))

    def test_rounding_is_idempotent(self):
        m = self._map()
        for k in range(1, 6):
            r = m.expanded_inflation_radius(0.3 * k)
            self.assertAlmostEqual(m.expanded_inflation_radius(r), r, places=12)

    def test_verbatim_and_no_obstacles(self):
        m = self._map()
        m.set_inflation_radius(1.3, use_expand=False)
        self.assertEqual(m.inflation_radius, 1.3)
        empty = GraphSampler(bounds=[[0, 9], [0, 9]], resolution=1.0, start=[], goal=[], use_discrete_space=True)
        empty.set_inflation_radius(1.0)
        self.assertAlmostEqual(empty.inflation_radius, math.sqrt(2.0))
        self.assertEqual(empty.requested_inflation_radius, 1.0)


class TestCheckerDictRadii(unittest.TestCase):
    @staticmethod
    def _pair(xb):
        return {
            "a": [_point((0.0, 0.0), 0), _point((0.0, 0.0), 1)],
            "b": [_point((xb, 0.0), 0), _point((xb, 0.0), 1)],
        }

    def test_pairwise_sum_of_radii(self):
        radii = {"a": 0.2, "b": 1.0}
        self.assertTrue(check_collision(self._pair(1.1), radii, verbose=False))
        self.assertFalse(check_collision(self._pair(1.3), radii, verbose=False))

    def test_scalar_still_means_two_r(self):
        self.assertFalse(check_collision(self._pair(1.1), 0.5, verbose=False))
        self.assertTrue(check_collision(self._pair(0.9), 0.5, verbose=False))

    def test_missing_name_raises(self):
        with self.assertRaises(ValueError):
            check_collision(self._pair(1.1), {"a": 0.2}, verbose=False)

    def test_per_agent_velocity_anomaly(self):
        sol = {"a": [_point((0.0, 0.0), 0), _point((2.0, 0.0), 1)]}
        self.assertTrue(check_velocity_anomaly(sol, verbose=False, velocity={"a": 2.0}))
        self.assertFalse(check_velocity_anomaly(sol, verbose=False, velocity={"a": 1.0}))
        self.assertTrue(check_velocity_anomaly(sol, verbose=False, velocity=None))


class TestSimulationHeterogeneous(unittest.TestCase):
    def _sim_and_algo(self, paths, agents, **sim_kwargs):
        from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import _Waypoint
        from path_planning.multi_agent_planner.decentralized.neural_attf.simulation import Simulation

        sim = Simulation(tasks=[], agents=agents, rng=random.Random(0), **sim_kwargs)

        class _Algo:
            agent_radius = 0.0

            def __init__(self):
                self.token = {"agents": {n: [_Waypoint(q) for q in p] for n, p in paths.items()}}

            def time_forward(self, *a, **k):
                pass

            def get_token(self):
                return self.token

        return sim, _Algo()

    def test_per_agent_arc_budget(self):
        paths = {
            "fast": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
            "slow": [(0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (3.0, 5.0)],
        }
        agents = [
            {"name": "fast", "start": (0.0, 0.0), "velocity": 2.0},
            {"name": "slow", "start": (0.0, 5.0), "velocity": 0.5},
        ]
        sim, algo = self._sim_and_algo(paths, agents, timestep_duration=1.0)
        self.assertEqual(sim.arc_budgets, {"fast": 2.0, "slow": 0.5})
        sim.time_forward(algo)
        self.assertEqual(sim.actual_paths["fast"][-1]["x"], 2.0)
        self.assertEqual(sim.actual_paths["slow"][-1]["x"], 1.0)  # first segment always taken

    def test_pairwise_hold_clearance(self):
        paths = {"a": [(0.0, 0.0), (2.0, 0.0)], "b": [(1.0, 1.1)]}
        agents = [{"name": "a", "start": (0.0, 0.0), "radius": 0.2}, {"name": "b", "start": (1.0, 1.1), "radius": 1.0}]
        sim, algo = self._sim_and_algo(paths, agents)
        sim.time_forward(algo)
        self.assertEqual(sim.actual_paths["a"][-1]["x"], 0.0)  # 1.1 < 0.2 + 1.0: held

        agents = [{"name": "a", "start": (0.0, 0.0), "radius": 0.2}, {"name": "b", "start": (1.0, 1.1), "radius": 0.5}]
        sim, algo = self._sim_and_algo(paths, agents)
        sim.time_forward(algo)
        self.assertEqual(sim.actual_paths["a"][-1]["x"], 2.0)  # 1.1 > 0.7: clear

    def test_scalar_radius_backstop_unchanged(self):
        paths = {"a": [(0.0, 0.0), (2.0, 0.0)], "b": [(1.0, 0.0)]}
        agents = [{"name": "a", "start": (0.0, 0.0)}, {"name": "b", "start": (1.0, 0.0)}]
        sim, algo = self._sim_and_algo(paths, agents, agent_radius=0.5)
        sim.time_forward(algo)
        self.assertEqual(sim.actual_paths["a"][-1]["x"], 0.0)


class TestSippHeterogeneous(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_ = _build_grid_map()

    def _agents(self):
        return [
            {"name": "big", "start": P, "goal": Q, "radius": 1.0, "velocity": 0.5},
            {"name": "small", "start": Q, "goal": P, "radius": 0.5, "velocity": 2.0},
        ]

    def test_big_slow_and_small_fast_agents_do_not_collide(self):
        from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner

        set_global_seed(1)
        planner = SippPlanner(self.map_, {}, self._agents(), radius=1.0, velocity=1.0,
                              heuristic_type="euclidean", time_limit=30, max_iterations=10)
        self.assertEqual(planner.agent_radii, {"big": 1.0, "small": 0.5})
        self.assertEqual(planner.agent_velocities, {"big": 0.5, "small": 2.0})
        solution, info = planner.compute_plan()
        self.assertTrue(info["success"])
        self.assertFalse(check_collision(solution, {"big": 1.0, "small": 0.5}, verbose=False))
        self.assertGreater(_min_pair_distance(solution), 1.5 - 1e-6)
        for name, v in (("big", 0.5), ("small", 2.0)):
            for s in _segment_speeds(solution[name]):
                self.assertAlmostEqual(s, v, places=6)
        self.assertEqual(set(planner.sipp_graphs), {1.0, 0.5})

    def test_homogeneous_dicts_match_scalar_run(self):
        from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner

        # ``b`` starts on ``a``'s row and must clear it: the two interact, and a
        # radius-1 solution exists (a head-on swap on this row does not).
        b_start, b_goal = (7.5, 5.5), (5.5, 8.5)
        agents_scalar = [{"name": "a", "start": P, "goal": Q}, {"name": "b", "start": b_start, "goal": b_goal}]
        agents_dict = [
            {"name": "a", "start": P, "goal": Q, "radius": RADIUS, "velocity": VELOCITY},
            {"name": "b", "start": b_start, "goal": b_goal, "radius": RADIUS, "velocity": VELOCITY},
        ]
        results = []
        for agents in (agents_scalar, agents_dict):
            set_global_seed(42)
            planner = SippPlanner(self.map_, {}, agents, radius=RADIUS, velocity=VELOCITY,
                                  heuristic_type="euclidean", time_limit=30, max_iterations=10)
            solution, info = planner.compute_plan()
            self.assertTrue(info["success"])
            self.assertEqual(len(planner.sipp_graphs), 1)
            results.append(solution)
        self.assertEqual(results[0], results[1])

    def test_obstacle_table_uses_sum_of_radii(self):
        from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippGraph

        obstacle = {"other": [_point(Q, 0.0)]}
        hetero = SippGraph(self.map_, obstacle, radius=1.0, velocity=1.0, agent_radii={"other": 0.5})
        equiv = SippGraph(self.map_, obstacle, radius=0.75, velocity=1.0)  # 0.75 + 0.75 == 1.0 + 0.5
        for pos in (P, Q, (7.5, 5.5)):
            self.assertEqual(hetero.sipp_graph[pos].interval_list, equiv.sipp_graph[pos].interval_list)


class TestCcbsHeterogeneous(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_ = _build_grid_map()

    def test_check_collision_r_sum(self):
        from path_planning.multi_agent_planner.centralized.ccbs.graph_generation import Environment

        agents = [{"name": "a", "start": P, "goal": Q}, {"name": "b", "start": Q, "goal": P}]
        env = Environment(self.map_, {}, agents, radius=1.0, velocity=1.0, agent_radii={"a": 1.0, "b": 0.5})
        self.assertEqual(env._pair_clearance("a", "b"), 1.5)
        p = np.array([0.0, 0.0])
        z = np.zeros(2)
        self.assertTrue(env.check_collision(p, z, p + np.array([1.4, 0.0]), z, 1.0, r_sum=1.5))
        self.assertFalse(env.check_collision(p, z, p + np.array([1.6, 0.0]), z, 1.0, r_sum=1.5))
        self.assertTrue(env.check_collision(p, z, p + np.array([1.6, 0.0]), z, 1.0))  # default 2r = 2.0

    def test_get_conflicts_uses_pairwise_clearance(self):
        from path_planning.multi_agent_planner.centralized.ccbs.graph_generation import Environment
        from path_planning.multi_agent_planner.centralized.sipp.graph_generation import State

        # Two agents stepping east on adjacent rows (1 apart), both over [0, 1].
        agents = [{"name": "a", "start": P, "goal": Q}, {"name": "b", "start": (5.5, 6.5), "goal": (8.5, 6.5)}]
        plans = {
            "a": [State((5.5, 5.5), 0.0), State((6.5, 5.5), 1.0)],
            "b": [State((5.5, 6.5), 0.0), State((6.5, 6.5), 1.0)],
        }
        costs = {"a": [(0.0, 1.0), (0.0, 0.0)], "b": [(0.0, 1.0), (0.0, 0.0)]}
        # r_a + r_b = 0.9 < 1: clear; the scalar 0.6 (2r = 1.2 > 1) collides.
        env = Environment(self.map_, {}, agents, radius=0.6, velocity=1.0, agent_radii={"a": 0.4, "b": 0.5})
        self.assertEqual(env.get_conflicts(plans, costs, get_first_conflict=False), [])
        env = Environment(self.map_, {}, agents, radius=0.6, velocity=1.0)
        self.assertTrue(env.get_conflicts(plans, costs, get_first_conflict=False))

    def test_search_uses_each_agents_speed(self):
        from path_planning.multi_agent_planner.centralized.ccbs.graph_generation import Environment
        from path_planning.multi_agent_planner.centralized.ccbs.ccbs import CCBS

        # Non-interacting legs so the (pre-existing) CCBS constraint branch is not hit.
        agents = [
            {"name": "slow", "start": P, "goal": Q, "radius": 1.0, "velocity": 0.5},
            {"name": "fast", "start": (5.5, 9.5), "goal": (8.5, 9.5), "radius": 0.5, "velocity": 2.0},
        ]
        set_global_seed(3)
        env = Environment(self.map_, {}, agents, radius=1.0, velocity=1.0, heuristic_type="euclidean", time_limit=30)
        solution, info = CCBS(env, time_limit=30, max_iterations=200, verbose=False).search()
        self.assertTrue(info["success"])
        for name, v in (("slow", 0.5), ("fast", 2.0)):
            for s in _segment_speeds(solution[name]):
                self.assertAlmostEqual(s, v, places=6)


class TestCbsHeterogeneous(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_ = _build_grid_map()

    def _solution(self):
        from path_planning.multi_agent_planner.centralized.cbs.cbs import Location, State

        # Two agents driving adjacent rows (y = 5.5 and y = 6.5, 1 apart), one step east.
        a = [State(0, Location((5.5, 5.5))), State(1, Location((6.5, 5.5)))]
        b = [State(0, Location((5.5, 6.5))), State(1, Location((6.5, 6.5)))]
        return {"a": a, "b": b}

    def test_pair_clearance_decides_conflicts(self):
        from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment

        agents = [{"name": "a", "start": P, "goal": Q}, {"name": "b", "start": (5.5, 6.5), "goal": (8.5, 6.5)}]
        # r_a + r_b = 0.9 < 1: no conflict; the scalar 0.6 (2r = 1.2 > 1) conflicts.
        env = Environment(self.map_, agents, radius=0.6, agent_radii={"a": 0.4, "b": 0.5})
        self.assertEqual(env.get_conflicts(self._solution(), get_first_conflict=False), [])
        env = Environment(self.map_, agents, radius=0.6)
        self.assertTrue(env.get_conflicts(self._solution(), get_first_conflict=False))

    def test_constraint_segment_signature(self):
        m = self.map_
        hits = m.get_constraint_segment((0.0, 0.0), (2.0, 0.0), (0.0, 1.2), (2.0, 1.2), 0.0, None, 1.5)
        self.assertTrue(hits)
        self.assertFalse(m.get_constraint_segment((0.0, 0.0), (2.0, 0.0), (0.0, 1.6), (2.0, 1.6), 0.0, None, 1.5))
        # Different speeds: a slow second agent is still met head-on.
        self.assertTrue(m.get_constraint_segment((0.0, 0.0), (4.0, 0.0), (4.0, 0.0), (0.0, 0.0), 2.0, 0.5, 1.0))


class TestNeuralAttfAccessors(unittest.TestCase):
    def test_bare_object_accessors(self):
        from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import NeuralATTF

        obj = NeuralATTF.__new__(NeuralATTF)
        obj.agent_radius = 0.5
        obj.velocity = 1.0
        self.assertEqual(obj._radius_of("me"), 0.5)
        self.assertEqual(obj._pair_clearance("me", "o"), 1.0)
        obj.agent_radii = {"me": 0.5, "o": 1.0}
        obj.agent_velocities = {"o": 2.0}
        self.assertEqual(obj._pair_clearance("me", "o"), 1.5)
        self.assertEqual(obj._max_radius(), 1.0)
        self.assertEqual(obj._any_clearance("me"), 1.5)
        self.assertEqual(obj._velocity_of("me"), 1.0)
        self.assertEqual(obj._velocity_of("o"), 2.0)


class TestSummarizeSolutionRecordsPerAgent(unittest.TestCase):
    def test_per_agent_values_in_summary(self):
        from path_planning.multi_agent_planner.mapf_solver import summarize_solution

        class _Map:
            nodes = []
            edges = []

        cfg = {"mapf_solver_name": "sipp", "agent_radius": 0.5, "agent_velocity": 1.0,
               "time_limit": 1, "max_iterations": 1, "heuristic_type": "euclidean"}
        agents = [{"name": "a", "radius": 0.2}, {"name": "b", "velocity": 2.0}]
        summary = summarize_solution({}, {"success": False, "runtime": 0.0}, cfg, _Map(), agents=agents)
        self.assertEqual(summary["agent_radii"], {"a": 0.2, "b": 0.5})
        self.assertEqual(summary["agent_velocities"], {"a": 1.0, "b": 2.0})
        self.assertEqual(summary["agent_radius"], 0.5)


if __name__ == "__main__":
    unittest.main()
