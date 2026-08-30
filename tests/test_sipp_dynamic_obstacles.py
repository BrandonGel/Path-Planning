"""Unit tests for SIPP dynamic-obstacle interval semantics and the NeuralATTF
time-based schedule resampling / obstacle-schedule builder.

Run with:  python -m unittest tests.test_sipp_dynamic_obstacles -v
"""

import math
import random
import unittest

import numpy as np

from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippGraph
from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import NeuralATTF
from path_planning.utils.util import read_graph_sampler_from_yaml, set_global_seed

MAP_YAML = "path_planning/maps/2d/2d.yaml"
RADIUS = 1.0
VELOCITY = 1.0
# Two free grid cells 3 apart in the lower-left room of 2d.yaml.
P = (5.0, 5.0)
Q = (8.0, 5.0)


def _build_grid_map():
    """Discrete 4-neighbour grid roadmap on 2d.yaml (same build as run_sipp.py)."""
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


def _safe_at(node, t):
    return any(a <= t <= b for a, b in node.interval_list)


class TestResampleByTime(unittest.TestCase):
    def test_wait_then_move_samples_once_per_tick(self):
        schedule = [_point((0, 0), 0.0), _point((0, 0), 0.4), _point((3, 0), 3.4)]
        out = NeuralATTF._resample_by_time(schedule, (3.0, 0.0), dt=1.0)
        expected = [(0.0, 0.0), (0.6, 0.0), (1.6, 0.0), (2.6, 0.0), (3.0, 0.0)]
        self.assertEqual(len(out), len(expected))
        for got, exp in zip(out, expected):
            self.assertAlmostEqual(got[0], exp[0], places=9)
            self.assertAlmostEqual(got[1], exp[1], places=9)

    def test_pure_wait_repeats_position(self):
        schedule = [_point((2, 2), 0.0), _point((2, 2), 3.0)]
        out = NeuralATTF._resample_by_time(schedule, (2.0, 2.0), dt=1.0)
        self.assertEqual(out, [(2.0, 2.0)] * 4)

    def test_single_point_schedule(self):
        out = NeuralATTF._resample_by_time([_point((1, 1), 0.0)], (1.0, 1.0), dt=1.0)
        self.assertEqual(out, [(1.0, 1.0)])

    def test_vias_record_vertices_passed_inside_a_tick(self):
        # Two 0.6-unit edges with a corner at t=0.6: tick 1 passes the corner (0.6, 0)
        # and ends 0.4 along the second edge; the corner is a via of sample 1.
        schedule = [_point((0, 0), 0.0), _point((0.6, 0), 0.6), _point((0.6, 0.6), 1.2)]
        out = NeuralATTF._resample_by_time(schedule, (0.6, 0.6), dt=1.0)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].via, ())
        self.assertEqual(out[1].via, ((0.6, 0.0),))
        self.assertAlmostEqual(out[1][0], 0.6, places=9)
        self.assertAlmostEqual(out[1][1], 0.4, places=9)
        self.assertAlmostEqual(out[1].via_t[0], 0.6, places=9)
        # The schedule ends at t=1.2, inside tick 2: the goal is reached at fraction
        # 0.2 and the agent rests there until the tick ends.
        self.assertEqual(out[2].via, ((0.6, 0.6),))
        self.assertAlmostEqual(out[2].via_t[0], 0.2, places=9)
        self.assertEqual(tuple(out[2]), (0.6, 0.6))
        # Vias never duplicate a neighbouring sample (exact hit at a tick boundary).
        out2 = NeuralATTF._resample_by_time([_point((0, 0), 0.0), _point((1, 0), 1.0), _point((1, 1), 2.0)], (1.0, 1.0), dt=1.0)
        self.assertTrue(all(w.via == () for w in out2))

    def test_wait_inside_a_tick_is_kept_with_its_times(self):
        # Reach (1, 0) at t=1, wait there until t=2.5, then move on: with dt=2 the
        # second sample (t=2) sits mid-wait and the third tick (2, 4] departs at
        # fraction 0.25. Both the arrival and the departure must survive as vias so
        # the executor holds the agent instead of smearing the wait into motion.
        schedule = [_point((0, 0), 0.0), _point((1, 0), 1.0), _point((1, 0), 2.5), _point((4, 0), 5.5)]
        out = NeuralATTF._resample_by_time(schedule, (4.0, 0.0), dt=2.0)
        self.assertEqual([tuple(w) for w in out], [(0.0, 0.0), (1.0, 0.0), (2.5, 0.0), (4.0, 0.0)])
        self.assertEqual(out[1].via, ((1.0, 0.0),))
        self.assertAlmostEqual(out[1].via_t[0], 0.5, places=9)
        self.assertEqual(out[2].via, ((1.0, 0.0),))
        self.assertAlmostEqual(out[2].via_t[0], 0.25, places=9)
        self.assertEqual(out[3].via, ((4.0, 0.0),))  # arrives at 5.5 = fraction 0.75 of tick (4, 6]
        self.assertAlmostEqual(out[3].via_t[0], 0.75, places=9)

    def test_exact_unit_speed_moves_one_unit_per_tick(self):
        schedule = [_point((0, 0), 0.0), _point((2, 0), 2.0), _point((2, 2), 4.0)]
        out = NeuralATTF._resample_by_time(schedule, (2.0, 2.0), dt=1.0)
        self.assertEqual(len(out), 5)
        for a, b in zip(out, out[1:]):
            self.assertAlmostEqual(math.dist(a, b), 1.0, places=9)
        self.assertEqual(out[-1], (2.0, 2.0))


class TestOtherAgentSchedules(unittest.TestCase):
    def _planner(self, agents):
        obj = NeuralATTF.__new__(NeuralATTF)
        obj.timestep_duration = 1.0
        obj.token = {"agents": agents}
        return obj

    def test_idle_agent_blocks_forever(self):
        obj = self._planner({"me": [(0, 0), (1, 0)], "idle": [(5.0, 5.0)]})
        sched, horizons = obj._other_agent_schedules("me")
        self.assertEqual(list(sched), ["idle"])
        self.assertEqual(len(sched["idle"]), 1)
        self.assertEqual(sched["idle"][0]["t"], 0.0)
        self.assertIsNone(horizons["idle"])

    def test_mover_is_sliced_by_offset_and_held_one_tick(self):
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        obj = self._planner({"me": [(9, 9)], "mover": path})
        sched, horizons = obj._other_agent_schedules("me", offset_steps=2)
        self.assertEqual([(s["x"], s["y"]) for s in sched["mover"]], [(2.0, 0.0), (3.0, 0.0)])
        self.assertEqual([s["t"] for s in sched["mover"]], [0.0, 1.0])
        self.assertEqual(horizons["mover"], 1.0)

    def test_mover_shorter_than_offset_is_one_resting_point(self):
        obj = self._planner({"me": [(9, 9)], "mover": [(0.0, 0.0), (1.0, 0.0)]})
        sched, horizons = obj._other_agent_schedules("me", offset_steps=5)
        self.assertEqual([(s["x"], s["y"], s["t"]) for s in sched["mover"]], [(1.0, 0.0, 0.0)])
        self.assertEqual(horizons["mover"], 1.0)

    def test_vias_are_expanded_with_arc_fraction_timestamps(self):
        from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import _Waypoint

        path = [_Waypoint((0.0, 0.0)), _Waypoint((0.6, 0.4), via=[(0.6, 0.0)]), _Waypoint((0.6, 1.4))]
        obj = self._planner({"me": [(9, 9)], "mover": path})
        sched, horizons = obj._other_agent_schedules("me")
        got = [((s["x"], s["y"]), round(s["t"], 6)) for s in sched["mover"]]
        self.assertEqual(got, [((0.0, 0.0), 0.0), ((0.6, 0.0), 0.6), ((0.6, 0.4), 1.0), ((0.6, 1.4), 2.0)])
        self.assertEqual(horizons["mover"], 1.0)

    def test_vias_with_times_keep_the_planners_intra_tick_timing(self):
        from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import _Waypoint

        # Wait at (1, 0) from 0.5 to 0.75 of tick 1, then move to (2, 0) by the tick end.
        path = [_Waypoint((0.0, 0.0)), _Waypoint((2.0, 0.0), via=[(1.0, 0.0), (1.0, 0.0)], via_t=[0.5, 0.75])]
        obj = self._planner({"me": [(9, 9)], "mover": path})
        sched, _ = obj._other_agent_schedules("me")
        got = [((s["x"], s["y"]), round(s["t"], 6)) for s in sched["mover"]]
        self.assertEqual(got, [((0.0, 0.0), 0.0), ((1.0, 0.0), 0.5), ((1.0, 0.0), 0.75), ((2.0, 0.0), 1.0)])
    def test_exclude_drops_agent(self):
        obj = self._planner({"me": [(9, 9)], "idle": [(5.0, 5.0)]})
        sched, _ = obj._other_agent_schedules("me", exclude={"idle"})
        self.assertEqual(sched, {})


class TestReachSegmentSafe(unittest.TestCase):
    """Reach segment on the grid edge P=(5,5) -- (6,5) of 2d.yaml, RADIUS 1 (2r = 2)."""

    @classmethod
    def setUpClass(cls):
        cls.map_ = _build_grid_map()
        cls.map_.set_constraint_sweep()  # NeuralATTF.__init__ does this on the real roadmap

    def _obj(self):
        obj = NeuralATTF.__new__(NeuralATTF)
        obj.graph_map = self.map_
        obj.agent_radius = RADIUS
        obj.sipp_clearance_margin = 0.0
        obj.timestep_duration = 1.0
        obj.velocity = VELOCITY
        return obj

    A = (5.5, 5.0)      # mid-edge actual position
    START = (6.0, 5.0)  # snapped node ahead
    OTHER = (5.0, 5.0)  # the edge's other endpoint

    def test_oncoming_agent_on_the_same_edge_is_unsafe(self):
        # "o" drives (8,5) -> (5,5) over [0, 3]: its footprint sweeps our edge in the window.
        sched = {"o": [_point((8, 5), 0.0), _point((5, 5), 3.0)]}
        self.assertFalse(self._obj()._reach_segment_safe(self.A, self.START, self.OTHER, 0.5, sched, {"o": 1.0}))

    def test_agent_that_passes_later_is_safe(self):
        # Same sweep, but it only reaches the edge's neighbourhood after t_reach.
        sched = {"o": [_point((12, 5), 0.0), _point((9, 5), 3.0), _point((5, 5), 7.0)]}
        self.assertTrue(self._obj()._reach_segment_safe(self.A, self.START, self.OTHER, 0.5, sched, {"o": 1.0}))

    def test_resting_agent_on_the_edge_is_unsafe_and_moving_away_from_one_is_safe(self):
        sched = {"o": [_point((7, 5), 0.0)]}  # rests forever 1 unit past START: inside 2r of the edge
        self.assertFalse(self._obj()._reach_segment_safe(self.A, self.START, self.OTHER, 0.5, sched, {"o": None}))
        # Held 0.5 from an idle agent at (5,5) and driving away toward (6,5): allowed.
        sched = {"o": [_point((5, 5), 0.0)]}
        self.assertTrue(self._obj()._reach_segment_safe(self.A, self.START, self.OTHER, 0.5, sched, {"o": None}))

    def test_resting_agent_touching_the_edge_but_not_the_driven_part_is_safe(self):
        # (4,5) rests forever: its 2r footprint covers the edge's (5,5) end, which the sweep
        # reports, but not the sub-segment (5.5,5) -> (6,5) we actually drive.
        sched = {"o": [_point((4, 5), 0.0)]}
        self.assertTrue(self._obj()._reach_segment_safe(self.A, self.START, self.OTHER, 0.5, sched, {"o": None}))


class TestStaticRoute(unittest.TestCase):
    def test_route_follows_roadmap_from_start_to_goal(self):
        map_ = _build_grid_map()
        obj = NeuralATTF.__new__(NeuralATTF)
        obj.graph_map = map_
        obj._node_kdtree = None
        route = obj._static_route(P, Q)
        self.assertEqual(route[0], P)
        self.assertEqual(route[-1], Q)
        self.assertEqual(len(route), 4)  # 3 unit edges
        for a, b in zip(route, route[1:]):
            self.assertAlmostEqual(math.dist(a, b), 1.0, places=9)

    def test_blocked_nodes_are_avoided_or_route_is_empty(self):
        map_ = _build_grid_map()
        obj = NeuralATTF.__new__(NeuralATTF)
        obj.graph_map = map_
        obj._node_kdtree = None
        detour = obj._static_route(P, Q, blocked={(6.0, 5.0)})
        self.assertTrue(detour and (6.0, 5.0) not in detour and detour[-1] == Q)
        sealed = obj._static_route(P, Q, blocked={(6.0, 5.0), (6.0, 4.0), (6.0, 6.0), (5.0, 4.0), (5.0, 6.0), (4.0, 5.0)})
        self.assertEqual(sealed, [])


class TestSippGoalSafeUntil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_ = _build_grid_map()

    def _arrival(self, goal_safe_until):
        from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner

        # "other" occupies Q over [4, 6] and rests there one more second; Q is free
        # before t=4 and after t=7.
        planner = SippPlanner(
            self.map_,
            dynamic_obstacles={"other": [_point(Q, 4.0), _point(Q, 6.0)]},
            agents=[{"name": "a", "start": P, "goal": Q}],
            radius=RADIUS,
            velocity=VELOCITY,
            use_constraint_sweep=True,
            heuristic_type="euclidean",
            time_limit=10.0,
            sipp_max_iterations=100000,
            obstacle_horizon={"other": 1.0},
            require_goal_safe_forever=False,
            goal_safe_until=goal_safe_until,
        )
        solution, info = planner.compute_plan()
        self.assertTrue(info["success"])
        return solution["a"][-1]["t"]

    def test_bounded_goal_accepts_early_arrival_by_default(self):
        # P -> Q is 3 units at unit speed: arrive at 3, inside the free window [0, 4).
        self.assertLess(self._arrival(None), 4.0)

    def test_goal_must_stay_safe_until_the_requested_time(self):
        # Needing Q safe until t+2 rules out the early window (blocked from 4), so the
        # plan has to wait for the window that opens after the hold ends at 7.
        self.assertGreaterEqual(self._arrival(lambda t: t + 2.0), 7.0 - 1e-6)


class TestPlanTaskRouteJoin(unittest.TestCase):
    def test_join_keeps_the_vias_of_the_tick_that_reaches_a_waypoint(self):
        from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import _SegState, _Waypoint

        obj = NeuralATTF.__new__(NeuralATTF)
        obj.num_goal_wait_steps = 1
        obj.low_level = "grid"  # skip the SIPP-only static reachability pre-check
        legs = {
            ((0.0, 0.0), (2.0, 0.0)): [_Waypoint((0.0, 0.0)), _Waypoint((2.0, 0.0), via=[(1.0, 0.0)], via_t=[0.4])],
            ((2.0, 0.0), (2.0, 2.0)): [_Waypoint((2.0, 0.0)), _Waypoint((2.0, 2.0), via=[(2.0, 1.0)], via_t=[0.5])],
        }
        obj.plan = lambda name, s, g, *a, **k: {name: [_SegState(p) for p in legs[(tuple(s), tuple(g))]]}
        joined, n = obj._plan_task_route("a", (0.0, 0.0), [(2.0, 0.0), (2.0, 2.0)], {}, [])
        self.assertEqual(n, 2)
        self.assertEqual([tuple(p) for p in joined], [(0.0, 0.0), (2.0, 0.0), (2.0, 0.0), (2.0, 2.0), (2.0, 2.0)])
        # Tick 1 reaches (2, 0) through (1, 0) at 0.4: the join must keep that.
        self.assertEqual(joined[1].via, ((1.0, 0.0),))
        self.assertEqual(joined[1].via_t, (0.4,))
        # Goal-wait copies are plain (no replayed vias); the last leg keeps its own.
        self.assertEqual(getattr(joined[2], "via", ()), ())
        self.assertEqual(joined[3].via, ((2.0, 1.0),))
        self.assertEqual(getattr(joined[4], "via", ()), ())


class TestSimulationRadiusBackstop(unittest.TestCase):
    def _sim_and_algo(self, paths, delay_names=()):
        from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import _Waypoint
        from path_planning.multi_agent_planner.decentralized.neural_attf.simulation import Simulation

        agents = [{"name": n, "start": tuple(p[0])} for n, p in paths.items()]
        sim = Simulation(tasks=[], agents=agents, agent_radius=0.5, rng=random.Random(0))

        class _Algo:
            agent_radius = 0.5

            def __init__(self):
                self.token = {"agents": {n: [_Waypoint(q) for q in p] for n, p in paths.items()}}
                self.seen_delayed = []

            def time_forward(self, t, position, delayed):
                self.seen_delayed.append(list(delayed))

            def get_token(self):
                return self.token

        return sim, _Algo()

    def test_mover_sweeping_through_a_stopped_agent_is_held_and_reported(self):
        # "b" rests on (1, 0); "a" was planned to drive (0,0) -> (2,0) straight through it.
        sim, algo = self._sim_and_algo({"a": [(0.0, 0.0), (2.0, 0.0)], "b": [(1.0, 0.0)]})
        sim.time_forward(algo)
        self.assertEqual(sim.actual_paths["a"][-1]["x"], 0.0)   # held
        self.assertEqual(len(algo.token["agents"]["a"]), 2)     # plan kept for the re-sync
        sim.time_forward(algo)
        self.assertEqual(algo.seen_delayed[-1], ["a"])          # reported next tick

    def test_escape_from_inside_a_footprint_is_not_held(self):
        # "a" was stopped 0.3 from "b" (inside 2r = 1.0) and now drives straight away.
        sim, algo = self._sim_and_algo({"a": [(0.3, 0.0), (2.0, 0.0)], "b": [(0.0, 0.0)]})
        sim.time_forward(algo)
        self.assertEqual(sim.actual_paths["a"][-1]["x"], 2.0)

    def test_random_delay_is_reported_to_the_planner(self):
        sim, algo = self._sim_and_algo({"a": [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]})
        sim.delay_probability = 1.0
        sim.time_forward(algo)
        self.assertEqual(sim.actual_paths["a"][-1]["x"], 0.0)
        self.assertEqual(len(algo.token["agents"]["a"]), 3)
        sim.time_forward(algo)
        self.assertEqual(algo.seen_delayed[-1], ["a"])

    def test_clear_pass_is_not_held(self):
        sim, algo = self._sim_and_algo({"a": [(0.0, 0.0), (2.0, 0.0)], "b": [(1.0, 1.5)]})
        sim.time_forward(algo)
        self.assertEqual(sim.actual_paths["a"][-1]["x"], 2.0)


class TestSimulationExecutesTimedWaypoints(unittest.TestCase):
    def test_one_time_sampled_waypoint_per_tick_and_planner_stamps(self):
        from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import _Waypoint
        from path_planning.multi_agent_planner.decentralized.neural_attf.simulation import Simulation

        # Arc budget 10 would swallow both short samples; time-sampled waypoints must
        # still be consumed exactly one per tick, stamped at their via_t fractions.
        sim = Simulation(tasks=[], agents=[{"name": "a", "start": (0.0, 0.0)}], velocity=10.0, timestep_duration=1.0)
        path = [
            _Waypoint((0.0, 0.0)),
            _Waypoint((2.0, 0.0), via=[(1.0, 0.0), (1.0, 0.0)], via_t=[0.5, 0.75]),
            _Waypoint((3.0, 0.0), via=[], via_t=[]),
        ]

        class _Algo:
            def __init__(self):
                self.token = {"agents": {"a": list(path)}}

            def time_forward(self, *a, **k):
                pass

            def get_token(self):
                return self.token

        algo = _Algo()
        sim.time_forward(algo)
        self.assertEqual(len(algo.token["agents"]["a"]), 2)
        got = [(p["t"], p["x"]) for p in sim.actual_paths["a"]]
        self.assertEqual(got, [(0, 0.0), (0.5, 1.0), (0.75, 1.0), (1.0, 2.0)])
        sim.time_forward(algo)
        self.assertEqual(sim.actual_paths["a"][-1], {"t": 2.0, "x": 3.0, "y": 0.0})



class TestSippGraphDynamicObstacles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_ = _build_grid_map()

    def _graph(self, schedule, horizon):
        return SippGraph(
            self.map_,
            dynamic_obstacles={"other": schedule},
            radius=RADIUS,
            velocity=VELOCITY,
            use_constraint_sweep=True,
            obstacle_horizon=horizon,
        )

    def test_nodes_exist(self):
        g = self._graph([_point(Q, 0.0)], None)
        self.assertIn(P, g.sipp_graph)
        self.assertIn(Q, g.sipp_graph)

    def test_wait_is_bounded_and_last_point_respects_horizon(self):
        # Wait at P for [0, 2], move P -> Q arriving at 4 (unit speed, 3 units in 2 s
        # is faster than nominal; the sweep uses the implied speed), rest at Q for 1 s.
        schedule = [_point(P, 0.0), _point(P, 2.0), _point(Q, 4.0)]
        g = self._graph(schedule, {"other": 1.0})
        p, q = g.sipp_graph[P], g.sipp_graph[Q]

        # P: blocked during the wait, free again well before the end of time.
        self.assertFalse(_safe_at(p, 1.0))
        self.assertTrue(math.isinf(p.interval_list[-1][1]))
        self.assertLess(p.interval_list[-1][0], 10.0)

        # Q: safe before the arrival, blocked over the 1 s hold, free afterwards.
        self.assertTrue(_safe_at(q, 0.5))
        self.assertFalse(_safe_at(q, 4.5))
        self.assertTrue(math.isinf(q.interval_list[-1][1]))
        self.assertLessEqual(q.interval_list[-1][0], 5.0 + 1e-6)

        # The resting footprint also blocks edges (self-loop at Q) for the hold only.
        self_loop = g.sipp_graph[(Q, Q)]
        self.assertTrue(self_loop.is_in_unsafe_interval(4.5, 4.5))
        self.assertFalse(self_loop.is_in_unsafe_interval(7.0, 7.0))

    def test_scalar_none_rests_forever(self):
        schedule = [_point(P, 0.0), _point(P, 2.0), _point(Q, 4.0)]
        g = self._graph(schedule, None)
        q = g.sipp_graph[Q]
        self.assertTrue(_safe_at(q, 0.5))
        self.assertFalse(any(math.isinf(b) for _, b in q.interval_list))
        self.assertTrue(g.sipp_graph[(Q, Q)].is_in_unsafe_interval(100.0, 100.0))
        # The bounded wait at P is unaffected by the horizon: P is free again later.
        self.assertTrue(math.isinf(g.sipp_graph[P].interval_list[-1][1]))

    def test_idle_single_point_blocks_from_zero(self):
        g = self._graph([_point(Q, 0.0)], {"other": None})
        self.assertEqual(g.sipp_graph[Q].interval_list, [])
        g2 = self._graph([_point(Q, 0.0)], {"other": 1.0})
        self.assertFalse(_safe_at(g2.sipp_graph[Q], 0.5))
        self.assertTrue(_safe_at(g2.sipp_graph[Q], 2.0))


if __name__ == "__main__":
    unittest.main()
