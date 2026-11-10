"""Direction zones: one-way regions and angle-restricted regions of the roadmap."""

import heapq
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from path_planning.common.environment.map.direction_zones import (  # noqa: E402
    DirectionZone, parse_direction_zone, parse_direction_zones, segment_allowed, violated_zone,
)
from path_planning.common.environment.map.graph_sampler import GraphSampler  # noqa: E402
from path_planning.multi_agent_planner.mapf_solver import solve_mapf  # noqa: E402
from path_planning.utils.checker import check_collision, check_direction_anomaly, check_solution_full  # noqa: E402
from path_planning.utils.util import (  # noqa: E402
    max_agent_radius, normalize_agent_specs, read_graph_sampler_from_yaml, set_global_seed,
)

BACKENDS = ("cgal", "shapely")
# eastbound lane on the row y = 5.5; the tight tolerance also closes the band to crossing
LANE = {"x": [2.0, 11.0], "y": [5.2, 5.8], "direction_deg": 0, "tolerance_deg": 30, "name": "east lane"}
# same lane, but vertical moves may still cross it (only westbound is forbidden)
LANE_CROSSABLE = dict(LANE, tolerance_deg=90, name="east lane (crossable)")


def _grid_map(backend="auto", agents=(), zones=None, neighbors=4, size=13.0):
    set_global_seed(1)
    m = GraphSampler(bounds=[[0.0, size], [0.0, size]], resolution=1.0, start=[], goal=[],
                     use_discrete_space=True, sweep_backend=backend)
    m.inflate_obstacles(radius=max_agent_radius(list(agents), 0.3) + np.sqrt(2) / 2)
    m.set_parameters(sample_num=0, num_neighbors=float(neighbors), min_edge_len=0.0,
                     max_edge_len=1.1 if neighbors <= 4 else 1.5)
    m.set_start([a["start"] for a in agents])
    m.set_goal([a["goal"] for a in agents])
    m.generate_roadmap(m.generateRandomNodes(generate_grid_nodes=True))
    m.set_constraint_sweep()
    if zones is not None:
        m.set_direction_zones(zones)
    return m


def _solve(m, agents, solver="sipp", **cfg):
    config = {"mapf_solver_name": solver, "time_limit": 60, "max_iterations": 2000, "agent_radius": 0.3,
              "agent_velocity": 1.0, "heuristic_type": "euclidean" if solver == "sipp" else "dijkstra"}
    config.update(cfg)
    set_global_seed(1)
    return solve_mapf(m, agents, config)


def _pts(m):
    return [tuple(n.current) for n in m.nodes]


def _directed_shortest(m, start, goal):
    """Brute-force Dijkstra over the directed roadmap (edge lengths)."""
    pts = _pts(m)
    s, g = pts.index(tuple(start)), pts.index(tuple(goal))
    dist = {s: 0.0}
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == g:
            return d
        if d > dist.get(u, math.inf):
            continue
        for v in m.road_map[u]:
            nd = d + math.dist(pts[u], pts[v])
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return math.inf


class TestDirectionZoneGeometry(unittest.TestCase):
    def test_one_way_rule(self):
        z = parse_direction_zone({"x": [0, 4], "y": [0, 4], "direction_deg": 90, "tolerance_deg": 45})
        self.assertFalse(z.bidirectional)
        self.assertTrue(z.allows((1, 1), (1, 2)))          # north
        self.assertTrue(z.allows((1, 1), (2, 2)))          # north-east, exactly 45 deg off
        self.assertFalse(z.allows((1, 1), (2, 1)))         # east
        self.assertFalse(z.allows((1, 2), (1, 1)))         # south
        self.assertTrue(z.allows((6, 2), (6, 1)))          # outside: unconstrained
        self.assertFalse(z.allows((5, 1), (4, 1)))         # touches the boundary: constrained
        self.assertTrue(z.allows((1, 1), (1, 1)))          # zero length

    def test_angle_rule(self):
        z = parse_direction_zone({"bounds": [[0, 4], [0, 4]], "angles_deg": [0, 90]})
        self.assertTrue(z.bidirectional)
        for q in ((2, 1), (0, 1), (1, 2), (1, 0)):
            self.assertTrue(z.allows((1, 1), q), q)
        for q in ((2, 2), (0, 2), (2, 0), (0, 0)):
            self.assertFalse(z.allows((1, 1), q), q)
        z45 = parse_direction_zone({"x": [0, 4], "y": [0, 4], "angles_deg": [45]})
        self.assertTrue(z45.allows((1, 1), (2, 2)) and z45.allows((2, 2), (1, 1)))
        self.assertFalse(z45.allows((1, 1), (2, 0)))

    def test_parsing_and_specs(self):
        zs = parse_direction_zones([
            {"x": [0, 1], "y": [0, 1], "direction_deg": [0, 90], "name": "a"},
            {"x_min": 2, "x_max": 3, "y_min": 0, "y_max": 1, "angles": [0], "tolerance": 5},
        ])
        self.assertEqual(zs[0].tolerance_deg, 45.0)
        self.assertEqual(zs[1].tolerance_deg, 5.0)
        self.assertEqual(parse_direction_zones([z.to_spec() for z in zs]), zs)
        self.assertEqual(parse_direction_zone({"x": [0, 1], "y": [0, 1], "direction_deg": -90}).headings_deg, (270.0,))
        with self.assertRaises(ValueError):
            parse_direction_zone({"x": [0, 1], "y": [0, 1]})
        with self.assertRaises(ValueError):
            parse_direction_zone({"x": [0, 1], "y": [0, 1], "direction_deg": 0, "angles_deg": [0]})
        with self.assertRaises(ValueError):
            DirectionZone(0, 0, 0, 1, (0,), 10)

    def test_overlap_is_intersection(self):
        zs = parse_direction_zones([
            {"x": [0, 4], "y": [0, 4], "direction_deg": 0, "tolerance_deg": 90},   # no westward
            {"x": [0, 4], "y": [0, 4], "angles_deg": [0]},                         # horizontal only
        ])
        self.assertTrue(segment_allowed(zs, (1, 1), (2, 1)))
        self.assertFalse(segment_allowed(zs, (2, 1), (1, 1)))
        self.assertFalse(segment_allowed(zs, (1, 1), (1, 2)))
        self.assertIs(violated_zone(zs, (1, 1), (1, 2)), zs[1])


class TestGraphSamplerDirectionZones(unittest.TestCase):
    def test_one_way_lane_filters_edges(self):
        m = _grid_map()
        n_full = len(m.edges)
        full = set(m.edges)
        m.set_direction_zones([LANE])
        pts = _pts(m)
        self.assertLess(len(m.edges), n_full)
        self.assertTrue(set(m.edges) <= full)
        for a, b in m.edges:
            self.assertTrue(m.edge_allowed(pts[a], pts[b]))
        lane = [(a, b) for a, b in m.edges if pts[a][1] == 5.5 and pts[b][1] == 5.5 and 2 <= pts[a][0] <= 11
                and 2 <= pts[b][0] <= 11]
        self.assertTrue(lane)
        for a, b in lane:
            self.assertGreater(pts[b][0], pts[a][0])           # eastbound only
            self.assertNotIn((b, a), m.edge_indices_dict)
            self.assertFalse(m.is_bidirectional(a, b))
        # edge tables stay aligned with the adjacency
        self.assertEqual(sum(len(r) for r in m.road_map), len(m.edges))
        self.assertEqual([len(r) for r in m.road_map], [len(w) for w in m.road_map_edge_weights])
        pred = m.predecessors()
        self.assertEqual({(u, v) for v, us in enumerate(pred) for u in us}, set(m.edges))

    def test_crossable_lane_keeps_vertical_edges(self):
        m = _grid_map(zones=[LANE_CROSSABLE])
        pts = _pts(m)
        i = pts.index((5.5, 5.5))
        nbrs = {pts[j] for j in m.road_map[i]}
        self.assertEqual(nbrs, {(6.5, 5.5), (5.5, 6.5), (5.5, 4.5)})

    def test_axis_zone_drops_diagonals(self):
        m = _grid_map(neighbors=8)
        zone = {"x": [4.0, 9.0], "y": [4.0, 9.0], "angles_deg": [0, 90]}
        n_full = len(m.edges)
        m.set_direction_zones([zone])
        pts = _pts(m)
        z = parse_direction_zone(zone)
        diag_touching = [(a, b) for a, b in m.edges
                         if pts[a][0] != pts[b][0] and pts[a][1] != pts[b][1] and z.intersects_segment(pts[a], pts[b])]
        self.assertEqual(diag_touching, [])
        self.assertLess(len(m.edges), n_full)
        # an angle zone keeps the adjacency symmetric
        self.assertTrue(all((b, a) in m.edge_indices_dict for a, b in m.edges))
        # diagonals elsewhere survive
        self.assertIn(pts.index((2.5, 2.5)), m.road_map[pts.index((1.5, 1.5))])

    def test_replace_and_clear_restore_edges(self):
        m = _grid_map()
        full = list(m.edges)
        v0 = m.direction_zones_version
        m.set_direction_zones([LANE])
        lane_edges = list(m.edges)
        m.set_direction_zones([LANE_CROSSABLE])
        self.assertGreater(len(m.edges), len(lane_edges))
        m.set_direction_zones([])
        self.assertEqual(m.edges, full)
        self.assertEqual(m.direction_zones_version, v0 + 3)

    def test_zones_set_before_roadmap_generation(self):
        m = GraphSampler(bounds=[[0.0, 13.0], [0.0, 13.0]], resolution=1.0, start=[], goal=[], use_discrete_space=True)
        m.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
        m.set_direction_zones([LANE])
        m.generate_roadmap(m.generateRandomNodes(generate_grid_nodes=True))
        self.assertEqual(set(m.edges), set(_grid_map(zones=[LANE]).edges))

    def test_yaml_pickle_and_pruned_copy_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            yml = os.path.join(d, "map.yaml")
            with open(yml, "w") as f:
                yaml.safe_dump({"map": {"dimensions": [13, 13], "obstacles": [], "direction_zones": [LANE]},
                                "agents": [{"name": "a", "start": [1, 1], "goal": [5, 5]}]}, f)
            m = read_graph_sampler_from_yaml(yml, use_discrete_space=True)
            self.assertEqual([z.name for z in m.direction_zones], ["east lane"])
            m.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
            m.generate_roadmap(m.generateRandomNodes(generate_grid_nodes=True))
            m.set_constraint_sweep()
            edges = list(m.edges)
            pkl = os.path.join(d, "graph.pkl")
            m.save_graph_sampler(pkl)
            m2 = GraphSampler(bounds=[[0.0, 13.0], [0.0, 13.0]], resolution=1.0, start=[], goal=[], use_discrete_space=True)
            m2.load_graph_sampler(pkl)
            self.assertEqual(m2.edges, edges)
            self.assertEqual(m2.direction_zones, m.direction_zones)
            m2.set_direction_zones([])                      # the pickle kept the unfiltered adjacency
            self.assertEqual(set(m2.edges), set(_grid_map().edges))
            pruned = m.create_pruned_copy(list(range(len(m.nodes))))
            self.assertEqual(set(pruned.edges), set(edges))
            self.assertEqual(pruned.direction_zones, m.direction_zones)

    def test_rotate_rejects_zones(self):
        m = _grid_map(zones=[LANE])
        with self.assertRaises(NotImplementedError):
            m.rotate(1, (0, 1))


class TestSweepOneWayEdges(unittest.TestCase):
    """A one-way edge is reported under its own directed key only, and its exact
    refinement models the direction that exists."""

    def test_directed_keys(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                m = _grid_map(backend, zones=[LANE])
                for use_interval in (False, True):
                    res = m.get_constraint_sweep((5.5, 4.5), (5.5, 6.5), 1.0, 0.6, use_interval=use_interval)
                    keys = set(res[1]) if use_interval else set(res)   # interval form: (vertices, edges)
                    self.assertIn(((5.5, 5.5), (6.5, 5.5)), keys)
                    self.assertNotIn(((6.5, 5.5), (5.5, 5.5)), keys)
                    pts = _pts(m)
                    for a, b in keys:
                        if a != b:
                            self.assertIn((pts.index(a), pts.index(b)), m.edge_indices_dict)

    def test_refinement_uses_existing_direction(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                m = _grid_map(backend)
                sweep = m.constraint_sweep
                pts = _pts(m)
                a, b = pts.index((5.5, 5.5)), pts.index((6.5, 5.5))
                idx = next(i for i, e in sweep.edge_indices.items() if set(e) == {a, b})
                self.assertEqual(sweep._edge_motion(idx), sweep.edge_indices[idx])     # bidirectional: canonical
                m.set_direction_zones([dict(LANE, direction_deg=180)])                # westbound only: max -> min
                sweep = m.constraint_sweep
                idx = next(i for i, e in sweep.edge_indices.items() if set(e) == {a, b})
                self.assertEqual(sweep.edge_directed_map[idx], [(b, a)])
                self.assertEqual(sweep._edge_motion(idx), (b, a))


class TestSippDirectionZones(unittest.TestCase):
    AGENTS = [{"name": "a", "start": [9.5, 5.5], "goal": [3.5, 5.5]},
              {"name": "b", "start": [1.5, 5.5], "goal": [11.5, 5.5]}]

    def test_one_way_lane_forces_detour(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                agents = normalize_agent_specs([dict(a) for a in self.AGENTS])
                free = _solve(_grid_map(backend, agents), agents)
                m = _grid_map(backend, agents, zones=[LANE])
                s = _solve(m, agents)
                self.assertTrue(free["success"] and s["success"])
                self.assertTrue(check_direction_anomaly(free["schedule"], [LANE], verbose=False))  # head-on without the rule
                self.assertEqual(check_direction_anomaly(s["schedule"], m.direction_zones, verbose=False), [])
                self.assertEqual(len(check_collision(s["schedule"], s["agent_footprints"], verbose=False)), 0)
                self.assertGreater(s["makespan"], free["makespan"])
                self.assertEqual(s["direction_zones"], [z.to_spec() for z in m.direction_zones])
                full = check_solution_full(s["schedule"], s["agent_footprints"], verbose=False,
                                           velocity=s["agent_velocities"], direction_zones=m.direction_zones)
                self.assertEqual(full["direction_violations"], [])

    def test_single_agent_cost_is_directed_shortest_path(self):
        for zones in ([LANE], [LANE_CROSSABLE], [{"x": [4.0, 9.0], "y": [4.0, 9.0], "angles_deg": [0, 90]}]):
            for heuristic in ("euclidean", "dijkstra"):
                with self.subTest(zone=zones[0].get("name", "axis"), heuristic=heuristic):
                    agents = normalize_agent_specs([{"name": "a", "start": [9.5, 5.5], "goal": [3.5, 6.5]}])
                    m = _grid_map("auto", agents, zones=zones, neighbors=8)
                    s = _solve(m, agents, heuristic_type=heuristic)
                    self.assertTrue(s["success"])
                    self.assertAlmostEqual(s["makespan"], _directed_shortest(m, (9.5, 5.5), (3.5, 6.5)), places=6)
                    self.assertEqual(check_direction_anomaly(s["schedule"], zones, verbose=False), [])

    def test_aligned_rectangle_with_turning_and_sink_goal(self):
        # goal in the middle of the closed lane: reachable only through the one-way edge into it
        agents = normalize_agent_specs([{
            "name": "r", "start": [1.5, 5.5], "goal": [6.5, 5.5], "velocity": 1.0, "turning_speed": math.pi / 2,
            "heading_deg": 90.0, "shape": {"type": "rectangle", "width": 1.6, "height": 0.6}}])
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                m = _grid_map(backend, agents, zones=[dict(LANE, x=[2.0, 6.2])])
                pts = _pts(m)
                g = pts.index((6.5, 5.5))
                self.assertEqual({pts[u] for u in m.predecessors()[g]} & {(5.5, 5.5)}, {(5.5, 5.5)})
                s = _solve(m, agents)
                self.assertTrue(s["success"], s.get("failure_reason"))
                self.assertEqual(check_direction_anomaly(s["schedule"], m.direction_zones, verbose=False), [])
                self.assertAlmostEqual(s["makespan"], 5.0 + 1.0, places=6)   # 5 m + one 90-degree turn

    def test_goal_only_reachable_by_one_way_edges(self):
        # the whole row is eastbound-only and closed to crossing: the goal's only
        # incoming edge is one-way and it has no outgoing edge at all inside the zone
        zone = {"x": [0.0, 13.0], "y": [5.2, 5.8], "direction_deg": 0, "tolerance_deg": 30}
        agents = normalize_agent_specs([{"name": "r", "start": [2.5, 5.5], "goal": [8.5, 5.5], "velocity": 1.0,
                                         "shape": {"type": "rectangle", "width": 1.2, "height": 0.6}}])
        m = _grid_map("auto", agents, zones=[zone])
        s = _solve(m, agents)
        self.assertTrue(s["success"], s.get("failure_reason"))
        self.assertAlmostEqual(s["makespan"], 6.0, places=6)
        back = normalize_agent_specs([{"name": "r", "start": [8.5, 5.5], "goal": [2.5, 5.5]}])
        self.assertFalse(_solve(_grid_map("auto", back, zones=[zone]), back)["success"])


class TestOtherSolversDirectionZones(unittest.TestCase):
    AGENTS = TestSippDirectionZones.AGENTS

    def test_cbs_and_icbs_respect_one_way_lane(self):
        for solver in ("cbs", "icbs"):
            with self.subTest(solver=solver):
                agents = normalize_agent_specs([dict(a) for a in self.AGENTS])
                m = _grid_map("auto", agents, zones=[LANE])
                s = _solve(m, agents, solver=solver)
                self.assertTrue(s["success"])
                self.assertEqual(check_direction_anomaly(s["schedule"], m.direction_zones, verbose=False), [])
                self.assertEqual(len(check_collision(s["schedule"], 0.3, verbose=False)), 0)

    def test_cbs_goal_distance_is_directed(self):
        from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment
        agents = normalize_agent_specs([dict(a) for a in self.AGENTS])
        m = _grid_map("auto", agents, zones=[LANE])
        env = Environment(m, agents, radius=0.3, velocity=1.0, heuristic_type="dijkstra")
        for start in ((9.5, 5.5), (5.5, 4.5), (1.5, 1.5)):
            self.assertAlmostEqual(env._dijkstra_heuristic(start, (3.5, 5.5)),
                                   _directed_shortest(m, start, (3.5, 5.5)), places=9)

    def test_ccbs_respects_one_way_lane(self):
        agents = normalize_agent_specs([dict(a) for a in self.AGENTS])
        m = _grid_map("auto", agents, zones=[LANE])
        s = _solve(m, agents, solver="ccbs")
        self.assertTrue(s["success"])
        self.assertEqual(check_direction_anomaly(s["schedule"], m.direction_zones, verbose=False), [])

    def _integer_map(self, zones):
        """LaCAM works on integer node coordinates."""
        set_global_seed(1)
        m = GraphSampler(bounds=[[0.0, 13.0], [0.0, 13.0]], resolution=1.0, start=[], goal=[], use_discrete_space=False)
        m.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
        m.set_start([(9.0, 5.0), (1.0, 5.0)])
        m.set_goal([(3.0, 5.0), (11.0, 5.0)])
        nodes = m.generate_custom_nodes([(float(x), float(y)) for x in range(1, 12) for y in range(1, 12)],
                                        filter_points=False)
        m.generate_roadmap(nodes)
        m.set_constraint_sweep()
        m.set_direction_zones(zones)
        return m

    def test_lacam_respects_one_way_lane(self):
        from path_planning.multi_agent_planner.centralized.lacam.dist_table import DistTable
        from path_planning.multi_agent_planner.centralized.lacam.lacam import LaCAM
        from path_planning.multi_agent_planner.centralized.lacam.utility import (
            get_in_neighbors, get_neighbors, set_starts_goals_config,
        )
        zone = {"x": [2.0, 10.0], "y": [4.8, 5.2], "direction_deg": 0, "tolerance_deg": 30}
        m = self._integer_map([zone])
        self.assertEqual(get_neighbors(m, (5, 5)), [(6, 5)])
        self.assertEqual(get_in_neighbors(m, (5, 5)), [(4, 5)])
        # hop distance TO the goal (3, 5) from the node just east of it: all the way round
        table = DistTable(m, (3, 5))
        self.assertEqual(table.get((2, 5)), 1)
        self.assertGreater(table.get((4, 5)), 10)
        starts, goals = set_starts_goals_config([(9.0, 5.0), (1.0, 5.0)], [(3.0, 5.0), (11.0, 5.0)])
        planner = LaCAM()
        config = planner.solve(m, starts, goals, seed=0, time_limit_ms=20000, max_iterations=5000, verbose=0)
        solution, _ = planner.get_solution_dict(config)
        self.assertTrue(solution)
        self.assertEqual(check_direction_anomaly(solution, [zone], verbose=False), [])
        for path in solution.values():
            self.assertGreater(len(path), 1)


class TestNeuralATTFSnap(unittest.TestCase):
    def test_snap_never_offers_reverse_endpoint(self):
        from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import NeuralATTF
        m = _grid_map(zones=[LANE])
        attf = NeuralATTF.__new__(NeuralATTF)
        attf.graph_map = m
        attf.velocity = 1.0
        from scipy.spatial import KDTree
        attf._node_kdtree = KDTree(np.asarray(_pts(m), dtype=float))
        attf._max_edge_len = lambda: 1.0
        attf._snap_to_node = lambda pt: min(_pts(m), key=lambda q: math.dist(q, pt))
        on_lane = attf._snap_start_candidates((5.8, 5.5))
        self.assertEqual([c[0] for c in on_lane], [(6.5, 5.5)])
        free = attf._snap_start_candidates((5.8, 8.5))
        self.assertEqual({c[0] for c in free}, {(5.5, 8.5), (6.5, 8.5)})


class TestVisualizerDirectionZones(unittest.TestCase):
    def test_overlay_and_arrows_draw(self):
        import matplotlib
        matplotlib.use("Agg")
        from path_planning.common.visualizer.visualizer_2d import Visualizer2D
        m = _grid_map(zones=[LANE, {"x": [3.0, 6.0], "y": [8.0, 11.0], "angles_deg": [0, 90]}])
        vis = Visualizer2D()
        vis.plot_grid_map(m)
        n_before = len(vis.ax.texts)
        vis.plot_road_map(m, m.nodes, m.road_map, map_frame=False)
        one_way = sum(1 for a, b in m.edges if (b, a) not in m.edge_indices_dict)
        self.assertEqual(len(vis.ax.texts) - n_before, one_way)       # one arrow annotation per one-way edge
        self.assertGreaterEqual(len(vis.ax.patches), 2)
        vis.close()


if __name__ == "__main__":
    unittest.main()
