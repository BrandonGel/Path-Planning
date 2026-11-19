"""SIPP with agent bodies: shapes, clearance, speeds, 3D, body-carrying obstacles and the input rules."""
import random
import unittest


from path_planning.agent import Agent
from path_planning.common.world.collision_checking.sweep.agent_shape import Box, Rectangle
from path_planning.multi_agent_planner.centralized.ccbs.ccbs import Environment as ExactCheck
from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner
from tests.test_cbs_shapes import DISC, MOVES, agents_of, grid_map, with_body

LANES = [((1.5, 3.5), (7.5, 3.5)), ((7.5, 4.5), (1.5, 4.5))]          # passing in adjacent rows, 1 apart


def solve(m, agents, **options):
    random.seed(0)
    return SippPlanner(m, agents, heuristic_type="dijkstra", time_limit=30, **options).compute_plan()


def cost(plan) -> float:
    return sum(rows[-1]["t"] for rows in plan.values())


def collision_free(m, agents, plan) -> bool:
    """No two bodies overlap along the plan (the exact test CCBS uses, at the agents' true shapes)."""
    exact = ExactCheck(m, agents)
    paths = {a: tuple((exact._node_index(tuple(p[k] for k in "xyz" if k in p)), p["t"]) for p in rows)
             for a, rows in plan.items()}
    return not exact.find_conflicts(paths)


class TestSippShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = grid_map()

    def check(self, agents, **options):
        plan, info = solve(self.m, agents, **options)
        self.assertTrue(info["success"])
        self.assertTrue(collision_free(self.m, agents, plan))
        return plan

    def test_size_decides_whether_lanes_can_pass(self):
        """Discs of radius 0.45 pass in rows 1 apart; at 0.55, or as wide rectangles, one has to wait."""
        self.assertEqual(cost(self.check(with_body(agents_of(LANES), Agent.disc(0.45, 1.0)))), 12.0)
        self.assertGreater(cost(self.check(with_body(agents_of(LANES), Agent.disc(0.55, 1.0)))), 12.0)
        self.assertGreater(cost(self.check(with_body(agents_of(LANES), Agent(Rectangle(2.0, 1.4), 1.0)))), 12.0)

    def test_slim_rectangles_pass_in_adjacent_lanes(self):
        """A rectangle on a vertex is tested at the heading it arrived with, not as its bounding ball:
        2 x 0.8 carts (bounding radius 1.08) pass in rows 1 apart without waiting, as CCBS finds (exact_vertices=True)."""
        agents = with_body(agents_of(LANES), Agent(Rectangle(2.0, 0.8), 1.0))
        self.assertEqual(cost(self.check(agents, exact_vertices=True)), 12.0)

    def test_heading_intervals_refine_the_bounding_ball(self):
        """Per-heading safe intervals contain the bounding ball's, and differ by heading."""
        from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippGraph
        cart = Agent(Rectangle(2.0, 0.8), 1.0)
        passing = [{"t": 0.0, "x": 1.5, "y": 3.5}, {"t": 6.0, "x": 7.5, "y": 3.5}, {"t": 7.0, "x": 7.5, "y": 3.5}]
        graph = SippGraph(self.m, {"o": passing}, cart, obstacle_horizon=1.0, exact_vertices=True)
        v = (4.5, 4.5)                                               # one row up from the obstacle's lane
        ball = graph.sipp_graph[v].interval_list
        along, across = graph.vertex_node(v, 0.0).interval_list, graph.vertex_node(v, 90.0).interval_list
        self.assertNotEqual(ball, [(0, float("inf"))])               # the bounding ball reaches the lane
        self.assertEqual(along, [(0, float("inf"))])                 # parallel to the lane: never touched
        self.assertNotEqual(across, [(0, float("inf"))])            # across it: reaches into the lane
        covered = lambda intervals, t: any(a <= t <= b for a, b in intervals)
        for t in (0.0, 1.0, 2.5, 3.0, 3.5, 4.0, 6.5, 9.0):
            if covered(ball, t):
                self.assertTrue(covered(along, t) and covered(across, t), t)

    def test_exact_vertices_switch(self):
        """By default (exact_vertices=False) a rectangle on a vertex is checked as its bounding ball: still
        collision-free, but the slim carts can no longer pass side by side."""
        agents = with_body(agents_of(LANES), Agent(Rectangle(2.0, 0.8), 1.0))
        plan, info = solve(self.m, agents)
        self.assertTrue(info["success"] and collision_free(self.m, agents, plan))
        self.assertGreater(cost(plan), 12.0)

    def test_clearance_grows_the_body(self):
        thin = Agent.disc(0.45, 1.0)
        self.assertEqual(cost(self.check(with_body(agents_of(LANES), thin))), 12.0)
        buffered = Agent.disc(0.45, 1.0, clearance=0.1)              # 0.55 each with the buffer
        self.assertGreater(cost(self.check(with_body(agents_of(LANES), buffered))), 12.0)

    def test_polygons_and_mixed_bodies(self):
        self.check(with_body(agents_of(MOVES), Agent(Rectangle(0.9, 0.5), 1.0)))
        mixed = [DISC, Agent(Rectangle(0.9, 0.5), 1.0), Agent.disc(0.3, 2.0), Agent(Rectangle(0.6, 0.6), 0.5)]
        planner = SippPlanner(self.m, agents_of(MOVES, mixed), heuristic_type="dijkstra")
        random.seed(0)
        plan, info = planner.compute_plan()
        self.assertTrue(info["success"] and collision_free(self.m, agents_of(MOVES, mixed), plan))
        self.assertEqual(len(planner._tables), 4)                    # one safe-interval table per body

    def test_speeds(self):
        """A move takes length / max_speed; without a speed, one time unit."""
        for body, expected in ((Agent.disc(0.35, 2.0), 3.0), (Agent.disc(0.35, 0.5), 12.0), (Agent.disc(0.35), 6.0)):
            plan, info = solve(self.m, with_body(agents_of(MOVES[:1]), body))
            self.assertAlmostEqual(cost(plan), expected)

    def test_3d(self):
        m3 = grid_map((5, 5, 3))
        moves = [((0.5, 2.5, 1.5), (4.5, 2.5, 1.5)), ((4.5, 2.5, 1.5), (0.5, 2.5, 1.5)), ((2.5, 0.5, 1.5), (2.5, 4.5, 1.5))]
        agents = agents_of(moves, [DISC, Agent(Box(0.8, 0.5, 0.5), 1.0), Agent(Box(0.6, 0.6, 0.6), 1.0)])
        plan, info = solve(m3, agents)
        self.assertTrue(info["success"] and collision_free(m3, agents, plan))
        self.assertTrue(all(set(p) == {"t", "x", "y", "z", "yaw", "pitch"} for rows in plan.values() for p in rows))

    def test_plan_headings(self):
        """Bodies get the heading of the move they last made; before moving, the start heading (0 unless
        given)."""
        agents = with_body(agents_of([((1.5, 4.5), (1.5, 6.5))]), DISC)
        plan, _ = solve(self.m, agents)
        self.assertEqual([p["yaw"] for p in plan["a0"]], [0.0, 90.0, 90.0])
        agents[0]["yaw"] = 45.0
        plan, _ = solve(self.m, agents)
        self.assertEqual([p["yaw"] for p in plan["a0"]], [45.0, 90.0, 90.0])
        plan, _ = solve(self.m, agents_of(MOVES[:2]))               # point particles: no heading
        self.assertTrue(all(set(p) == {"t", "x", "y"} for rows in plan.values() for p in rows))

    def test_obstacle_bodies(self):
        """A plain schedule has the planned agent's body; a dict entry brings its own."""
        agents = with_body(agents_of(MOVES[:1]), DISC)                 # along row 4.5
        parked = [{"t": 0.0, "x": 4.5, "y": 5.5}]                     # one row up, for good
        plan, info = solve(self.m, agents, dynamic_obstacles={"o": parked})
        self.assertEqual(cost(plan), 6.0)                             # 0.35 + 0.35 < 1: straight past
        big = {"o": {"agent": Agent.disc(0.8, 1.0), "schedule": parked}}
        plan, info = solve(self.m, agents, dynamic_obstacles=big)
        self.assertTrue(info["success"])
        self.assertGreater(cost(plan), 6.0)                           # 0.8 + 0.35 > 1: round it
        self.assertTrue(all(abs(p["y"] - 5.5) + abs(p["x"] - 4.5) > 1.0 for p in plan["a0"]))

    def test_inputs(self):
        with self.assertRaises(ValueError):                           # bodies and points cannot be mixed
            SippPlanner(self.m, [{**agents_of(MOVES[:1])[0], "agent": DISC}, agents_of(MOVES[1:2])[0] | {"name": "a1"}])
        with self.assertRaises(TypeError):                            # the body comes from the agent dicts
            SippPlanner(self.m, agents_of(MOVES[:1]), radius=0.5, velocity=1.0)


class TestSippTableBuild(unittest.TestCase):
    """A new ``SippGraph`` blocks its dynamic obstacles in one pass; ``init_intervals`` blocks them piece
    by piece. Both must give the same table."""

    @classmethod
    def setUpClass(cls):
        cls.m = grid_map()

    def obstacles(self, body, count=6, steps=25):
        """Random walks with waits, time-sampled at uneven steps (so some moves are slower than the body)."""
        rng = random.Random(3)
        nodes = self.m.nodes
        out = {}
        for k in range(count):
            u, t = rng.randrange(len(nodes)), rng.choice([0.0, 0.5])
            rows = [{"t": t, "x": nodes[u].current[0], "y": nodes[u].current[1]}]
            for _ in range(steps):
                if rng.random() < 0.7:
                    u = rng.choice([n for n in self.m.road_map[u]])
                t += rng.choice([1.0, 1.0, 1.5])
                rows.append({"t": t, "x": nodes[u].current[0], "y": nodes[u].current[1]})
            out[f"o{k}"] = {"agent": body, "schedule": rows} if k % 2 else rows
        return out

    def array_rows(self, graph, speed):
        """The interval lists the compiled search reads: ``(per vertex, per edge)``."""
        data = graph.arrays(speed, lambda length: length / speed)
        nd_lo, nd_hi, nd_n, ed_lo, ed_hi, ed_n = data[5:]
        cut = lambda lo, hi, n, count: [(lo[k, :n[k]].tolist(), hi[k, :n[k]].tolist()) for k in range(count)]
        return cut(nd_lo, nd_hi, nd_n, len(self.m.nodes)), cut(ed_lo, ed_hi, ed_n, len(ed_n))

    @staticmethod
    def records(graph):
        return {key: (list(r) if isinstance(key[0], tuple) else r.interval_list) for key, r in graph.sipp_graph.items()}

    def same_tables(self, agent, **options):
        from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippGraph
        obstacles = self.obstacles(agent)
        horizon = {"o0": 3.0, "o1": None}
        one_pass = SippGraph(self.m, obstacles, agent, obstacle_horizon=horizon, **options)
        by_piece = SippGraph(self.m, None, agent, obstacle_horizon=horizon, **options)
        by_piece.init_intervals(obstacles)
        self.assertEqual(list(one_pass.sipp_graph), list(by_piece.sipp_graph))
        expected = self.records(by_piece)
        self.assertEqual(self.records(one_pass), expected)
        self.assertGreater(sum(bool(r) for key, r in expected.items() if isinstance(key[0], tuple)), 50)
        starts = lambda graph: [type(r.interval_list[0][0]) for r in graph._nobj[:len(self.m.nodes)] if r.interval_list]
        self.assertEqual(starts(one_pass), starts(by_piece))        # the start of time stays the int 0
        self.assertEqual(one_pass._pieces, by_piece._pieces)
        self.assertEqual(one_pass._blockers, by_piece._blockers)
        # The arrays the compiled search reads come straight from the one pass: no record is made for them.
        fresh = SippGraph(self.m, obstacles, agent, obstacle_horizon=horizon, **options)
        self.assertEqual(self.array_rows(fresh, agent.max_speed), self.array_rows(by_piece, agent.max_speed))
        self.assertEqual(len(dict.keys(fresh.sipp_graph)), 0)

    def test_discs(self):
        self.same_tables(Agent.disc(0.35, 1.0))

    def test_rectangles_per_heading(self):
        self.same_tables(Agent(Rectangle(2.0, 0.8), 1.0), exact_vertices=True)

    def test_a_later_plan_is_blocked_on_top(self):
        """Blocking after the one pass (a committed plan) changes the records and the arrays alike."""
        from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippGraph, State
        agent = Agent.disc(0.35, 1.0)
        obstacles = self.obstacles(agent)
        plan = [State((1.5, 3.5), 0.0), State((2.5, 3.5), 1.0), State((3.5, 3.5), 2.0)]
        costs = [(0.0, 1.0), (0.0, 1.0), (0, 0)]
        one_pass, by_piece = SippGraph(self.m, obstacles, agent), SippGraph(self.m, None, agent)
        by_piece.init_intervals(obstacles)
        for graph in (one_pass, by_piece):
            self.array_rows(graph, 1.0)                     # (the arrays exist before the plan is blocked)
            graph.update_intervals([plan], [costs], ["a"])
        self.assertEqual(self.array_rows(one_pass, 1.0), self.array_rows(by_piece, 1.0))
        self.assertEqual(self.records(one_pass), self.records(by_piece))

    def test_swaps_are_indexed_for_agents_without_a_body(self):
        """An agent without a body checks swaps against the obstacles' steps, also when an obstacle has a body."""
        from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippGraph
        step = [{"t": 0.0, "x": 1.5, "y": 3.5}, {"t": 1.0, "x": 2.5, "y": 3.5}]
        expected = {((1.5, 3.5), (2.5, 3.5), 0.0, 1.0)}
        self.assertEqual(SippGraph(self.m, {"o": step}).swaps, expected)
        self.assertEqual(SippGraph(self.m, {"o": {"agent": Agent.disc(0.35, 1.0), "schedule": step}}).swaps, expected)
        self.assertEqual(SippGraph(self.m, {"o": step}, Agent.disc(0.35, 1.0)).swaps, set())
