"""CCBS (continuous time): unsafe intervals, the low-level search, and the solver with its options."""
import unittest


import numpy as np

from path_planning.agent import Agent
from path_planning.common.world.collision_checking.sweep.agent_shape import Box, Rectangle
from path_planning.multi_agent_planner.centralized.ccbs.ccbs import CCBS, Constraint, Environment, Move, INF
from tests.test_cbs_shapes import DISC, MOVES, agents_of, grid_map, with_body

# Two agents swapping ends of a row, one crossing it, one passing nearby.
OPTIONS = [{}, dict(use_cardinal=True), dict(use_cardinal=True, hlh="lp"), dict(use_cardinal=True, hlh="greedy"),
           dict(disjoint=True), dict(use_cardinal=True, disjoint=True, hlh="greedy")]


def solve(m, agents, **options):
    env = Environment(m, agents)
    plan, info = CCBS(env, time_limit=30, **options).search()
    return env, plan, info


def as_paths(env, plan):
    """A returned plan as the solver's ``{agent: ((vertex, time), ...)}``."""
    return {a: tuple((env._node_index(tuple(p[k] for k in "xyz" if k in p)), p["t"]) for p in rows) for a, rows in plan.items()}


def collision_free(env, plan) -> bool:
    return not env.find_conflicts(as_paths(env, plan))


class TestUnsafeIntervals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = grid_map()

    def env(self, body, moves=MOVES[:2]):
        return Environment(self.m, with_body(agents_of(moves), body))

    def collide(self, env, m1, m2) -> bool:
        return len(env._overlaps([env._piece("a0", m1), env._piece("a1", m2)])[0]) > 0

    def check_move_window(self, body):
        """Every start inside the window collides; the first start after it does not."""
        env = self.env(body)
        v = lambda x, y: env._node_index((x, y))
        mine = Move(v(3.5, 4.5), v(4.5, 4.5), 2.0, 3.0, None)                 # heading right along the row
        theirs = Move(v(4.5, 3.5), v(4.5, 4.5), 2.2, 3.2, None)               # coming up into the same cell
        self.assertTrue(self.collide(env, mine, theirs))
        c = env.constraint_for("a0", mine, "a1", theirs, 2.5)
        self.assertEqual((c.a, c.b, c.t1, c.positive), (mine.a, mine.b, 2.0, False))
        self.assertGreater(c.t2, c.t1)
        self.assertLessEqual(c.t2, theirs.t2 + 1e-9)
        for start in np.linspace(c.t1, c.t2 - 1e-4, 7):
            self.assertTrue(self.collide(env, mine._replace(t1=start, t2=start + 1.0), theirs), start)
        self.assertFalse(self.collide(env, mine._replace(t1=c.t2 + 1e-4, t2=c.t2 + 1.0001), theirs))
        return c

    def test_move_window_discs(self):
        self.check_move_window(DISC)

    def test_move_window_rectangles(self):
        """The bisection with the exact polygon test gives a window with the same property."""
        self.check_move_window(Agent(Rectangle(0.8, 0.5), 1.0))

    def test_wait_window(self):
        """A waiting agent may not be at its vertex while the other's body covers it."""
        env = self.env(DISC)
        v = lambda x, y: env._node_index((x, y))
        wait = Move(v(4.5, 4.5), v(4.5, 4.5), 0.0, 5.0, None)
        cross = Move(v(4.5, 3.5), v(4.5, 5.5), 2.0, 4.0, None)                # a 2-long pass through the vertex
        # (not a roadmap edge; the geometry is what matters here)
        c = env.constraint_for("a0", wait, "a1", cross, 2.5)
        self.assertEqual((c.a, c.b), (wait.a, wait.a))
        self.assertAlmostEqual(c.t1, 2.0 + 0.3, places=5)                     # centres 0.7 apart at 2.3 and 3.7
        self.assertAlmostEqual(c.t2, 4.0 - 0.3, places=5)

    def test_parked_agent(self):
        env = self.env(DISC)
        v = lambda x, y: env._node_index((x, y))
        mine = Move(v(3.5, 4.5), v(4.5, 4.5), 2.0, 3.0, None)
        parked = Move(v(4.5, 4.5), v(4.5, 4.5), 1.0, INF, None)
        c = env.constraint_for("a0", mine, "a1", parked, 2.3)
        self.assertEqual((c.t1, c.t2), (2.0, INF))                            # never again from then on
        back = env.constraint_for("a1", parked, "a0", mine, 2.3)             # and the parked one: not there meanwhile
        self.assertEqual((back.a, back.b), (parked.a, parked.a))
        self.assertLess(back.t1, back.t2)
        self.assertLessEqual(back.t2, 3.0 + 1e-6)


class TestLowLevel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = grid_map()
        cls.env = Environment(cls.m, with_body(agents_of([((1.5, 4.5), (4.5, 4.5))]), DISC))
        cls.v = staticmethod(lambda x, y: cls.env._node_index((x, y)))

    def find(self, *constraints):
        return self.env.sipp.find_path("a0", list(constraints))

    def test_free_path(self):
        path, cost = self.find()
        self.assertEqual(cost, 3.0)
        self.assertEqual([v for v, _ in path], [self.v(x, 4.5) for x in (1.5, 2.5, 3.5, 4.5)])

    def test_vertex_interval(self):
        """It may not be at (2.5, 4.5) during [0.5, 2.5): it waits at the start or goes round."""
        block = self.v(2.5, 4.5)
        path, cost = self.find(Constraint("a0", block, block, 0.5, 2.5))
        self.assertGreater(cost, 3.0)
        for (u, t1), (w, t2) in zip(path, path[1:]):
            for vertex, lo, hi in ((u, t1, t1 if u != w else t2), (w, t2, t2)):
                if vertex == block:
                    self.assertFalse(lo < 2.5 - 1e-9 and hi > 0.5 + 1e-9, path)

    def test_edge_window(self):
        """It may not start its first move during [0, 1.25): it starts it at 1.25 (waiting is cheaper
        than the detour)."""
        a, b = self.v(1.5, 4.5), self.v(2.5, 4.5)
        path, cost = self.find(Constraint("a0", a, b, 0.0, 1.25))
        self.assertAlmostEqual(cost, 4.25)
        self.assertEqual(path[:3], ((a, 0.0), (a, 1.25), (b, 2.25)))

    def test_goal_is_safe(self):
        """It may not finish at a goal it would have to leave: it arrives after the unsafe interval."""
        goal = self.v(4.5, 4.5)
        path, cost = self.find(Constraint("a0", goal, goal, 5.0, 6.0))
        self.assertEqual(path[-1][0], goal)
        self.assertAlmostEqual(cost, 6.0)
        path, cost = self.find(Constraint("a0", goal, goal, 5.0, INF))        # never safe: no path
        self.assertIsNone(path)

    def test_landmark(self):
        """A positive constraint: it has to start a given move inside a window, even off its way."""
        a, b = self.v(2.5, 4.5), self.v(2.5, 5.5)
        path, cost = self.find(Constraint("a0", a, b, 3.0, 4.0, True))
        k = next(k for k, ((u, _), (w, _)) in enumerate(zip(path, path[1:])) if (u, w) == (a, b))
        self.assertTrue(3.0 - 1e-9 <= path[k][1] < 4.0)
        self.assertEqual(path[-1][0], self.v(4.5, 4.5))
        self.assertAlmostEqual(cost, 3.0 + 1.0 + 3.0)                         # leave at 3, up, then 3 moves to the goal
        # A window wholly forbidden by a negative constraint on the same move cannot be met.
        path, _ = self.find(Constraint("a0", a, b, 3.0, 4.0, True), Constraint("a0", a, b, 2.0, 5.0))
        self.assertIsNone(path)


class TestHeadedWaits(unittest.TestCase):
    """A wait constraint on a body that is not a ball holds only for the heading it was worked out with.

    a0 is a long thin rod; a1 a short one going up the column x = 5.5. At V = (4.5, 4.5) a0 facing right
    reaches into a1's lane, facing up it does not."""
    LONG, SHORT = Agent(Rectangle(1.9, 0.2), 1.0), Agent(Rectangle(0.5, 0.2), 1.0)

    @classmethod
    def setUpClass(cls):
        cls.m = grid_map()

    def env(self, a0_start, a0_yaw):
        agents = [{"name": "a0", "start": a0_start, "goal": (4.5, 4.5), "agent": self.LONG, "yaw": a0_yaw},
                  {"name": "a1", "start": (5.5, 2.5), "goal": (5.5, 6.5), "agent": self.SHORT, "yaw": 90.0}]
        env = Environment(self.m, agents)
        return env, (lambda x, y: env._node_index((x, y)))

    def test_constraint_keeps_heading(self):
        env, v = self.env((4.5, 3.5), 90.0)
        V, up = v(4.5, 4.5), Move(v(5.5, 3.5), v(5.5, 4.5), 1.0, 2.0, None)
        c = env.constraint_for("a0", Move(V, V, 0.0, 10.0, 0.0), "a1", up, 1.65)
        self.assertEqual((c.a, c.b, c.facing), (V, V, 0.0))
        self.assertAlmostEqual(c.t1, 1.65, places=5)
        disc = Environment(self.m, with_body(agents_of(MOVES[:2]), DISC))                # a ball: any heading
        w = disc._node_index((4.5, 4.5))
        cross = Move(disc._node_index((4.5, 3.5)), disc._node_index((4.5, 5.5)), 2.0, 4.0, None)
        self.assertIsNone(disc.constraint_for("a0", Move(w, w, 0.0, 5.0, None), "a1", cross, 2.5).facing)

    def test_other_heading_still_allowed(self):
        """Banned at V facing right over [1.65, 2): it may still arrive there from below (facing up) at 1."""
        env, v = self.env((4.5, 3.5), 90.0)
        V = v(4.5, 4.5)
        path, cost = env.sipp.find_path("a0", [Constraint("a0", V, V, 1.65, 2.0, facing=0.0)])
        self.assertAlmostEqual(cost, 1.0)
        self.assertEqual(path, ((v(4.5, 3.5), 0.0), (V, 1.0)))

    def test_banned_heading_waits(self):
        """Coming from the left it faces right at V: it may not be there over [0.5, 2), so it waits (cheaper
        than going round below)."""
        env, v = self.env((3.5, 4.5), 0.0)
        V = v(4.5, 4.5)
        path, cost = env.sipp.find_path("a0", [Constraint("a0", V, V, 0.5, 2.0, facing=0.0)])
        self.assertAlmostEqual(cost, 2.0)
        self.assertEqual(path, ((v(3.5, 4.5), 0.0), (v(3.5, 4.5), 1.0), (V, 2.0)))

    def test_headed_goal_needs_its_heading_safe_for_good(self):
        """Facing right is never safe at the goal again, facing up is: it has to arrive from below."""
        env, v = self.env((3.5, 4.5), 0.0)
        V = v(4.5, 4.5)
        path, cost = env.sipp.find_path("a0", [Constraint("a0", V, V, 0.5, INF, facing=0.0)])
        self.assertAlmostEqual(cost, 3.0)
        self.assertEqual(path[-2][0], v(4.5, 3.5))

    def test_landmark_at_headed_vertex(self):
        """It has to start V -> (4.5, 5.5) in [3, 4) but may never be at V facing right: it gets to V from
        below by 3, goes up, and comes back down (facing down) to its goal."""
        env, v = self.env((3.5, 4.5), 0.0)
        V, above = v(4.5, 4.5), v(4.5, 5.5)
        path, cost = env.sipp.find_path("a0", [Constraint("a0", V, V, 0.5, INF, facing=0.0),
                                               Constraint("a0", V, above, 3.0, 4.0, True)])
        self.assertAlmostEqual(cost, 5.0)
        k = next(k for k, ((u, _), (w, _)) in enumerate(zip(path, path[1:])) if (u, w) == (V, above))
        self.assertEqual(path[k - 1][0], v(4.5, 3.5))                           # entered V from below
        self.assertTrue(3.0 - 1e-9 <= path[k][1] < 4.0)

    def test_solver(self):
        for start, yaw in (((4.5, 3.5), 90.0), ((3.5, 4.5), 0.0)):
            env, _ = self.env(start, yaw)
            for options in OPTIONS:
                plan, info = CCBS(env, time_limit=30, **options).search()
                self.assertTrue(info["success"], options)
                self.assertTrue(collision_free(env, plan), options)


class TestCCBS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = grid_map()

    def test_options_agree(self):
        costs = []
        for options in OPTIONS:
            env, plan, info = solve(self.m, with_body(agents_of(MOVES), DISC), **options)
            self.assertTrue(info["success"], options)
            self.assertTrue(collision_free(env, plan), options)
            self.assertAlmostEqual(info["cost"], sum(rows[-1]["t"] for rows in plan.values()))
            costs.append(info["cost"])
        self.assertTrue(all(abs(c - costs[0]) < 1e-6 for c in costs), costs)
        self.assertGreater(costs[0], 24.0)                                    # 4 x 6 moves without anyone in the way

    def test_plan_format(self):
        env, plan, info = solve(self.m, with_body(agents_of(MOVES[:2]), DISC), use_cardinal=True)
        for rows in plan.values():
            self.assertTrue(all(set(p) == {"t", "x", "y", "yaw"} for p in rows))
            self.assertTrue(all(b["t"] > a["t"] for a, b in zip(rows, rows[1:])))
            self.assertEqual(rows[0]["t"], 0.0)
        self.assertEqual((plan["a0"][0]["x"], plan["a0"][-1]["x"]), (1.5, 7.5))
        self.assertEqual(info["total_iterations"], info["high_level_nodes_expanded"])

    def test_parked_agent_on_route(self):
        """a0 parks on a1's row after one move: a1 goes round it (or a0 lets it by first)."""
        moves = [((3.5, 4.5), (4.5, 4.5)), ((1.5, 4.5), (7.5, 4.5))]
        for options in ({}, dict(use_cardinal=True, disjoint=True, hlh="lp")):
            env, plan, info = solve(self.m, with_body(agents_of(moves), DISC), **options)
            self.assertTrue(info["success"])
            self.assertTrue(collision_free(env, plan))
            self.assertAlmostEqual(info["cost"], 1.0 + 8.0)                   # a1: two extra moves round a0

    def test_speeds(self):
        """A move takes length / max_speed; without a speed, 1 time unit."""
        for body, expected in ((Agent.disc(0.35, 2.0), 3.0), (Agent.disc(0.35, 0.5), 12.0), (Agent.disc(0.35), 6.0)):
            _, plan, info = solve(self.m, with_body(agents_of(MOVES[:1]), body))
            self.assertAlmostEqual(info["cost"], expected)
        fast, slow = Agent.disc(0.35, 2.0), Agent.disc(0.35, 1.0)             # per-agent speeds
        env, plan, info = solve(self.m, agents_of(MOVES[:2], [fast, slow]), use_cardinal=True)
        self.assertTrue(info["success"] and collision_free(env, plan))
        self.assertGreater(plan["a1"][-1]["t"], plan["a0"][-1]["t"])

    def test_shapes_and_clearance(self):
        lanes = [((1.5, 3.5), (7.5, 3.5)), ((7.5, 4.5), (1.5, 4.5))]          # passing in adjacent rows
        slim, wide = Agent(Rectangle(2.0, 0.8), 1.0), Agent(Rectangle(2.0, 1.4), 1.0)
        env, plan, info = solve(self.m, with_body(agents_of(lanes), slim))
        self.assertTrue(info["success"] and collision_free(env, plan))
        self.assertAlmostEqual(info["cost"], 12.0)                            # they fit side by side
        env, plan, info = solve(self.m, with_body(agents_of(lanes), wide), use_cardinal=True)
        self.assertTrue(info["success"] and collision_free(env, plan))
        self.assertGreater(info["cost"], 12.0)                                # too wide: one gives way
        buffered = Agent(Rectangle(2.0, 0.8), 1.0, clearance=0.3)             # the clearance makes the slim one wide
        env, plan, info = solve(self.m, with_body(agents_of(lanes), buffered), use_cardinal=True)
        self.assertTrue(info["success"] and collision_free(env, plan))
        self.assertGreater(info["cost"], 12.0)
        env, plan, info = solve(self.m, agents_of(MOVES[:2], [DISC, slim]), use_cardinal=True)   # a disc and a rectangle
        self.assertTrue(info["success"] and collision_free(env, plan))
        self.assertTrue(all("yaw" in p for rows in plan.values() for p in rows))

    def test_3d(self):
        m3 = grid_map((5, 5, 3))
        moves = [((0.5, 2.5, 1.5), (4.5, 2.5, 1.5)), ((4.5, 2.5, 1.5), (0.5, 2.5, 1.5))]
        env, plan, info = solve(m3, agents_of(moves, [Agent.disc(0.35, 1.0), Agent(Box(0.8, 0.5, 0.5), 1.0)]), use_cardinal=True)
        self.assertTrue(info["success"] and collision_free(env, plan))
        self.assertGreater(info["cost"], 8.0)
        self.assertTrue(all(set(p) == {"t", "x", "y", "z", "yaw", "pitch"} for rows in plan.values() for p in rows))

    def test_arguments(self):
        with self.assertRaises(ValueError):                                   # every agent needs a body
            Environment(self.m, agents_of(MOVES[:2]))
        env = Environment(self.m, with_body(agents_of(MOVES[:2]), DISC))
        with self.assertRaises(ValueError):                                   # the bound comes from cardinal conflicts
            CCBS(env, hlh="lp")
        with self.assertRaises(ValueError):
            CCBS(env, use_cardinal=True, hlh="simplex")
        with self.assertRaises(ValueError):                                   # off the roadmap
            Environment(self.m, with_body(agents_of([((1.2, 4.5), (7.5, 4.5))]), DISC))

    def test_time_limit(self):
        env = Environment(self.m, with_body(agents_of(MOVES), DISC))
        plan, info = CCBS(env, time_limit=1e-9).search()
        self.assertEqual((plan, info["success"]), ({}, False))

    def test_solve_mapf(self):
        from path_planning.multi_agent_planner.mapf_solver import solve_mapf
        costs = {}
        for name, extra in (("ccbs", {}), ("ccbs", {"ccbs_use_cardinal": True}),
                            ("ccbs", {"ccbs_use_cardinal": True, "ccbs_disjoint": True, "ccbs_hlh": "greedy"})):
            config = {"mapf_solver_name": name, "time_limit": 30, "agent_radius": 0.35, "agent_velocity": 1.0,
                      "max_iterations": 10000, "heuristic_type": "dijkstra", **extra}
            summary = solve_mapf(self.m, agents_of(MOVES), config)
            self.assertTrue(summary["success"])
            self.assertFalse(summary["collision"])
            costs[(name, len(extra))] = sum(rows[-1]["t"] for rows in summary["schedule"].values())
        self.assertEqual(len({round(c, 6) for c in costs.values()}), 1)
