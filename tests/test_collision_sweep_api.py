"""Map-level collision queries: Agent types, sweep_paths / agent_conflicts / get_constraint_segment, and the planner sweep cache."""
import unittest

import numpy as np


from path_planning.agent import Agent
from path_planning.common.world.collision_checking.sweep.agent_shape import Rectangle
from path_planning.common.world.collision_checking.sweep.disc_sweep import disc_agent_conflicts
from path_planning.multi_agent_planner.centralized.sipp.graph_generation import cached_collision_sweep
from path_planning.multi_agent_planner.sweep_cache import SweepCache, shared_sweep_cache
from path_planning.utils.util import read_graph_sampler_from_yaml, set_global_seed
from tests.helpers import MAP_2D



def build(backend):
    set_global_seed(0)
    m = read_graph_sampler_from_yaml(MAP_2D, use_discrete_space=False, sweep_backend=backend)
    m.set_roadmap_config(sample_num=250, num_neighbors=8.0, min_edge_length=0.0, max_edge_length=5.0)
    m.set_start([])
    m.set_goal([])
    m.generateRandomNodes(generate_grid_nodes=False)
    m.generate_roadmap("prm")
    m.set_collision_sweep()
    return m


def move(m, agent, p1, p2, **kw):
    """``sweep_paths`` for one move ``p1 -> p2``, or a wait held forever when ``p1 == p2``."""
    wait = tuple(p1) == tuple(p2)
    return m.sweep_paths(agent, [np.array([p1] if wait else [p1, p2])], goal_hold=np.inf if wait else 0.0, **kw)[0]


def windows(arrays):
    """``{element: [(t1, t2), ...]}`` for the vertex part and the edge part of a query result."""
    (vid, t1, t2), (src, tgt, e1, e2) = arrays
    v, e = {}, {}
    for i, a, b in zip(vid.tolist(), t1.tolist(), t2.tolist()):
        v.setdefault(i, []).append((a, b))
    for i, j, a, b in zip(src.tolist(), tgt.tolist(), e1.tolist(), e2.tolist()):
        e.setdefault((i, j), []).append((a, b))
    return v, e


class TestAgent(unittest.TestCase):
    def test_equal_types_share_a_key(self):
        a, b = Agent(Rectangle(2, 1), 1.5, 1.0, "cart"), Agent(Rectangle(2, 1), 1.5, 1.0, "cart")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, Agent(Rectangle(2, 1), 1.5, 1.0, "truck"))
        self.assertNotEqual(a, Agent(Rectangle(2, 1.5), 1.5, 1.0, "cart"))

    def test_round_trip_and_validation(self):
        for a in (Agent.disc(0.5, 1.0), Agent(Rectangle(2, 1, 0.3, z_height=1.0), None, 0.0, "cart")):
            self.assertEqual(Agent.from_dict(a.to_dict()), a)
        self.assertTrue(Agent.disc(0.5).is_ball)
        self.assertFalse(Agent(Rectangle(2, 1)).is_ball)
        with self.assertRaises(ValueError):
            Agent(-1.0)
        with self.assertRaises(TypeError):
            Agent("disc")
        with self.assertRaises(AttributeError):
            Agent.disc(0.5).max_speed = 2.0


class TestClearance(unittest.TestCase):
    """``Agent.clearance``: a buffer that expands the body radially."""

    @classmethod
    def setUpClass(cls):
        cls.m = build("cgal")

    def test_agent_field(self):
        a = Agent(Rectangle(2, 1), clearance=0.25)
        self.assertEqual((a.clearance, Agent(Rectangle(2, 1)).clearance), (0.25, 0.0))
        self.assertAlmostEqual(a.reach, a.radius + 0.25)
        self.assertNotEqual(a, Agent(Rectangle(2, 1)))                       # part of the type's identity
        self.assertEqual(Agent.from_dict(a.to_dict()), a)
        self.assertEqual(Agent.disc(0.3, clearance=0.2).clearance, 0.2)
        with self.assertRaises(ValueError):
            Agent(0.5, clearance=-0.1)

    def test_buffered_disc_is_a_bigger_disc(self):
        rng = np.random.default_rng(0)
        paths = [np.cumsum(rng.normal(0, 1.5, (5, 2)), 0) + rng.uniform(5, 25, 2) for _ in range(10)]
        thin, wide = Agent.disc(0.3, clearance=0.2), Agent.disc(0.5)
        for x, y in zip(self.m.agent_conflicts(thin, paths, goal_hold=np.inf, as_arrays=True),
                        self.m.agent_conflicts(wide, paths, goal_hold=np.inf, as_arrays=True)):
            np.testing.assert_allclose(x, y)
        i, j = self.m.edges[5]
        path = [np.array([self.m.nodes[i].current, self.m.nodes[j].current])]
        for x, y in zip(sum(self.m.sweep_paths(thin, path)[0], ()), sum(self.m.sweep_paths(wide, path)[0], ())):
            np.testing.assert_allclose(x, y)

    def test_polygons_keep_the_sum_of_their_clearances(self):
        # Two 1 x 1 squares head-on along x, 8 apart, each moving 4 in one step: their faces meet at t = 7/8.
        paths = [np.array([[0.0, 0.0], [4.0, 0.0]]), np.array([[8.0, 0.0], [4.0, 0.0]])]
        square = Rectangle(1, 1)
        touch = self.m.agent_conflicts(Agent(square), paths, first_only=True)[(0, 1)]
        self.assertAlmostEqual(touch, 7.0 / 8.0)
        early = self.m.agent_conflicts([Agent(square, clearance=0.3), Agent(square, clearance=0.1)], paths,
                                       first_only=True)[(0, 1)]
        self.assertAlmostEqual(early, touch - 0.4 / 8.0)                     # 0.4 sooner at a closing speed of 8
        lanes = [np.array([[0.0, 0.0], [6.0, 0.0]]), np.array([[6.0, 1.2], [0.0, 1.2]])]   # faces 0.2 apart
        self.assertFalse(self.m.agent_conflicts(Agent(square, clearance=0.09), lanes))
        self.assertTrue(self.m.agent_conflicts(Agent(square, clearance=0.11), lanes))
        mixed = [Agent(square, clearance=0.05), Agent.disc(0.5, clearance=0.2)]             # 0.25 > 0.2
        self.assertTrue(self.m.agent_conflicts(mixed, lanes))

    def test_clearance_widens_the_roadmap_sweep(self):
        i, j = self.m.edges[5]
        path = [np.array([self.m.nodes[i].current, self.m.nodes[j].current])]
        plain = set(self.m.sweep_paths(Agent(Rectangle(2, 1)), path)[0][0][0].tolist())
        buffered = set(self.m.sweep_paths(Agent(Rectangle(2, 1), clearance=0.5), path)[0][0][0].tolist())
        self.assertTrue(plain < buffered)


class TestMapQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = {b: build(b) for b in ("shapely", "cgal")}

    def edge_points(self, m, k=0):
        i, j = m.edges[k]
        return tuple(m.nodes[i].current), tuple(m.nodes[j].current)

    def test_ball_move_equals_legacy_query(self):
        for backend, m in self.maps.items():
            nodes = [tuple(n.current) for n in m.nodes]
            for k in range(0, 60, 3):
                p1, p2 = self.edge_points(m, k)
                lv, le = m.get_collision_sweep(p1, p2, 1.0, 0.7, use_interval=True, get_time_interval=True)
                v, e = windows(move(m, Agent.disc(0.7, 1.0), p1, p2))
                self.assertEqual({nodes[i]: w[0] for i, w in v.items()}, lv, backend)
                got = {(nodes[i], nodes[j]): w[0] for (i, j), w in e.items()}
                got.update({(p, p): w for p, w in lv.items()})          # the legacy self-loop keys
                self.assertEqual(got, le, backend)

    def test_wait_covers_forever(self):
        m = self.maps["cgal"]
        p1, _ = self.edge_points(m)
        (vid, t1, t2), _ = move(m, Agent.disc(0.5), p1, p1)
        self.assertIn(m._index_of(p1), vid.tolist())
        self.assertTrue(np.all(t1 == 0.0) and np.all(np.isinf(t2)))

    def test_both_backends_agree_for_polygons(self):
        agent = Agent(Rectangle(2, 1), 1.0, None, "cart")
        p1, p2 = self.edge_points(self.maps["cgal"], 5)
        a, b = (move(m, agent, p1, p2) for m in self.maps.values())
        for x, y in zip(a[0] + a[1], b[0] + b[1]):
            np.testing.assert_allclose(x, y, atol=1e-9)

    def test_clearance_grows_the_cover(self):
        m = self.maps["cgal"]
        agent = Agent(Rectangle(2, 1), 1.0, None, "cart")
        p1, p2 = self.edge_points(m, 5)
        plain, _ = windows(move(m, agent, p1, p2))
        grown, _ = windows(move(m, agent, p1, p2, other=0.6))
        exact, _ = windows(move(m, agent, p1, p2, other=Agent(Rectangle(1, 1)), clearance="minkowski"))
        self.assertTrue(set(plain) <= set(grown))
        self.assertTrue(set(plain) <= set(exact))
        self.assertGreater(len(grown), len(plain))
        with self.assertRaises(ValueError):
            move(m, agent, p1, p2, other=0.5, clearance="exact")

    def test_map_does_not_cache(self):
        m = self.maps["cgal"]
        p1, p2 = self.edge_points(m, 3)
        agent = Agent(Rectangle(2, 1), 1.0)
        self.assertIsNot(move(m, agent, p1, p2), move(m, agent, p1, p2))
        self.assertIsNot(m.get_collision_sweep(p1, p2, 1.0, 0.5, use_interval=True, get_time_interval=True),
                         m.get_collision_sweep(p1, p2, 1.0, 0.5, use_interval=True, get_time_interval=True))

    def test_planner_cache(self):
        m = build("cgal")
        p1, p2 = self.edge_points(m, 3)
        first = cached_collision_sweep(m, p1, p2, 1.0, 0.5)
        self.assertIs(cached_collision_sweep(m, p1, p2, 1.0, 0.5), first)            # shared across planners
        self.assertEqual(first, m.get_collision_sweep(p1, p2, 1.0, 0.5, use_interval=True, get_time_interval=True))
        self.assertIs(shared_sweep_cache(m), shared_sweep_cache(m))
        m.set_collision_sweep(force=True)                                            # rebinding renews it
        self.assertEqual(len(shared_sweep_cache(m)), 0)
        self.assertIsNot(cached_collision_sweep(m, p1, p2, 1.0, 0.5), first)
        small = SweepCache(max_entries=2)
        for k in range(3):
            small.get_or_compute(k, lambda: k)
        self.assertEqual(dict(small), {2: 2})                                        # emptied when full

    def test_sweep_paths_and_conflicts(self):
        m = self.maps["cgal"]
        p1, p2 = self.edge_points(m, 5)
        paths = [np.array([p1, p2]), np.array([p2, p1])]
        balls = [Agent.disc(0.5, 1.0), Agent.disc(0.8, 1.0)]
        self.assertEqual(len(m.sweep_paths(balls, paths)), 2)
        self.assertEqual(m.agent_conflicts(balls, paths), disc_agent_conflicts(paths, [0.5, 0.8], 1.0))
        mixed = m.agent_conflicts([balls[0], Agent(Rectangle(2, 1), 1.0)], paths)
        self.assertIn((0, 1), mixed)                                                 # head-on on one edge

    def test_prev_shifts_to_arrival(self):
        m = self.maps["cgal"]
        agent = Agent(Rectangle(2, 1), 1.0, 1.0, "cart")
        p0, p1 = self.edge_points(m, 5)
        _, p2 = self.edge_points(m, 9)
        (vid, t1, t2), (_, _, e1, _) = move(m, agent, p1, p2, prev=p0)
        self.assertTrue(np.all(t1 >= 0.0) and np.all(e1 >= 0.0) and np.all(t2 > 0.0))
        never_turns = Agent(Rectangle(2, 1), 1.0, 0.0, "cart")
        for x, y in zip(*(sum(move(m, never_turns, p1, p2, prev=q), ()) for q in (p0, None))):
            np.testing.assert_array_equal(x, y)                                      # prev ignored

    def test_constraint_segment(self):
        m = self.maps["cgal"]
        self.assertTrue(m.get_constraint_segment((0, 0), (2, 0), (2, 0), (0, 0), 1.0, 0.5))   # head-on swap
        self.assertFalse(m.get_constraint_segment((0, 0), (2, 0), (0, 3), (2, 3), 1.0, 0.5))  # parallel, apart
        # A waiter beside the end of the other's move: hit late in the step (missed when only t = 0 was checked).
        self.assertTrue(m.get_constraint_segment((4, 0.5), (4, 0.5), (0, 0), (4, 0), 1.0, 0.5))
        self.assertFalse(m.get_constraint_segment((0, 0), (1, 0), (5, 5), (6, 5), None, 0.0))

    def test_constraint_segment_any_shape(self):
        m = self.maps["cgal"]
        cart, disc = Agent(Rectangle(2, 1), 1.0, 1.0, "cart"), Agent.disc(0.4, 1.0)
        lanes = ((0, 0), (4, 0), (0, 1.2), (4, 1.2))                                  # parallel, 1.2 apart
        self.assertFalse(m.get_constraint_segment(*lanes, agents=cart))              # half-widths 0.5 clear
        self.assertTrue(m.get_constraint_segment(*lanes, agents=Agent.disc(cart.radius, 1.0)))
        wait = ((2, 1.0), (2, 1.0), (0, 0), (4, 0))                                  # a cart parked by a lane
        self.assertTrue(m.get_constraint_segment(*wait, agents=[cart, disc]))        # attitude unknown: ball
        self.assertFalse(m.get_constraint_segment(*wait, agents=[cart, disc], heading=[0.0, None]))
        self.assertTrue(m.get_constraint_segment(*wait, agents=[cart, disc], heading=[np.pi / 2, None]))
