"""ICBS options for point agents: disjoint splitting and meta-agents with merge and restart."""
import math
import unittest


from path_planning.agent import Agent
from path_planning.common.world.collision_checking.sweep.agent_shape import Rectangle
from path_planning.multi_agent_planner.centralized.cbs.cbs import Constraints, Location, State, VertexConstraint
from path_planning.multi_agent_planner.centralized.icbs.icbs import ICBS, IEnvironment
from tests.test_cbs_shapes import MOVES, agents_of, cost, grid_map, with_body

# Two agents swapping ends of a row, one crossing it, one passing nearby.


def solve(m, agents, **options):
    env = IEnvironment(m, agents, astar_max_iterations=20000, heuristic_type="dijkstra")
    plan, info = ICBS(env, time_limit=30, max_iterations=5000, **options).search()
    states = {n: [State(p["t"], Location((p["x"], p["y"]))) for p in path] for n, path in plan.items()}
    return env, plan, info, states


class TestICBSOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = grid_map()
        cls.agents = agents_of(MOVES)
        cls.plain = solve(cls.m, cls.agents)

    def check(self, **options):
        env, plan, info, states = solve(self.m, self.agents, **options)
        self.assertTrue(info["success"], options)
        self.assertEqual(cost(plan), cost(self.plain[1]), options)          # still optimal
        self.assertFalse(env.get_conflicts(states, get_first_conflict=False), options)
        return info

    def test_plain_is_conflict_free(self):
        env, plan, info, states = self.plain
        self.assertTrue(info["success"])
        self.assertFalse(env.get_conflicts(states, get_first_conflict=False))
        self.assertEqual((info["meta_agents"], info["restarts"]), ([], 0))

    def test_disjoint_splitting(self):
        info = self.check(disjoint=True)
        self.assertEqual(info["meta_agents"], [])

    def test_merge_and_restart(self):
        info = self.check(merge_bound=0)                                    # merge on the first conflict
        self.assertGreaterEqual(info["restarts"], 1)
        self.assertTrue(info["meta_agents"])
        self.assertTrue(all(len(group) >= 2 for group in info["meta_agents"]))
        self.assertEqual(self.check(merge_bound=1000)["restarts"], 0)       # never reached: no merge

    def test_both(self):
        self.check(disjoint=True, merge_bound=1)

    def test_joint_search(self):
        env = IEnvironment(self.m, agents_of(MOVES[:2]), heuristic_type="dijkstra")
        free = {a: Constraints() for a in env.agent_dict}
        paths, costs = env.joint_search(["a0", "a1"], free)
        self.assertEqual(sum(costs.values()), 14)                           # 12 straight + one sidestep and back
        self.assertEqual([p[-1].location.point for p in paths.values()], [(7.5, 4.5), (1.5, 4.5)])
        self.assertFalse(env.get_conflicts(paths, get_first_conflict=False))
        # Each member obeys its own constraints: a0 has to be at (1.5, 5.5) at t = 1.
        free["a0"].must_be[1] = Location((1.5, 5.5))
        free["a1"].vertex_constraints.add(VertexConstraint(1, Location((6.5, 4.5))))
        paths, costs = env.joint_search(["a0", "a1"], free)
        self.assertEqual(paths["a0"][1].location.point, (1.5, 5.5))
        self.assertNotEqual(paths["a1"][1].location.point, (6.5, 4.5))
        self.assertFalse(env.joint_search(["a0", "a1"], free, stop_at=0.0))   # already past its time limit

    def test_goal_constraints_are_honoured(self):
        """An agent may not end its plan at its goal while a constraint still forbids it being there."""
        env = IEnvironment(self.m, agents_of([((1.5, 4.5), (3.5, 4.5))]), heuristic_type="dijkstra")
        goal = Location((3.5, 4.5))
        con = Constraints()
        self.assertEqual(con.free_from(goal), float("-inf"))
        con.vertex_constraints.add(VertexConstraint(5, goal))               # not at the goal at t = 5
        self.assertEqual(con.free_from(goal), 6)
        env.constraints = con
        path, path_cost = env.a_star.search("a0")
        self.assertEqual(path[-1].location, goal)
        self.assertGreaterEqual(path[-1].time, 6)                           # it used to stop at t = 2
        self.assertNotEqual(path[5].location, goal)
        self.assertEqual(path_cost, 6)                                      # 2 moves + 4 steps of waiting
        legacy = IEnvironment(self.m, agents_of([((1.5, 4.5), (3.5, 4.5))]), heuristic_type="dijkstra",
                              move_parked_agents=False)
        legacy.constraints = con
        path, path_cost = legacy.a_star.search("a0")
        self.assertEqual((path[-1].time, path_cost), (2, 2))                # switched off: it parks on arrival

    def test_bypass_search_is_bounded(self):
        env = IEnvironment(self.m, agents_of([((1.5, 4.5), (3.5, 4.5))]), heuristic_type="dijkstra")
        env.constraints = Constraints()
        self.assertTrue(env.a_star.search("a0", cost_limit=2)[0])           # the shortest path costs 2
        self.assertFalse(env.a_star.search("a0", cost_limit=1.5)[0])        # nothing cheaper exists

    def test_point_agents_only(self):
        agents = agents_of(MOVES[:2])
        with self.assertRaises(ValueError):                                 # disjoint splitting: points only
            ICBS(IEnvironment(self.m, with_body(agents, Agent.disc(0.4))), disjoint=True)
        with self.assertRaises(ValueError):                                 # meta-agents: points and balls
            ICBS(IEnvironment(self.m, with_body(agents, Agent(Rectangle(2, 1)))), merge_bound=3)
        ICBS(IEnvironment(self.m, with_body(agents, Agent.disc(0.4))), merge_bound=3)

    def test_meta_agents_of_discs(self):
        from path_planning.utils.checker import is_conflict_free
        agents = agents_of(MOVES[:3])                                        # (four discs in one group is slow)
        disc = Agent(0.4, max_rotation=math.pi)                             # discs that have to turn (one step a turn)
        env = IEnvironment(self.m, with_body(agents, disc), heuristic_type="dijkstra")
        plain, _ = ICBS(env, time_limit=30, max_iterations=5000).search()
        env = IEnvironment(self.m, with_body(agents, disc), heuristic_type="dijkstra")
        merged, info = ICBS(env, time_limit=30, max_iterations=5000, merge_bound=0).search()
        self.assertTrue(info["success"])
        self.assertTrue(info["meta_agents"])
        self.assertTrue(is_conflict_free(merged, 0.4))
        self.assertLessEqual(cost(merged), cost(plain))                     # a joint plan is never worse here
        paths, costs = env.joint_search(["a0", "a1"], {a: Constraints() for a in env.agent_dict})
        self.assertFalse(env.get_conflicts(paths, get_first_conflict=False))  # the pair's joint plan is collision-free
        for path in paths.values():                                         # discs move only the way they face
            for a, b in zip(path, path[1:]):
                if a.location != b.location:
                    self.assertEqual(b.ori, math.degrees(math.atan2(b.location[1] - a.location[1],
                                                                    b.location[0] - a.location[0])) % 360)
                    self.assertIn(a.ori, (b.ori, "free"))
        free = IEnvironment(self.m, with_body(agents, Agent.disc(0.4)), heuristic_type="dijkstra")   # instant turns
        merged, info = ICBS(free, time_limit=30, max_iterations=5000, merge_bound=0).search()
        self.assertTrue(info["success"] and info["meta_agents"])
        self.assertTrue(is_conflict_free(merged, 0.4))
        for path in merged.values():                                        # a heading is shown, no step spent turning
            self.assertTrue(all("yaw" in p for p in path))
            self.assertFalse(any((a["x"], a["y"]) == (b["x"], b["y"]) and a["yaw"] != b["yaw"]
                                 for a, b in zip(path, path[1:])))

    def test_goal_conflicts(self):
        """A disc parked on another's route: one branching either delays it or keeps everyone clear of it."""
        from path_planning.multi_agent_planner.centralized.cbs.cbs import CBS, Environment
        from path_planning.utils.checker import is_conflict_free
        moves = [((3.5, 4.5), (4.5, 4.5)), ((1.5, 4.5), (7.5, 4.5))]         # a0 parks on a1's row after one move
        results = {}
        for flag in (False, True):
            env = Environment(self.m, with_body(agents_of(moves), Agent.disc(0.4)), heuristic_type="dijkstra", goal_conflicts=flag)
            plan, info = CBS(env, time_limit=30, max_iterations=5000).search()
            self.assertTrue(info["success"])
            self.assertTrue(is_conflict_free(plan, 0.4))
            results[flag] = (cost(plan), info["total_iterations"])
        self.assertEqual(results[True][0], results[False][0])
        self.assertLessEqual(results[True][1], results[False][1])
        env = Environment(self.m, with_body(agents_of(moves), Agent.disc(0.4)), heuristic_type="dijkstra", goal_conflicts=True)
        root, _ = env.compute_solution()
        conflict = env.get_conflicts(root)
        self.assertEqual(conflict.parked[0], "a0")
        delayed, clear = env.branches(conflict)
        self.assertEqual(list(delayed), ["a0"])                             # a0 is not stopped there yet
        self.assertEqual(clear["a0"].deadline, conflict.time)               # or it is, and a1 keeps out for good
        self.assertTrue(clear["a1"].regions)
