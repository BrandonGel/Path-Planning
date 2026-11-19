"""CBS / ICBS with Agent bodies of any shape: one agent_conflicts pass per plan, one sweep per conflict."""
import math
import unittest

import numpy as np
import shapely
from shapely import affinity


from path_planning.agent import Agent
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.world.collision_checking.sweep.agent_shape import Box, Rectangle, Sphere
from path_planning.multi_agent_planner.centralized.cbs.cbs import CBS, Conflict, Environment
from path_planning.multi_agent_planner.centralized.icbs.icbs import ICBS, IEnvironment

CART = Agent(Rectangle(2, 1), max_rotation=math.pi, type="cart")  # 2 long, 1 wide; any turn takes it one step


DISC = Agent.disc(0.35, 1.0)                 # radius 0.35, speed 1: a move between cells takes 1
#: four crossing trips on the 9x9 grid, shared by the CBS / ICBS / CCBS / SIPP tests
MOVES = [((1.5, 4.5), (7.5, 4.5)), ((7.5, 4.5), (1.5, 4.5)), ((4.5, 1.5), (4.5, 7.5)), ((2.5, 5.5), (6.5, 3.5))]


def grid_map(size=(9, 9)):
    """An empty unit grid (nodes at cell centres, 4-connected)."""
    m = GraphSampler(bounds=[[0.0, float(s)] for s in size], resolution=1.0, start=[], goal=[],
                     use_discrete_space=True)
    m.set_roadmap_config(sample_num=0, num_neighbors=4.0 if len(size) == 2 else 6.0, min_edge_length=0.0,
                         max_edge_length=1.1)
    m.generateRandomNodes(generate_grid_nodes=True)
    m.generate_roadmap("prm")
    return m


def agents_of(moves, types=None):
    return [{"name": f"a{i}", "start": tuple(map(float, s)), "goal": tuple(map(float, g)),
             **({"agent": types[i]} if types else {})} for i, (s, g) in enumerate(moves)]


def with_body(agents, body):
    """``agents`` with ``body`` (an Agent) for every agent that has none of its own."""
    return [{"agent": body, **a} for a in agents]


def solve(m, agents, solver="cbs", **env_kw):
    env_cls, cls = (Environment, CBS) if solver == "cbs" else (IEnvironment, ICBS)
    env = env_cls(m, agents, astar_max_iterations=20000, heuristic_type="dijkstra", **env_kw)
    plan, info = cls(env, time_limit=30, max_iterations=2000).search()
    return env, plan, info


def cost(plan):
    return sum(len(path) - 1 for path in plan.values())


def rectangle_overlap(plan, length=2.0, width=1.0, samples=20):
    """Largest overlap area of any two rectangles over the plan, by sampling time (independent of the sweep
    code). With ``yaw`` in the plan: the body faces ``yaw`` and is its bounding disc during a turn step.
    Without: a move faces along itself; a wait keeps the last move's attitude, or the first move's."""
    names = list(plan)
    pts = {n: np.array([[p["x"], p["y"]] for p in plan[n]]) for n in names}
    ori = {n: [p.get("yaw") for p in plan[n]] for n in names}
    steps = max(len(p) for p in pts.values())

    def pose(p, t):
        k = min(int(t), len(p) - 1)
        a, b = p[k], p[min(k + 1, len(p) - 1)]
        moves = [(p[i], p[i + 1]) for i in range(len(p) - 1) if np.any(p[i] != p[i + 1])]
        if np.any(a != b):
            d = b - a
        else:
            before = [(u, v) for i, (u, v) in enumerate(zip(p[:-1], p[1:])) if i < k and np.any(u != v)]
            u, v = before[-1] if before else (moves[0] if moves else (np.zeros(2), np.array([1.0, 0.0])))
            d = v - u
        return a + (t - k) * (b - a), math.degrees(math.atan2(d[1], d[0]))

    def body(n, t):
        p, o = pts[n], ori[n]
        c, ang = pose(p, t)
        k = min(int(t), len(p) - 1)
        if o[k] is not None:
            k1 = min(k + 1, len(p) - 1)
            if o[k1] != o[k] and np.all(p[k] == p[k1]) and t > k:       # turning in place
                return shapely.Point(c[0], c[1]).buffer(math.hypot(length, width) / 2, 64)
            ang = o[k1] if np.any(p[k] != p[k1]) else o[k]
        box = shapely.box(-length / 2, -width / 2, length / 2, width / 2)
        return affinity.translate(affinity.rotate(box, ang, origin=(0, 0)), c[0], c[1])

    worst = 0.0
    for t in np.linspace(0, steps, steps * samples + 1):
        shapes = [body(n, t) for n in names]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                worst = max(worst, shapes[i].intersection(shapes[j]).area)
    return worst


class TestCBSShapes(unittest.TestCase):
    def test_head_on_rectangles(self):
        m = grid_map()
        agents = agents_of([((1.5, 4.5), (7.5, 4.5)), ((7.5, 4.5), (1.5, 4.5))])
        for solver in ("cbs", "icbs"):
            env, plan, info = solve(m, with_body(agents, CART), solver)
            self.assertTrue(info["success"], solver)
            self.assertFalse(env.get_conflicts(self._states(env, plan)), solver)
            self.assertLess(rectangle_overlap(plan), 1e-6, solver)
            self.assertGreater(cost(plan), 12, solver)                  # someone had to give way

    def test_rectangles_pass_where_bounding_discs_cannot(self):
        m = grid_map()
        lanes = [((1.5, 3.5), (7.5, 3.5)), ((7.5, 4.5), (1.5, 4.5))]     # adjacent rows, opposite directions
        agents = with_body(agents_of(lanes), CART)
        agents[1]["yaw"] = 180                                          # already facing its way
        _, carts, info = solve(m, agents)
        self.assertTrue(info["success"])
        self.assertEqual(cost(carts), 12)                               # straight through, side by side
        self.assertLess(rectangle_overlap(carts), 1e-6)
        _, carts, _ = solve(m, with_body(agents_of(lanes), CART))       # both start facing 0: a1 turns first
        self.assertEqual(cost(carts), 13)
        _, discs, info = solve(m, with_body(agents_of(lanes), Agent.disc(CART.radius)))  # bounding discs (r = 1.118)
        self.assertTrue(info["success"])
        self.assertGreater(cost(discs), 12)

    def test_parked_rectangle_keeps_its_attitude(self):
        m = grid_map()
        moves = [((1.5, 3.5), (7.5, 3.5)), ((4.5, 4.5), (4.5, 4.5))]     # a cart parked beside the lane
        _, plan, info = solve(m, with_body(agents_of(moves), CART))
        self.assertTrue(info["success"])
        self.assertEqual(cost(plan), 6)                                 # it never moves: it lies along +x
        self.assertLess(rectangle_overlap(plan), 1e-6)

    def test_mixed_shapes(self):
        m = grid_map()
        moves = [((1.5, 4.5), (7.5, 4.5)), ((4.5, 1.5), (4.5, 7.5)), ((7.5, 4.5), (1.5, 4.5))]
        types = [CART, Agent.disc(0.4), Agent(Rectangle(1, 1), type="pallet")]
        env, plan, info = solve(m, agents_of(moves, types))
        self.assertTrue(info["success"])
        self.assertFalse(env.get_conflicts(self._states(env, plan)))
        with self.assertRaises(ValueError):                             # a body for some, a point for others
            Environment(m, agents_of(moves[:2], [CART, None]))

    def test_3d_bodies(self):
        m = grid_map((5, 5, 3))
        moves = [((0.5, 2.5, 1.5), (4.5, 2.5, 1.5)), ((4.5, 2.5, 1.5), (0.5, 2.5, 1.5))]
        for types in ([Agent(Sphere(0.4))] * 2, [Agent(Box(1.6, 0.8, 0.8)), Agent(Sphere(0.4))]):
            env, plan, info = solve(m, agents_of(moves, types))
            self.assertTrue(info["success"])
            self.assertFalse(env.get_conflicts(self._states(env, plan)))
            self.assertGreater(cost(plan), 8)

    def test_fans(self):
        """A fan holds the move the agent made (which really collides) and otherwise only moves out of its
        vertex along roadmap edges; it may hold moves that would not collide, since an edge counts as
        covered whenever the other body reaches it during the step, wherever the agent is on it."""
        m = grid_map()
        agents = agents_of([((1.5, 4.5), (7.5, 4.5)), ((7.5, 4.5), (1.5, 4.5))])
        env = Environment(m, with_body(agents, CART), heuristic_type="dijkstra")
        plan, _ = env.compute_solution()
        c = env.get_conflicts(plan)
        self.assertEqual(c.type, Conflict.SWEEP)
        moves = {n: (plan[n][min(c.time, len(plan[n]) - 1)].location.point,
                     plan[n][min(c.time + 1, len(plan[n]) - 1)].location.point) for n in plan}
        self.assertTrue(m.agent_conflicts(CART, [np.array(moves[c.agent_1]), np.array(moves[c.agent_2])]))
        for name, src, fan in ((c.agent_1, c.location_1, c.location_1_f), (c.agent_2, c.location_2, c.location_2_f)):
            self.assertEqual(moves[name][0], src.point)
            reachable = {src.point} | {tuple(n.current) for n in m.get_neighbors(m._index_of(src.point))}
            self.assertIn(moves[name][1], {loc.point for loc in fan})
            self.assertTrue({loc.point for loc in fan} <= reachable)

    def test_heading_in_the_plan(self):
        m = grid_map()
        corner = [((1.5, 1.5), (3.5, 3.5))]                              # an L: 4 moves and one turn
        _, plan, info = solve(m, with_body(agents_of(corner), CART))
        path = plan["a0"]
        self.assertEqual(cost(plan), 5)
        self.assertTrue(all(set(p) == {"t", "x", "y", "yaw"} for p in path))
        turns = [(a, b) for a, b in zip(path, path[1:]) if a["yaw"] != b["yaw"]]
        self.assertEqual(len(turns), 1)
        a, b = turns[0]
        self.assertEqual((a["x"], a["y"], b["t"] - a["t"]), (b["x"], b["y"], 1))    # in place, one step
        for a, b in zip(path, path[1:]):                                 # it only drives the way it faces
            if (a["x"], a["y"]) != (b["x"], b["y"]):
                facing = math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"])) % 360
                self.assertEqual((a["yaw"], b["yaw"]), (facing, facing))
        # a start heading is kept: facing +y, going +x costs a turn first
        agents = agents_of([((1.5, 1.5), (3.5, 1.5))])
        agents[0]["yaw"] = 90
        _, plan, _ = solve(m, with_body(agents, CART))
        self.assertEqual([p["yaw"] for p in plan["a0"]], [90.0, 0.0, 0.0, 0.0])
        self.assertEqual(cost(plan), 3)
        # a finite max_rotation makes any body turn, a disc too (one step at the corner)
        _, discs, _ = solve(m, with_body(agents_of(corner), Agent(0.4, max_rotation=math.pi)))
        self.assertEqual(cost(discs), 5)
        self.assertEqual(len({p["yaw"] for p in discs["a0"]}), 2)
        # None or infinite: instant turns, for a disc and for a cart alike. No step is spent turning, and
        # the plan shows the direction of the move last made (the start heading, 0, before it).
        for body in (Agent.disc(0.4), Agent(0.4, max_rotation=math.inf), Agent(Rectangle(2, 1))):
            _, free, _ = solve(m, with_body(agents_of(corner), body))
            self.assertEqual(cost(free), 4, body)
            path = free["a0"]
            facing = [math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"])) % 360 for a, b in zip(path, path[1:])]
            self.assertEqual([p["yaw"] for p in path], [0.0] + facing, body)
        agents = agents_of([((1.5, 1.5), (3.5, 1.5))])                    # a start heading is shown until it leaves
        agents[0]["yaw"] = 90
        _, free, _ = solve(m, with_body(agents, Agent.disc(0.4)))
        self.assertEqual([p["yaw"] for p in free["a0"]], [90.0, 0.0, 0.0])
        _, points, _ = solve(m, agents_of(corner))                       # point particles have no heading
        self.assertEqual(cost(points), 4)
        self.assertFalse(any("yaw" in p for p in points["a0"]))
        with self.assertRaises(ValueError):                              # a body that can never turn
            Environment(m, with_body(agents_of(corner), Agent(Rectangle(2, 1), max_rotation=0.0)))
        # each agent follows its own rule
        both = agents_of([corner[0], ((1.5, 6.5), (3.5, 8.5))], [Agent(0.4, max_rotation=math.pi), Agent.disc(0.4)])
        _, mixed, _ = solve(m, both)
        self.assertEqual((len(mixed["a0"]) - 1, len(mixed["a1"]) - 1), (5, 4))
        in_place = lambda path: sum((a["x"], a["y"]) == (b["x"], b["y"]) and a["yaw"] != b["yaw"] for a, b in zip(path, path[1:]))
        self.assertEqual((in_place(mixed["a0"]), in_place(mixed["a1"])), (1, 0))

    def test_turning_rate(self):
        """``max_rotation`` is in radians per step: a turn through ``a`` takes ``ceil(a / max_rotation)`` steps."""
        m = grid_map()
        corner = [((1.5, 1.5), (3.5, 3.5))]                              # an L: 4 moves and a 90 degree turn
        for rate, steps in ((math.radians(15), 6), (math.radians(60), 2), (math.radians(90), 1), (math.radians(45), 2)):
            for body in (Agent(Rectangle(2, 1), max_rotation=rate), Agent(0.4, max_rotation=rate)):
                _, plan, info = solve(m, with_body(agents_of(corner), body))
                path = plan["a0"]
                self.assertEqual(cost(plan), 4 + steps, (rate, body))
                turning = [(a["yaw"], b["yaw"]) for a, b in zip(path, path[1:]) if (a["x"], a["y"]) == (b["x"], b["y"])]
                self.assertEqual(len(turning), steps)
                for before, after in turning:                             # never more than the rate in one step
                    self.assertLessEqual(abs((after - before + 180) % 360 - 180), math.degrees(rate) + 1e-6)
        # the headings it passes through at 15 degrees per step
        _, plan, _ = solve(m, with_body(agents_of(corner), Agent(0.4, max_rotation=math.radians(15))))
        yaws = [p["yaw"] for p in plan["a0"]]
        k = next(i for i in range(1, len(yaws)) if yaws[i] != yaws[i - 1])
        first, last = yaws[k - 1], yaws[k + 5]
        sign = 1 if (last - first + 180) % 360 - 180 > 0 else -1
        self.assertEqual(yaws[k - 1:k + 6], [(first + sign * 15 * i) % 360 for i in range(7)])
        # a start heading the other way round: 180 degrees at 90 per step takes 2 steps, then 2 moves
        agents = agents_of([((1.5, 1.5), (3.5, 1.5))])
        agents[0]["yaw"] = 180
        _, plan, _ = solve(m, with_body(agents, Agent(0.4, max_rotation=math.pi / 2)))
        self.assertEqual(cost(plan), 4)
        self.assertEqual([p["yaw"] for p in plan["a0"]][-3:], [0.0, 0.0, 0.0])
        # 3D: a box pitching up 90 degrees at 45 per step
        m3 = grid_map((5, 5, 3))
        _, plan, _ = solve(m3, with_body(agents_of([((0.5, 2.5, 0.5), (2.5, 2.5, 2.5))]), Agent(Box(1.6, 0.8, 0.8), max_rotation=math.pi / 4)))
        self.assertEqual(cost(plan), 6)                                  # 4 moves and 2 turn steps
        self.assertIn(45.0, {round(p["pitch"], 6) for p in plan["a0"]})
        # while it turns it is checked as its bounding disc, every step of the turn
        moves = [((2.5, 4.5), (4.5, 6.5)), ((7.5, 5.5), (1.5, 5.5))]
        env, plan, info = solve(m, with_body(agents_of(moves), Agent(Rectangle(2, 1), max_rotation=math.radians(30))))
        self.assertTrue(info["success"])
        self.assertLess(rectangle_overlap(plan), 1e-6)
        self.assertFalse(env.get_conflicts(self._states(env, plan)))

    def test_turning_needs_room(self):
        m = grid_map()
        # a0 turns at (4.5, 4.5) on its way up; a1 drives past on the next row while it would be turning
        moves = [((2.5, 4.5), (4.5, 6.5)), ((7.5, 5.5), (1.5, 5.5))]
        for solver in ("cbs", "icbs"):
            env, plan, info = solve(m, with_body(agents_of(moves), CART), solver)
            self.assertTrue(info["success"], solver)
            self.assertLess(rectangle_overlap(plan), 1e-6, solver)
            self.assertFalse(env.get_conflicts(self._states(env, plan)), solver)

    def test_3d_heading(self):
        m = grid_map((5, 5, 3))
        _, plan, info = solve(m, with_body(agents_of([((0.5, 2.5, 0.5), (2.5, 2.5, 2.5))]), Agent(Box(1.6, 0.8, 0.8), max_rotation=math.pi)))
        self.assertTrue(info["success"])
        self.assertEqual(cost(plan), 5)                                  # 4 moves and one turn (to pitch 90)
        self.assertTrue(all(set(p) == {"t", "x", "y", "z", "yaw", "pitch"} for p in plan["a0"]))
        self.assertEqual({(p["yaw"], p["pitch"]) for p in plan["a0"]}, {(0.0, 0.0), (0.0, 90.0)})
        agents = agents_of([((0.5, 2.5, 0.5), (2.5, 2.5, 0.5))])
        agents[0].update(agent=Agent(Box(1.6, 0.8, 0.8), max_rotation=math.pi), yaw=90, pitch=0)   # a start heading: turn first
        _, plan, _ = solve(m, agents)
        self.assertEqual([(p["yaw"], p["pitch"]) for p in plan["a0"]], [(90.0, 0.0)] + [(0.0, 0.0)] * 3)

    def test_agent_clearance(self):
        """Carts on adjacent rows touch as they pass; with a clearance they have to keep apart."""
        m = grid_map()
        lanes = [((1.5, 3.5), (7.5, 3.5)), ((7.5, 4.5), (1.5, 4.5))]
        agents = with_body(agents_of(lanes), CART)
        agents[1]["yaw"] = 180                                          # already facing its way
        _, touching, _ = solve(m, agents)
        self.assertEqual(cost(touching), 12)
        buffered = Agent(Rectangle(2, 1), type="cart", clearance=0.1)
        env, plan, info = solve(m, with_body(agents_of(lanes), buffered))
        self.assertTrue(info["success"])
        self.assertGreater(cost(plan), 12)
        self.assertEqual(env.agent_type("a0").clearance, 0.1)
        self.assertFalse(env.get_conflicts(self._states(env, plan)))
        self.assertLess(rectangle_overlap(plan, length=2.2, width=1.2), 1e-6)      # the buffered boxes stay apart

    def test_conflict_span(self):
        """Window (the default): one conflict per overlap, covering all its steps. Step: one per time step."""
        m = grid_map()
        agents = agents_of([((1.5, 4.5), (7.5, 4.5)), ((7.5, 4.5), (1.5, 4.5))])        # head-on, 2 long
        spans = {}
        for span in ("window", "step"):
            env = Environment(m, with_body(agents, CART), heuristic_type="dijkstra", conflict_span=span)
            root, _ = env.compute_solution()
            conflicts = env.get_conflicts(root, get_first_conflict=False)
            spans[span] = [[c.time] + [later.time for later in c.later] for c in conflicts]
            branch = env.branches(conflicts[0])[0]
            times = {e.time for con in branch.values() for e in con.edge_constraints}
            self.assertEqual(times, set(spans[span][0]))                 # one rule set per step it covers
            _, plan, info = solve(m, with_body(agents, CART), conflict_span=span)
            self.assertTrue(info["success"])
            self.assertLess(rectangle_overlap(plan), 1e-6)
        self.assertEqual(Environment(m, with_body(agents, CART)).conflict_span, "window")
        self.assertEqual(len(spans["window"]), 1)                        # the whole pass is one conflict
        self.assertGreater(len(spans["window"][0]), 1)
        self.assertEqual([t for [t] in spans["step"]], spans["window"][0])
        with self.assertRaises(ValueError):
            Environment(m, with_body(agents, CART), conflict_span="all")

    def test_point_agents_unchanged(self):
        m = grid_map()
        env, plan, info = solve(m, agents_of([((1.5, 4.5), (7.5, 4.5)), ((7.5, 4.5), (1.5, 4.5))]))
        self.assertTrue(info["success"])
        self.assertFalse(env.has_bodies)
        self.assertEqual(cost(plan), 14)                                # one sidestep and back

    @staticmethod
    def _states(env, plan):
        """The returned plan as the State lists ``get_conflicts`` reads."""
        from path_planning.multi_agent_planner.centralized.cbs.cbs import Location, State
        ori = lambda p: (p["yaw"], p["pitch"]) if "pitch" in p else p.get("yaw")
        return {n: [State(p["t"], Location(tuple(p[k] for k in ("x", "y", "z") if k in p)), ori(p)) for p in path]
                for n, path in plan.items()}
