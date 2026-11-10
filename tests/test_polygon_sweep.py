"""Convex-polygon footprints (regular polygons, rectangles) in the sweep backends.

Checks the exact polygon sweep of ``CGAL_Sweep`` / ``ShapelySweep`` against a
time-sampled Shapely brute force, cross-backend agreement, bidirectional
reuse, the disc-vs-polygon containment bounds, the shape algebra, and the
GraphSampler pass-through.

Run with:  python -m pytest tests/test_polygon_sweep.py -q
"""

import math
import random
import unittest

import numpy as np
import shapely
from shapely import LineString, Point, Polygon

from path_planning.utils.agent_shape import (
    AlignedClearance,
    AlignedShape,
    ConvexPolygon,
    Disc,
    convex_hull,
    heading_key,
    heading_of,
    minkowski_sum,
    moving_point_interval,
    oriented_pair_clearance,
    pair_clearance,
    path_headings,
    rectangle,
    regular_polygon,
    schedule_headings,
    shape_from_spec,
    square,
)
from path_planning.utils.cgal_sweep import CGAL_Sweep
from path_planning.utils.shapely_sweep import ShapelySweep

try:
    from path_planning.common.environment.map.graph_sampler import GraphSampler

    _HAS_SAMPLER = True
except Exception:  # pragma: no cover
    _HAS_SAMPLER = False


# ----------------------------------------------------------------------------- fixtures


def _random_roadmap(rng, n_vertices=60, box=10.0, k=3):
    """Random 2D point roadmap with k-nearest undirected edges (both directions listed)."""
    verts = [(round(rng.uniform(0, box), 3), round(rng.uniform(0, box), 3)) for _ in range(n_vertices)]
    pts = np.asarray(verts)
    edges = set()
    for i in range(n_vertices):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.argsort(d)[1 : k + 1]:
            edges.add((i, int(j)))
            edges.add((int(j), i))
    return verts, sorted(edges)


def _random_queries(rng, verts, n, box=10.0):
    queries = []
    for _ in range(n):
        roll = rng.random()
        u = verts[rng.randrange(len(verts))] if roll < 0.5 else (rng.uniform(0, box), rng.uniform(0, box))
        if roll < 0.15:
            v = u
        else:
            v = (rng.uniform(0, box), rng.uniform(0, box))
        queries.append((u, v))
    return queries


SHAPES = {
    "rect": rectangle(2.0, 0.8),
    "rect_rot": rectangle(2.0, 0.8, angle=0.6),
    "tri": regular_polygon(3, 1.0, angle=0.3),
    "hex": regular_polygon(6, inradius=0.7),
    "octagon": regular_polygon(8, 1.1, angle=0.1),
    "offcentre": rectangle(1.5, 0.6, center=(0.5, 0.2)),
}


# ----------------------------------------------------------------------------- brute force


def _brute_force(verts, sweep, u, v, velocity, poly, n_steps=3000):
    """Time-sample the moving polygon; return ({vertex: (t_lo, t_hi, n_samples)},
    {canonical_edge_idx: (t_lo, t_hi, n_samples)}) using strict-interior contact."""
    u_arr = np.asarray(u, float)
    v_arr = np.asarray(v, float)
    d = float(np.linalg.norm(v_arr - u_arr))
    if u == v:
        tdur = 0.0
        ts = np.array([0.0])
    else:
        tdur = 1.0 if velocity == 0.0 else d / velocity
        ts = np.linspace(0.0, tdur, n_steps)
    vel = (v_arr - u_arr) / (tdur if tdur > 0 else 1.0)
    pts = poly.points
    polys = [Polygon(pts + u_arr + t * vel) for t in ts]
    region = Polygon(poly.swept(u_arr, v_arr).points).buffer(1e-6)

    vhits = {}
    for i, q in enumerate(verts):
        p = Point(q)
        if not region.intersects(p):
            continue
        inside = np.array([pg.contains(p) for pg in polys])
        if inside.any():
            idx = np.flatnonzero(inside)
            vhits[i] = (ts[idx[0]], ts[idx[-1]], len(idx))
    ehits = {}
    for e_idx, (s, t) in sweep.edge_indices.items():
        geom = Point(verts[s]) if s == t else LineString([verts[s], verts[t]])
        if not region.intersects(geom):
            continue
        inside = np.array([pg.intersects(geom) and not pg.touches(geom) for pg in polys])
        if inside.any():
            idx = np.flatnonzero(inside)
            ehits[e_idx] = (ts[idx[0]], ts[idx[-1]], len(idx))
    return vhits, ehits, (ts[1] - ts[0] if len(ts) > 1 else 0.0)


def _directed_to_canonical(sweep, edges_dict_or_set):
    """Map directed-key results back to canonical edge indices."""
    rev = {}
    for idx, directed in sweep.edge_directed_map.items():
        for dk in directed:
            rev[dk] = idx
    if isinstance(edges_dict_or_set, dict):
        return {rev[k]: iv for k, iv in edges_dict_or_set.items()}
    return {rev[k] for k in edges_dict_or_set}


# ----------------------------------------------------------------------------- tests


class TestShapeAlgebra(unittest.TestCase):
    def test_regular_polygon_geometry(self):
        hexa = regular_polygon(6, 1.0)
        self.assertEqual(len(hexa.vertices), 6)
        self.assertAlmostEqual(hexa.bounding_radius, 1.0)
        self.assertAlmostEqual(hexa.area, 3 * math.sqrt(3) / 2)
        # inradius form: apothem == offsets
        hex_in = regular_polygon(6, inradius=0.5)
        np.testing.assert_allclose(hex_in.offsets, 0.5)
        with self.assertRaises(ValueError):
            regular_polygon(2, 1.0)
        with self.assertRaises(ValueError):
            regular_polygon(4)

    def test_rectangle_and_square(self):
        r = rectangle(2.0, 1.0)
        self.assertEqual(r.bbox, (-1.0, -0.5, 1.0, 0.5))
        np.testing.assert_allclose(sorted(r.offsets), [0.5, 0.5, 1.0, 1.0])
        self.assertTrue(r.contains(np.array([[0.9, 0.4]]))[0])
        self.assertFalse(r.contains(np.array([[1.0, 0.0]]))[0])  # boundary is not strict interior
        self.assertTrue(r.contains(np.array([[1.0, 0.0]]), strict=False)[0])
        rot = rectangle(2.0, 1.0, angle=math.pi / 2)
        self.assertAlmostEqual(rot.bbox[0], -0.5)
        self.assertAlmostEqual(rot.bbox[1], -1.0)
        self.assertEqual(square(2.0), rectangle(2.0, 2.0))
        np.testing.assert_allclose(square(2.0).points, regular_polygon(4, math.sqrt(2), angle=math.pi / 4).points, atol=1e-12)

    def test_canonical_vertices_and_hashing(self):
        a = ConvexPolygon(((1, 1), (-1, 1), (-1, -1), (1, -1)))  # clockwise input
        b = ConvexPolygon.from_points([(1, -1), (-1, -1), (0, -1), (1, 1), (-1, 1)])  # collinear extra
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(a.vertices[0], (-1.0, -1.0))  # lexicographic start
        self.assertGreater(a.area, 0)  # CCW
        with self.assertRaises(ValueError):
            ConvexPolygon(((0, 0), (1, 1), (2, 2)))
        self.assertEqual(len({a, b, square(2.0)}), 1)

    def test_hull(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.5, 0.0]])
        hull = convex_hull(pts)
        self.assertEqual(hull.shape, (4, 2))

    def test_minkowski_and_pair_clearance(self):
        self.assertEqual(square(1.0).minkowski(square(1.0)), square(2.0))
        self.assertEqual(minkowski_sum(Disc(0.3), Disc(0.4)), Disc(0.7))
        self.assertEqual(pair_clearance(Disc(0.3), Disc(0.4)), Disc(0.7))
        tri = regular_polygon(3, 1.0)
        self.assertNotEqual(tri.reflected(), tri)
        np.testing.assert_allclose(tri.reflected().points, tri.rotated(math.pi).points, atol=1e-12)
        self.assertEqual(square(1.0).reflected(), square(1.0))
        # P (+) (-P) is centrally symmetric even for a triangle (a hexagon)
        combo = pair_clearance(tri, tri)
        np.testing.assert_allclose(combo.points, combo.reflected().points, atol=1e-12)
        self.assertEqual(len(combo.vertices), 6)
        # disc + polygon is conservative: contains the exact rounded shape's extreme points
        buf = minkowski_sum(rectangle(2.0, 1.0), Disc(0.5), n_sides=16)
        self.assertGreaterEqual(buf.bbox[2], 1.5 - 1e-12)
        self.assertGreaterEqual(buf.bbox[3], 1.0 - 1e-12)
        self.assertTrue(buf.contains(np.array([[1.0 + 0.5 * math.cos(0.7), 0.5 + 0.5 * math.sin(0.7)]]))[0])
        self.assertEqual(minkowski_sum(Disc(0.0), square(1.0)), square(1.0))

    def test_disc_polygonisation_bounds(self):
        d = Disc(1.0)
        outer = d.as_polygon(12, circumscribed=True)
        inner = d.as_polygon(12, circumscribed=False)
        np.testing.assert_allclose(outer.offsets, 1.0)
        np.testing.assert_allclose(np.linalg.norm(inner.points, axis=1), 1.0)

    def test_moving_point_interval_square(self):
        sq = square(2.0)  # [-1, 1]^2
        # Start left outside, move right through the square.
        t1, t2, ok = moving_point_interval(np.array([[-3.0, 0.0]]), np.array([1.0, 0.0]), sq.normals, sq.offsets, 10.0)
        self.assertTrue(ok[0])
        self.assertAlmostEqual(t1[0], 2.0)
        self.assertAlmostEqual(t2[0], 4.0, places=8)
        # Grazing the top edge: boundary-only contact is not a hit.
        _, _, ok = moving_point_interval(np.array([[-3.0, 1.0]]), np.array([1.0, 0.0]), sq.normals, sq.offsets, 10.0)
        self.assertFalse(ok[0])
        # Stationary inside / outside.
        _, _, ok = moving_point_interval(np.array([[0.0, 0.0], [2.0, 0.0]]), np.zeros(2), sq.normals, sq.offsets, 1.0)
        self.assertEqual(ok.tolist(), [True, False])
        # Entering after tdur is not a hit; exiting before 0 is not a hit.
        _, _, ok = moving_point_interval(np.array([[-3.0, 0.0], [3.0, 0.0]]), np.array([1.0, 0.0]), sq.normals, sq.offsets, 1.5)
        self.assertEqual(ok.tolist(), [False, False])


class _SweepTestBase:
    backend = None  # set by subclasses

    @classmethod
    def setUpClass(cls):
        rng = random.Random(7)
        cls.verts, cls.edges = _random_roadmap(rng)
        cls.queries = _random_queries(rng, cls.verts, 24)
        cls.rng = rng

    def _make(self, **kw):
        s = self.backend(**kw)
        s.set_graph(self.verts, self.edges)
        return s

    def test_against_brute_force(self):
        for name in ("rect_rot", "tri", "hex", "offcentre"):
            poly = SHAPES[name]
            for velocity in (0.0, 1.7):
                sweep = self._make(record_sweep=False, use_exact_collision_check=False)
                for (u, v) in self.queries[:12]:
                    with self.subTest(shape=name, velocity=velocity, u=u, v=v):
                        ov, oe = sweep.overlapping_interval_cgal(u, v, velocity, poly, get_time_interval=True)
                        oe = _directed_to_canonical(sweep, oe)
                        bv, be, dt = _brute_force(self.verts, sweep, u, v, velocity, poly)
                        self._compare(ov, bv, dt, "vertex")
                        self._compare(oe, be, dt, "edge")
                        # elements query without refinement == interval edge set
                        el = _directed_to_canonical(sweep, sweep.overlapping_graph_elements_cgal(u, v, velocity, poly))
                        self.assertEqual(el, set(oe.keys()))

    def _compare(self, computed, brute, dt, what):
        thin = 3.0 * dt
        for k, (lo, hi, n) in brute.items():
            if n >= 3:
                self.assertIn(k, computed, f"{what} {k} contact missed (brute {lo:.4f}..{hi:.4f})")
        for k, (t1, t2) in computed.items():
            if math.isinf(t2):
                self.assertIn(k, brute, f"{what} {k} stationary contact not found by brute force")
                continue
            if t2 - t1 >= thin:
                self.assertIn(k, brute, f"{what} {k} spurious contact {t1:.4f}..{t2:.4f}")
            if k in brute:
                lo, hi, _ = brute[k]
                self.assertLessEqual(abs(t1 - lo), 2.0 * dt + 1e-9, f"{what} {k} entry {t1} vs {lo}")
                self.assertLessEqual(abs(t2 - hi), 2.0 * dt + 1e-9, f"{what} {k} exit {t2} vs {hi}")

    def test_point_query_matches_containment(self):
        poly = SHAPES["octagon"]
        sweep = self._make(record_sweep=False)
        for (u, _) in self.queries:
            ov, oe = sweep.overlapping_interval_cgal(u, u, 1.0, poly, get_time_interval=True)
            expect = set(np.flatnonzero(poly.translated(u).contains(np.asarray(self.verts))).tolist())
            self.assertEqual(set(ov), expect)
            for iv in ov.values():
                self.assertEqual(iv, (0.0, float("inf")))
            el = sweep.overlapping_graph_elements_cgal(u, u, 1.0, poly)
            self.assertEqual(el, set(oe))

    def test_exact_refinement_is_subset(self):
        poly = SHAPES["rect"]
        loose = self._make(record_sweep=False, use_exact_collision_check=False)
        exact = self._make(record_sweep=False, use_exact_collision_check=True)
        any_pruned = False
        for (u, v) in self.queries:
            a = loose.overlapping_graph_elements_cgal(u, v, 1.0, poly)
            b = exact.overlapping_graph_elements_cgal(u, v, 1.0, poly)
            self.assertTrue(b <= a)
            any_pruned |= b != a
        self.assertTrue(any_pruned)

    def test_bidirectional_reuse_matches_direct(self):
        poly = SHAPES["tri"]  # not centrally symmetric: exercises the general flip
        for velocity in (0.0, 2.0):
            cached = self._make(record_sweep=True)
            fresh = self._make(record_sweep=False)
            for (u, v) in self.queries:
                if u == v:
                    continue
                cached.overlapping_interval_cgal(v, u, velocity, poly, get_time_interval=True)
                rv, re_ = cached.overlapping_interval_cgal(u, v, velocity, poly, get_time_interval=True)
                dv, de = fresh.overlapping_interval_cgal(u, v, velocity, poly, get_time_interval=True)
                self.assertEqual(set(rv), set(dv))
                self.assertEqual(set(re_), set(de))
                for k in dv:
                    np.testing.assert_allclose(rv[k], dv[k], atol=1e-8)
                for k in de:
                    np.testing.assert_allclose(re_[k], de[k], atol=1e-8)
            # Element query (no exact check) is reused symmetrically too.
            cached_el = self._make(record_sweep=True, use_exact_collision_check=False)
            fresh_el = self._make(record_sweep=False, use_exact_collision_check=False)
            for (u, v) in self.queries:
                cached_el.overlapping_graph_elements_cgal(v, u, velocity, poly)
                self.assertEqual(
                    cached_el.overlapping_graph_elements_cgal(u, v, velocity, poly),
                    fresh_el.overlapping_graph_elements_cgal(u, v, velocity, poly),
                )

    def test_polygon_brackets_disc(self):
        """Inscribed n-gon hits ⊆ disc hits ⊆ circumscribed n-gon hits."""
        r = 0.9
        inner = Disc(r).as_polygon(64, circumscribed=False)
        outer = Disc(r).as_polygon(64, circumscribed=True)
        sweep = self._make(record_sweep=False)
        for (u, v) in self.queries:
            dv, de = sweep.overlapping_interval_cgal(u, v, 1.0, r)
            iv, ie = sweep.overlapping_interval_cgal(u, v, 1.0, inner)
            ov, oe = sweep.overlapping_interval_cgal(u, v, 1.0, outer)
            self.assertTrue(iv <= dv <= ov, (u, v))
            self.assertTrue(ie <= de <= oe, (u, v))

    def test_disc_object_is_float_path(self):
        a = self._make(record_sweep=False)
        for (u, v) in self.queries:
            self.assertEqual(
                a.overlapping_interval_cgal(u, v, 1.0, 0.75, get_time_interval=True),
                a.overlapping_interval_cgal(u, v, 1.0, Disc(0.75), get_time_interval=True),
            )
            self.assertEqual(
                a.overlapping_graph_elements_cgal(u, v, 1.0, 0.75),
                a.overlapping_graph_elements_cgal(u, v, 1.0, Disc(0.75)),
            )

    def test_three_dimensional_polygon_rejected(self):
        if self.backend is ShapelySweep:
            self.skipTest("Shapely backend is 2D only anyway")
        s = self.backend(record_sweep=False)
        s.set_graph([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [(0, 1)])
        with self.assertRaises(ValueError):
            s.overlapping_interval_cgal((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0, SHAPES["rect"])


class TestCGALPolygonSweep(_SweepTestBase, unittest.TestCase):
    backend = CGAL_Sweep


class TestShapelyPolygonSweep(_SweepTestBase, unittest.TestCase):
    backend = ShapelySweep


class TestCrossBackend(unittest.TestCase):
    def test_backends_agree(self):
        rng = random.Random(11)
        verts, edges = _random_roadmap(rng, n_vertices=120, k=4)
        queries = _random_queries(rng, verts, 80)
        for name, poly in SHAPES.items():
            for exact in (False, True):
                c = CGAL_Sweep(record_sweep=False, use_exact_collision_check=exact)
                s = ShapelySweep(record_sweep=False, use_exact_collision_check=exact)
                c.set_graph(verts, edges)
                s.set_graph(verts, edges)
                for (u, v) in queries:
                    with self.subTest(shape=name, exact=exact, u=u, v=v):
                        self.assertEqual(
                            c.overlapping_graph_elements_cgal(u, v, 1.3, poly),
                            s.overlapping_graph_elements_cgal(u, v, 1.3, poly),
                        )
                        cv, ce = c.overlapping_interval_cgal(u, v, 1.3, poly, get_time_interval=True)
                        sv, se = s.overlapping_interval_cgal(u, v, 1.3, poly, get_time_interval=True)
                        self.assertEqual(set(cv), set(sv))
                        self.assertEqual(set(ce), set(se))
                        for k in cv:
                            np.testing.assert_allclose(cv[k], sv[k], atol=1e-9)
                        for k in ce:
                            np.testing.assert_allclose(ce[k], se[k], atol=1e-9)


@unittest.skipUnless(_HAS_SAMPLER, "GraphSampler not importable")
class TestGraphSamplerPassThrough(unittest.TestCase):
    def _map(self, backend):
        m = GraphSampler(bounds=[[0.0, 6.0], [0.0, 6.0]], resolution=1.0, start=[], goal=[],
                         use_discrete_space=True, sweep_backend=backend)
        m.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
        nodes = m.generateRandomNodes(generate_grid_nodes=True)
        m.generate_roadmap(nodes)
        m.set_constraint_sweep()
        return m

    def test_polygon_clearance_through_graph_sampler(self):
        for backend in ("cgal", "shapely"):
            m = self._map(backend)
            p1 = tuple(m.nodes[0].current)
            nbr = m.get_neighbors(m.nodes[0])
            p2 = tuple((nbr[0].current if hasattr(nbr[0], "current") else m.nodes[nbr[0]].current))
            poly = pair_clearance(rectangle(1.2, 0.6), rectangle(1.2, 0.6))
            vi, ei = m.get_constraint_sweep(p1, p2, 1.0, poly, use_interval=True, get_time_interval=True)
            self.assertIn(p1, vi)
            self.assertIn(p2, vi)
            self.assertIn((p1, p2), ei)
            self.assertTrue(all(isinstance(k, tuple) and len(k) == 2 for k in vi))
            # non-interval path (cached by (p1, p2, v, shape) key)
            edges1 = m.get_constraint_sweep(p1, p2, 1.0, poly)
            edges2 = m.get_constraint_sweep(p1, p2, 1.0, poly)
            self.assertIs(edges1, edges2)
            self.assertIn((p1, p2), edges1)
            # closest-approach test with a polygon clearance
            self.assertTrue(m.get_constraint_segment(p1, p2, p2, p1, 1.0, r_sum=poly))
            far = tuple(np.asarray(p1) + np.array([4.0, 4.0]))
            self.assertFalse(m.get_constraint_segment(p1, p1, far, far, 1.0, r_sum=poly))


@unittest.skipUnless(_HAS_SAMPLER, "GraphSampler not importable")
class TestSippPolygonAgents(unittest.TestCase):
    """Head-on + crossing agents with asymmetric footprints planned by SIPP:
    the exact footprint checker must find no collision (regression for the
    sweep-clearance orientation: query = obstacle, element = planning agent)."""

    def test_sipp_plans_are_footprint_collision_free(self):
        from path_planning.multi_agent_planner.mapf_solver import solve_mapf
        from path_planning.utils.checker import check_collision
        from path_planning.utils.util import max_agent_radius, normalize_agent_specs, set_global_seed

        agents = normalize_agent_specs([
            {"name": "tri", "start": [6.5, 1.5], "goal": [6.5, 11.5], "velocity": 0.5,
             "shape": {"type": "regular_polygon", "n_sides": 3, "circumradius": 0.6, "angle_deg": 90}},
            {"name": "hexa", "start": [12.5, 6.5], "goal": [0.5, 6.5], "velocity": 1.0,
             "shape": {"type": "regular_polygon", "n_sides": 6, "inradius": 0.45}},
            {"name": "rect", "start": [0.5, 6.5], "goal": [12.5, 6.5], "velocity": 2.0,
             "shape": {"type": "rectangle", "width": 1.2, "height": 0.6}},
            {"name": "disc", "start": [2.5, 11.5], "goal": [10.5, 1.5], "velocity": 1.5, "radius": 0.35},
        ])
        for backend in ("cgal", "shapely"):
            set_global_seed(1)
            m = GraphSampler(bounds=[[0.0, 13.0], [0.0, 13.0]], resolution=1.0, start=[], goal=[],
                             use_discrete_space=True, sweep_backend=backend)
            m.inflate_obstacles(radius=max_agent_radius(agents, 0.3) + np.sqrt(2) / 2)
            m.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
            m.set_start([a["start"] for a in agents]); m.set_goal([a["goal"] for a in agents])
            nodes = m.generateRandomNodes(generate_grid_nodes=True)
            m.generate_roadmap(nodes)
            m.set_constraint_sweep()
            summary = solve_mapf(m, agents, {
                "mapf_solver_name": "sipp", "time_limit": 60, "max_iterations": 50,
                "agent_radius": 0.3, "agent_velocity": 1.0, "heuristic_type": "euclidean",
            })
            self.assertTrue(summary["success"], summary.get("failure_reason"))
            self.assertFalse(summary["collision"])
            fps = summary["agent_footprints"]
            self.assertIsInstance(fps["tri"], AlignedShape)  # YAML polygons are heading-aligned by default
            self.assertEqual(fps["disc"], 0.35)
            self.assertEqual(check_collision(summary["schedule"], fps, verbose=False), {})
            # Speeds are honoured per agent.
            for name, path in summary["schedule"].items():
                moves = [(a, b) for a, b in zip(path, path[1:]) if (a["x"], a["y"]) != (b["x"], b["y"])]
                for a, b in moves:
                    self.assertAlmostEqual(math.dist((a["x"], a["y"]), (b["x"], b["y"])) / (b["t"] - a["t"]),
                                           summary["agent_velocities"][name], places=6)


# ----------------------------------------------------------------------------- heading-aligned footprints


class TestHeadings(unittest.TestCase):
    def test_heading_key_canonical(self):
        self.assertEqual(heading_key(-math.pi), heading_key(math.pi))
        self.assertEqual(heading_key(-0.0), 0.0)
        self.assertEqual(heading_key(2 * math.pi), 0.0)
        self.assertEqual(heading_of((0, 0), (-1, 0)), heading_of((0, 0), (-1, -0.0)))
        self.assertIsNone(heading_of((1, 1), (1, 1)))
        self.assertEqual(heading_of((0, 0), (0, 2)), heading_key(math.pi / 2))

    def test_path_headings_carry_rule(self):
        east, north = 0.0, heading_key(math.pi / 2)
        pts = [(0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (1, 1)]
        # start wait faces the first edge; waits keep the arrival heading; the end holds it
        self.assertEqual(path_headings(pts), [east, east, east, north, north, north])
        self.assertEqual(path_headings([(3, 3), (3, 3)]), [0.0, 0.0])
        self.assertEqual(path_headings([(3, 3), (3, 3)], default=1.0), [1.0, 1.0])
        # explicit headings win (an in-place turn before departing)
        self.assertEqual(path_headings(pts, headings=[north, None, None, None, None, None]),
                         [north, east, east, north, north, north])
        sched = [{"x": 0, "y": 0, "t": 0, "heading": north}, {"x": 0, "y": 0, "t": 1}, {"x": 1, "y": 0, "t": 2}]
        self.assertEqual(schedule_headings(sched), [north, east, east])

    def test_shape_from_spec_align_flag(self):
        aligned = shape_from_spec({"type": "rectangle", "width": 1.2, "height": 0.6})
        fixed = shape_from_spec({"type": "rectangle", "width": 1.2, "height": 0.6, "align": False})
        self.assertIsInstance(aligned, AlignedShape)
        self.assertIsInstance(fixed, ConvexPolygon)
        self.assertEqual(aligned.body, fixed)
        self.assertIsInstance(shape_from_spec({"type": "disc", "radius": 0.3}), Disc)
        self.assertIsInstance(shape_from_spec([[0, 0], [1, 0], [0, 1]]), AlignedShape)
        self.assertEqual(aligned.bounding_radius, fixed.bounding_radius)
        self.assertEqual(hash(aligned), hash(AlignedShape(rectangle(1.2, 0.6))))

    def test_aligned_shape_rotation(self):
        rect = AlignedShape(rectangle(1.2, 0.6))
        self.assertEqual(rect.at(0.0), rect.body)
        self.assertEqual(rect.at(math.pi / 2).bbox, (-0.3, -0.6, 0.3, 0.6))  # exact quarter turn
        self.assertEqual(rect.at(math.pi), rect.body)  # centrally symmetric
        tri = AlignedShape(regular_polygon(3, 1.0))
        np.testing.assert_allclose(tri.at(math.pi).points, tri.body.reflected().points, atol=1e-12)
        np.testing.assert_allclose(tri.at(0.7).points, tri.body.rotated(0.7).points, atol=1e-12)
        self.assertEqual(rect.envelope, Disc(rect.body.bounding_radius))

    def test_aligned_clearance_resolution(self):
        rect = AlignedShape(rectangle(2.0, 0.8))
        tri = AlignedShape(regular_polygon(3, 1.0))
        clr = pair_clearance(rect, tri)
        self.assertIsInstance(clr, AlignedClearance)
        self.assertTrue(clr.aligned_query and clr.aligned_element)
        combo = clr.resolve(0.3, 1.1)
        np.testing.assert_allclose(combo.points, rect.at(0.3).minkowski(tri.at(1.1).reflected()).points, atol=1e-12)
        np.testing.assert_allclose(oriented_pair_clearance(rect, 0.3, tri, 1.1).points, combo.points, atol=1e-12)
        # unknown headings -> envelope discs (two discs resolve to the float sum)
        self.assertAlmostEqual(clr.resolve(None, None), rect.bounding_radius + tri.bounding_radius, places=12)
        env = clr.resolve(0.3, None)
        self.assertAlmostEqual(env.bounding_radius, rect.at(0.3).buffered(tri.bounding_radius).bounding_radius, places=12)
        # disc pair resolves to the float sum; non-aligned query keeps cache keys stable
        d = AlignedClearance(Disc(0.3), tri)
        self.assertIs(d.with_query_heading(0.5), d)
        self.assertEqual(AlignedClearance(Disc(0.3), Disc(0.2)).resolve(None, None), 0.5)
        self.assertIsNot(clr.with_query_heading(0.5), clr)
        self.assertEqual(clr.with_query_heading(0.5).query_heading, heading_key(0.5))
        # stationary query: query_heading wins, else envelope
        self.assertEqual(clr.with_query_heading(0.5).query_footprint(None), rect.at(0.5))
        self.assertEqual(clr.query_footprint(None), rect.envelope)
        self.assertEqual(clr.with_query_heading(0.5).query_footprint(1.0), rect.at(1.0))


def _incident_headings(sweep, n_verts):
    inc = [set() for _ in range(n_verts)]
    for s_i, t_i in sweep.edge_indices.values():
        if s_i == t_i:
            continue
        h = heading_of(sweep.vertex_positions[s_i], sweep.vertex_positions[t_i])
        hr = heading_of(sweep.vertex_positions[t_i], sweep.vertex_positions[s_i])
        inc[s_i].update((h, hr))
        inc[t_i].update((h, hr))
    return inc


def _brute_force_aligned(verts, sweep, u, v, velocity, clr, n_steps=2500):
    """Time-sampled contact of the query footprint (facing its motion) with an
    aligned element placed at every vertex in every incident heading and swept
    along every directed edge (facing that edge); self-loops use the element's
    envelope polygonised exactly as the sweep does (circumscribed 16-gon)."""
    u_arr = np.asarray(u, float)
    v_arr = np.asarray(v, float)
    d = float(np.linalg.norm(v_arr - u_arr))
    if u == v:
        tdur = 0.0
        ts = np.array([0.0])
    else:
        tdur = 1.0 if velocity == 0.0 else d / velocity
        ts = np.linspace(0.0, tdur, n_steps)
    vel = (v_arr - u_arr) / (tdur if tdur > 0 else 1.0)
    q_fp = clr.query_footprint(None if u == v else heading_of(u, v))
    if isinstance(q_fp, Disc):
        # a disc query (stationary aligned query with no heading) is polygonised
        # by the sweep exactly like this (circumscribed n-gon, angle 0)
        q_fp = q_fp.as_polygon(n_sides=clr.n_sides)
    q_pts = q_fp.points
    polys = [Polygon(q_pts + u_arr + t * vel) for t in ts]
    outer = q_fp.buffered(clr.element.bounding_radius)
    region = Polygon(outer.swept(u_arr, v_arr).points).buffer(1e-6)
    E = clr.element
    env16 = E.envelope.as_polygon(n_sides=clr.n_sides)
    inc = _incident_headings(sweep, len(verts))

    def contact(geoms):
        inside = np.array([pg.intersects(g) and not pg.touches(g) for pg, g in zip(polys, geoms)])
        if inside.any():
            idx = np.flatnonzero(inside)
            return (ts[idx[0]], ts[idx[-1]], len(idx))
        return None

    vhits = {}
    for i, q in enumerate(verts):
        if not region.intersects(Point(q)):
            continue
        for h in inc[i]:
            g = Polygon(E.at(h).points + np.asarray(q))
            c = contact([g] * len(polys))
            if c is not None:
                vhits[(i, h)] = c
    ehits = {}
    for s_i, t_i in sweep.edge_indices.values():
        for a, b in ((s_i, t_i), (t_i, s_i)) if s_i != t_i else ((s_i, s_i),):
            if a == b:
                g = Polygon(env16.points + np.asarray(verts[a]))
            else:
                h = heading_of(verts[a], verts[b])
                g = Polygon(E.at(h).swept(verts[a], verts[b]).points)
            if not region.intersects(g):
                continue
            c = contact([g] * len(polys))
            if c is not None:
                ehits[(a, b)] = c
    return vhits, ehits, (ts[1] - ts[0] if len(ts) > 1 else 0.0)


ALIGNED_CLEARANCES = {
    "rect-rect": AlignedClearance(AlignedShape(rectangle(2.0, 0.8)), AlignedShape(rectangle(2.0, 0.8))),
    "tri-rect": AlignedClearance(AlignedShape(regular_polygon(3, 1.0)), AlignedShape(rectangle(1.6, 0.6))),
    "fixedhex-tri": AlignedClearance(regular_polygon(6, inradius=0.7), AlignedShape(regular_polygon(3, 0.9, angle=0.4))),
    "disc-rect": AlignedClearance(Disc(0.5), AlignedShape(rectangle(2.0, 0.8))),
}


class _AlignedSweepTestBase:
    backend = None

    @classmethod
    def setUpClass(cls):
        rng = random.Random(11)
        cls.verts, cls.edges = _random_roadmap(rng)
        cls.queries = _random_queries(rng, cls.verts, 16)

    def _make(self, **kw):
        s = self.backend(**kw)
        s.set_graph(self.verts, self.edges)
        return s

    def test_aligned_against_brute_force(self):
        for name in ("rect-rect", "tri-rect", "fixedhex-tri"):
            clr = ALIGNED_CLEARANCES[name]
            for velocity in (0.0, 1.3):
                sweep = self._make(record_sweep=False, use_exact_collision_check=False)
                for (u, v) in self.queries[:10]:
                    with self.subTest(pair=name, velocity=velocity, u=u, v=v):
                        ov, oe = sweep.overlapping_interval_cgal(u, v, velocity, clr, get_time_interval=True)
                        bv, be, dt = _brute_force_aligned(self.verts, sweep, u, v, velocity, clr)
                        _SweepTestBase._compare(self, ov, bv, dt, "vertex")
                        _SweepTestBase._compare(self, oe, be, dt, "edge")
                        el = sweep.overlapping_graph_elements_cgal(u, v, velocity, clr)
                        self.assertEqual(el, set(oe.keys()))

    def test_disc_query_is_conservative_and_tight(self):
        # A disc query is polygonised (circumscribed 16-gon): never misses a
        # brute-force contact against the exact disc, only adds thin ones.
        clr = ALIGNED_CLEARANCES["disc-rect"]
        sweep = self._make(record_sweep=False)
        r = clr.query.radius
        for (u, v) in self.queries[:8]:
            ov, oe = sweep.overlapping_interval_cgal(u, v, 1.0, clr, get_time_interval=True)
            exact_clr = AlignedClearance(Disc(r).as_polygon(64, circumscribed=False), clr.element)
            ev, ee = sweep.overlapping_interval_cgal(u, v, 1.0, exact_clr, get_time_interval=True)
            self.assertTrue(set(ev) <= set(ov))
            self.assertTrue(set(ee) <= set(oe))

    def test_aligned_query_only_uses_fixed_path(self):
        tri = AlignedShape(regular_polygon(3, 1.0, angle=0.2))
        elem = rectangle(1.6, 0.6, angle=0.3)
        clr = AlignedClearance(tri, elem)
        sweep = self._make(record_sweep=False)
        for (u, v) in self.queries:
            h = None if u == v else heading_of(u, v)
            fixed = pair_clearance(tri.at(h) if h is not None else tri.envelope.as_polygon(16), elem)
            a = sweep.overlapping_interval_cgal(u, v, 1.0, clr, get_time_interval=True)
            b = sweep.overlapping_interval_cgal(u, v, 1.0, fixed, get_time_interval=True)
            self.assertEqual(a[0].keys(), b[0].keys())
            self.assertEqual(a[1].keys(), b[1].keys())
            # a stationary aligned query with an explicit heading faces it
            if u == v:
                c = sweep.overlapping_interval_cgal(u, v, 1.0, clr.with_query_heading(0.9), get_time_interval=True)
                d = sweep.overlapping_interval_cgal(u, v, 1.0, pair_clearance(tri.at(0.9), elem), get_time_interval=True)
                self.assertEqual(c[0].keys(), d[0].keys())

    def test_direction_dependence(self):
        # An aligned triangle element on edge a->b faces b-a: the two directed
        # keys of one edge carry different windows (unlike the fixed path).
        clr = ALIGNED_CLEARANCES["tri-rect"]
        clr = AlignedClearance(AlignedShape(rectangle(2.0, 0.8)), AlignedShape(regular_polygon(3, 1.0)))
        sweep = self._make(record_sweep=False)
        differs = False
        for (u, v) in self.queries:
            if u == v:
                continue
            _, oe = sweep.overlapping_interval_cgal(u, v, 1.0, clr, get_time_interval=True)
            for (a, b), iv in oe.items():
                if a != b and (b, a) in oe and oe[(b, a)] != iv:
                    differs = True
            if differs:
                break
        self.assertTrue(differs)

    def test_vertex_keys_cover_incident_headings_only(self):
        clr = ALIGNED_CLEARANCES["rect-rect"]
        sweep = self._make(record_sweep=False)
        inc = _incident_headings(sweep, len(self.verts))
        for (u, v) in self.queries:
            ov, _ = sweep.overlapping_interval_cgal(u, v, 1.0, clr, get_time_interval=True)
            for (vi, h) in ov:
                self.assertIn(h, inc[vi])


class TestCGALAlignedSweep(_AlignedSweepTestBase, unittest.TestCase):
    backend = CGAL_Sweep


class TestShapelyAlignedSweep(_AlignedSweepTestBase, unittest.TestCase):
    backend = ShapelySweep


class TestCrossBackendAligned(unittest.TestCase):
    def test_backends_agree(self):
        rng = random.Random(5)
        verts, edges = _random_roadmap(rng, n_vertices=80)
        queries = _random_queries(rng, verts, 40)
        a = CGAL_Sweep(record_sweep=True)
        b = ShapelySweep(record_sweep=True)
        a.set_graph(verts, edges)
        b.set_graph(verts, edges)
        for name, clr in ALIGNED_CLEARANCES.items():
            for (u, v) in queries:
                with self.subTest(pair=name, u=u, v=v):
                    av, ae = a.overlapping_interval_cgal(u, v, 1.5, clr, get_time_interval=True)
                    bv, be = b.overlapping_interval_cgal(u, v, 1.5, clr, get_time_interval=True)
                    self.assertEqual(set(av), set(bv))
                    self.assertEqual(set(ae), set(be))
                    for k in av:
                        np.testing.assert_allclose(av[k], bv[k], atol=1e-9)
                    for k in ae:
                        np.testing.assert_allclose(ae[k], be[k], atol=1e-9)
                    self.assertEqual(a.overlapping_graph_elements_cgal(u, v, 1.5, clr),
                                     b.overlapping_graph_elements_cgal(u, v, 1.5, clr))


@unittest.skipUnless(_HAS_SAMPLER, "GraphSampler unavailable")
class TestGraphSamplerAligned(unittest.TestCase):
    def test_keys_and_segment_test(self):
        for backend in ("cgal", "shapely"):
            m = TestGraphSamplerPassThrough._map(self, backend)
            m.set_constraint_sweep()
            rect = AlignedShape(rectangle(1.2, 0.6))
            clr = pair_clearance(rect, rect)
            p = tuple(m.nodes[0].current)
            q = tuple(m.nodes[1].current)
            ov, oe = m.get_constraint_sweep(p, q, 1.0, clr, use_interval=True, get_time_interval=True)
            self.assertTrue(ov)
            for key in ov:
                self.assertEqual(len(key), 2)
                self.assertEqual(len(key[0]), 2)  # (position, heading)
                self.assertIsInstance(key[1], float)
            for (a, b) in oe:
                self.assertEqual(len(a), 2)
            # elements form (CBS) returns directed position pairs
            el = m.get_constraint_sweep(p, q, 1.0, clr)
            self.assertTrue(all(len(e) == 2 and len(e[0]) == 2 for e in el))
            # closest-approach test resolves headings from both motions
            self.assertTrue(m.get_constraint_segment(p, q, p, q, 1.0, 1.0, clr))


@unittest.skipUnless(_HAS_SAMPLER, "GraphSampler unavailable")
class TestSippAlignedAgents(unittest.TestCase):
    """SIPP with heading-aligned footprints."""

    @staticmethod
    def _empty_map(backend, agents, default_radius=0.3):
        from path_planning.utils.util import max_agent_radius, set_global_seed

        set_global_seed(1)
        m = GraphSampler(bounds=[[0.0, 13.0], [0.0, 13.0]], resolution=1.0, start=[], goal=[],
                         use_discrete_space=True, sweep_backend=backend)
        m.inflate_obstacles(radius=max_agent_radius(agents, default_radius) + np.sqrt(2) / 2)
        m.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
        m.set_start([a["start"] for a in agents])
        m.set_goal([a["goal"] for a in agents])
        nodes = m.generateRandomNodes(generate_grid_nodes=True)
        m.generate_roadmap(nodes)
        m.set_constraint_sweep()
        return m

    @staticmethod
    def _solve(m, agents):
        from path_planning.multi_agent_planner.mapf_solver import solve_mapf

        return solve_mapf(m, agents, {
            "mapf_solver_name": "sipp", "time_limit": 60, "max_iterations": 50,
            "agent_radius": 0.3, "agent_velocity": 1.0, "heuristic_type": "euclidean",
        })

    def test_adjacent_lanes_need_alignment(self):
        """Two 1.6 x 0.6 rectangles on adjacent lanes, opposite directions: aligned
        with the lane they pass (lateral clearance 0.6 < 1); fixed in the world
        frame (1.6 wide across the lanes) one of them has to give way."""
        from path_planning.utils.checker import check_collision
        from path_planning.utils.util import normalize_agent_specs

        def agents(align):
            shape = {"type": "rectangle", "width": 1.6, "height": 0.6, "align": align}
            return normalize_agent_specs([
                {"name": "up", "start": [5.5, 1.5], "goal": [5.5, 11.5], "velocity": 1.0, "shape": shape},
                {"name": "down", "start": [6.5, 11.5], "goal": [6.5, 1.5], "velocity": 1.0, "shape": shape},
            ])

        for backend in ("cgal", "shapely"):
            aligned = agents(True)
            s = self._solve(self._empty_map(backend, aligned), aligned)
            self.assertTrue(s["success"], s.get("failure_reason"))
            self.assertFalse(s["collision"])
            self.assertAlmostEqual(s["makespan"], 10.0, places=6)
            self.assertEqual(check_collision(s["schedule"], s["agent_footprints"], verbose=False), {})
            for path in s["schedule"].values():
                self.assertTrue(all("heading" in p for p in path))
            self.assertTrue(all(p["heading"] == heading_key(math.pi / 2) for p in s["schedule"]["up"]))
            self.assertTrue(all(p["heading"] == heading_key(-math.pi / 2) for p in s["schedule"]["down"]))

            fixed = agents(False)
            s = self._solve(self._empty_map(backend, fixed), fixed)
            self.assertTrue(s["success"], s.get("failure_reason"))
            self.assertFalse(s["collision"])
            self.assertGreater(s["makespan"], 10.0 + 1e-6)
            self.assertEqual(check_collision(s["schedule"], s["agent_footprints"], verbose=False), {})
            for path in s["schedule"].values():
                self.assertTrue(all("heading" not in p for p in path))

    def test_turning_mixed_agents_are_collision_free(self):
        """Crossing / head-on agents whose paths turn: the exact heading-aware
        checker must find no collision and move headings must match the edges."""
        from path_planning.utils.checker import check_collision
        from path_planning.utils.util import normalize_agent_specs

        agents = normalize_agent_specs([
            {"name": "rect_a", "start": [1.5, 1.5], "goal": [11.5, 11.5], "velocity": 2.0,
             "shape": {"type": "rectangle", "width": 1.6, "height": 0.6}},
            {"name": "rect_b", "start": [11.5, 1.5], "goal": [1.5, 11.5], "velocity": 1.0,
             "shape": {"type": "rectangle", "width": 1.4, "height": 0.5}},
            {"name": "tri", "start": [6.5, 0.5], "goal": [6.5, 12.5], "velocity": 0.5,
             "shape": {"type": "regular_polygon", "n_sides": 3, "circumradius": 0.6}},
            {"name": "hexa", "start": [12.5, 6.5], "goal": [0.5, 6.5], "velocity": 1.5,
             "shape": {"type": "regular_polygon", "n_sides": 6, "inradius": 0.45}},
            {"name": "disc", "start": [3.5, 0.5], "goal": [9.5, 12.5], "velocity": 1.0, "radius": 0.35},
        ])
        for backend in ("cgal", "shapely"):
            s = self._solve(self._empty_map(backend, agents), agents)
            self.assertTrue(s["success"], s.get("failure_reason"))
            self.assertFalse(s["collision"])
            self.assertEqual(check_collision(s["schedule"], s["agent_footprints"], verbose=False), {})
            for name, path in s["schedule"].items():
                if name == "disc":
                    continue
                for a, b in zip(path, path[1:]):
                    if (a["x"], a["y"]) != (b["x"], b["y"]):
                        self.assertEqual(a["heading"], heading_of((a["x"], a["y"]), (b["x"], b["y"])))
                        self.assertAlmostEqual(math.dist((a["x"], a["y"]), (b["x"], b["y"])) / (b["t"] - a["t"]),
                                               s["agent_velocities"][name], places=6)
                # the schedule's headings reproduce under the carry rule
                self.assertEqual(schedule_headings(path), [p["heading"] for p in path])
            turned = any(len({p["heading"] for p in path}) > 1 for n, path in s["schedule"].items() if n != "disc")
            self.assertTrue(turned)


class TestCheckerHeadings(unittest.TestCase):
    def test_orientation_decides_collision(self):
        from path_planning.utils.checker import check_collision

        rect = AlignedShape(rectangle(2.0, 0.4))
        a = [{"x": 0.0, "y": 0.0, "t": 0.0}, {"x": 10.0, "y": 0.0, "t": 10.0}]
        fps = {"a": rect, "b": rect}
        # b parked beside the lane facing east: lateral gap 0.5 > 0.4 -> clear
        b = [{"x": 5.0, "y": 0.5, "t": 0.0, "heading": 0.0}, {"x": 5.0, "y": 0.5, "t": 20.0, "heading": 0.0}]
        self.assertEqual(check_collision({"a": a, "b": b}, fps, verbose=False), {})
        # facing north it reaches into the lane
        b = [{"x": 5.0, "y": 0.5, "t": 0.0, "heading": math.pi / 2}, {"x": 5.0, "y": 0.5, "t": 20.0, "heading": math.pi / 2}]
        hits = check_collision({"a": a, "b": b}, fps, verbose=False)
        self.assertIn(("a", "b"), hits)
        (t0, t1, *_), = hits[("a", "b")]
        self.assertAlmostEqual(t0, 5.0 - 1.2, places=6)  # 1.0 (a half length) + 0.2 (b half width)
        self.assertAlmostEqual(t1, 5.0 + 1.2, places=6)
        # no explicit headings: b drove south into the spot, so it keeps facing south
        b = [{"x": 5.0, "y": 2.5, "t": 0.0}, {"x": 5.0, "y": 0.5, "t": 1.0}, {"x": 5.0, "y": 0.5, "t": 20.0}]
        self.assertIn(("a", "b"), check_collision({"a": a, "b": b}, fps, verbose=False))
        # fixed world-frame rectangles ignore headings entirely
        fixed = {"a": rect.body, "b": rect.body}
        self.assertEqual(check_collision({"a": a, "b": b}, fixed, verbose=False), {})


if __name__ == "__main__":
    unittest.main()
