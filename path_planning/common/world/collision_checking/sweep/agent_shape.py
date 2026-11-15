"""Agent bodies: a disc or polygon (2D) and a sphere, box or convex polyhedron (3D), in the body frame.

Every shape has its centroid at the origin. ``orientation`` is the body's fixed rotation relative to the
direction of travel: at ``orientation = 0`` the reference direction below points along the move.
``radius()`` is the bounding radius about the centroid.

Reference direction at ``orientation = 0``:
    Triangle        centroid -> the vertex joined by side 1 and side 2
    Rectangle       along the width
    RegularPolygon  centroid -> a vertex (odd number of sides) or the middle of a side (even)
    CustomPolygon   the given points' own +x axis
    Circle          rotation-invariant

3D: a 2D shape with a ``z_height`` is the prism of its outline, from ``-z_height / 2`` to
``+z_height / 2`` (a ``Circle`` becomes a 32-gon prism, never smaller than the cylinder). Native 3D shapes:

    Sphere            rotation-invariant
    Box               length along body +x, width along +y, height along +z
    ConvexPolyhedron  the convex hull of the given points

Their ``orientation`` is ``(yaw, pitch, roll)`` in radians, or just a yaw. The 3D travel frame has +x
along the direction of travel and no roll.
"""
import math

import numpy as np
import shapely
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.polygon import orient


def _check_z_height(z_height):
    if z_height is None:
        return None
    z = float(z_height)
    if not (math.isfinite(z) and z > 0):
        raise ValueError(f"z_height must be positive and finite, or None, got {z_height}")
    return z


class Shape:
    dim = 2                                     # native dimension: 2 for outlines, 3 for the 3D shapes

    def __init__(self, radius: float = 1, orientation: float = 0.0, z_height=None):
        self._radius = radius
        self._orientation = float(orientation)
        self._z_height = _check_z_height(z_height)

    def z_height(self):
        """Vertical thickness of the body in 3D (the prism of its outline), or None (2D only)."""
        return self._z_height

    def _with_z(self, d: dict) -> dict:
        """``d`` plus ``z_height`` when it is set."""
        if self._z_height is not None:
            d["z_height"] = self._z_height
        return d

    def to_dict(self):
        return self._with_z({"radius": self._radius, "orientation": self._orientation})

    @staticmethod
    def _orientation_from(data: dict) -> float:
        """``orientation`` of a saved dict (older dicts call it ``heading``)."""
        return data.get("orientation", data.get("heading", 0.0))

    @staticmethod
    def from_dict(data: dict):
        return Shape(radius=data["radius"], orientation=Shape._orientation_from(data), z_height=data.get("z_height"))

    def radius(self):
        """Bounding radius about the centroid."""
        return self._radius

    def orientation(self):
        """Rotation of the body relative to the direction of travel, in radians."""
        return self._orientation

    def outline(self) -> np.ndarray:
        """``(K, 2)`` body-frame vertices at ``orientation = 0``, counter-clockwise."""
        raise NotImplementedError(f"{type(self).__name__} has no outline")

    def extent(self) -> tuple:
        """``(x_min, x_max, y_min, y_max)`` of the body in the travel frame (+x along travel, ``orientation``
        applied): its reach behind and ahead of the centroid, and its lateral reach."""
        if getattr(self, "_extent", None) is None:
            try:
                pts = self.outline()
            except NotImplementedError:                      # a bare radius is a disc
                r = float(self._radius)
                self._extent = (-r, r, -r, r)
            else:
                c, s = math.cos(self._orientation), math.sin(self._orientation)
                x = c * pts[:, 0] - s * pts[:, 1]
                y = s * pts[:, 0] + c * pts[:, 1]
                self._extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
        return self._extent

    def front_face(self) -> float:
        """The body's lateral extent (perpendicular to travel), with ``orientation`` applied."""
        _, _, y_min, y_max = self.extent()
        return y_max - y_min


class Circle(Shape):
    def __init__(self, radius: float, z_height=None):
        super().__init__(radius, 0.0, z_height)

    def to_dict(self):
        return self._with_z({"radius": self._radius})

    @staticmethod
    def from_dict(data: dict):
        return Circle(radius=data["radius"], z_height=data.get("z_height"))

    def extent(self) -> tuple:
        r = float(self._radius)
        return (-r, r, -r, r)

    def outline(self, quad_segs: int = 32) -> np.ndarray:
        n = 4 * quad_segs
        a = 2.0 * math.pi * np.arange(n) / n
        return self._radius * np.column_stack([np.cos(a), np.sin(a)])


class Triangle(Shape):
    # orientation = 0: the direction from the centroid to the vertex joined by side 1 and side 2
    def __init__(self, side1: float, side2: float, side3: float, orientation: float = 0.0, z_height=None):
        if min(side1, side2, side3) <= 0 or side1 + side2 <= side3 or side1 + side3 <= side2 or side2 + side3 <= side1:
            raise ValueError(f"Triangle: sides {side1}, {side2}, {side3} do not form a triangle")
        self._side1 = side1
        self._side2 = side2
        self._side3 = side3
        # Vertex A joins side 1 (A-B) and side 2 (A-C); side 3 is B-C.
        ax, ay = 0.0, 0.0
        bx, by = side1, 0.0
        cx = (side1**2 + side2**2 - side3**2) / (2.0 * side1)
        cy = math.sqrt(max(side2**2 - cx**2, 0.0))
        pts = np.array([[ax, ay], [bx, by], [cx, cy]])
        pts -= pts.mean(axis=0)                                  # centroid at the origin
        theta = -math.atan2(pts[0, 1], pts[0, 0])                # rotate A onto +x
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        self._outline = pts @ rot.T
        super().__init__(float(np.linalg.norm(self._outline, axis=1).max()), orientation, z_height)

    def outline(self) -> np.ndarray:
        return self._outline.copy()

    def to_dict(self):
        return self._with_z({"side1": self._side1, "side2": self._side2, "side3": self._side3,
                             "radius": self._radius, "orientation": self._orientation})

    @staticmethod
    def from_dict(data: dict):
        return Triangle(side1=data["side1"], side2=data["side2"], side3=data["side3"],
                        orientation=Shape._orientation_from(data), z_height=data.get("z_height"))


class Rectangle(Shape):
    # orientation = 0: the direction from the centroid along the width (perpendicular to the height sides)
    def __init__(self, width: float, height: float, orientation: float = 0.0, z_height=None):
        if width <= 0 or height <= 0:
            raise ValueError(f"Rectangle: width and height must be positive, got {width}, {height}")
        self.width = width
        self.height = height
        super().__init__(0.5 * math.hypot(width, height), orientation, z_height)     # half the diagonal

    def outline(self) -> np.ndarray:
        w, h = 0.5 * self.width, 0.5 * self.height
        return np.array([[w, -h], [w, h], [-w, h], [-w, -h]])

    def to_dict(self):
        return self._with_z({"width": self.width, "height": self.height,
                             "radius": self._radius, "orientation": self._orientation})

    @staticmethod
    def from_dict(data: dict):
        return Rectangle(width=data["width"], height=data["height"], orientation=Shape._orientation_from(data),
                         z_height=data.get("z_height"))


class RegularPolygon(Shape):
    # orientation = 0: centroid -> vertex for an odd number of sides,
    #                  centroid -> the middle of a side for an even number of sides
    def __init__(self, side: float, num_sides: int, orientation: float = 0.0, z_height=None):
        if num_sides < 3 or side <= 0:
            raise ValueError(f"RegularPolygon: need side > 0 and num_sides >= 3, got {side}, {num_sides}")
        self.side = side
        self.num_sides = num_sides
        super().__init__(side / (2.0 * math.sin(math.pi / num_sides)), orientation, z_height)   # circumradius

    def outline(self) -> np.ndarray:
        n = self.num_sides
        start = 0.0 if n % 2 == 1 else math.pi / n
        a = start + 2.0 * math.pi * np.arange(n) / n
        return self._radius * np.column_stack([np.cos(a), np.sin(a)])

    def to_dict(self):
        return self._with_z({"side": self.side, "num_sides": self.num_sides,
                             "radius": self._radius, "orientation": self._orientation})

    @staticmethod
    def from_dict(data: dict):
        return RegularPolygon(side=data["side"], num_sides=data["num_sides"], orientation=Shape._orientation_from(data),
                              z_height=data.get("z_height"))


class CustomPolygon(Shape):
    # orientation = 0: the given points keep their axes, translated so the area centroid is the origin;
    #                  body +x points along the direction of travel
    def __init__(self, points, edges=None, orientation: float = 0.0, z_height=None):
        """Polygon from ``points`` (``(N, 2)``) joined by ``edges`` (``(N, 2)`` index pairs; default: in
        sequence, closing the loop). The edges must form one simple closed loop."""
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
            raise ValueError(f"CustomPolygon: points must be a finite (N, 2) array, got shape {pts.shape}")
        n = len(pts)
        if n < 3:
            raise ValueError(f"CustomPolygon: need at least 3 points, got {n}")

        if edges is None:
            edge_arr = np.column_stack([np.arange(n), (np.arange(n) + 1) % n])
        else:
            edge_arr = np.asarray(edges)
            if edge_arr.ndim != 2 or edge_arr.shape[1] != 2 or not np.issubdtype(edge_arr.dtype, np.integer):
                raise ValueError("CustomPolygon: edges must be an (M, 2) array of integer point indices")
        if edge_arr.min() < 0 or edge_arr.max() >= n:
            raise ValueError(f"CustomPolygon: edge index out of range for {n} points")
        if (edge_arr[:, 0] == edge_arr[:, 1]).any():
            raise ValueError("CustomPolygon: an edge joins a point to itself")
        keys = {tuple(sorted(map(int, e))) for e in edge_arr}
        if len(keys) != len(edge_arr):
            raise ValueError("CustomPolygon: duplicate edges")

        degree = np.bincount(edge_arr.ravel(), minlength=n)
        bad = np.flatnonzero(degree != 2)
        if len(bad):
            raise ValueError("CustomPolygon: every point needs exactly 2 edges; "
                             f"point(s) {bad.tolist()} have {degree[bad].tolist()}")

        nbrs = [[] for _ in range(n)]
        for a, b in keys:
            nbrs[a].append(b)
            nbrs[b].append(a)
        order, prev, cur = [0], -1, 0           # walk the loop from point 0
        while True:
            nxt = nbrs[cur][0] if nbrs[cur][0] != prev else nbrs[cur][1]
            if nxt == 0:
                break
            order.append(nxt)
            prev, cur = cur, nxt
        if len(order) != n:
            raise ValueError("CustomPolygon: edges form several separate loops, expected one")

        poly = ShapelyPolygon(pts[order])
        if not poly.is_valid:
            raise ValueError(f"CustomPolygon: invalid polygon ({shapely.is_valid_reason(poly)})")
        if poly.area < 1e-12:
            raise ValueError("CustomPolygon: polygon has zero area")
        poly = orient(poly)                      # counter-clockwise
        centroid = np.asarray(poly.centroid.coords[0])

        self._points = pts.copy()
        self._edges = edge_arr.copy()
        self._outline = np.asarray(poly.exterior.coords)[:-1] - centroid
        super().__init__(float(np.linalg.norm(self._outline, axis=1).max()), orientation, z_height)

    def outline(self) -> np.ndarray:
        return self._outline.copy()

    def to_dict(self):
        return self._with_z({"points": self._points.tolist(), "edges": self._edges.tolist(),
                             "radius": self._radius, "orientation": self._orientation})

    @staticmethod
    def from_dict(data: dict):
        return CustomPolygon(points=data["points"], edges=data.get("edges"),
                             orientation=Shape._orientation_from(data), z_height=data.get("z_height"))




# ---------------------------------------------------------------------------
# Native 3D shapes
# ---------------------------------------------------------------------------

def euler_rotation(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """The 3x3 rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (radians)."""
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch),
                              math.cos(roll), math.sin(roll))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _orientation3(orientation) -> tuple:
    """``orientation`` as ``(yaw, pitch, roll)``: a number is a yaw."""
    if np.ndim(orientation) == 0:
        return (float(orientation), 0.0, 0.0)
    o = tuple(float(x) for x in orientation)
    if len(o) != 3:
        raise ValueError(f"3D orientation: expected a yaw or (yaw, pitch, roll), got {orientation}")
    return o


class Shape3D(Shape):
    """A body in 3D: ``orientation()`` is ``(yaw, pitch, roll)``."""
    dim = 3

    def __init__(self, radius: float, orientation=0.0):
        self._radius = float(radius)
        self._orientation = _orientation3(orientation)
        self._z_height = None

    def rotation_matrix(self) -> np.ndarray:
        """The fixed body rotation relative to the travel frame."""
        return euler_rotation(*self._orientation)

    def vertices(self) -> np.ndarray:
        """``(K, 3)`` body-frame vertices before ``orientation``."""
        raise NotImplementedError(f"{type(self).__name__} has no vertices")

    def outline(self) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} is a 3D shape; see vertices()")

    def extent(self) -> tuple:
        raise NotImplementedError(f"{type(self).__name__} is a 3D shape")


class Sphere(Shape3D):
    def __init__(self, radius: float):
        if not (np.isfinite(radius) and radius > 0):
            raise ValueError(f"Sphere: radius must be positive, got {radius}")
        super().__init__(radius, 0.0)

    def to_dict(self):
        return {"radius": self._radius}

    @staticmethod
    def from_dict(data: dict):
        return Sphere(radius=data["radius"])


class ConvexPolyhedron(Shape3D):
    # orientation = 0: the given points keep their axes, translated so the volume centroid is the origin;
    #                  body +x points along the direction of travel
    def __init__(self, points, orientation=0.0):
        """The convex hull of ``points`` (``(N, 3)``, at least 4, not coplanar)."""
        from scipy.spatial import ConvexHull
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4 or not np.isfinite(pts).all():
            raise ValueError(f"ConvexPolyhedron: points must be a finite (N >= 4, 3) array, got shape {pts.shape}")
        hull = ConvexHull(pts)
        if hull.volume < 1e-12:
            raise ValueError("ConvexPolyhedron: the points are coplanar (zero volume)")
        # Volume centroid: the tetrahedra (interior point, facet triangle).
        inner = pts[hull.vertices].mean(axis=0)
        tri = pts[hull.simplices]
        vol = np.abs(np.einsum("ij,ij->i", tri[:, 0] - inner, np.cross(tri[:, 1] - inner, tri[:, 2] - inner))) / 6.0
        centroid = (vol[:, None] * (inner + tri.sum(axis=1)) / 4.0).sum(axis=0) / vol.sum()
        self._points = pts.copy()
        self._vertices = pts[hull.vertices] - centroid
        super().__init__(float(np.linalg.norm(self._vertices, axis=1).max()), orientation)

    def vertices(self) -> np.ndarray:
        return self._vertices.copy()

    def to_dict(self):
        return {"points": self._points.tolist(), "radius": self._radius, "orientation": list(self._orientation)}

    @staticmethod
    def from_dict(data: dict):
        return ConvexPolyhedron(points=data["points"], orientation=data.get("orientation", 0.0))


class Box(ConvexPolyhedron):
    # orientation = 0: length along body +x (the direction of travel), width along +y, height along +z
    def __init__(self, length: float, width: float, height: float, orientation=0.0):
        if min(length, width, height) <= 0:
            raise ValueError(f"Box: length, width and height must be positive, got {length}, {width}, {height}")
        self.length, self.width, self.height = float(length), float(width), float(height)
        corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
        super().__init__(corners * [self.length, self.width, self.height], orientation)

    def to_dict(self):
        return {"length": self.length, "width": self.width, "height": self.height,
                "radius": self._radius, "orientation": list(self._orientation)}

    @staticmethod
    def from_dict(data: dict):
        return Box(length=data["length"], width=data["width"], height=data["height"],
                   orientation=data.get("orientation", 0.0))
