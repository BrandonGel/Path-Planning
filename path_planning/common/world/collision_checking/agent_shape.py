"""Agent footprints: a disc or a polygon in the agent's body frame.

Every shape has its reference point (the centroid) at the origin and an
``orientation`` in radians: the rotation of the body relative to the direction
of travel. While an agent moves, the shape keeps following the direction of
travel; ``orientation`` is a fixed extra rotation on top of it, so
``orientation = 0`` means the shape's reference direction (below) points along
the move. ``radius()`` is the bounding radius about the centroid.

Reference direction at ``orientation = 0``:
    Triangle        centroid -> the vertex joined by side 1 and side 2
    Rectangle       along the width (perpendicular to the height sides)
    RegularPolygon  centroid -> a vertex for an odd number of sides,
                    centroid -> the middle of a side for an even number
    CustomPolygon   the given points' own +x axis, shifted so the area centroid is the origin
    Circle          rotation-invariant (orientation is always 0)
"""
import math

import numpy as np
import shapely
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.polygon import orient


class Shape:
    def __init__(self, radius: float = 1, orientation: float = 0.0):
        self._radius = radius
        self._orientation = float(orientation)

    def to_dict(self):
        return {"radius": self._radius, "orientation": self._orientation}

    @staticmethod
    def _orientation_from(data: dict) -> float:
        """``orientation`` of a saved dict; dicts saved before the rename call it ``heading``."""
        return data.get("orientation", data.get("heading", 0.0))

    @staticmethod
    def from_dict(data: dict):
        return Shape(radius=data["radius"], orientation=Shape._orientation_from(data))

    def radius(self):
        """Bounding radius about the centroid."""
        return self._radius

    def orientation(self):
        """Rotation of the body relative to the direction of travel, in radians."""
        return self._orientation

    def outline(self) -> np.ndarray:
        """``(K, 2)`` body-frame vertices (centroid at the origin) at ``orientation = 0``,
        counter-clockwise. The orientation is applied by the caller."""
        raise NotImplementedError(f"{type(self).__name__} has no outline")

    def extent(self) -> tuple:
        """``(x_min, x_max, y_min, y_max)`` of the body in the travel frame (centroid at the origin,
        +x along the direction of travel, ``orientation`` applied): ``x_min`` is how far the body
        reaches behind the centroid, ``x_max`` ahead of it, and ``y`` is the lateral reach."""
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
        """Length of the face the body presents to the direction of travel: its lateral extent
        (perpendicular to the travel direction) with ``orientation`` applied."""
        _, _, y_min, y_max = self.extent()
        return y_max - y_min


class Circle(Shape):
    def __init__(self, radius: float):
        super().__init__(radius, 0.0)

    def to_dict(self):
        return {"radius": self._radius}

    @staticmethod
    def from_dict(data: dict):
        return Circle(radius=data["radius"])

    def extent(self) -> tuple:
        r = float(self._radius)
        return (-r, r, -r, r)

    def outline(self, quad_segs: int = 32) -> np.ndarray:
        n = 4 * quad_segs
        a = 2.0 * math.pi * np.arange(n) / n
        return self._radius * np.column_stack([np.cos(a), np.sin(a)])


class Triangle(Shape):
    # orientation = 0: the direction from the centroid to the vertex joined by side 1 and side 2
    def __init__(self, side1: float, side2: float, side3: float, orientation: float = 0.0):
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
        super().__init__(float(np.linalg.norm(self._outline, axis=1).max()), orientation)

    def outline(self) -> np.ndarray:
        return self._outline.copy()

    def to_dict(self):
        return {"side1": self._side1, "side2": self._side2, "side3": self._side3,
                "radius": self._radius, "orientation": self._orientation}

    @staticmethod
    def from_dict(data: dict):
        return Triangle(side1=data["side1"], side2=data["side2"], side3=data["side3"],
                        orientation=Shape._orientation_from(data))


class Rectangle(Shape):
    # orientation = 0: the direction from the centroid along the width (perpendicular to the height sides)
    def __init__(self, width: float, height: float, orientation: float = 0.0):
        if width <= 0 or height <= 0:
            raise ValueError(f"Rectangle: width and height must be positive, got {width}, {height}")
        self.width = width
        self.height = height
        super().__init__(0.5 * math.hypot(width, height), orientation)     # half the diagonal

    def outline(self) -> np.ndarray:
        w, h = 0.5 * self.width, 0.5 * self.height
        return np.array([[w, -h], [w, h], [-w, h], [-w, -h]])

    def to_dict(self):
        return {"width": self.width, "height": self.height,
                "radius": self._radius, "orientation": self._orientation}

    @staticmethod
    def from_dict(data: dict):
        return Rectangle(width=data["width"], height=data["height"], orientation=Shape._orientation_from(data))


class Square(Rectangle):
    def __init__(self, side: float, orientation: float = 0.0):
        super().__init__(side, side, orientation)
        self.side = side

    def to_dict(self):
        return {"side": self.side, "radius": self._radius, "orientation": self._orientation}

    @staticmethod
    def from_dict(data: dict):
        return Square(side=data["side"], orientation=Shape._orientation_from(data))


class RegularPolygon(Shape):
    # orientation = 0: centroid -> vertex for an odd number of sides,
    #                  centroid -> the middle of a side for an even number of sides
    def __init__(self, side: float, num_sides: int, orientation: float = 0.0):
        if num_sides < 3 or side <= 0:
            raise ValueError(f"RegularPolygon: need side > 0 and num_sides >= 3, got {side}, {num_sides}")
        self.side = side
        self.num_sides = num_sides
        super().__init__(side / (2.0 * math.sin(math.pi / num_sides)), orientation)   # circumradius

    def outline(self) -> np.ndarray:
        n = self.num_sides
        start = 0.0 if n % 2 == 1 else math.pi / n
        a = start + 2.0 * math.pi * np.arange(n) / n
        return self._radius * np.column_stack([np.cos(a), np.sin(a)])

    def to_dict(self):
        return {"side": self.side, "num_sides": self.num_sides,
                "radius": self._radius, "orientation": self._orientation}

    @staticmethod
    def from_dict(data: dict):
        return RegularPolygon(side=data["side"], num_sides=data["num_sides"], orientation=Shape._orientation_from(data))


class CustomPolygon(Shape):
    # orientation = 0: the given points keep their axes, translated so the area centroid is the origin;
    #                  body +x points along the direction of travel
    def __init__(self, points, edges=None, orientation: float = 0.0):
        """Polygon from ``points`` (``(N, 2)``) joined by ``edges`` (``(N, 2)`` index pairs).

        Without ``edges`` the points are joined in sequence and the last connects to the first.
        Every point must have exactly two edges and the edges must form one simple closed loop.
        """
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
        super().__init__(float(np.linalg.norm(self._outline, axis=1).max()), orientation)

    def outline(self) -> np.ndarray:
        return self._outline.copy()

    def to_dict(self):
        return {"points": self._points.tolist(), "edges": self._edges.tolist(),
                "radius": self._radius, "orientation": self._orientation}

    @staticmethod
    def from_dict(data: dict):
        return CustomPolygon(points=data["points"], edges=data.get("edges"),
                             orientation=Shape._orientation_from(data))


Polygon = RegularPolygon        # the earlier name; it clashes with ``shapely.Polygon``, prefer RegularPolygon
