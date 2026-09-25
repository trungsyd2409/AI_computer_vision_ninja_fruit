"""Small 2D geometry helpers (pure numpy, no OpenCV).

Polygons are numpy arrays of shape (N, 2) with float points in order.
They can be concave (star, heart, cross).
"""
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import split


def cross2(a, b):
    """2D cross product (z of a x b)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid(poly):
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    c = x * y1 - x1 * y
    a = c.sum() / 2.0
    if abs(a) < 1e-9:
        return poly.mean(axis=0)
    cx = ((x + x1) * c).sum() / (6 * a)
    cy = ((y + y1) * c).sum() / (6 * a)
    return np.array([cx, cy])


def point_in_polygon(pt, poly):
    """Ray casting test."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def segments_intersect(p1, p2, q1, q2):
    """True if segment p1-p2 crosses segment q1-q2."""
    r = p2 - p1
    s = q2 - q1
    denom = cross2(r, s)
    if abs(denom) < 1e-9:
        return False  # parallel
    qp = q1 - p1
    t = cross2(qp, s) / denom
    u = cross2(qp, r) / denom
    return 0 <= t <= 1 and 0 <= u <= 1


def segment_hits_polygon(a, b, poly):
    """True if the segment a-b touches the polygon (crosses an edge or is inside)."""
    if point_in_polygon(a, poly) or point_in_polygon(b, poly):
        return True
    n = len(poly)
    for i in range(n):
        if segments_intersect(a, b, poly[i], poly[(i + 1) % n]):
            return True
    return False


def split_by_line(poly, point, direction):
    """Cut ANY simple polygon (convex or not, e.g. a star) with an infinite line.

    Returns a list of pieces (numpy arrays). A convex shape gives 2 pieces,
    a star cut through two arms can give 3 or more. [] if the line misses.
    Uses shapely (a GIS geometry library) because splitting concave polygons
    correctly is tricky to write by hand.
    """
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-9)
    far = 10000.0
    line = LineString([tuple(point - d * far), tuple(point + d * far)])
    shape = Polygon(poly)
    if not shape.is_valid:
        shape = shape.buffer(0)          # fixes small self-intersections
    if not shape.intersects(line):
        return []
    pieces = []
    for g in split(shape, line).geoms:
        if isinstance(g, Polygon) and g.area > 1.0:
            pieces.append(np.array(g.exterior.coords)[:-1])   # last point = first point
    return pieces if len(pieces) >= 2 else []


def edges_on_line(poly, point, direction, tol=0.5):
    """Boolean per edge i (poly[i] -> poly[i+1]): True if the edge lies on the cut line."""
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-9)
    dist = np.abs(cross2(np.broadcast_to(d, poly.shape), poly - point))
    on = dist < tol
    return on & np.roll(on, -1)
