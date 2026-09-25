"""Small 2D geometry helpers (pure numpy, no OpenCV).

Polygons are numpy arrays of shape (N, 2) with float points in order.
"""
import numpy as np


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


def split_polygon(poly, point, direction):
    """Cut a CONVEX polygon with an infinite line (point + t * direction).

    Returns (piece_a, piece_b, cut_edge) or None if the line misses the polygon.
    cut_edge is a (2, 2) array with the two points where the line crosses.
    """
    d = direction / (np.linalg.norm(direction) + 1e-9)
    side = cross2(np.broadcast_to(d, poly.shape), poly - point)  # >0 left, <0 right
    piece_a, piece_b, cut_pts = [], [], []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        sp, sq = side[i], side[(i + 1) % n]
        if sp >= 0:
            piece_a.append(p)
        if sp <= 0:
            piece_b.append(p)
        if (sp > 0 and sq < 0) or (sp < 0 and sq > 0):
            t = sp / (sp - sq)
            x = p + t * (q - p)
            piece_a.append(x)
            piece_b.append(x)
            cut_pts.append(x)
    if len(piece_a) < 3 or len(piece_b) < 3 or len(cut_pts) < 2:
        return None
    return np.array(piece_a), np.array(piece_b), np.array(cut_pts[:2])
