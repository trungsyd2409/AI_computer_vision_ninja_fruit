"""The blade = trail of one finger tip. It cuts fruits when it moves fast."""
import cv2
import numpy as np

import config


def catmull_rom(points, steps):
    """Smooth curve through the points (only for drawing, cutting uses raw points)."""
    if len(points) < 3 or steps <= 1:
        return points
    p = np.vstack([points[0], points, points[-1]])      # repeat ends
    out = []
    t = np.linspace(0, 1, steps, endpoint=False)[:, None]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(points[-1:])
    return np.vstack(out)


class Blade:
    def __init__(self):
        self.points = []       # trail for drawing: list of (x, y, t)
        self.last = None       # last accepted point (x, y, t), kept even when the trail fades
        self.speed = 0.0
        self.rejected = 0

    def add(self, pos, t):
        """Add a tracked finger position (t = camera time). Return a cut segment (a, b) or None."""
        p = np.array(pos, dtype=float)
        segment = None
        if self.last is not None and t - self.last[2] > config.BLADE_LOST_TIMEOUT:
            self.last, self.points, self.speed = None, [], 0.0   # hand was lost -> new stroke
        if self.last is not None:
            a = np.array(self.last[:2])
            dt = max(t - self.last[2], 1e-3)
            v = np.linalg.norm(p - a) / dt
            if v > config.BLADE_MAX_SPEED and self.rejected == 0:
                # impossible speed = tracking glitch: skip this point once.
                # If the next point agrees with it, the hand really moved there.
                self.rejected += 1
                return None
            if v > config.BLADE_MAX_SPEED:
                self.points, self.speed = [], 0.0          # second time -> restart the stroke
            else:
                # smooth the speed a little so one slow frame does not stop a swipe
                self.speed = 0.5 * self.speed + 0.5 * v
                if self.speed >= config.BLADE_MIN_SPEED:
                    segment = (a, p)
        self.rejected = 0
        self.last = (p[0], p[1], t)
        self.points.append(self.last)
        return segment

    def age(self, now):
        """Remove old trail points. Call once per game frame."""
        self.points = [pt for pt in self.points if now - pt[2] <= config.BLADE_TRAIL_TIME]
        if self.last is not None and now - self.last[2] > config.BLADE_LOST_TIMEOUT:
            self.last, self.speed = None, 0.0

    @property
    def alive(self):
        return self.last is not None or bool(self.points)

    @property
    def is_cutting(self):
        return self.speed >= config.BLADE_MIN_SPEED and len(self.points) >= 2

    def _segments(self):
        """Yield (a, b, k) for each trail piece; k goes 0 (tail) -> 1 (finger)."""
        if len(self.points) < 2:
            return
        pts = catmull_rom(np.array([pt[:2] for pt in self.points]), config.BLADE_SMOOTH_STEPS)
        pts = pts.astype(int)
        n = len(pts)
        for i in range(1, n):
            yield tuple(pts[i - 1]), tuple(pts[i]), i / (n - 1)

    def draw_glow(self, glow_layer):
        for a, b, k in self._segments():
            w = max(1, int(config.BLADE_WIDTH * k))
            cv2.line(glow_layer, a, b, config.BLADE_GLOW_COLOR, w * 3, cv2.LINE_AA)

    def draw_core(self, canvas):
        """Tapered trail: thin at the tail, thick at the finger + a finger cursor."""
        for a, b, k in self._segments():
            w = max(1, int(config.BLADE_WIDTH * k))
            cv2.line(canvas, a, b, config.BLADE_CORE_COLOR, w, cv2.LINE_AA)
        if self.last is not None:
            tip = (int(self.last[0]), int(self.last[1]))
            color = (80, 255, 80) if self.is_cutting else (255, 255, 255)  # green = fast enough to cut
            cv2.circle(canvas, tip, 9, color, 2, cv2.LINE_AA)
