"""Fruits (simple 2D shapes), their halves after a cut, and the spawner."""
import colorsys
import math
import random

import cv2
import numpy as np

import config
from geometry import polygon_centroid, segment_hits_polygon, split_polygon


def make_shape(shape, r):
    """Return polygon points (local, centred at 0,0) for a shape of 'radius' r."""
    if shape == "circle":
        k = 32
    elif shape == "square":
        k = 4
    elif shape == "triangle":
        k = 3
    else:
        raise ValueError(shape)
    start = math.pi / 4 if shape == "square" else -math.pi / 2
    ang = start + np.arange(k) * 2 * math.pi / k
    return np.stack([np.cos(ang) * r, np.sin(ang) * r], axis=1)


# square and triangle look smaller than a circle with the same radius
SHAPE_SCALE = {"circle": 1.0, "square": 1.2, "triangle": 1.4}


def random_color():
    """Bright random colour in BGR."""
    h = random.random()
    r, g, b = colorsys.hsv_to_rgb(h, random.uniform(0.7, 1.0), random.uniform(0.85, 1.0))
    return (int(b * 255), int(g * 255), int(r * 255))


def mix(c1, c2, t):
    return tuple(int(a * (1 - t) + b * t) for a, b in zip(c1, c2))


class Piece:
    """A flying polygon. A whole fruit and each half are both Pieces."""

    def __init__(self, local_poly, pos, vel, color, angle=0.0, spin=0.0,
                 is_half=False, cut_edge=None, shape="", radius=0.0):
        self.local = local_poly
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.angle = angle
        self.spin = spin
        self.is_half = is_half
        self.cut_edge = cut_edge          # local points of the cut (halves only)
        self.shape = shape
        self.radius = radius
        self.alive = True

    # ---------- physics ----------
    def update(self, dt):
        self.vel[1] += config.GRAVITY * dt
        self.pos += self.vel * dt
        self.angle += self.spin * dt

    def _rot(self, pts):
        c, s = math.cos(self.angle), math.sin(self.angle)
        R = np.array([[c, -s], [s, c]])
        return pts @ R.T + self.pos

    def world_poly(self):
        return self._rot(self.local)

    def is_off_screen(self):
        return self.vel[1] > 0 and self.pos[1] - self.radius > config.GAME_HEIGHT + 20

    # ---------- slicing ----------
    def hit_by(self, a, b):
        # quick circle check first (cheap), then exact polygon check
        ab = b - a
        t = np.clip(np.dot(self.pos - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
        if np.linalg.norm(a + t * ab - self.pos) > self.radius:
            return False
        return segment_hits_polygon(a, b, self.world_poly())

    def slice(self, a, b):
        """Cut along the blade direction. Return two half Pieces (or [])."""
        poly = self.world_poly()
        d = b - a
        d = d / (np.linalg.norm(d) + 1e-9)
        # point on the blade line closest to the fruit centre,
        # pulled towards the centre so we never cut off a tiny sliver
        foot = a + np.dot(self.pos - a, d) * d
        offset = foot - self.pos
        max_off = 0.35 * self.radius
        if np.linalg.norm(offset) > max_off:
            offset = offset / np.linalg.norm(offset) * max_off
        result = split_polygon(poly, self.pos + offset, d)
        if result is None:
            result = split_polygon(poly, self.pos, d)
            if result is None:
                return []
        piece_a, piece_b, cut = result
        normal = np.array([-d[1], d[0]])   # points to the side of piece_a
        halves = []
        for piece, sign in ((piece_a, 1), (piece_b, -1)):
            c = polygon_centroid(piece)
            vel = self.vel * 0.6 + sign * normal * config.HALF_PUSH_SPEED
            spin = self.spin + sign * random.uniform(1.5, 4.0)
            halves.append(Piece(piece - c, c, vel, self.color, 0.0, spin,
                                is_half=True, cut_edge=cut - c, radius=self.radius))
        return halves

    # ---------- drawing ----------
    def draw(self, canvas):
        pts = self.world_poly().astype(np.int32)
        dark = mix(self.color, (0, 0, 0), 0.45)
        if not self.is_half:
            cv2.fillPoly(canvas, [pts], self.color, cv2.LINE_AA)
            # lighter inner shape -> simple "3D" look
            inner = (self.world_poly() - self.pos) * 0.6 + self.pos + np.array([-self.radius * 0.12, -self.radius * 0.12])
            cv2.fillPoly(canvas, [inner.astype(np.int32)], mix(self.color, (255, 255, 255), 0.25), cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, dark, 3, cv2.LINE_AA)
        else:
            flesh = mix(self.color, (255, 255, 255), 0.55)
            cv2.fillPoly(canvas, [pts], flesh, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, self.color, 7, cv2.LINE_AA)   # the "skin"
            cut = self._rot(self.cut_edge).astype(np.int32)
            cv2.line(canvas, tuple(cut[0]), tuple(cut[1]), flesh, 7, cv2.LINE_AA)
            cv2.line(canvas, tuple(cut[0]), tuple(cut[1]), (255, 255, 255), 2, cv2.LINE_AA)


def launch_fruit():
    """Create one fruit below the screen, thrown upwards."""
    W, H = config.GAME_WIDTH, config.GAME_HEIGHT
    shape = random.choice(config.SHAPES)
    r = random.uniform(config.FRUIT_RADIUS_MIN, config.FRUIT_RADIUS_MAX)
    r *= SHAPE_SCALE[shape]   # so all shapes look about the same size
    x0 = random.uniform(0.15 * W, 0.85 * W)
    y0 = H + r
    peak_y = random.uniform(config.PEAK_HEIGHT_MIN, config.PEAK_HEIGHT_MAX) * H
    vy = -math.sqrt(2 * config.GRAVITY * (y0 - peak_y))
    flight_time = 2 * -vy / config.GRAVITY
    target_x = random.uniform(0.25 * W, 0.75 * W)
    vx = (target_x - x0) / flight_time
    return Piece(make_shape(shape, r), (x0, y0), (vx, vy), random_color(),
                 angle=random.uniform(0, 2 * math.pi),
                 spin=random.uniform(-config.SPIN_MAX, config.SPIN_MAX),
                 shape=shape, radius=r)


class Spawner:
    """Throws waves of fruits. Waves get bigger and faster over time."""

    def __init__(self):
        self.play_time = 0.0
        self.next_wave = 1.0   # small delay before the first wave
        self.queue = []        # (time, fruit) - fruits in a wave come one by one

    def update(self, dt):
        self.play_time += dt
        new_fruits = []
        self.next_wave -= dt
        if self.next_wave <= 0:
            size = min(config.WAVE_SIZE_MAX,
                       config.WAVE_SIZE_START + int(self.play_time // config.WAVE_SIZE_GROW_EVERY))
            size = random.randint(max(1, size - 1), size)
            for i in range(size):
                self.queue.append([i * random.uniform(0.05, 0.25), launch_fruit()])
            interval = max(config.SPAWN_INTERVAL_MIN,
                           config.SPAWN_INTERVAL_START - config.SPAWN_INTERVAL_DECAY * self.play_time)
            self.next_wave = interval + 0.3 * size
        for item in self.queue:
            item[0] -= dt
        new_fruits = [f for t, f in self.queue if t <= 0]
        self.queue = [item for item in self.queue if item[0] > 0]
        return new_fruits
