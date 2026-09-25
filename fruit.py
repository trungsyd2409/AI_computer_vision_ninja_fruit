"""Fruits (2D neon shapes), the pieces after a cut, and the spawner."""
import math
import random

import cv2
import numpy as np

import config
from geometry import (edges_on_line, polygon_area, polygon_centroid,
                      segment_hits_polygon, split_by_line)


# ---------------------------------------------------------------- shapes
def _regular(k, r, start=-math.pi / 2):
    ang = start + np.arange(k) * 2 * math.pi / k
    return np.stack([np.cos(ang) * r, np.sin(ang) * r], axis=1)


def _base_shape(shape):
    """Shape with 'radius' about 1, centred at (0, 0)."""
    if shape == "circle":
        return _regular(32, 1.0)
    if shape == "oval":
        p = _regular(32, 1.0)
        return p * [1.0, 0.68]
    if shape == "square":
        return _regular(4, 1.0, math.pi / 4)
    if shape == "rectangle":
        return np.array([[-1.0, -0.6], [1.0, -0.6], [1.0, 0.6], [-1.0, 0.6]])
    if shape == "triangle":
        return _regular(3, 1.0)
    if shape == "diamond":
        return np.array([[0, -1.0], [0.62, 0], [0, 1.0], [-0.62, 0]])
    if shape == "pentagon":
        return _regular(5, 1.0)
    if shape == "hexagon":
        return _regular(6, 1.0, 0)
    if shape == "octagon":
        return _regular(8, 1.0, math.pi / 8)
    if shape == "star":                                    # 5 points, concave
        ang = -math.pi / 2 + np.arange(10) * math.pi / 5
        rad = np.where(np.arange(10) % 2 == 0, 1.0, 0.45)
        return np.stack([np.cos(ang) * rad, np.sin(ang) * rad], axis=1)
    if shape == "heart":                                   # classic heart curve, concave
        t = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        return np.stack([x, y], axis=1) / 16.0
    if shape == "cross":                                   # plus sign, concave
        a, b = 1.0, 0.34
        return np.array([[-b, -a], [b, -a], [b, -b], [a, -b], [a, b], [b, b],
                         [b, a], [-b, a], [-b, b], [-a, b], [-a, -b], [-b, -b]])
    raise ValueError(f"unknown shape {shape}")


def make_shape(shape, r):
    """Polygon for `shape` with the SAME AREA as a circle of radius r
    (so a star is not tiny next to a circle). Centred at its centroid."""
    p = _base_shape(shape)
    p = p - polygon_centroid(p)
    scale = r * math.sqrt(math.pi / polygon_area(p))
    return p * scale


# ---------------------------------------------------------------- colours
def random_color():
    return random.choice(config.NEON_COLORS)


def mix(c1, c2, t):
    return tuple(int(a * (1 - t) + b * t) for a, b in zip(c1, c2))


def scale_color(c, k):
    return tuple(int(v * k) for v in c)


# ---------------------------------------------------------------- piece
class Piece:
    """A flying polygon. A whole fruit and every cut piece are Pieces.

    generation 0 = whole fruit, 1 = piece after 1 cut, 2 = piece of a piece...
    """

    def __init__(self, local_poly, pos, vel, color, angle=0.0, spin=0.0,
                 generation=0, cut_edges=None, shape="", radius=None):
        self.local = np.asarray(local_poly, dtype=float)
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.angle = angle
        self.spin = spin
        self.generation = generation
        self.cut_edges = cut_edges            # bool per edge: True = fresh cut (drawn white-hot)
        self.shape = shape
        self.radius = radius if radius is not None else float(np.linalg.norm(self.local, axis=1).max())
        self.area = polygon_area(self.local)
        self.age = 0.0
        self.alive = True

    @property
    def is_half(self):
        return self.generation > 0

    @property
    def can_be_cut(self):
        if self.generation == 0:
            return True
        return (self.generation < config.MAX_CUTS_PER_FRUIT
                and self.area >= config.PIECE_MIN_AREA
                and self.age >= config.PIECE_CUT_COOLDOWN)

    # ---------- physics ----------
    def update(self, dt):
        self.vel[1] += config.GRAVITY * dt
        self.pos += self.vel * dt
        self.angle += self.spin * dt
        self.age += dt

    def _rot(self, pts):
        c, s = math.cos(self.angle), math.sin(self.angle)
        R = np.array([[c, -s], [s, c]])
        return pts @ R.T + self.pos

    def world_poly(self):
        return self._rot(self.local)

    def is_off_screen(self):
        """Gone for good: below the screen and falling, or past a side and moving away."""
        x, y, r = self.pos[0], self.pos[1], self.radius
        if self.vel[1] > 0 and y - r > config.GAME_HEIGHT + 20:
            return True
        return (x + r < -20 and self.vel[0] < 0) or (x - r > config.GAME_WIDTH + 20 and self.vel[0] > 0)

    # ---------- slicing ----------
    def hit_by(self, a, b):
        # quick circle check first (cheap), then exact polygon check
        ab = b - a
        t = np.clip(np.dot(self.pos - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
        if np.linalg.norm(a + t * ab - self.pos) > self.radius:
            return False
        return segment_hits_polygon(a, b, self.world_poly())

    def slice(self, a, b):
        """Cut along the blade direction. Return the new pieces (2 or more), or []."""
        poly = self.world_poly()
        d = b - a
        d = d / (np.linalg.norm(d) + 1e-9)
        # point on the blade line closest to the centre, pulled towards the centre
        # so we never cut off a tiny sliver
        foot = a + np.dot(self.pos - a, d) * d
        offset = foot - self.pos
        max_off = 0.35 * self.radius
        if np.linalg.norm(offset) > max_off:
            offset = offset / np.linalg.norm(offset) * max_off
        p = self.pos + offset
        parts = split_by_line(poly, p, d)
        if not parts:
            p = self.pos
            parts = split_by_line(poly, p, d)
        normal = np.array([-d[1], d[0]])
        pieces = []
        for part in parts:
            c = polygon_centroid(part)
            side = 1.0 if np.dot(c - p, normal) >= 0 else -1.0
            vel = self.vel * 0.6 + side * normal * config.HALF_PUSH_SPEED * random.uniform(0.8, 1.2)
            spin = self.spin + side * random.uniform(1.5, 4.0)
            pieces.append(Piece(part - c, c, vel, self.color, 0.0, spin,
                                generation=self.generation + 1,
                                cut_edges=edges_on_line(part, p, d)))
        return pieces

    # ---------- drawing ----------
    def _fresh_cut(self):
        """How hot the fresh cut edge is: 1 just after the cut -> 0 after 0.6 s."""
        if self.cut_edges is None or not self.cut_edges.any():
            return 0.0
        return max(0.0, 1.0 - self.age / 0.6)

    def draw_glow(self, glow):
        """Pass 1: wide border on the glow layer (it gets blurred later).
        LINE_8 is enough here (faster than LINE_AA) because the glow is blurred anyway."""
        pts = self.world_poly().astype(np.int32)
        cv2.polylines(glow, [pts], True, self.color, config.GLOW_WIDTH, cv2.LINE_8)
        heat = self._fresh_cut()
        if heat > 0:
            hot = mix(self.color, (255, 255, 255), 0.35 + 0.65 * heat)
            for i in np.flatnonzero(self.cut_edges):
                cv2.line(glow, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), hot,
                         config.GLOW_WIDTH, cv2.LINE_8)

    def draw(self, canvas):
        """Pass 2 (after the glow): flat colour inside + sharp bright border.
        Drawing the fill AFTER the glow keeps the inside flat (no glow inside)."""
        pts = self.world_poly().astype(np.int32)
        cv2.fillPoly(canvas, [pts], scale_color(self.color, config.FILL_BRIGHTNESS), cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, mix(self.color, (255, 255, 255), 0.35),
                      config.BORDER_WIDTH, cv2.LINE_AA)
        heat = self._fresh_cut()
        if heat > 0:
            hot = mix(self.color, (255, 255, 255), 0.35 + 0.65 * heat)
            for i in np.flatnonzero(self.cut_edges):
                cv2.line(canvas, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), hot,
                         config.BORDER_WIDTH, cv2.LINE_AA)


def launch_fruit(side=None):
    """Create one fruit just outside the screen and throw it in.

    side = "bottom": shot up from below (like Fruit Ninja)
           "left" / "right": flies in from the side in an arc, falls out at the bottom
    """
    W, H = config.GAME_WIDTH, config.GAME_HEIGHT
    g = config.GRAVITY
    if side is None:
        sides = list(config.SPAWN_SIDES)
        side = random.choices(sides, weights=[config.SPAWN_SIDES[s] for s in sides])[0]
    shape = random.choice(config.SHAPES)
    r = random.uniform(config.FRUIT_RADIUS_MIN, config.FRUIT_RADIUS_MAX)
    local = make_shape(shape, r)
    radius = float(np.linalg.norm(local, axis=1).max())

    if side == "bottom":
        x0 = random.uniform(0.15 * W, 0.85 * W)
        y0 = H + radius
        peak_y = random.uniform(config.PEAK_HEIGHT_MIN, config.PEAK_HEIGHT_MAX) * H
        exit_x = random.uniform(0.25 * W, 0.75 * W)       # where it falls back out
    else:
        x0 = -radius if side == "left" else W + radius
        y0 = random.uniform(*config.SIDE_START_HEIGHT) * H
        top = min(config.PEAK_HEIGHT_MAX * H, y0 - 0.1 * H)  # must go up at least 10% of H
        peak_y = random.uniform(config.PEAK_HEIGHT_MIN * H, top)
        far = random.uniform(0.55 * W, 1.0 * W)             # lands on the other half
        exit_x = far if side == "left" else W - far

    # physics: go up to peak_y, then fall until below the screen (y = H + radius)
    vy = -math.sqrt(2 * g * (y0 - peak_y))
    t_up = -vy / g
    t_down = math.sqrt(2 * (H + radius - peak_y) / g)
    vx = (exit_x - x0) / (t_up + t_down)
    return Piece(local, (x0, y0), (vx, vy), random_color(),
                 angle=random.uniform(0, 2 * math.pi),
                 spin=random.uniform(-config.SPIN_MAX, config.SPIN_MAX),
                 shape=shape, radius=radius)


class Spawner:
    """Throws fruits in WAVES.

    1. throw a wave (fruits come very close together)
    2. wait until every whole fruit is gone (cut or fell off the screen)
    3. rest WAVE_REST seconds, then the next wave (random size WAVE_SIZE_MIN..MAX)
    """

    def __init__(self):
        self.wave = 0
        self.next_wave = config.WAVE_FIRST_DELAY   # rest countdown (runs only when screen is clear)
        self.queue = []        # [delay, fruit] - fruits of the current wave not thrown yet

    def _start_wave(self):
        self.wave += 1
        size = random.randint(config.WAVE_SIZE_MIN, config.WAVE_SIZE_MAX)   # random 1..5
        size = min(size, config.MAX_FRUITS_ON_SCREEN)
        delay = 0.0
        for _ in range(size):
            self.queue.append([delay, launch_fruit()])
            delay += random.uniform(0, config.WAVE_STAGGER)

    def update(self, dt, on_screen=0):
        """on_screen = number of whole fruits still flying. Returns fruits to add now."""
        if not self.queue and on_screen == 0:     # wave cleared -> rest, then next wave
            self.next_wave -= dt
            if self.next_wave <= 0:
                self._start_wave()
                self.next_wave = config.WAVE_REST
        for item in self.queue:
            item[0] -= dt
        new_fruits = [f for t, f in self.queue if t <= 0]
        self.queue = [item for item in self.queue if item[0] > 0]
        return new_fruits
