"""Visual effects: juice particles, cut flash, floating score text."""
import random

import cv2
import numpy as np

import config


class Particle:
    def __init__(self, pos, color):
        ang = random.uniform(0, 2 * np.pi)
        speed = random.uniform(150, 550)
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([np.cos(ang), np.sin(ang)]) * speed + np.array([0, -150])
        self.color = color
        self.size = random.uniform(3, 8)
        self.life = self.max_life = random.uniform(0.5, 1.0) * config.PARTICLE_LIFE

    def update(self, dt):
        self.vel[1] += config.GRAVITY * 0.8 * dt
        self.pos += self.vel * dt
        self.life -= dt

    def _center_radius(self):
        k = max(self.life / self.max_life, 0)
        return (int(self.pos[0]), int(self.pos[1])), max(1, int(self.size * k))

    def draw(self, canvas, glow):
        c, r = self._center_radius()
        cv2.circle(glow, c, r * 2 + 2, self.color, -1, cv2.LINE_8)            # neon spark

    def draw_top(self, canvas):
        c, r = self._center_radius()
        cv2.circle(canvas, c, max(1, r // 2), (255, 255, 255), -1, cv2.LINE_AA)


class CutFlash:
    """Bright line that shows where the fruit was cut, fades quickly."""

    def __init__(self, a, b, center, radius):
        d = b - a
        d = d / (np.linalg.norm(d) + 1e-9)
        self.p1 = center - d * radius * 1.4
        self.p2 = center + d * radius * 1.4
        self.life = self.max_life = 0.18

    def update(self, dt):
        self.life -= dt

    def draw(self, canvas, glow_layer):
        k = max(self.life / self.max_life, 0)
        cv2.line(glow_layer, tuple(self.p1.astype(int)), tuple(self.p2.astype(int)),
                 (255, 255, 255), max(1, int(10 * k)), cv2.LINE_AA)


class FloatingText:
    def __init__(self, text, pos, color=(255, 255, 255), scale=1.0, life=0.8):
        self.text = text
        self.pos = np.array(pos, dtype=float)
        self.color = color
        self.scale = scale
        self.life = self.max_life = life

    def update(self, dt):
        self.pos[1] -= 60 * dt
        self.life -= dt

    def draw(self, canvas, glow):
        pass                                   # text is drawn on top, see draw_top

    def draw_top(self, canvas):
        org = (int(self.pos[0]), int(self.pos[1]))
        draw_text(canvas, self.text, org, self.scale, self.color, center=True)


_TEXT_CACHE = {}


def _render_text(text, scale, color, thickness):
    """Render text once into a small image + mask (putText is slow, ~9 calls per text)."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (w, h), base = cv2.getTextSize(text, font, scale, thickness)
    d = 2 if scale > 1 else 1
    pad = d + thickness + 2
    img = np.zeros((h + base + 2 * pad, w + 2 * pad, 3), np.uint8)
    mask = np.zeros(img.shape[:2], np.uint8)
    org = (pad, pad + h)
    # Outline = the same text in black, shifted a little in 8 directions.
    # (Not a thicker stroke: in OpenCV 5 a thicker stroke also makes letters wider,
    #  so the black and white text no longer line up.)
    for dx in (-d, 0, d):
        for dy in (-d, 0, d):
            if dx or dy:
                cv2.putText(mask, text, (org[0] + dx, org[1] + dy), font, scale, 255, thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(mask, text, org, font, scale, 255, thickness, cv2.LINE_AA)
    return img, mask.astype(np.float32)[..., None] / 255.0, org


def draw_text(canvas, text, org, scale, color, thickness=2, center=False):
    """Text with a black outline so it is readable on any camera background.
    Rendered texts are cached, so drawing the same text again is very cheap."""
    key = (text, scale, color, thickness)
    if key not in _TEXT_CACHE:
        if len(_TEXT_CACHE) > 300:
            _TEXT_CACHE.clear()
        _TEXT_CACHE[key] = _render_text(text, scale, color, thickness)
    img, alpha, (ox, oy) = _TEXT_CACHE[key]
    h, w = img.shape[:2]
    x, y = org[0] - ox, org[1] - oy
    if center:
        x, y = org[0] - w // 2, org[1] - h // 2
    # clip to the canvas
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = canvas[y0:y1, x0:x1]
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    t = img[y0 - y:y1 - y, x0 - x:x1 - x]
    roi[:] = (t * a + roi * (1 - a)).astype(np.uint8)


class Effects:
    def __init__(self):
        self.items = []

    def splash(self, pos, color, count=config.PARTICLES_PER_SLICE):
        for _ in range(count):
            self.items.append(Particle(pos, color))

    def add(self, item):
        self.items.append(item)

    def update(self, dt):
        for it in self.items:
            it.update(dt)
        self.items = [it for it in self.items if it.life > 0]

    def draw(self, canvas, glow_layer):
        """Glow pass (before blur)."""
        for it in self.items:
            it.draw(canvas, glow_layer)

    def draw_top(self, canvas):
        """Sharp pass (after the glow is added)."""
        for it in self.items:
            if hasattr(it, "draw_top"):
                it.draw_top(canvas)
