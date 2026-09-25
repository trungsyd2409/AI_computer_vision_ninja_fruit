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

    def draw(self, canvas):
        k = max(self.life / self.max_life, 0)
        r = max(1, int(self.size * k))
        cv2.circle(canvas, (int(self.pos[0]), int(self.pos[1])), r, self.color, -1, cv2.LINE_AA)


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

    def draw(self, glow_layer):
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

    def draw(self, canvas):
        org = (int(self.pos[0]), int(self.pos[1]))
        draw_text(canvas, self.text, org, self.scale, self.color, center=True)


def draw_text(canvas, text, org, scale, color, thickness=2, center=False):
    """Text with a black outline so it is readable on any camera background."""
    font = cv2.FONT_HERSHEY_DUPLEX
    if center:
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        org = (org[0] - w // 2, org[1] + h // 2)
    cv2.putText(canvas, text, org, font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(canvas, text, org, font, scale, color, thickness, cv2.LINE_AA)


class Effects:
    def __init__(self):
        self.items = []

    def splash(self, pos, color):
        for _ in range(config.PARTICLES_PER_SLICE):
            self.items.append(Particle(pos, color))

    def add(self, item):
        self.items.append(item)

    def update(self, dt):
        for it in self.items:
            it.update(dt)
        self.items = [it for it in self.items if it.life > 0]

    def draw(self, canvas, glow_layer):
        for it in self.items:
            if isinstance(it, CutFlash):
                it.draw(glow_layer)
            else:
                it.draw(canvas)
