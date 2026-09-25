"""Game logic + drawing. It does not know about the camera,
so it can be tested with fake finger positions."""
import json

import cv2
import numpy as np

import config
from blade import Blade
from effects import CutFlash, Effects, FloatingText, draw_text
from fruit import Spawner


def load_highscore():
    try:
        with open(config.HIGHSCORE_FILE) as f:
            return int(json.load(f).get("best", 0))
    except (OSError, ValueError):
        return 0


def save_highscore(best):
    try:
        with open(config.HIGHSCORE_FILE, "w") as f:
            json.dump({"best": best}, f)
    except OSError:
        pass


class Game:
    def __init__(self):
        self.best = load_highscore()
        self.reset()

    def reset(self):
        self.fruits = []
        self.halves = []
        self.blades = {}           # hand id -> Blade
        self.effects = Effects()
        self.spawner = Spawner()
        self.score = 0
        self.sliced = 0            # whole fruits cut
        self.cuts = 0              # all cuts (fruits + pieces)
        self.missed = 0
        self.combo_count = 0
        self.combo_pos = None
        self.last_cut_time = -10.0
        self.best_combo = 0
        self.paused = False

    # ------------------------------------------------------------------ update
    def update(self, dt, now, tracking=()):
        """tracking = list of (tips, capture_time) from HandTracker.poll(),
        tips = {hand_id: (x, y)} finger positions in game pixels."""
        dt = min(dt, 0.1)   # avoid huge jumps if a frame was very slow

        # 1) blades follow the fingers (always, even when paused, so you see the cursor)
        segments = []
        for tips, t in tracking:
            for hand_id, pos in tips.items():
                seg = self.blades.setdefault(hand_id, Blade()).add(pos, t)
                if seg is not None:
                    segments.append(seg)
        for blade in self.blades.values():
            blade.age(now)
        self.blades = {k: b for k, b in self.blades.items() if b.alive}

        if self.paused:
            return

        # 2) spawn + move
        self.fruits.extend(self.spawner.update(dt, len(self.fruits)))
        for p in self.fruits + self.halves:
            p.update(dt)

        # 3) slicing: whole fruits AND pieces can be cut (pieces: see Piece.can_be_cut)
        new_pieces = []
        for a, b in segments:
            for piece in self.fruits + self.halves:
                if piece.alive and piece.can_be_cut and piece.hit_by(a, b):
                    new_pieces.extend(self._cut(piece, a, b, now))
        self.fruits = [f for f in self.fruits if f.alive]
        self.halves = [h for h in self.halves if h.alive] + new_pieces

        # 4) remove pieces that fell off the screen
        for f in self.fruits:
            if f.is_off_screen():
                f.alive = False
                self.missed += 1
        self.fruits = [f for f in self.fruits if f.alive]
        self.halves = [h for h in self.halves if not h.is_off_screen()]

        # 5) combo ends when no cut happened for COMBO_WINDOW seconds
        if self.combo_count and now - self.last_cut_time > config.COMBO_WINDOW:
            self._finish_combo()

        self.effects.update(dt)

    def _cut(self, fruit, a, b, now):
        """Cut one fruit or piece. Returns the new pieces."""
        pieces = fruit.slice(a, b)
        if not pieces:
            return []
        fruit.alive = False
        size = min(1.0, fruit.area / (3.14 * config.FRUIT_RADIUS_MAX ** 2))   # small piece = small splash
        self.effects.splash(fruit.pos, fruit.color, max(4, int(config.PARTICLES_PER_SLICE * size)))
        self.effects.add(CutFlash(a, b, fruit.pos.copy(), fruit.radius))
        points = config.POINTS_PER_FRUIT if fruit.generation == 0 else config.POINTS_PER_PIECE
        self.score += points
        self.cuts += 1
        if fruit.generation == 0:
            self.sliced += 1
        self.effects.add(FloatingText(f"+{points}", fruit.pos.copy(), scale=0.9 if fruit.generation == 0 else 0.7))

        if now - self.last_cut_time <= config.COMBO_WINDOW:
            self.combo_count += 1
        else:
            self._finish_combo()
            self.combo_count = 1
        self.last_cut_time = now
        self.combo_pos = fruit.pos.copy()
        return pieces

    def _finish_combo(self):
        n = self.combo_count
        if n >= config.COMBO_MIN:
            bonus = n                      # like Fruit Ninja: combo of n fruits = +n bonus
            self.score += bonus
            self.best_combo = max(self.best_combo, n)
            pos = np.clip(self.combo_pos, [200, 100], [config.GAME_WIDTH - 200, config.GAME_HEIGHT - 100])
            self.effects.add(FloatingText(f"COMBO x{n}  +{bonus}", pos, (0, 215, 255), 1.6, 1.2))
        self.combo_count = 0
        if self.score > self.best:
            self.best = self.score
            save_highscore(self.best)

    # ------------------------------------------------------------------ draw
    def draw(self, frame, now, stats=None, landmarks=None):
        canvas = frame.copy()
        if config.BACKGROUND_DIM > 0:
            canvas = cv2.convertScaleAbs(canvas, alpha=1.0 - config.BACKGROUND_DIM)
        glow = np.zeros_like(canvas)

        if landmarks:
            for hand in landmarks:
                for i, (x, y) in enumerate(hand):
                    color = (0, 0, 255) if i in (5, 6, 8) else (0, 255, 255)   # index finger in red
                    cv2.circle(canvas, (int(x), int(y)), 4 if i in (5, 6, 8) else 3, color, -1)

        pieces = self.halves + self.fruits
        # pass 1: everything that glows goes on the glow layer
        for p in pieces:
            p.draw_glow(glow)
        self.effects.draw(canvas, glow)
        for blade in self.blades.values():
            blade.draw_glow(glow)

        # soft glow: blur a small copy (fast) and ADD it to the image (light adds up)
        small = cv2.resize(glow, (canvas.shape[1] // 4, canvas.shape[0] // 4), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), config.GLOW_BLUR)
        canvas = cv2.add(canvas, cv2.resize(small, (canvas.shape[1], canvas.shape[0])))

        # pass 2: sharp shapes on top (flat fill + bright border), then blade cores
        for p in pieces:
            p.draw(canvas)
        self.effects.draw_top(canvas)
        for blade in self.blades.values():
            blade.draw_core(canvas)

        self._draw_hud(canvas, stats or {})
        return canvas

    def _draw_hud(self, canvas, stats):
        W, H = config.GAME_WIDTH, config.GAME_HEIGHT
        draw_text(canvas, f"SCORE {self.score}", (25, 55), 1.4, (255, 255, 255), 3)
        draw_text(canvas, f"BEST {self.best}   WAVE {self.spawner.wave}", (25, 95), 0.8, (200, 255, 200))
        draw_text(canvas, f"Fruits {self.sliced}   Cuts {self.cuts}   Missed {self.missed}   Best combo {self.best_combo}",
                  (25, H - 25), 0.6, (230, 230, 230), 1)
        fps_text = "   ".join(f"{k} {v:.0f}" for k, v in stats.get("fps", {}).items())
        if fps_text:
            (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            draw_text(canvas, fps_text, (W - tw - 20, 35), 0.6, (230, 230, 230), 1)
        for i, line in enumerate(stats.get("lines", [])):
            draw_text(canvas, line, (W - 330, 65 + i * 26), 0.55, (180, 255, 255), 1)
        if self.combo_count >= 2:
            draw_text(canvas, f"x{self.combo_count}", (W // 2 - 30, 60), 1.2, (0, 215, 255), 2)
        if not self.blades:
            draw_text(canvas, "Show your hand - swipe with your INDEX finger", (W // 2, H - 70),
                      0.8, (255, 255, 255), 2, center=True)
        if self.paused:
            draw_text(canvas, "PAUSED", (W // 2, H // 2 - 30), 2.2, (255, 255, 255), 4, center=True)
            draw_text(canvas, "P: resume   R: restart   Q: quit", (W // 2, H // 2 + 40),
                      0.9, (255, 255, 255), 2, center=True)
