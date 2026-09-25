"""Tests that run without a webcam:  python -m pytest tests  (or python tests/test_logic.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from fruit import Piece, make_shape  # noqa: E402
from game import Game  # noqa: E402
from geometry import polygon_area, segment_hits_polygon, split_polygon  # noqa: E402


def test_split_square_in_half():
    sq = make_shape("square", 50)
    a, b, cut = split_polygon(sq, np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert abs(polygon_area(a) + polygon_area(b) - polygon_area(sq)) < 1e-6
    assert abs(polygon_area(a) - polygon_area(b)) < 1e-6
    assert cut.shape == (2, 2)


def test_split_misses():
    tri = make_shape("triangle", 50)
    assert split_polygon(tri, np.array([0.0, 500.0]), np.array([1.0, 0.0])) is None


def test_segment_hit():
    circle = make_shape("circle", 50)
    assert segment_hits_polygon(np.array([-100.0, 0]), np.array([100.0, 0]), circle)
    assert not segment_hits_polygon(np.array([-100.0, 80]), np.array([100.0, 80]), circle)


def test_slice_keeps_area():
    for shape in config.SHAPES:
        f = Piece(make_shape(shape, 50), (300, 300), (0, 0), (0, 0, 255), angle=0.7, shape=shape, radius=50)
        halves = f.slice(np.array([200.0, 290]), np.array([400.0, 320]))
        assert len(halves) == 2
        total = sum(polygon_area(h.world_poly()) for h in halves)
        assert abs(total - polygon_area(f.world_poly())) < 1e-3


def swipe(game, fruits, y, t0, steps=10, dt=1 / 30):
    """Move a fake finger left->right across the screen at height y."""
    t = t0
    for i in range(steps + 1):
        x = 100 + i * (config.GAME_WIDTH - 200) / steps
        game.update(dt, t, [({"Right": (x, y)}, t)])
        t += dt
    return t


def test_combo_scoring(tmp_path=None):
    config.HIGHSCORE_FILE = os.path.join(tmp_path or "/tmp", "hs_test.json")
    game = Game()
    game.spawner.next_wave = 1e9           # no random fruits
    for i in range(4):                      # 4 fruits on one line, not moving
        f = Piece(make_shape("circle", 45), (250 + i * 250, 360), (0, 0), (0, 200, 0), shape="circle", radius=45)
        game.fruits.append(f)
    config_g = config.GRAVITY
    config.GRAVITY = 0
    try:
        t = swipe(game, game.fruits, 360, 0.0)
        game.update(1 / 30, t + 1.0)    # wait -> combo ends
    finally:
        config.GRAVITY = config_g
    assert game.sliced == 4
    assert game.score == 4 + 4               # 4 fruits + combo x4 bonus
    assert game.best_combo == 4


def test_slow_finger_does_not_cut():
    game = Game()
    game.spawner.next_wave = 1e9
    config_g = config.GRAVITY
    config.GRAVITY = 0
    try:
        game.fruits.append(Piece(make_shape("square", 45), (640, 360), (0, 0), (0, 0, 200), radius=45))
        t = 0.0
        for i in range(60):                 # 200 px in 2 s = 100 px/s -> too slow
            game.update(1 / 30, t, [({"Left": (540 + i * 3.3, 360)}, t)])
            t += 1 / 30
    finally:
        config.GRAVITY = config_g
    assert game.sliced == 0


def test_finger_guard_fixes_folded_tip():
    from hand_tracker import guard_fingertip
    mcp, pip = np.array([100.0, 300]), np.array([100.0, 250])      # finger points up
    ok_tip = np.array([102.0, 198])
    tip, guarded, _ = guard_fingertip(ok_tip, pip, mcp)
    assert not guarded and np.allclose(tip, ok_tip)
    folded = np.array([105.0, 262])                                   # blur: tip fell to the palm
    tip, guarded, _ = guard_fingertip(folded, pip, mcp, ratio_ref=1.0)
    assert guarded and np.allclose(tip, [100, 200])


def test_blade_skips_single_glitch():
    from blade import Blade
    b = Blade()
    b.add((100, 100), 0.0)
    b.add((200, 100), 1 / 30)                 # 3000 px/s, normal swipe
    assert b.add((1200, 700), 2 / 30) is None  # 1 glitch point (35000 px/s) -> ignored
    seg = b.add((400, 100), 3 / 30)            # back on track: stroke continues
    assert seg is not None and np.allclose(seg[0], [200, 100])


def test_blade_keeps_stroke_at_low_fps():
    from blade import Blade
    b = Blade()
    b.add((100, 300), 0.0)
    b.age(0.1)
    seg = b.add((700, 300), 0.1)              # 10 fps, 600 px jump = 6000 px/s -> real swipe
    assert seg is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
