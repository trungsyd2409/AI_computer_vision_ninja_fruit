"""Tests that run without a webcam:  python -m pytest tests  (or python tests/test_logic.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from fruit import Piece, make_shape  # noqa: E402
from game import Game  # noqa: E402
from geometry import polygon_area, segment_hits_polygon, split_by_line  # noqa: E402


def test_split_square_in_half():
    sq = make_shape("square", 50)
    a, b = split_by_line(sq, np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert abs(polygon_area(a) + polygon_area(b) - polygon_area(sq)) < 1e-6
    assert abs(polygon_area(a) - polygon_area(b)) < 1e-6


def test_split_misses():
    tri = make_shape("triangle", 50)
    assert split_by_line(tri, np.array([0.0, 500.0]), np.array([1.0, 0.0])) == []


def test_split_star_two_arms_gives_3_pieces():
    star = make_shape("star", 50)
    low = star[:, 1].max()                        # cut near the bottom: through 2 arms
    pieces = split_by_line(star, np.array([0.0, low * 0.7]), np.array([1.0, 0.0]))
    assert len(pieces) == 3
    assert abs(sum(polygon_area(p) for p in pieces) - polygon_area(star)) < 1e-3


def test_all_shapes_same_area():
    for shape in config.SHAPES:
        assert abs(polygon_area(make_shape(shape, 50)) - np.pi * 50 ** 2) < 1.0, shape


def test_segment_hit():
    circle = make_shape("circle", 50)
    assert segment_hits_polygon(np.array([-100.0, 0]), np.array([100.0, 0]), circle)
    assert not segment_hits_polygon(np.array([-100.0, 80]), np.array([100.0, 80]), circle)


def test_slice_keeps_area():
    for shape in config.SHAPES:
        f = Piece(make_shape(shape, 50), (300, 300), (0, 0), (0, 0, 255), angle=0.7, shape=shape, radius=50)
        halves = f.slice(np.array([200.0, 290]), np.array([400.0, 320]))
        assert len(halves) >= 2
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


def test_fps_counter_bursty_frames():
    from fps import FpsCounter
    c, t = FpsCounter(), 0.0
    for i in range(60):                 # frames in pairs: 2 ms apart, then 64 ms gap
        t += 0.002 if i % 2 else 0.064
        c.tick(t)
    assert 28 < c.value < 33            # real rate ~30 fps (old 1/dt average said ~250)


def test_piece_can_be_cut_again():
    f = Piece(make_shape("hexagon", 60), (300, 300), (0, 0), (0, 0, 255), shape="hexagon")
    halves = f.slice(np.array([150.0, 300]), np.array([450.0, 300]))
    h = halves[0]
    assert h.generation == 1 and not h.can_be_cut        # cooldown: not by the same swipe
    h.age = config.PIECE_CUT_COOLDOWN + 0.01
    assert h.can_be_cut
    quarters = h.slice(h.pos - [0, 100], h.pos + [0, 100])
    assert len(quarters) == 2 and quarters[0].generation == 2
    deep = Piece(make_shape("circle", 60), (0, 0), (0, 0), (0, 0, 255), generation=config.MAX_CUTS_PER_FRUIT)
    deep.age = 1.0
    assert not deep.can_be_cut                           # max depth reached


def test_game_multi_cut_scores():
    config.HIGHSCORE_FILE = "/tmp/hs_test2.json"
    game = Game()
    game.spawner.next_wave = 1e9
    g = config.GRAVITY
    config.GRAVITY = 0
    try:
        game.fruits.append(Piece(make_shape("square", 60), (640, 360), (0, 0), (0, 0, 255), shape="square"))
        t = swipe(game, game.fruits, 360, 0.0)          # cut 1 -> 2 halves
        pieces = list(game.halves)
        for p in pieces:
            p.vel[:] = 0
        t += 0.3                                        # wait for the cooldown
        for i in range(8):                              # vertical swipe through both halves
            game.update(1 / 30, t, [({"Right": (640, 150 + i * 60)}, t)])
            t += 1 / 30
    finally:
        config.GRAVITY = g
    assert game.sliced == 1
    assert game.cuts >= 2


def test_spawner_waits_for_wave_to_clear():
    from fruit import Spawner
    sp, t, dt = Spawner(), 0.0, 1 / 60
    thrown = []
    while not thrown:                           # first wave after WAVE_FIRST_DELAY
        thrown += sp.update(dt, 0)
        t += dt
    assert abs(t - config.WAVE_FIRST_DELAY) < 0.05
    while sp.queue:                             # rest of the wave (staggered)
        thrown += sp.update(dt, len(thrown))
    n_first = len(thrown)
    for _ in range(300):                        # 5 s with fruits still flying -> no new wave
        assert sp.update(dt, on_screen=1) == []
    t_clear, new = 0.0, []
    while not new:                              # screen cleared -> next wave after WAVE_REST
        new = sp.update(dt, 0)
        t_clear += dt
    assert abs(t_clear - config.WAVE_REST) < 0.05
    assert sp.wave == 2 and n_first >= 1


def test_every_fruit_is_visible():
    """Fruits from any side must fly through the screen for a while, not just skim an edge."""
    from fruit import launch_fruit
    W, H = config.GAME_WIDTH, config.GAME_HEIGHT
    for side in ("bottom", "left", "right"):
        for _ in range(200):
            f = launch_fruit(side)
            visible, t = 0.0, 0.0
            while not f.is_off_screen() and t < 6:
                f.update(1 / 60)
                t += 1 / 60
                if 0 < f.pos[0] < W and 0 < f.pos[1] < H:
                    visible += 1 / 60
            assert f.is_off_screen(), side          # it leaves the screen in the end
            assert visible > 0.8, (side, visible)   # and was on screen long enough to cut
            assert f.pos[1] > H or f.vel[1] > 0     # it left falling (bottom), not flying up


def test_wave_size_is_random_1_to_5():
    from fruit import Spawner
    sizes = set()
    for _ in range(200):
        sp = Spawner()
        sp._start_wave()
        sizes.add(len(sp.queue))
    assert sizes == set(range(config.WAVE_SIZE_MIN, config.WAVE_SIZE_MAX + 1))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
