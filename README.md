# Ninja Fruit CV 🍉✋

A Fruit Ninja style game played with your **webcam**. Move your **index finger** (one hand or both hands) fast to cut the fruits that fly up from the bottom of the screen.

Fruits are 12 neon 2D shapes (circle, oval, square, rectangle, triangle, diamond, pentagon, hexagon, octagon, star, heart, cross): flat colour inside, glowing border. You can cut a fruit, then cut its pieces again (up to 3 levels).

## How it works

```
Webcam frame ──► flip (mirror) ──► MediaPipe Hand Landmarker ──► index finger tip (landmark 8)
                                                                         │
                                                             One Euro filter (less jitter)
                                                                         │
                                        Blade: trail of finger points + speed (px/s)
                                                                         │
             if speed > BLADE_MIN_SPEED: segment (last point → new point) cuts every fruit it touches
                                                                         │
                         the fruit polygon is split along the cut line → 2 halves fly apart
```

| File | What it does |
|---|---|
| `main.py` | Game loop, keyboard, window, timing |
| `camera.py` | Webcam in a background thread (MJPG, buffer size 1) |
| `hand_tracker.py` | MediaPipe Tasks API (async LIVE_STREAM), finger guard, One Euro filter |
| `blade.py` | Finger trail, speed, draws the blade |
| `fruit.py` | 12 shapes (same area), neon drawing, physics, multi-cut slicing, spawner |
| `geometry.py` | Hit test (numpy) + polygon split by a line (shapely, works for concave shapes) |
| `effects.py` | Juice particles, cut flash, floating text |
| `game.py` | Game rules: score, combo, draw HUD |
| `config.py` | **All settings** (speed, gravity, sizes, colours...) |
| `tests/test_logic.py` | Tests that run without a webcam |
| `tools/diagnose.py` | Measures camera FPS / MediaPipe time / drawing time |

## Setup (Windows)

Python **3.10 – 3.12** is recommended (MediaPipe does not always support the newest Python).

```bash
cd AI_computer_vision_ninja_fruit
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The first run downloads the hand model (`models/hand_landmarker.task`, ~7 MB). If the download fails, get it manually from
<https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task>
and put it in the `models/` folder.

## Controls

| Key | Action |
|---|---|
| Index finger (fast) | Cut |
| `P` | Pause / resume |
| `R` | Restart |
| `D` | Show / hide hand landmarks (debug, index finger joints in red) |
| `T` | Show / hide timing: ms spent in each step of the loop |
| `L` | Start / stop recording finger data to `logs/track_*.csv` |
| `F` | Fullscreen |
| `Q` / `Esc` | Quit |

The circle on your finger turns **green** when you move fast enough to cut.

## Scoring

- Each fruit: **+1**, each extra cut of a piece: **+1** (`MAX_CUTS_PER_FRUIT`, `PIECE_MIN_AREA` in `config.py`)
- **Combo**: cut 3 or more fruits with less than 0.35 s between cuts → bonus **+N** (N = number of fruits in the combo)
- Best score is saved in `highscore.json`

## Low FPS? Run the diagnostic first

```bash
python tools/diagnose.py
```

It tests your webcam with different settings and tells you which one is fastest, how long MediaPipe takes per frame, and how long drawing takes. Put the best camera settings in `config.py` (`CAMERA_BACKEND`, `CAMERA_FOURCC`, `CAPTURE_WIDTH/HEIGHT`).

The top-right corner of the game shows 3 numbers:
- **camera**: frames per second the webcam really gives. If this is ~10, the problem is the camera (format or low light), not the code.
- **tracking**: hand results per second from MediaPipe.
- **game**: how often the screen is redrawn.

## Tuning tips

- Cuts do not register → lower `BLADE_MIN_SPEED` in `config.py`.
- Waves: `WAVE_REST` (rest after a wave is cleared), `WAVE_SIZE_*`, `WAVE_STAGGER`.
- Neon look → `NEON_COLORS`, `FILL_BRIGHTNESS`, `GLOW_WIDTH`, `GLOW_BLUR`.
- Game is slow (low FPS) → lower `DETECT_WIDTH` (e.g. 480) or `GAME_WIDTH/GAME_HEIGHT` (e.g. 960×540).
- Finger cursor shakes → lower `FILTER_MIN_CUTOFF` (more smoothing). Blade feels laggy → raise it.
- Use good light and a plain background: MediaPipe loses fast hands when the image is blurry.
  In a dark room the webcam uses a long exposure → fewer FPS **and** more motion blur.
  Try `CAMERA_EXPOSURE = -6` (or -7) in `config.py` with a bright room.
- Fingertip jumps towards the palm during fast swipes → keep `FINGER_GUARD = True`
  (it rebuilds the tip from the two finger joints when the tip looks folded).
- Wrong camera → change `CAMERA_INDEX`.

## Run tests

```bash
pip install pytest
python -m pytest tests
```

## Ideas for next versions

- Real fruit images (PNG with transparency) + sound effects
- Bombs, lives, 60-second mode
- Start menu controlled by the hand (hover to select)
- Train your own model (e.g. a small CNN or YOLO) to detect a real object used as a "sword"
