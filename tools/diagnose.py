"""Find why the game is slow.  Run from the project folder:

    python tools/diagnose.py

It measures:
  1. real camera FPS for different settings (backend / MJPG / resolution)
  2. MediaPipe time per frame (keep ONE hand in front of the camera for this part)
  3. game drawing time
and saves a report to logs/diagnose_<time>.txt
"""
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from camera import describe, open_capture, set_auto_exposure, set_exposure  # noqa: E402

lines = []


def log(msg=""):
    print(msg, flush=True)
    lines.append(msg)


def measure_camera(backend, fourcc, w, h, seconds=2.0, exposure=None):
    """Returns dict: reads/s, NEW frames/s (duplicates removed), brightness, ..."""
    t_open = time.perf_counter()
    cap = open_capture(backend=backend, fourcc=fourcc, width=w, height=h, exposure=None)
    open_s = time.perf_counter() - t_open
    if not cap.isOpened():
        return None
    accepted = None
    if exposure is not None:
        accepted = set_exposure(cap, backend, exposure)
    info = describe(cap)
    for _ in range(10):                     # warm up (first frames are slow, exposure settles)
        cap.read()
    reads, unique, prev, light = 0, 0, None, []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        reads += 1
        small = frame[::24, ::24]
        if prev is None or not np.array_equal(small, prev):
            unique += 1
            light.append(float(small.mean()))
        prev = small.copy()
    el = time.perf_counter() - t0
    r = dict(reads=reads / el, unique=unique / el, info=info, open_s=open_s,
             light=np.mean(light) if light else 0.0, accepted=accepted,
             exposure=cap.get(cv2.CAP_PROP_EXPOSURE), auto=cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    cap.release()
    return r


def main():
    log("=== System ===")
    log(f"Python {platform.python_version()} | OpenCV {cv2.__version__} | MediaPipe {mp.__version__}")
    log(f"{platform.system()} {platform.release()} | CPU: {platform.processor()} | cores: {os.cpu_count()}")

    # ---------------------------------------------------------------- camera
    log("\n=== 1. Camera FPS (no MediaPipe, no drawing) ===")
    log("reads = frames returned by cap.read(), NEW = really new images (copies removed)")
    log("light = mean brightness 0-255 (below ~70 = dark room -> webcam slows down)")
    backends = ["dshow", "msmf"] if platform.system() == "Windows" else ["any"]
    results = []
    for backend in backends:
        for fourcc in (None, "MJPG"):
            for w, h in ((640, 480), (1280, 720)):
                r = measure_camera(backend, fourcc, w, h)
                name = f"{backend:<5} {fourcc or 'default':<7} ask {w}x{h}"
                if r is None:
                    log(f"{name}: cannot open")
                    continue
                results.append((r["unique"], backend, fourcc, w, h))
                log(f"{name}: reads {r['reads']:5.1f}/s | NEW {r['unique']:5.1f} fps | light {r['light']:5.1f}"
                    f" | got {r['info']} | open {r['open_s']:.1f}s")
    if not results:
        log("No camera found. Check CAMERA_INDEX in config.py")
        return
    best = max(results, key=lambda r: (round(r[0] / 3), r[3]))   # fps first, then bigger size
    log(f"\nBest: {best[1]} {best[2] or 'default'} {best[3]}x{best[4]} -> {best[0]:.1f} NEW fps")

    log("\n=== 1b. Exposure test (is the room too dark?) ===")
    log("If NEW fps goes up with a shorter exposure, auto exposure is what limits your FPS.")
    for exp in (None, -5, -6, -7, -8):
        r = measure_camera(best[1], best[2], best[3], best[4], exposure=exp)
        if r is None:
            continue
        label = "auto" if exp is None else f"{exp} (1/{2 ** -exp} s)"
        log(f"exposure {label:<13}: NEW {r['unique']:5.1f} fps | light {r['light']:5.1f} | "
            f"driver accepted={r['accepted']} | reports exposure {r['exposure']} auto {r['auto']}")
    # the driver may remember the last manual value -> switch auto exposure back on
    cap = open_capture(backend=best[1], fourcc=best[2], width=best[3], height=best[4], exposure=None)
    log(f"auto exposure restored: {set_auto_exposure(cap, best[1])}")
    cap.release()

    # ---------------------------------------------------------------- mediapipe
    log("\n=== 2. MediaPipe speed ===")
    log("Show ONE hand to the camera and keep moving it until the test ends (~10 s)")
    for i in (3, 2, 1):
        print(f"  starting in {i}...", flush=True)
        time.sleep(1)
    from mediapipe.tasks.python import BaseOptions, vision
    from hand_tracker import ensure_model
    cap = open_capture(backend=best[1], fourcc=best[2], width=best[3], height=best[4], exposure=None)
    for num_hands in (2, 1):
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE)
        lm = vision.HandLandmarker.create_from_options(opts)
        times, found, ts = [], 0, 0
        t_end = time.perf_counter() + 5
        while time.perf_counter() < t_end:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (config.DETECT_WIDTH, int(h * config.DETECT_WIDTH / w)))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            ts += 33
            t0 = time.perf_counter()
            res = lm.detect_for_video(img, ts)
            times.append((time.perf_counter() - t0) * 1000)
            found += bool(res.hand_landmarks)
        lm.close()
        if times:
            t = np.array(times[3:] or times)
            log(f"num_hands={num_hands}: mean {t.mean():.1f} ms, p90 {np.percentile(t, 90):.1f} ms "
                f"-> max {1000 / t.mean():.0f} fps | hand found in {found}/{len(times)} frames")
    cap.release()

    # ---------------------------------------------------------------- drawing
    log("\n=== 3. Game drawing ===")
    from game import Game
    from fruit import launch_fruit
    game = Game()
    for _ in range(8):
        f = launch_fruit()
        f.pos[1] = config.GAME_HEIGHT / 2
        game.fruits.append(f)
    frame = np.full((config.GAME_HEIGHT, config.GAME_WIDTH, 3), 90, np.uint8)
    t0 = time.perf_counter()
    for _ in range(30):
        game.draw(frame, time.perf_counter())
    log(f"game.draw: {(time.perf_counter() - t0) / 30 * 1000:.1f} ms per frame")

    # ---------------------------------------------------------------- save
    os.makedirs(config.TRACK_LOG_DIR, exist_ok=True)
    path = os.path.join(config.TRACK_LOG_DIR, time.strftime("diagnose_%Y%m%d_%H%M%S.txt"))
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()
