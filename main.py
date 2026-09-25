"""Ninja Fruit with your webcam.

Run:  python main.py
Keys: Q/Esc quit | P pause | R restart | F fullscreen
      D show hand landmarks | T show timing (ms per step) | L record finger data to CSV
"""
import time

import cv2

import config
from camera import Camera
from effects import draw_text
from game import Game
from hand_tracker import HandTracker


def fit_to_game(frame):
    """Crop the camera image to the game aspect ratio, then resize (no stretching)."""
    h, w = frame.shape[:2]
    if (w, h) == (config.GAME_WIDTH, config.GAME_HEIGHT):
        return frame
    target = config.GAME_WIDTH / config.GAME_HEIGHT
    if w / h > target:                      # too wide -> crop left/right
        new_w = int(h * target)
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    elif w / h < target:                    # too tall -> crop top/bottom
        new_h = int(w / target)
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h]
    return cv2.resize(frame, (config.GAME_WIDTH, config.GAME_HEIGHT))


class Timer:
    """Average time (ms) of each step of the loop, to find what is slow."""

    def __init__(self):
        self.ms = {}
        self.t = time.perf_counter()

    def mark(self, name):
        now = time.perf_counter()
        self.ms[name] = 0.9 * self.ms.get(name, 0.0) + 0.1 * (now - self.t) * 1000
        self.t = now


def main():
    cam = Camera()
    tracker = HandTracker()
    game = Game()
    timer = Timer()
    show_landmarks = False
    show_timing = config.SHOW_TIMING
    fullscreen = config.FULLSCREEN

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.GAME_WIDTH, config.GAME_HEIGHT)
    if fullscreen:
        cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_id = 0
    last = time.perf_counter()
    fps = 0.0
    try:
        while True:
            timer.t = time.perf_counter()
            item = cam.read(last_id)            # waits only if no new frame yet
            if item is None:
                print("No camera frame for 1 second...")
                continue
            last_id, frame, capture_time = item
            timer.mark("wait camera")

            if config.MIRROR:
                frame = cv2.flip(frame, 1)
            frame = fit_to_game(frame)
            timer.mark("flip+resize")

            tracker.submit(frame, capture_time)   # async: returns at once
            tracking = tracker.poll()             # results that are ready now
            timer.mark("mediapipe" if not tracker.async_mode else "mediapipe (submit)")

            now = time.perf_counter()
            dt = now - last
            last = now
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-3))
            game.update(dt, now, tracking)
            timer.mark("game update")

            stats = {"fps": {"game": fps, "camera": cam.fps, "tracking": tracker.fps}}
            if show_timing:
                stats["lines"] = [f"{k:<18} {v:5.1f} ms" for k, v in timer.ms.items()]
                stats["lines"].append(f"{'track latency':<18} {tracker.latency_ms:5.1f} ms")
            out = game.draw(frame, now, stats, tracker.last_landmarks if show_landmarks else None)
            if tracker.log_rows is not None:
                cv2.circle(out, (config.GAME_WIDTH // 2 - 60, 100), 10, (0, 0, 255), -1)
                draw_text(out, f"REC {len(tracker.log_rows)}", (config.GAME_WIDTH // 2 - 40, 110),
                          0.7, (0, 0, 255))
            timer.mark("draw")

            cv2.imshow(config.WINDOW_NAME, out)
            key = cv2.waitKey(1) & 0xFF
            timer.mark("imshow")

            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                game.paused = not game.paused
            elif key == ord("r"):
                game.reset()
            elif key == ord("d"):
                show_landmarks = not show_landmarks
            elif key == ord("t"):
                show_timing = not show_timing
            elif key == ord("l"):
                path = tracker.toggle_log()
                if path:
                    print("Finger data saved to", path)
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            # closing the window with the X button
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if tracker.log_rows is not None:
            print("Finger data saved to", tracker.toggle_log())
        cam.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
