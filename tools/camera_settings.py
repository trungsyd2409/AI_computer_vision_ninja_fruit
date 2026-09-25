"""Open the webcam driver settings + a live preview that shows the REAL fps.

    python tools/camera_settings.py

In the settings window, look for (names depend on your webcam):
  - "Exposure" -> untick "Auto", move the slider left (shorter exposure)
  - "Low light compensation" / "Low-light boost" -> turn OFF (it halves the fps!)
  - "Power line frequency" / "Anti-flicker" -> 50 Hz (Australia), stops light flicker
Watch the "NEW fps" number in the preview while you change things.
Most drivers remember these settings, so the game will use them too.

Keys: S = open settings again | A = auto exposure back ON | Q = quit
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from camera import set_auto_exposure  # noqa: E402
from effects import draw_text  # noqa: E402
from fps import FpsCounter  # noqa: E402


def main():
    # The settings dialog only exists in the DirectShow backend
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_SETTINGS, 1)
    counter, prev = FpsCounter(), None
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        small = frame[::24, ::24]
        if prev is None or not np.array_equal(small, prev):
            counter.tick(time.perf_counter())
        prev = small.copy()
        frame = cv2.flip(frame, 1)
        draw_text(frame, f"NEW fps {counter.value:4.1f}   light {small.mean():5.1f}", (15, 35),
                  0.8, (255, 255, 255))
        draw_text(frame, "S: settings  A: auto exposure  Q: quit", (15, 465), 0.6, (255, 255, 255), 1)
        cv2.imshow("Camera settings test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
        if key == ord("a"):
            set_auto_exposure(cap, "dshow")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
