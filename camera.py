"""Webcam reader that runs in its own thread.

Why a thread? `cap.read()` blocks until the camera has a new frame (~33 ms at 30 fps).
If the game loop calls it directly, that waiting time is added to MediaPipe + drawing
time every frame. With a thread, the camera keeps reading in the background and the
game always takes the newest frame.
"""
import platform
import threading
import time

import cv2
import numpy as np

import config
from fps import FpsCounter

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


def fourcc_to_str(value):
    v = int(value)
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)) if v > 0 else "?"


def open_capture(index=config.CAMERA_INDEX, backend=config.CAMERA_BACKEND,
                 fourcc=config.CAMERA_FOURCC, width=config.CAPTURE_WIDTH,
                 height=config.CAPTURE_HEIGHT, fps=config.CAPTURE_FPS,
                 exposure=config.CAMERA_EXPOSURE):
    """Open the webcam with the given settings. Returns cv2.VideoCapture."""
    if platform.system() != "Windows" and backend in ("dshow", "msmf"):
        backend = "any"
    cap = cv2.VideoCapture(index, BACKENDS.get(backend, cv2.CAP_ANY))
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    # The order matters for many webcams on DirectShow: FOURCC first, then size.
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # do not keep old frames
    if exposure is not None:
        set_exposure(cap, backend, exposure)
    return cap


def set_exposure(cap, backend, value):
    """Turn off auto exposure and set a fixed one. value is log2(seconds):
    -5 = 1/32 s, -6 = 1/64 s, -7 = 1/128 s. Shorter = less blur, darker image.
    Returns True if the driver accepted it (some drivers ignore it)."""
    manual = 0.25 if backend == "dshow" else 0   # the "manual" value differs per backend
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
    return bool(cap.set(cv2.CAP_PROP_EXPOSURE, value))


def set_auto_exposure(cap, backend):
    """Give exposure control back to the camera (auto mode)."""
    return bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if backend == "dshow" else 1))


def describe(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return f"{w}x{h} {fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))} (driver says {cap.get(cv2.CAP_PROP_FPS):.0f} fps)"


class Camera:
    def __init__(self):
        self.cap = open_capture()
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open the webcam. Try another CAMERA_INDEX in config.py")
        print("Camera:", describe(self.cap))
        self.frame = None
        self.frame_time = 0.0
        self.frame_id = 0
        self.counter = FpsCounter()
        self.duplicates = 0            # frames that were the same image as the one before
        self.total = 0
        self.running = True
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        prev_small = None
        while self.running:
            ok, frame = self.cap.read()
            now = time.perf_counter()          # time the frame arrived
            if not ok:
                time.sleep(0.01)
                continue
            # Some drivers return the SAME frame again when no new one is ready.
            # Compare a tiny sample of pixels to skip these copies.
            small = frame[::24, ::24]
            self.total += 1
            if prev_small is not None and np.array_equal(small, prev_small):
                self.duplicates += 1
                continue
            prev_small = small.copy()
            self.counter.tick(now)
            with self.cond:
                self.frame, self.frame_time = frame, now
                self.frame_id += 1
                self.cond.notify_all()

    @property
    def fps(self):
        return self.counter.value

    def read(self, last_id, timeout=1.0):
        """Wait for a frame newer than last_id. Returns (id, frame, time) or None."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.frame_id != last_id, timeout):
                return None
            return self.frame_id, self.frame, self.frame_time

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
