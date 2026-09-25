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

import config

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
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual on DirectShow
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return cap


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
        self.fps = 0.0
        self.running = True
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        last = time.perf_counter()
        while self.running:
            ok, frame = self.cap.read()
            now = time.perf_counter()          # time the frame arrived
            if not ok:
                time.sleep(0.01)
                continue
            dt = now - last
            last = now
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-3))
            with self.cond:
                self.frame, self.frame_time = frame, now
                self.frame_id += 1
                self.cond.notify_all()

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
