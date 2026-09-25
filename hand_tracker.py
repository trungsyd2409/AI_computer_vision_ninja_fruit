"""Find index finger tips with MediaPipe Hand Landmarker (Tasks API).

Note: new MediaPipe versions (0.10.30+) removed the old `mp.solutions.hands`,
so we use the Tasks API. It needs a model file, which is downloaded once.

Two modes (config.TRACKING_ASYNC):
- async (LIVE_STREAM): detect_async() returns at once, MediaPipe works in its own
  thread and calls _on_result() later. The game loop never waits for the model.
- sync (VIDEO): detect_for_video() blocks until the result is ready.
"""
import csv
import math
import os
import threading
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import config


def ensure_model(path=config.MODEL_PATH, url=config.MODEL_URL):
    """Download the hand model the first time the game runs."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading hand model to {path} ...")
    urllib.request.urlretrieve(url, path)
    print("Done.")
    return path


class OneEuroFilter:
    """Simple 1D One Euro filter: strong smoothing when slow, little lag when fast.
    Paper: Casiez et al., CHI 2012.
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def guard_fingertip(tip, pip, mcp, ratio_ref=1.0):
    """Check if the fingertip looks wrong (folded towards the palm).

    Returns (tip_to_use, is_guarded, ratio).
    The index finger has 2 parts in our model: MCP->PIP (stable, near the palm)
    and PIP->TIP. For a straight finger |PIP->TIP| is about |MCP->PIP|.
    """
    v1 = pip - mcp
    v2 = tip - pip
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6:
        return tip, False, 0.0
    ratio = n2 / n1
    cos = np.dot(v1, v2) / (n1 * n2 + 1e-9)
    bend = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
    if ratio < config.FINGER_MIN_RATIO or bend > config.FINGER_MAX_BEND_DEG:
        return pip + v1 * ratio_ref, True, ratio   # rebuild a straight finger
    return tip, False, ratio


class HandTracker:
    def __init__(self, async_mode=config.TRACKING_ASYNC):
        self.async_mode = async_mode
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.LIVE_STREAM if async_mode else vision.RunningMode.VIDEO,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            result_callback=self._on_result if async_mode else None,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.start = time.perf_counter()
        self.last_ts = -1
        self._lock = threading.Lock()
        self._results = deque()        # (result, ts_ms) waiting for poll()
        self._frames = {}              # ts_ms -> (capture_time, width, height, submit_time)
        self.filters = {}              # hand id -> (filter_x, filter_y)
        self.ratio_ref = {}            # hand id -> normal |PIP->TIP| / |MCP->PIP|
        self.last_landmarks = []       # for debug drawing
        self.latency_ms = 0.0          # time from submit to result
        self.fps = 0.0                 # tracking results per second
        self._last_result_time = None
        self.log_rows = None           # list when logging (press L)

    # ------------------------------------------------------------ input
    def submit(self, frame_bgr, capture_time):
        """Send a frame to MediaPipe. capture_time = when the camera gave the frame."""
        h, w = frame_bgr.shape[:2]
        small_h = int(h * config.DETECT_WIDTH / w)
        small = cv2.resize(frame_bgr, (config.DETECT_WIDTH, small_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # timestamps must be strictly increasing (milliseconds)
        ts = max(int((capture_time - self.start) * 1000), self.last_ts + 1)
        self.last_ts = ts
        with self._lock:
            self._frames[ts] = (capture_time, w, h, time.perf_counter())
        if self.async_mode:
            self.landmarker.detect_async(image, ts)
        else:
            self._on_result(self.landmarker.detect_for_video(image, ts), image, ts)

    def _on_result(self, result, image, ts):
        # In async mode this runs in MediaPipe's thread -> keep it short
        with self._lock:
            self._results.append((result, ts))

    # ------------------------------------------------------------ output
    def poll(self):
        """Return a list of (tips, capture_time) for all new results (oldest first).
        tips = {hand_id: (x, y)} in pixels of the submitted frame."""
        with self._lock:
            items = list(self._results)
            self._results.clear()
            infos = []
            for _, ts in items:
                infos.append(self._frames.pop(ts, None))
            if items:   # frames that MediaPipe skipped (busy) never get a result
                newest = items[-1][1]
                for k in [k for k in self._frames if k < newest]:
                    del self._frames[k]
        out = []
        for (result, ts), info in zip(items, infos):
            if info is None:
                continue
            capture_time, w, h, submit_time = info
            now = time.perf_counter()
            self.latency_ms = 0.8 * self.latency_ms + 0.2 * (now - submit_time) * 1000
            if self._last_result_time is not None:
                self.fps = 0.9 * self.fps + 0.1 / max(capture_time - self._last_result_time, 1e-3)
            self._last_result_time = capture_time
            out.append((self._tips_from_result(result, w, h, capture_time), capture_time))
        return out

    def _tips_from_result(self, result, w, h, t):
        tips = {}
        self.last_landmarks = []
        for i, landmarks in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name if result.handedness else str(i)
            if label in tips:          # both hands got the same label -> rename
                label = label + "_2"
            pts = np.array([(p.x * w, p.y * h) for p in landmarks])   # normalised 0..1 -> px
            self.last_landmarks.append(pts)
            raw = pts[config.INDEX_FINGER_TIP]
            tip, guarded = raw, False
            if config.FINGER_GUARD:
                ref = self.ratio_ref.get(label, 1.0)
                tip, guarded, ratio = guard_fingertip(raw, pts[config.INDEX_FINGER_PIP],
                                                      pts[config.INDEX_FINGER_MCP], ref)
                if not guarded:   # learn this person's normal finger ratio
                    self.ratio_ref[label] = float(np.clip(0.9 * ref + 0.1 * ratio, 0.8, 1.4))
            if label not in self.filters:
                self.filters[label] = (
                    OneEuroFilter(config.FILTER_MIN_CUTOFF, config.FILTER_BETA),
                    OneEuroFilter(config.FILTER_MIN_CUTOFF, config.FILTER_BETA))
            fx, fy = self.filters[label]
            tips[label] = (fx(tip[0], t), fy(tip[1], t))
            if self.log_rows is not None:
                p6, p5 = pts[config.INDEX_FINGER_PIP], pts[config.INDEX_FINGER_MCP]
                self.log_rows.append([f"{t - self.start:.4f}", label,
                                      *[f"{v:.1f}" for v in (*raw, *p6, *p5, *tip, *tips[label])],
                                      int(guarded)])

        # forget filters of hands that left, so they restart cleanly
        for label in list(self.filters):
            if label not in tips:
                del self.filters[label]
        return tips

    # ------------------------------------------------------------ logging
    def toggle_log(self):
        """Start/stop saving finger data. Returns the CSV path when stopping."""
        if self.log_rows is None:
            self.log_rows = []
            return None
        rows, self.log_rows = self.log_rows, None
        os.makedirs(config.TRACK_LOG_DIR, exist_ok=True)
        path = os.path.join(config.TRACK_LOG_DIR, time.strftime("track_%Y%m%d_%H%M%S.csv"))
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t", "hand", "tip_x", "tip_y", "pip_x", "pip_y", "mcp_x", "mcp_y",
                         "used_x", "used_y", "filtered_x", "filtered_y", "guarded"])
            wr.writerows(rows)
        return path

    def close(self):
        self.landmarker.close()
