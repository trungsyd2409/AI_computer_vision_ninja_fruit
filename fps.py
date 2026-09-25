"""Correct FPS counter.

Wrong way (used before): average of 1/dt. If frames come in bursts
(e.g. 2 ms, 60 ms, 2 ms, 60 ms...) the values are 500, 16, 500, 16 -> "258 fps",
but the real rate is 2 frames per 62 ms = 32 fps.

Right way: count frames in the last second.
"""
from collections import deque


class FpsCounter:
    def __init__(self, window=1.0):
        self.window = window
        self.times = deque()

    def tick(self, t):
        self.times.append(t)
        while self.times and t - self.times[0] > self.window:
            self.times.popleft()

    @property
    def value(self):
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0
