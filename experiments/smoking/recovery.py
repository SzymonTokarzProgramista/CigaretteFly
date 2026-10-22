"""Detect failed locomotion and request a bounded physical recovery maneuver."""
from collections import deque
import math


class StuckRecovery:
    def __init__(self, window_sec=1.2, min_progress_mm=0.5):
        self.window_sec = window_sec
        self.min_progress_mm = min_progress_mm
        self.history = deque()
        self.started = None
        self.direction = 1
        self.attempts = 0

    def cancel(self):
        self.history.clear()
        self.started = None

    def update(self, time_sec, position, walking, turn=0.0):
        if not walking:
            self.cancel()
            return '', None
        if self.started is not None:
            elapsed = time_sec - self.started
            if elapsed < 0.45 - 1e-9:
                return 'REPOSITION', (0.8, 0.8)
            if elapsed < 1.05 - 1e-9:
                return 'TURN', ((1.0, 0.6) if self.direction > 0 else (0.6, 1.0))
            if elapsed < 1.50 - 1e-9:
                return 'RESUME', (0.8, 0.8)
            self.cancel()
        self.history.append((time_sec, tuple(position[:2])))
        # Retain the sample just before the window's start as well.
        while len(self.history) > 1 and self.history[1][0] <= time_sec - self.window_sec:
            self.history.popleft()
        if time_sec - self.history[0][0] < self.window_sec - 1e-9:
            return '', None
        anchor = self.history[0][1]
        excursion = max(math.dist(anchor, p) for _, p in self.history)
        if excursion >= self.min_progress_mm:
            return '', None
        self.attempts += 1
        self.direction = (1 if turn >= 0 else -1) * (1 if self.attempts % 2 else -1)
        self.started = time_sec
        self.history.clear()
        return 'REPOSITION', (0.8, 0.8)
