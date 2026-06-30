"""
src/utils/timer.py
──────────────────
Lightweight per-module latency profiler for the NoviSentra pipeline.

Usage:
    from utils.timer import PipelineTimer

    timer = PipelineTimer()

    timer.start("detection")
    boxes = detector.detect(frame)
    timer.stop("detection")

    timer.start("face_recognition")
    faces = recognize_faces(frame)
    timer.stop("face_recognition")

    # Get summary every N frames
    if frame_count % 100 == 0:
        print(timer.summary())
        # → detection: 45ms | face_recognition: 120ms | clip: 210ms

    # Get as dict for API/dashboard
    stats = timer.stats()
    # → {"detection": {"avg_ms": 45.2, "last_ms": 43.1, "calls": 100}, ...}
"""

import time
from collections import defaultdict, deque


class PipelineTimer:
    """
    Rolling-window latency tracker for named pipeline stages.
    Thread-safe for reading; not intended for concurrent writes to same key.
    """

    WINDOW = 60   # keep last N measurements for rolling average

    def __init__(self):
        self._starts: dict  = {}
        self._history: dict = defaultdict(lambda: deque(maxlen=self.WINDOW))
        self._counts: dict  = defaultdict(int)
        self._last_ms: dict = {}

    def start(self, name: str):
        """Mark start of a timed section."""
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """
        Mark end of a timed section.
        Returns elapsed milliseconds (or 0 if start() wasn't called).
        """
        t0 = self._starts.pop(name, None)
        if t0 is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._history[name].append(elapsed_ms)
        self._counts[name] += 1
        self._last_ms[name] = elapsed_ms
        return elapsed_ms

    def avg_ms(self, name: str) -> float:
        """Rolling average latency in ms for a named section."""
        hist = self._history.get(name)
        if not hist:
            return 0.0
        return sum(hist) / len(hist)

    def last_ms(self, name: str) -> float:
        """Most recent latency in ms."""
        return self._last_ms.get(name, 0.0)

    def stats(self) -> dict:
        """Return dict of all tracked stages with avg/last/calls."""
        return {
            name: {
                "avg_ms":  round(self.avg_ms(name), 1),
                "last_ms": round(self.last_ms(name), 1),
                "calls":   self._counts[name],
            }
            for name in self._history
        }

    def summary(self) -> str:
        """One-line summary string for console/log output."""
        parts = []
        for name, data in sorted(self.stats().items()):
            parts.append(f"{name}: {data['avg_ms']}ms")
        return " | ".join(parts) if parts else "(no data)"

    def reset(self):
        """Clear all history."""
        self._starts.clear()
        self._history.clear()
        self._counts.clear()
        self._last_ms.clear()