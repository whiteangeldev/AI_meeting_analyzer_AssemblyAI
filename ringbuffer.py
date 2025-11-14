import numpy as np
from collections import deque
from config import SAMPLE_RATE, MAX_BUFFER_SECS


class RingBuffer:
    def __init__(self):
        self.samples = deque()  # list of (start_time, np.ndarray)
        self._now = 0.0  # logical time in seconds

    def append(self, frame: np.ndarray):
        """
        Append frame; frame length defines time advance.
        """
        start_t = self._now
        self._now += len(frame) / SAMPLE_RATE
        self.samples.append((start_t, frame))
        self._trim()

    def _trim(self):
        min_t = self._now - MAX_BUFFER_SECS
        while (
            self.samples
            and self.samples[0][0] + len(self.samples[0][1]) / SAMPLE_RATE < min_t
        ):
            self.samples.popleft()

    @property
    def now(self):
        return self._now

    def slice(self, start_s: float, end_s: float):
        """
        Return np.ndarray for audio between [start_s, end_s].
        If nothing overlaps, returns None.
        """
        if start_s >= end_s:
            return None
        chunks = []
        for t, f in self.samples:
            ft0, ft1 = t, t + len(f) / SAMPLE_RATE
            if ft1 <= start_s or ft0 >= end_s:
                continue
            s_idx = max(0, int((start_s - ft0) * SAMPLE_RATE))
            e_idx = min(len(f), int((end_s - ft0) * SAMPLE_RATE))
            if s_idx < e_idx:
                chunks.append(f[s_idx:e_idx])
        if not chunks:
            return None
        return np.concatenate(chunks)
