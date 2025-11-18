import numpy as np
from collections import deque
from config import SAMPLE_RATE, MAX_BUFFER_SECS


class RingBuffer:
    """Simple ring buffer for audio samples"""
    
    def __init__(self):
        self.samples = deque()
        self._now = 0.0

    def append(self, frame: np.ndarray):
        """Append frame; frame length defines time advance"""
        start_t = self._now
        self._now += len(frame) / SAMPLE_RATE
        self.samples.append((start_t, frame))
        self._trim()

    def _trim(self):
        """Remove old samples beyond buffer size"""
        min_t = self._now - MAX_BUFFER_SECS
        while self.samples:
            t, _ = self.samples[0]
            if t + len(self.samples[0][1]) / SAMPLE_RATE >= min_t:
                break
            self.samples.popleft()

    @property
    def now(self):
        """Current time position in seconds"""
        return self._now

    def slice(self, start_s: float, end_s: float):
        """Return audio slice between [start_s, end_s]"""
        if start_s >= end_s:
            return None
        
        chunks = []
        for t, f in self.samples:
            ft0 = t
            ft1 = t + len(f) / SAMPLE_RATE
            if ft1 <= start_s or ft0 >= end_s:
                continue
            s_idx = max(0, int((start_s - ft0) * SAMPLE_RATE))
            e_idx = min(len(f), int((end_s - ft0) * SAMPLE_RATE))
            if s_idx < e_idx:
                chunks.append(f[s_idx:e_idx])
        
        return np.concatenate(chunks) if chunks else None
