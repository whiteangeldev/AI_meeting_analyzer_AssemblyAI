import webrtcvad, numpy as np
from config import SAMPLE_RATE

_vad = webrtcvad.Vad(2)  # 0-3 (3 = most aggressive)


def is_voice(frame_f32: np.ndarray) -> bool:
    # Need 20/30/40ms chunks; we just approximate here.
    frame = (np.clip(frame_f32, -1, 1) * 32767).astype("int16").tobytes()
    if len(frame) < int(0.03 * SAMPLE_RATE) * 2:
        return True
    return _vad.is_speech(frame, SAMPLE_RATE)
