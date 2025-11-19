# diarizer.py

import numpy as np
from resemblyzer import VoiceEncoder
from scipy.spatial.distance import cosine


class OnlineDiarizer:
    """
    Lightweight real-time diarizer:
    - Uses short waveform windows (≈1–1.2s)
    - Maintains speaker centroids
    - Returns stable labels: User1, User2, ...
    - Caps number of speakers (default 2) to avoid User3/User4 noise
    """

    def __init__(
        self, sample_rate: int, threshold: float = 0.72, max_speakers: int = 2
    ):
        self.sample_rate = sample_rate
        self.encoder = VoiceEncoder("cpu")
        print("Diarizer initialized (Resemblyzer on CPU)")

        self.threshold = threshold
        self.max_speakers = max_speakers

        # { "User1": centroid_embedding, ... }
        self.speakers: dict[str, np.ndarray] = {}
        self.next_id = 1

    # ------------- internal helpers -------------

    def _new_speaker_label(self) -> str:
        label = f"User{self.next_id}"
        self.next_id += 1
        return label

    def _best_match(self, emb: np.ndarray):
        """
        Return (best_label, best_similarity) or (None, -inf) if no speakers yet.
        """
        if not self.speakers:
            return None, float("-inf")

        best_label = None
        best_sim = float("-inf")

        for label, centroid in self.speakers.items():
            sim = 1.0 - cosine(centroid, emb)
            if sim > best_sim:
                best_sim = sim
                best_label = label

        return best_label, best_sim

    def _update_centroid(self, label: str, emb: np.ndarray, alpha: float = 0.2):
        """
        Exponential moving average update of speaker centroid.
        """
        old = self.speakers[label]
        new = alpha * emb + (1.0 - alpha) * old
        new /= np.linalg.norm(new) + 1e-9
        self.speakers[label] = new

    # ------------- public API -------------

    def diarize(self, waveform: np.ndarray) -> str | None:
        """
        Main entry point: given a waveform (1D numpy array),
        returns a label like 'User1', 'User2', ... or None if too short.
        Called on short windows (≈1.0–1.2s).
        """

        if waveform is None or len(waveform) < 0.4 * self.sample_rate:
            # Very short → skip
            return None

        # Ensure float32 mono
        wav = waveform.astype(np.float32)

        # Embed utterance (Resemblyzer handles variable length)
        emb = self.encoder.embed_utterance(wav)
        emb = emb / (np.linalg.norm(emb) + 1e-9)

        # Case 1: no speakers yet → create User1
        if not self.speakers:
            label = self._new_speaker_label()
            self.speakers[label] = emb
            print(f"🆕 Registered speaker: {label}")
            return label

        # Case 2: have speakers → find best match
        best_label, best_sim = self._best_match(emb)

        # If we already reached max_speakers → always reuse best match
        if len(self.speakers) >= self.max_speakers:
            self._update_centroid(best_label, emb)
            return best_label

        # If best match is good enough → reuse that speaker
        if best_sim >= self.threshold:
            self._update_centroid(best_label, emb)
            return best_label

        # Otherwise, create a new speaker (within max_speakers limit)
        label = self._new_speaker_label()
        self.speakers[label] = emb
        print(f"🆕 Registered speaker: {label} (sim={best_sim:.3f}, below threshold)")
        return label
