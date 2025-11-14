import numpy as np
from scipy.spatial.distance import cdist
from resemblyzer import VoiceEncoder
from config import SIM_THRESH, EMA, MAX_SPK


class SpeakerRegistry:
    def __init__(self, sim_thresh=SIM_THRESH, ema=EMA, max_spk=MAX_SPK):
        self.sim_thresh = sim_thresh
        self.ema = ema
        self.max_spk = max_spk
        self.labels = []
        self.centroids = []
        self.count = 0
        self.encoder = VoiceEncoder()

    def assign(self, audio_f32: np.ndarray, sr=16000) -> str:
        if audio_f32 is None or len(audio_f32) < sr * 0.3:  # <300ms too short
            return "User?"
        emb = self.encoder.embed_utterance(audio_f32.astype(np.float32))
        emb = emb / (np.linalg.norm(emb) + 1e-9)

        if not self.centroids:
            self.count += 1
            lbl = f"User{self.count}"
            self.labels.append(lbl)
            self.centroids.append(emb)
            return lbl

        sims = 1.0 - cdist([emb], self.centroids, metric="cosine")[0]
        j = int(np.argmax(sims))
        if sims[j] >= self.sim_thresh:
            self.centroids[j] = self.ema * self.centroids[j] + (1 - self.ema) * emb
            self.centroids[j] /= np.linalg.norm(self.centroids[j]) + 1e-9
            return self.labels[j]

        if len(self.labels) < self.max_spk:
            self.count += 1
            lbl = f"User{self.count}"
            self.labels.append(lbl)
            self.centroids.append(emb)
            return lbl

        return self.labels[j]
