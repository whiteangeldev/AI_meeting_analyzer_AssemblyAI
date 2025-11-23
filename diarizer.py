# diarizer.py

from collections import deque
from typing import Deque

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
        self,
        sample_rate: int,
        threshold: float = 0.65,
        max_speakers: int = 2,
        min_segment_sec: float = 0.5,
        update_guard: float = 0.08,
        history_size: int = 20,
        min_samples_before_new_speaker: int = 3,
        confidence_gap: float = 0.05,  # Lowered from 0.10 - too strict was blocking legitimate User2
    ):
        self.sample_rate = sample_rate
        self.encoder = VoiceEncoder("cpu")
        print("Diarizer initialized (Resemblyzer on CPU)")

        self.threshold = threshold
        self.max_speakers = max_speakers
        self.min_segment_sec = min_segment_sec
        self.update_guard = update_guard
        self.history_size = max(1, history_size)
        self.min_samples_before_new_speaker = min_samples_before_new_speaker
        self.confidence_gap = confidence_gap  # New speaker must be this much different

        # { "User1": centroid_embedding, ... }
        self.speakers: dict[str, np.ndarray] = {}
        self.histories: dict[str, Deque[np.ndarray]] = {}
        self.next_id = 1

    # ------------- internal helpers -------------

    def reset(self):
        """Reset all tracked speakers (new conversation)."""
        self.speakers.clear()
        self.histories.clear()
        self.next_id = 1

    def _new_speaker_label(self) -> str:
        label = f"User{self.next_id}"
        self.next_id += 1
        return label

    def _register_speaker(self, emb: np.ndarray, reason: str = "") -> str:
        """
        Register a new speaker with initial embedding boost.

        Adds the same embedding multiple times to give new speakers
        initial weight in the centroid, making them more stable.
        """
        label = self._new_speaker_label()
        self.speakers[label] = emb
        self.histories[label] = deque([emb], maxlen=self.history_size)
        # Add the same embedding 2 more times to give it initial weight
        # This helps new speakers be more stable from the start
        for _ in range(2):
            self.histories[label].append(emb)
        # Recalculate centroid with the boosted history
        stacked = np.stack(list(self.histories[label]), axis=0)
        centroid = stacked.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        self.speakers[label] = centroid
        log_reason = f" ({reason})" if reason else ""
        print(f"🆕 Registered speaker: {label}{log_reason}")
        return label

    def is_newly_registered(self, label: str, min_samples: int = 3) -> bool:
        """Check if speaker was recently registered (has few samples)."""
        if label not in self.histories:
            return False
        return len(self.histories[label]) < min_samples

    def get_speaker_stats(self) -> dict:
        """Get statistics about all registered speakers for debugging."""
        stats = {}
        for label in self.speakers.keys():
            stats[label] = {
                "samples": len(self.histories.get(label, [])),
                "centroid_norm": float(np.linalg.norm(self.speakers[label])),
            }
        return stats

    def _best_match(self, emb: np.ndarray):
        """
        Return (best_label, best_similarity) with confidence weighting.

        Similarity is weighted by speaker history size to give established speakers
        a slight advantage, reducing false new-speaker registrations.
        """
        if not self.speakers:
            return None, float("-inf")

        best_label = None
        best_sim = float("-inf")

        for label, centroid in self.speakers.items():
            # Base similarity (cosine distance)
            sim = 1.0 - cosine(centroid, emb)

            # Weight by history size: more samples = more confidence
            # This gives established speakers a small boost (up to 5% after 5 samples)
            history_size = len(self.histories.get(label, []))
            confidence_weight = min(
                1.0, history_size / 5.0
            )  # Full weight after 5 samples
            weighted_sim = sim * (0.95 + 0.05 * confidence_weight)  # Boost by up to 5%

            if weighted_sim > best_sim:
                best_sim = weighted_sim
                best_label = label

        return best_label, best_sim

    def _update_centroid(self, label: str, emb: np.ndarray):
        """
        Update centroid using a bounded history buffer to keep
        the embedding representative of the latest audio.
        """
        history = self.histories[label]
        history.append(emb)
        stacked = np.stack(history, axis=0)
        centroid = stacked.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        self.speakers[label] = centroid

    def _should_update(self, similarity: float) -> bool:
        """
        Decide whether to refresh the centroid for the matched speaker.
        """
        update_threshold = max(self.threshold - self.update_guard, 0.0)
        return similarity >= update_threshold

    # ------------- public API -------------

    def diarize(
        self, waveform: np.ndarray, update_centroid: bool = True
    ) -> tuple[str | None, float]:
        """
        Main entry point: given a waveform (1D numpy array),
        returns (label, similarity) like ('User1', 0.85) or (None, -inf) if too short.
        Called on short windows (≈1.0–1.2s).

        Args:
            waveform: Audio samples
            update_centroid: If False, don't update speaker centroids (use for inference only)

        Returns:
            (speaker_label, similarity_score)
        """

        if waveform is None or len(waveform) < self.min_segment_sec * self.sample_rate:
            # Very short → skip
            return None, float("-inf")

        # Ensure float32 mono
        wav = waveform.astype(np.float32)

        # Embed utterance (Resemblyzer handles variable length)
        emb = self.encoder.embed_utterance(wav)
        emb = emb / (np.linalg.norm(emb) + 1e-9)

        # Case 1: no speakers yet → create User1
        if not self.speakers:
            label = self._register_speaker(emb) if update_centroid else None
            return label, 1.0 if label else float("-inf")

        # Case 2: have speakers → find best match
        best_label, best_sim = self._best_match(emb)

        # Get all similarities for logging and confidence gap check
        all_similarities = {}
        for label, centroid in self.speakers.items():
            sim = 1.0 - cosine(centroid, emb)
            all_similarities[label] = sim

        # If we already reached max_speakers → always reuse best match
        if len(self.speakers) >= self.max_speakers:
            # Use adaptive threshold: lower if we have both speakers to avoid creating third
            # This helps when both speakers exist but similarity is borderline
            adaptive_threshold = (
                self.threshold - 0.05 if len(self.speakers) == 2 else self.threshold
            )
            if best_sim >= adaptive_threshold:
                if update_centroid and self._should_update(best_sim):
                    self._update_centroid(best_label, emb)
                elif update_centroid:
                    print(
                        f"ℹ️  Skipping centroid update for {best_label} (sim={best_sim:.3f})"
                    )
            else:
                # Log why we're not updating - helps debug
                print(
                    f"ℹ️  Low similarity for {best_label} (sim={best_sim:.3f} < {adaptive_threshold:.3f}), "
                    f"all sims: {all_similarities}"
                )
            return best_label, best_sim

        # If best match is good enough → reuse that speaker
        if best_sim >= self.threshold:
            if update_centroid:
                self._update_centroid(best_label, emb)
            return best_label, best_sim

        # Before creating new speaker, check requirements:
        # 1. First speaker must have minimum samples
        # 2. New speaker must be significantly different (confidence gap)
        if len(self.speakers) == 1:
            first_speaker = list(self.speakers.keys())[0]
            first_samples = len(self.histories.get(first_speaker, []))

            # Require minimum samples before allowing second speaker
            if first_samples < self.min_samples_before_new_speaker:
                # Force match to first speaker even if below threshold
                print(
                    f"ℹ️  Forcing match to {first_speaker} (only {first_samples} samples, "
                    f"need {self.min_samples_before_new_speaker}). Best sim: {best_sim:.3f}"
                )
                if update_centroid:
                    self._update_centroid(first_speaker, emb)
                return first_speaker, best_sim

            # Require confidence gap: new speaker must be significantly different
            first_sim = all_similarities.get(first_speaker, 0.0)
            gap = first_sim - best_sim

            # Adaptive gap logic:
            # - If best_sim is well below threshold (< 0.60), it's clearly not User1
            #   Allow User2 registration regardless of gap (skip gap check)
            # - If best_sim is borderline (0.60-0.65), require gap to prevent false splits
            # - If first_speaker is very confident (> 0.75), require larger gap
            if best_sim < 0.60:
                # Well below threshold - clearly different speaker, skip gap check
                # This allows User2 to register even if gap is small
                print(
                    f"ℹ️  Allowing new speaker (best_sim={best_sim:.3f} < 0.60, "
                    f"clearly different from {first_speaker} sim={first_sim:.3f})"
                )
            elif first_sim > 0.75:
                # First speaker is very confident, require larger gap to prevent false splits
                adaptive_gap = self.confidence_gap * 1.5
                if best_sim > first_sim - adaptive_gap:
                    print(
                        f"ℹ️  Too similar to {first_speaker} (sim={first_sim:.3f} vs best={best_sim:.3f}, "
                        f"gap={gap:.3f} < {adaptive_gap:.3f}). Forcing match."
                    )
                    if update_centroid:
                        self._update_centroid(first_speaker, emb)
                    return first_speaker, best_sim
            else:
                # Standard case - require normal confidence gap
                adaptive_gap = self.confidence_gap
                if best_sim > first_sim - adaptive_gap:
                    print(
                        f"ℹ️  Too similar to {first_speaker} (sim={first_sim:.3f} vs best={best_sim:.3f}, "
                        f"gap={gap:.3f} < {adaptive_gap:.3f}). Forcing match."
                    )
                    if update_centroid:
                        self._update_centroid(first_speaker, emb)
                    return first_speaker, best_sim

        # All checks passed - create a new speaker (within max_speakers limit)
        if update_centroid:
            label = self._register_speaker(
                emb, reason=f"sim={best_sim:.3f}, below threshold"
            )
            return label, best_sim
        return None, best_sim
