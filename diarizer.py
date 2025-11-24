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

        # If we already reached max_speakers → check if we should still register new speaker
        if len(self.speakers) >= self.max_speakers:
            # If best similarity is below threshold, it's clearly a new speaker
            # Allow registration by replacing the least confident speaker
            # Use threshold - 0.10 as replacement trigger (with 0.72 threshold, this is 0.62)
            replacement_threshold = max(0.60, self.threshold - 0.10)
            if best_sim < replacement_threshold:
                # Find the speaker with the fewest samples (least confident)
                least_confident = min(
                    self.speakers.keys(),
                    key=lambda l: len(self.histories.get(l, [])),
                )
                print(
                    f"🔄 Replacing least confident speaker {least_confident} "
                    f"(best_sim={best_sim:.3f} < {replacement_threshold:.2f}, clearly new speaker)"
                )
                # Remove the least confident speaker
                del self.speakers[least_confident]
                del self.histories[least_confident]
                # Register new speaker
                if update_centroid:
                    label = self._register_speaker(
                        emb, reason=f"replacing {least_confident}"
                    )
                    return label, best_sim
                return None, best_sim

            # Use adaptive threshold based on number of speakers
            # More speakers = lower threshold needed (each speaker gets 0.02 reduction)
            adaptive_threshold = self.threshold - (0.02 * (len(self.speakers) - 1))
            adaptive_threshold = max(0.50, adaptive_threshold)  # Don't go below 0.50

            if best_sim >= adaptive_threshold:
                if update_centroid and self._should_update(best_sim):
                    self._update_centroid(best_label, emb)
                elif update_centroid:
                    print(
                        f"ℹ️  Skipping centroid update for {best_label} (sim={best_sim:.3f})"
                    )
            else:
                # Similarity too low - might be a new speaker but we're at max
                # Log for debugging
                print(
                    f"ℹ️  Low similarity for {best_label} (sim={best_sim:.3f} < {adaptive_threshold:.3f}), "
                    f"but max_speakers reached. All sims: {all_similarities}"
                )
            return best_label, best_sim

        # If best match is good enough → reuse that speaker
        # Use adaptive threshold: slightly lower when we have fewer speakers, but keep it high
        adaptive_match_threshold = self.threshold
        if len(self.speakers) < self.max_speakers:
            # Small reduction to help register new speakers, but keep threshold high
            # With high base threshold (0.72), we only reduce slightly
            reduction = 0.02 * (self.max_speakers - len(self.speakers))
            adaptive_match_threshold = max(
                0.65, self.threshold - reduction
            )  # Keep minimum at 0.65

        if best_sim >= adaptive_match_threshold:
            if update_centroid:
                self._update_centroid(best_label, emb)
            return best_label, best_sim

        # Before creating new speaker, check requirements:
        # 1. All existing speakers must have minimum samples (or at least the first one)
        # 2. New speaker must be significantly different from ALL existing speakers
        if len(self.speakers) > 0:
            # Check minimum samples for first speaker
            first_speaker = list(self.speakers.keys())[0]
            first_samples = len(self.histories.get(first_speaker, []))

            if first_samples < self.min_samples_before_new_speaker:
                # Force match to first speaker
                print(
                    f"ℹ️  Forcing match to {first_speaker} (only {first_samples} samples, "
                    f"need {self.min_samples_before_new_speaker}). Best sim: {best_sim:.3f}"
                )
                if update_centroid:
                    self._update_centroid(first_speaker, emb)
                return first_speaker, best_sim

            # Check confidence gap against ALL existing speakers, not just first
            # New speaker must be significantly different from the closest existing speaker
            closest_speaker = best_label
            closest_sim = best_sim

            # Adaptive gap logic based on best similarity
            # With higher base threshold, we can be more lenient here
            # If similarity is below 0.70, it's clearly a different speaker
            new_speaker_threshold = 0.70
            if best_sim < new_speaker_threshold:
                # Well below threshold - clearly different speaker, skip gap check
                print(
                    f"ℹ️  Allowing new speaker (best_sim={best_sim:.3f} < {new_speaker_threshold:.2f}, "
                    f"clearly different from {closest_speaker} sim={closest_sim:.3f})"
                )
                # Skip the rest of the gap checks and proceed to registration
                # (fall through to line 300)
            elif closest_sim > 0.75:
                # Closest speaker is very confident, require larger gap
                adaptive_gap = self.confidence_gap * 1.5
                gap = closest_sim - best_sim
                if best_sim > closest_sim - adaptive_gap:
                    print(
                        f"ℹ️  Too similar to {closest_speaker} (sim={closest_sim:.3f} vs best={best_sim:.3f}, "
                        f"gap={gap:.3f} < {adaptive_gap:.3f}). Forcing match."
                    )
                    if update_centroid:
                        self._update_centroid(closest_speaker, emb)
                    return closest_speaker, best_sim
            else:
                # Standard case - require normal confidence gap
                adaptive_gap = self.confidence_gap
                gap = closest_sim - best_sim
                if best_sim > closest_sim - adaptive_gap:
                    print(
                        f"ℹ️  Too similar to {closest_speaker} (sim={closest_sim:.3f} vs best={best_sim:.3f}, "
                        f"gap={gap:.3f} < {adaptive_gap:.3f}). Forcing match."
                    )
                    if update_centroid:
                        self._update_centroid(closest_speaker, emb)
                    return closest_speaker, best_sim

        # All checks passed - create a new speaker (within max_speakers limit)
        if update_centroid:
            label = self._register_speaker(
                emb, reason=f"sim={best_sim:.3f}, below threshold"
            )
            print(f"✅ Successfully registered {label} with similarity {best_sim:.3f}")
            return label, best_sim
        else:
            print(
                f"⚠️  Would register new speaker but update_centroid=False (sim={best_sim:.3f})"
            )
        return None, best_sim
