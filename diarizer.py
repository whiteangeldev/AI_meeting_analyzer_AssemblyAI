import numpy as np
from scipy.spatial.distance import cdist
from config import SAMPLE_RATE, MIN_AUDIO_SECS


class SpeakerDiarizer:
    """Lightweight speaker diarization with speaker merging and stability"""

    def __init__(self, similarity_threshold=0.75, merge_threshold=0.85):
        """
        Initialize diarizer

        Args:
            similarity_threshold: Minimum similarity to match existing speaker (0.75 = 75%)
            merge_threshold: Similarity threshold to merge speakers (0.85 = 85%)
        """
        try:
            from resemblyzer import VoiceEncoder

            self.encoder = VoiceEncoder()
        except ImportError:
            raise ImportError("resemblyzer not installed. Run: pip install resemblyzer")

        self.similarity_threshold = similarity_threshold
        self.merge_threshold = merge_threshold
        self.speakers = []  # List of speaker labels: ["User1", "User2", ...]
        self.embeddings = []  # List of speaker embeddings (centroids)
        self.speaker_count = 0
        self.max_speakers = 2  # Expecting up to 2 speakers (e.g., dialogue)
        self.assignment_count = 0
        self.new_speaker_threshold = (
            0.68  # If similarity < 0.68, create new speaker (more sensitive)
        )

    def reset(self):
        """Reset diarizer for new session"""
        self.speakers = []
        self.embeddings = []
        self.speaker_count = 0
        self.assignment_count = 0

    def _merge_similar_speakers(self):
        """Merge speakers that are very similar (likely same person)"""
        if len(self.speakers) < 2:
            return

        # Check all pairs for high similarity
        merged = set()
        for i in range(len(self.speakers)):
            if i in merged:
                continue
            for j in range(i + 1, len(self.speakers)):
                if j in merged:
                    continue

                # Calculate similarity between speakers
                sim = np.dot(self.embeddings[i], self.embeddings[j])

                # Lower merge threshold to merge more aggressively (0.80 for very aggressive merging)
                if sim >= 0.80:
                    # Merge j into i
                    # Average embeddings
                    self.embeddings[i] = (
                        0.6 * self.embeddings[i] + 0.4 * self.embeddings[j]
                    )
                    self.embeddings[i] /= np.linalg.norm(self.embeddings[i]) + 1e-9

                    # Remove j
                    merged.add(j)
                    print(
                        f"🔀 Merged {self.speakers[j]} into {self.speakers[i]} (similarity: {sim:.3f})"
                    )

        # Remove merged speakers (in reverse order to maintain indices)
        for idx in sorted(merged, reverse=True):
            self.speakers.pop(idx)
            self.embeddings.pop(idx)

    def assign(self, audio_f32: np.ndarray, sr=SAMPLE_RATE) -> str:
        """
        Assign speaker label to audio segment

        Args:
            audio_f32: Audio samples as float32 array
            sr: Sample rate (default 16000)

        Returns:
            Speaker label: "User1", "User2", etc.
        """
        # Check minimum audio length
        min_samples = int(sr * MIN_AUDIO_SECS)
        if audio_f32 is None or len(audio_f32) < min_samples:
            return None

        # Generate embedding
        emb = self.encoder.embed_utterance(audio_f32.astype(np.float32))
        emb = emb / (np.linalg.norm(emb) + 1e-9)  # Normalize

        # First speaker
        if not self.speakers:
            self.speaker_count = 1
            label = f"User{self.speaker_count}"
            self.speakers.append(label)
            self.embeddings.append(emb)
            self.assignment_count += 1
            print(f"🎤 First speaker detected: {label}")
            return label

        # Compare with existing speakers
        embeddings_array = np.array(self.embeddings)
        similarities = 1.0 - cdist([emb], embeddings_array, metric="cosine")[0]
        max_similarity = similarities.max()
        best_match_idx = similarities.argmax()

        # Log similarities for debugging
        sim_details = [
            f"{self.speakers[i]}: {sim:.3f}" for i, sim in enumerate(similarities)
        ]
        print(
            f"📊 Similarities: {', '.join(sim_details)} | Max: {max_similarity:.3f} with {self.speakers[best_match_idx]}"
        )

        # If similarity is VERY high (>= 0.75), assign to existing speaker
        # This requires high confidence that it's the same person
        if max_similarity >= self.similarity_threshold:
            # Update centroid with exponential moving average
            alpha = 0.90  # Allow some adaptation
            self.embeddings[best_match_idx] = (
                alpha * self.embeddings[best_match_idx] + (1 - alpha) * emb
            )
            self.embeddings[best_match_idx] /= (
                np.linalg.norm(self.embeddings[best_match_idx]) + 1e-9
            )
            self.assignment_count += 1
            print(
                f"✓ High confidence: Assigned to {self.speakers[best_match_idx]} (similarity: {max_similarity:.3f})"
            )

            # Merge similar speakers periodically (every 5 assignments) - more frequent merging
            if self.assignment_count % 5 == 0:
                self._merge_similar_speakers()
                # After merging, check if best_match_idx is still valid
                if best_match_idx < len(self.speakers):
                    return self.speakers[best_match_idx]
                else:
                    return self.speakers[0] if self.speakers else "User1"

            return self.speakers[best_match_idx]

        # Check if similarity is low enough to warrant new speaker
        # For man/woman, similarity is typically 0.5-0.7
        # If similarity < 0.68, likely different speaker - create new one
        if max_similarity < self.new_speaker_threshold:
            # Low similarity - create new speaker (if we still have room)
            if self.speaker_count < self.max_speakers:
                self.speaker_count += 1
                label = f"User{self.speaker_count}"
                self.speakers.append(label)
                self.embeddings.append(emb)
                self.assignment_count += 1
                print(
                    f"🎤 NEW SPEAKER DETECTED: {label} (similarity to closest: {max_similarity:.3f})"
                )

                # Merge periodically (every 5 assignments)
                if self.assignment_count % 5 == 0:
                    self._merge_similar_speakers()

                return label
            else:
                print(
                    f"⚠ Max speakers ({self.max_speakers}) reached - assigning to closest match {self.speakers[best_match_idx]} (similarity: {max_similarity:.3f})"
                )
                return self.speakers[best_match_idx]

        # Moderate similarity (0.68 - 0.75) - ambiguous zone
        # For different speakers, this range is common (man/woman typically 0.65-0.72)
        # Be more aggressive about creating new speakers in this range
        if len(similarities) > 1:
            sorted_sims = np.sort(similarities)[::-1]
            second_best = sorted_sims[1] if len(sorted_sims) > 1 else 0
            diff = max_similarity - second_best

            # If difference is small (< 0.15) OR similarity is in ambiguous zone (< 0.72), prefer new speaker
            # This catches cases where similarity is 0.68-0.72 (likely different speaker)
            if (diff < 0.15) or (max_similarity < 0.72):
                if self.speaker_count < self.max_speakers:
                    self.speaker_count += 1
                    label = f"User{self.speaker_count}"
                    self.speakers.append(label)
                    self.embeddings.append(emb)
                    self.assignment_count += 1
                    print(
                        f"🎤 NEW SPEAKER (ambiguous zone): {label} (max: {max_similarity:.3f}, 2nd: {second_best:.3f}, diff: {diff:.3f})"
                    )

                    if self.assignment_count % 5 == 0:
                        self._merge_similar_speakers()

                    return label
                else:
                    print(
                        f"⚠ Ambiguous zone but max speakers reached - assigning to {self.speakers[best_match_idx]}"
                    )
                    return self.speakers[best_match_idx]

        # If similarity is 0.72-0.75 and second best is clearly lower, assign to best match
        # This is the only moderate case where we match (very close to threshold)
        alpha = 0.90
        self.embeddings[best_match_idx] = (
            alpha * self.embeddings[best_match_idx] + (1 - alpha) * emb
        )
        self.embeddings[best_match_idx] /= (
            np.linalg.norm(self.embeddings[best_match_idx]) + 1e-9
        )
        self.assignment_count += 1
        print(
            f"⚠ Moderate confidence (edge case): Assigned to {self.speakers[best_match_idx]} (similarity: {max_similarity:.3f})"
        )

        if self.assignment_count % 5 == 0:
            self._merge_similar_speakers()
            if best_match_idx < len(self.speakers):
                return self.speakers[best_match_idx]
            else:
                return self.speakers[0] if self.speakers else "User1"

        return self.speakers[best_match_idx]
