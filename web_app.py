# web_app.py

import eventlet

eventlet.monkey_patch()

import asyncio
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from threading import Thread
import time
import hashlib
import re
from collections import Counter
from dotenv import load_dotenv

from audio_capture import mic_frames, system_audio_frames, list_audio_devices
from ringbuffer import RingBuffer
from stream_aai import aai_stream
from config import SAMPLE_RATE, MAX_BUFFER_SECS
from diarizer import OnlineDiarizer

load_dotenv()

# ---------------- Flask / Socket.IO setup ----------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    logger=False,
    engineio_logger=False,
)

# ---------------- Global state ----------------

ring = RingBuffer()
is_recording = False
recording_thread = None
emitted_text_hashes = set()  # Track emitted text to prevent duplicates
recent_emitted_blocks = []  # Track recent finalized blocks
recent_blocks_max = 10
processed_final_transcripts = (
    set()
)  # Track processed final transcripts to avoid duplicate diarization

# diarizer: real-time speaker id
# Lower threshold (0.65) for better User2 detection, relaxed update_guard (0.08) for learning
diarizer = OnlineDiarizer(
    sample_rate=SAMPLE_RATE, threshold=0.65, max_speakers=2, update_guard=0.08
)

# AssemblyAI speaker label mapping (API -> UI-friendly "UserX")
MAX_UI_SPEAKERS = 4
api_speaker_map: dict[str, str] = {}
api_speaker_counts: dict[str, int] = {}
next_ui_speaker_id = 1


# ---------------- Duplicate detection / formatting ----------------


def normalize_text(text):
    """Normalize text for duplicate checking."""
    normalized = text.lower()
    normalized = re.sub(r"\bten\b", "10", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def check_duplicate(text):
    """Check if text is a duplicate (exact or near-exact match only)."""
    if not text:
        return False

    normalized = normalize_text(text)
    text_hash = hashlib.md5(normalized.encode()).hexdigest()
    if text_hash in emitted_text_hashes:
        return True

    # Only check for near-exact matches (95%+ similarity) to avoid false positives
    for block_text in recent_emitted_blocks:
        block_normalized = normalize_text(block_text)
        if not block_normalized:
            continue

        # Exact substring match (one contains the other completely)
        if normalized == block_normalized:
            return True

        # Very high similarity check (95%+ word overlap) for near-duplicates
        text_words = set(normalized.split())
        block_words = set(block_normalized.split())
        if text_words and block_words:
            overlap_ratio = len(text_words & block_words) / max(
                len(text_words), len(block_words)
            )
            # Only flag as duplicate if 95%+ overlap (was 70%, too aggressive)
            if overlap_ratio >= 0.95:
                return True

    return False


def format_transcript(text):
    """Format transcript text (capitalization, punctuation, spacing)."""
    if not text:
        return ""

    text = " ".join(text.split())

    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    # Add period if sentence is complete and no ending punctuation
    if text and text[-1] not in ".!?":
        if any(text.lower().endswith(word) for word in ["here", "work", "too", "true"]):
            text += "."

    # Fix spacing around punctuation
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", text)

    return text.strip()


def _word_time_bounds(words):
    """
    Return (start_s, end_s) in seconds covering all words that have timing info.
    """
    starts = []
    ends = []
    for word in words or []:
        start = word.get("start")
        end = word.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            if end > start:
                starts.append(float(start) / 1000.0)
                ends.append(float(end) / 1000.0)
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def slice_audio_window(
    ring: RingBuffer,
    words,
    pad_sec: float = 0.25,  # Increased from 0.15 to capture more context
    min_window_sec: float = 1.5,  # Increased from 1.2 for better embeddings
    max_window_sec: float = 3.0,  # Increased from 2.5 to allow longer segments
    fallback_window_sec: float = 2.2,  # Increased from 1.8 for more audio
):
    """
    Slice audio aligned with the AssemblyAI word timings when available.
    Returns (waveform, (start_s, end_s)).
    """
    bounds = _word_time_bounds(words)
    ring_now = ring.now

    if bounds:
        start_s, end_s = bounds
        start_s = max(0.0, start_s - pad_sec)
        end_s = min(ring_now, end_s + pad_sec)
        if end_s - start_s < min_window_sec:
            end_s = min(ring_now, start_s + min_window_sec)
        if end_s - start_s > max_window_sec:
            start_s = end_s - max_window_sec
    else:
        end_s = ring_now
        start_s = max(0.0, end_s - fallback_window_sec)

    buffer_floor = max(0.0, ring_now - MAX_BUFFER_SECS + 0.05)
    start_s = max(start_s, buffer_floor)
    if end_s <= start_s:
        return None, (start_s, end_s)

    waveform = ring.slice(start_s, end_s)
    return waveform, (start_s, end_s)


def reset_api_speakers():
    """Reset AssemblyAI->UI speaker mapping each time a session starts."""
    global api_speaker_map, api_speaker_counts, next_ui_speaker_id
    api_speaker_map = {}
    api_speaker_counts = {}
    next_ui_speaker_id = 1


def assign_ui_speaker_label(api_speaker_id):
    """
    Map AssemblyAI speaker ids (e.g., "A", "B") to stable UI labels (User1, User2...).
    Caps the number of concurrent UI speakers to avoid unbounded growth.
    """
    global api_speaker_map, api_speaker_counts, next_ui_speaker_id

    if not api_speaker_id:
        return None

    if api_speaker_id in api_speaker_map:
        api_speaker_counts[api_speaker_id] += 1
        return api_speaker_map[api_speaker_id]

    if len(api_speaker_map) >= MAX_UI_SPEAKERS:
        # Reuse the least-frequent mapping to keep UI speaker ids bounded
        oldest = min(api_speaker_counts, key=api_speaker_counts.get)
        label = api_speaker_map.pop(oldest)
        api_speaker_counts.pop(oldest)
    else:
        label = f"User{next_ui_speaker_id}"
        next_ui_speaker_id += 1

    api_speaker_map[api_speaker_id] = label
    api_speaker_counts[api_speaker_id] = 1
    return label


def resolve_api_speaker(words):
    """
    Extract speaker label from AssemblyAI's word-level speaker tags.

    AssemblyAI provides speaker IDs (like "A", "B", "C") for each word when
    speaker diarization is enabled. This function:
    1. Counts votes from all words (weighted by word duration)
    2. Picks the most common speaker ID
    3. Maps it to our UI-friendly label (User1, User2, etc.)

    Args:
        words: List of word dicts from AssemblyAI, each may have "speaker" field

    Returns:
        UI-friendly speaker label (e.g., "User1", "User2") or None if no speaker tags found
    """
    if not words:
        return None

    speaker_votes = Counter()
    for word in words:
        api_id = word.get("speaker")
        if not api_id:
            continue

        start = word.get("start")
        end = word.get("end")
        if (
            isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and end > start
        ):
            weight = float(end) - float(start)
        else:
            weight = max(len(word.get("text", "")), 1)
        speaker_votes[api_id] += weight

    if not speaker_votes:
        return None

    api_label, _ = speaker_votes.most_common(1)[0]
    return assign_ui_speaker_label(api_label)


def detect_speaker_boundaries(words):
    """
    Detect exact speaker change points within a transcript using AssemblyAI's word-level speaker tags.

    This function scans through words and identifies where the speaker changes, allowing us to
    split a single transcript into multiple segments, each with the correct speaker label.

    Example: If transcript has "Hello [Speaker A] ... How are you? [Speaker B] I'm good"
    This will return: [(0, 3, 'A'), (3, 6, 'B')] - meaning words 0-2 are Speaker A, words 3-5 are Speaker B.

    Args:
        words: List of word dicts from AssemblyAI, each may have a "speaker" field (e.g., "A", "B")

    Returns:
        List of tuples: [(start_idx, end_idx, speaker_id), ...]
        Each tuple represents one continuous segment spoken by the same speaker.
        Empty list if no speaker tags found.
    """
    if not words:
        return []

    segments = []
    current_speaker = None  # Track which speaker we're currently in
    segment_start = 0  # Where the current segment started

    # Scan through words looking for speaker changes
    for i, word in enumerate(words):
        api_id = word.get(
            "speaker"
        )  # Get speaker ID from AssemblyAI (e.g., "A", "B", None)

        # If this word has a different speaker than the current one, we found a boundary
        if api_id and api_id != current_speaker:
            # Speaker change detected!
            if current_speaker is not None:
                # Save the previous segment (from segment_start to this word)
                segments.append((segment_start, i, current_speaker))
            # Start tracking the new speaker
            current_speaker = api_id
            segment_start = i

    # Don't forget the last segment (from last boundary to end of words)
    if current_speaker is not None:
        segments.append((segment_start, len(words), current_speaker))

    return segments


def try_split_long_transcript(words, ring, diarizer, transcript_text):
    """
    Attempt to split a long transcript by analyzing audio in time-based chunks.

    This is a fallback when AssemblyAI doesn't provide speaker tags but the transcript
    is long enough that it likely contains multiple speakers.

    Args:
        words: List of word dicts with timing information
        ring: RingBuffer with audio
        diarizer: OnlineDiarizer instance
        transcript_text: Full transcript text

    Returns:
        List of (start_idx, end_idx, speaker_label) segments, or None if can't split
    """
    if not words or len(words) < 30:  # Only try for transcripts with 30+ words
        return None

    # Get time bounds
    bounds = _word_time_bounds(words)
    if not bounds:
        return None

    start_s, end_s = bounds
    duration = end_s - start_s

    # Only try if transcript is longer than 8 seconds
    if duration < 8.0:
        return None

    # Split into 4-5 second chunks and analyze each
    chunk_duration = 4.5
    segments = []
    current_chunk_start_idx = 0
    current_chunk_start_time = start_s
    last_speaker = None

    for chunk_idx in range(int(duration / chunk_duration) + 1):
        chunk_start_time = start_s + chunk_idx * chunk_duration
        chunk_end_time = min(end_s, chunk_start_time + chunk_duration)

        if chunk_end_time <= chunk_start_time:
            break

        # Find words in this time range
        chunk_word_indices = []
        for i, word in enumerate(words):
            word_start = word.get("start")
            word_end = word.get("end")
            if isinstance(word_start, (int, float)) and isinstance(
                word_end, (int, float)
            ):
                word_start_s = float(word_start) / 1000.0
                word_end_s = float(word_end) / 1000.0
                # Check if word overlaps with chunk
                if word_start_s < chunk_end_time and word_end_s > chunk_start_time:
                    chunk_word_indices.append(i)

        if not chunk_word_indices:
            continue

        chunk_words = [words[i] for i in chunk_word_indices]

        # Get audio for this chunk
        try:
            audio_segment, _ = slice_audio_window(ring, chunk_words)
            if audio_segment is not None and len(audio_segment) > 0:
                # Run diarization (don't update centroids to avoid contamination)
                speaker_label, similarity = diarizer.diarize(
                    audio_segment, update_centroid=False
                )

                if speaker_label and similarity >= 0.50:
                    # Check if speaker changed
                    if last_speaker is None:
                        last_speaker = speaker_label
                        current_chunk_start_idx = chunk_word_indices[0]
                    elif speaker_label != last_speaker:
                        # Speaker changed - save previous segment
                        segments.append(
                            (
                                current_chunk_start_idx,
                                chunk_word_indices[0],
                                last_speaker,
                            )
                        )
                        # Start new segment
                        last_speaker = speaker_label
                        current_chunk_start_idx = chunk_word_indices[0]
        except Exception as e:
            print(f"⚠ Error analyzing chunk for splitting: {e}")
            continue

    # Add final segment
    if last_speaker is not None and segments:
        segments.append((current_chunk_start_idx, len(words), last_speaker))
    elif last_speaker is not None:
        # Only one segment found, no need to split
        return None

    # Only return if we found multiple segments
    return segments if len(segments) > 1 else None


def emit_partial(text, speaker="Speaker"):
    """Emit partial in overwrite mode (Google Meet style)."""
    socketio.emit(
        "partial_update",
        {
            "speaker": speaker,
            "text": text,
            "timestamp": time.time(),
        },
        namespace="/",
    )


def emit_transcription(text, is_partial=False, speaker="Speaker"):
    """
    Emit transcription to frontend.
    - For finals: apply duplicate check + formatting.
    - For partials: (currently) not sent to UI to avoid 'Speaker...' noisy lines.
    """
    try:
        if not text or not text.strip():
            return

        # If partials are too noisy in UI, simply don't emit them:
        if is_partial:
            # If you later want UI partials, you can re-enable this,
            # but ensure the frontend overwrites instead of appending lines.
            # socketio.emit(... is_partial=True ...)
            return

        # Finalized text → duplicate filter
        if check_duplicate(text):
            print(f"⚠ Duplicate detected, skipping: {text[:70]}...")
            return

        # Mark as emitted
        normalized = normalize_text(text)
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        emitted_text_hashes.add(text_hash)

        # Track recent blocks
        recent_emitted_blocks.append(text)
        if len(recent_emitted_blocks) > recent_blocks_max:
            recent_emitted_blocks.pop(0)

        # Format final text
        formatted = format_transcript(text)

        socketio.emit(
            "transcription",
            {
                "speaker": speaker,
                "text": formatted,
                "is_partial": False,
                "timestamp": time.time(),
            },
            namespace="/",
        )

    except Exception as e:
        print(f"Error emitting transcription: {e}")


# ---------------- Transcription worker ----------------


async def transcription_worker(audio_mode="microphone"):
    """Main transcription worker: captures audio, sends to AAI, receives callbacks."""

    async def audio_gen():
        global is_recording, ring

        try:
            frame_source = (
                system_audio_frames() if audio_mode == "system" else mic_frames()
            )

            async for frame in frame_source:
                if not is_recording:
                    break
                # store into ring buffer for later diarization
                ring.append(frame)
                yield frame

        except RuntimeError as e:
            error_msg = str(e)
            print(f"Audio error: {error_msg}")
            socketio.emit("error", {"message": error_msg}, namespace="/")
            is_recording = False
            socketio.emit("recording_status", {"is_recording": False}, namespace="/")
        except Exception as e:
            error_msg = f"Audio capture error: {str(e)}"
            print(f"Audio error: {error_msg}")
            socketio.emit("error", {"message": error_msg}, namespace="/")
            is_recording = False
            socketio.emit("recording_status", {"is_recording": False}, namespace="/")

    async def on_result(data: dict):
        """
        Main callback for AssemblyAI streaming transcription responses.

        This function handles two types of transcripts:
        1. PARTIAL transcripts (end_of_turn=False): Live updates, shown in partial caption area
        2. FINAL transcripts (end_of_turn=True): Complete sentences, processed for speaker diarization

        Speaker Diarization Flow:
        - First tries AssemblyAI's built-in speaker labels (most accurate)
        - Falls back to Resemblyzer if AssemblyAI didn't provide speaker tags
        - Detects speaker boundaries within transcripts (splits multi-speaker transcripts)
        - Uses adaptive thresholds: lower for newly registered speakers (0.50) vs established (0.55)

        Args:
            data: Dict from AssemblyAI containing:
                - "transcript": The text transcription
                - "words": List of word dicts with timing and optional speaker tags
                - "end_of_turn": Boolean indicating if this is a final transcript
        """
        global is_recording, ring, diarizer, processed_final_transcripts
        if not is_recording:
            return

        # Skip non-transcript messages
        if "transcript" not in data and "words" not in data:
            return

        end_of_turn = data.get("end_of_turn", False)
        transcript = data.get("transcript", "").strip()
        words = data.get("words") or []

        if not transcript:
            return

        # ========================================================================
        # DUPLICATE DETECTION
        # ========================================================================
        # AssemblyAI sometimes sends the same final transcript multiple times.
        # We use a hash of the normalized transcript text to detect and skip duplicates.
        # This prevents duplicate diarization calls and duplicate UI messages.
        if end_of_turn:
            # Use normalized transcript text only for deduplication
            normalized = normalize_text(transcript)
            transcript_hash = hashlib.md5(normalized.encode()).hexdigest()
            if transcript_hash in processed_final_transcripts:
                print(f"ℹ️  Skipping duplicate final transcript: {transcript[:50]}...")
                return  # Already processed this exact final transcript
            processed_final_transcripts.add(transcript_hash)
            # Keep set size bounded
            if len(processed_final_transcripts) > 50:
                processed_final_transcripts.clear()

        # 1) Partials → don't send to UI (avoids noisy 'Speaker...' lines),
        # but we *could* use them internally if needed.
        if not end_of_turn:
            temp_speaker = resolve_api_speaker(words) or "Speaker"

            # Show partial caption (overwrite)
            emit_partial(transcript, speaker=temp_speaker)
            return

        # ========================================================================
        # 2) FINAL TRANSCRIPTS - Speaker Diarization & Boundary Detection
        # ========================================================================
        # For finalized transcripts, we need to:
        #   1. Detect if multiple speakers appear in this single transcript
        #   2. Split at exact boundaries where speaker changes occur
        #   3. Assign correct speaker label to each segment
        #   4. Use Resemblyzer as fallback if AssemblyAI didn't provide speaker tags

        # Step 1: Detect speaker boundaries using AssemblyAI's word-level speaker tags
        # This finds exact points where the speaker changes within the transcript
        segments = detect_speaker_boundaries(words)

        # Clear the partial caption area (the live-updating text) when final arrives
        socketio.emit("partial_update", {"text": "", "speaker": ""}, namespace="/")

        # Step 2: Check if we actually have multiple distinct speakers
        # We only split if:
        #   - We found more than one segment
        #   - All segments have valid speaker IDs (not None)
        #   - There are at least 2 different speakers (not just one speaker split weirdly)
        has_multiple_speakers = (
            len(segments) > 1
            and all(seg[2] is not None for seg in segments)
            and len(set(seg[2] for seg in segments))
            > 1  # At least 2 different speakers
        )

        # Step 2b: If no speaker boundaries detected but transcript is long, try fallback splitting
        if not has_multiple_speakers and len(words) >= 30:
            fallback_segments = try_split_long_transcript(
                words, ring, diarizer, transcript
            )
            if fallback_segments:
                print(
                    f"🔀 Fallback split: Detected {len(fallback_segments)} speaker segments in long transcript"
                )
                segments = fallback_segments
                has_multiple_speakers = True

        if has_multiple_speakers:
            # ============================================================
            # CASE A: Multiple Speakers in One Transcript
            # ============================================================
            # This happens when AssemblyAI sends one transcript that contains
            # speech from multiple speakers (e.g., rapid back-and-forth conversation).
            # We split it into separate segments, each with the correct speaker label.

            print(f"🔀 Detected {len(segments)} speaker segments in transcript")

            # Process each segment separately
            for seg_start, seg_end, speaker_id_or_label in segments:
                # Extract the words that belong to this segment
                seg_words = words[seg_start:seg_end]
                # Reconstruct the text for this segment
                seg_text = " ".join(
                    w.get("text", "") for w in seg_words if w.get("text")
                )

                # Skip empty segments
                if not seg_text.strip():
                    continue

                # Step 1: Determine speaker label
                # Check if it's already a UI label (from fallback splitting) or an API ID
                if speaker_id_or_label and speaker_id_or_label.startswith("User"):
                    # Already a UI label from fallback splitting
                    seg_speaker = speaker_id_or_label
                else:
                    # It's an API speaker ID - map to UI label
                    seg_speaker = (
                        assign_ui_speaker_label(speaker_id_or_label)
                        if speaker_id_or_label
                        else None
                    )

                # Step 2: If AssemblyAI didn't provide a speaker, use Resemblyzer as fallback
                if not seg_speaker:
                    try:
                        # Get the audio window that corresponds to this segment's words
                        audio_segment, (slice_start, slice_end) = slice_audio_window(
                            ring, seg_words
                        )

                        if audio_segment is not None and len(audio_segment) > 0:
                            # Run Resemblyzer diarization on this segment's audio
                            diarized_label, similarity = diarizer.diarize(
                                audio_segment, update_centroid=True
                            )

                            # IMPORTANT: Use lower threshold for newly registered speakers
                            # Why? When a speaker is first detected, their embedding might not be perfect yet.
                            # A newly registered speaker (first 3 samples) gets threshold 0.50 instead of 0.55.
                            # This prevents legitimate new speakers from being rejected as "Speaker".
                            min_sim = (
                                0.50  # Lower threshold for new speakers (more lenient)
                                if (
                                    diarized_label
                                    and diarizer.is_newly_registered(diarized_label)
                                )
                                else 0.55  # Normal threshold for established speakers
                            )

                            # Only accept if similarity is high enough
                            if diarized_label and similarity >= min_sim:
                                seg_speaker = diarized_label
                                print(
                                    f"✓ Diarized segment: {diarized_label} (sim={similarity:.3f}) "
                                    f"window {slice_start:.2f}-{slice_end:.2f}s"
                                )
                    except Exception as e:
                        print(f"⚠ Diarization error for segment: {e}")

                # Step 3: Final fallback - if we still don't have a speaker, use generic "Speaker"
                if not seg_speaker:
                    seg_speaker = "Speaker"

                # Emit this segment as a separate transcription with its speaker label
                emit_transcription(seg_text, is_partial=False, speaker=seg_speaker)
        else:
            # ============================================================
            # CASE B: Single Speaker in Transcript
            # ============================================================
            # Most common case: one transcript = one speaker.
            # We try AssemblyAI's speaker label first, then fall back to Resemblyzer.

            # Step 1: Try to get speaker from AssemblyAI's word-level speaker tags
            # This is usually more accurate than Resemblyzer because AssemblyAI has
            # access to the full audio context and better models.
            speaker_label = resolve_api_speaker(words)
            api_provided_speaker = speaker_label is not None

            # Step 2: Set confidence thresholds for Resemblyzer diarization
            # These determine when we accept a diarization result:
            MIN_DIARIZATION_SIM = 0.55  # Normal threshold for established speakers
            MIN_DIARIZATION_SIM_NEW = (
                0.50  # Lower threshold for newly registered speakers (first 3 samples)
                # Why lower? New speakers haven't built up a good centroid yet,
                # so their similarity scores might be slightly lower but still valid.
            )

            # Step 3: Run Resemblyzer diarization only if AssemblyAI didn't provide speaker
            # Why skip if API provided speaker? To avoid contaminating our centroids with
            # potentially noisy audio windows. We trust AssemblyAI's labels when available.
            if not api_provided_speaker:
                try:
                    # Get the audio window aligned with this transcript's words
                    audio_segment, (slice_start, slice_end) = slice_audio_window(
                        ring, words
                    )

                    if audio_segment is not None and len(audio_segment) > 0:
                        # Run Resemblyzer diarization
                        diarized_label, similarity = diarizer.diarize(
                            audio_segment, update_centroid=True
                        )

                        # Choose threshold based on whether speaker is newly registered
                        # New speakers get more lenient threshold (0.50) to avoid false rejections
                        # Also accept if similarity is close to threshold (within 0.03) for any registered speaker
                        is_new = diarized_label and diarizer.is_newly_registered(
                            diarized_label, min_samples=5
                        )  # Increased to 5 to catch more cases
                        min_sim = (
                            MIN_DIARIZATION_SIM_NEW  # 0.50 for new speakers
                            if is_new
                            else MIN_DIARIZATION_SIM  # 0.55 for established speakers
                        )

                        # Accept diarization result if:
                        # 1. Similarity is high enough, OR
                        # 2. It's a registered speaker and similarity is close (within 0.03 of threshold)
                        #    This prevents "Speaker" fallback for legitimate registered speakers
                        if diarized_label and (
                            similarity >= min_sim
                            or (
                                similarity >= min_sim - 0.03
                                and diarized_label in diarizer.speakers
                            )
                        ):
                            speaker_label = diarized_label
                            # Log speaker stats periodically for debugging
                            stats = diarizer.get_speaker_stats()
                            print(
                                f"✓ Diarized: {diarized_label} (sim={similarity:.3f}) "
                                f"window {slice_start:.2f}-{slice_end:.2f}s | "
                                f"Stats: {stats}"
                            )
                        elif diarized_label:
                            # Diarization found a speaker but confidence too low
                            print(
                                f"⚠ Low confidence diarization: {diarized_label} "
                                f"(sim={similarity:.3f} < {min_sim:.2f}), "
                                f"using fallback"
                            )
                        else:
                            # Diarizer couldn't determine speaker (too short, etc.)
                            print(
                                f"ℹ️  Diarizer returned None for window "
                                f"{slice_start:.2f}-{slice_end:.2f}s"
                            )
                    else:
                        # No audio available in ring buffer for this window
                        print(
                            f"ℹ️  No audio available for diarization window "
                            f"{slice_start:.2f}-{slice_end:.2f}s"
                        )
                except Exception as e:
                    print(f"⚠ Diarization error: {e}")
            else:
                # AssemblyAI provided speaker label - use it directly
                # We don't run Resemblyzer here to avoid contaminating centroids with
                # potentially misaligned audio windows
                print(f"✓ Using API speaker: {speaker_label}")

            # Step 4: Final fallback - if we still don't have a speaker, use generic "Speaker"
            if not speaker_label:
                speaker_label = "Speaker"

            # Step 5: Emit the final transcript with the determined speaker label
            emit_transcription(transcript, is_partial=False, speaker=speaker_label)

    await aai_stream(audio_gen(), on_result)


def run_transcription(audio_mode="microphone"):
    """Run transcription (async) in a separate thread."""
    global is_recording
    print(f"Transcription worker starting with audio mode: {audio_mode}...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(transcription_worker(audio_mode))
    except Exception as e:
        print(f"Transcription error: {e}")
        socketio.emit("error", {"message": str(e)}, namespace="/")
    finally:
        is_recording = False
        socketio.emit("recording_status", {"is_recording": False}, namespace="/")
        print("Transcription worker stopped")


# ---------------- Flask / Socket.IO handlers ----------------


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    print("Client connected")
    try:
        devices = list_audio_devices()
        emit("connected", {"status": "Connected", "audio_devices": devices})
    except Exception as e:
        print(f"Error listing audio devices: {e}")
        emit("connected", {"status": "Connected"})


@socketio.on("list_audio_devices")
def handle_list_audio_devices():
    """List all available audio input devices."""
    try:
        devices = list_audio_devices()
        emit("audio_devices", {"devices": devices})
    except Exception as e:
        emit("error", {"message": f"Failed to list audio devices: {str(e)}"})


@socketio.on("start_recording")
def handle_start_recording(data=None):
    """Start transcription."""
    global is_recording, recording_thread, ring
    global emitted_text_hashes, recent_emitted_blocks, processed_final_transcripts, diarizer

    audio_mode = "microphone"
    if data and isinstance(data, dict):
        audio_mode = data.get("audio_mode", "microphone")

    if audio_mode not in ["microphone", "system"]:
        emit("error", {"message": f"Invalid audio mode: {audio_mode}"})
        return

    print(f"Start recording with audio mode: {audio_mode}")
    if is_recording:
        emit("error", {"message": "Recording already in progress"})
        return

    is_recording = True
    emit("recording_status", {"is_recording": True}, broadcast=True)

    # Reset buffers + state
    ring = RingBuffer()
    emitted_text_hashes = set()
    recent_emitted_blocks = []
    processed_final_transcripts.clear()
    reset_api_speakers()

    # Reset diarizer (start new session speakers)
    diarizer.reset()

    # Start transcription thread
    recording_thread = Thread(target=run_transcription, args=(audio_mode,), daemon=True)
    recording_thread.start()


@socketio.on("stop_recording")
def handle_stop_recording():
    """Stop transcription."""
    global is_recording
    print("Stop recording requested")
    is_recording = False
    emit("recording_status", {"is_recording": False}, broadcast=True)


if __name__ == "__main__":
    print("Starting Live Caption Server...")
    print("Open http://localhost:5000 in your browser")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
