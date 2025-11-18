import asyncio
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from threading import Thread
import time
import hashlib
import re
from dotenv import load_dotenv

from audio_capture import mic_frames, system_audio_frames, list_audio_devices
from ringbuffer import RingBuffer
from stream_aai import aai_stream
from diarizer import SpeakerDiarizer
from config import SAMPLE_RATE, MIN_AUDIO_SECS

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)

# Global state
ring = RingBuffer()
diarizer = SpeakerDiarizer()
is_recording = False
recording_thread = None
emitted_text_hashes = set()  # Track emitted text to prevent duplicates
recent_emitted_blocks = []  # Track recent finalized blocks for containment checking
recent_blocks_max = 10  # Keep last 10 blocks

# Block management state
current_speaker = None  # Current active speaker (User1, User2, etc.)
current_block_text = ""  # Text for current block
current_block_words = []  # Accumulated words for current block
current_block_word_ids = set()  # Track unique word identifiers

# Speaker switch management
last_speaker_switch_time = 0
speaker_switch_cooldown = 1.0  # 1 second cooldown to prevent false switches


def normalize_text(text):
    """Normalize text for duplicate checking"""
    normalized = text.lower()
    normalized = re.sub(r"\bten\b", "10", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def check_duplicate(text):
    """Check if text is a duplicate (exact or containment)"""
    if not text:
        return False

    # Check exact duplicate
    normalized = normalize_text(text)
    text_hash = hashlib.md5(normalized.encode()).hexdigest()
    if text_hash in emitted_text_hashes:
        return True

    # Check containment duplicate (text is substring of recent blocks)
    for block_text in recent_emitted_blocks:
        block_normalized = normalize_text(block_text)
        if not block_normalized:
            continue

        # Check if text is substring of block or vice versa
        if len(normalized) >= len(block_normalized) * 0.7:
            if normalized in block_normalized or block_normalized in normalized:
                return True

        # Check word overlap (70% threshold)
        text_words = set(normalized.split())
        block_words = set(block_normalized.split())
        if text_words and block_words:
            overlap_ratio = len(text_words & block_words) / max(
                len(text_words), len(block_words)
            )
            if overlap_ratio >= 0.7:
                return True

    return False


def format_transcript(text):
    """Format transcript text"""
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

    # Fix spacing
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", text)

    return text.strip()


def reset_block_words():
    """Reset accumulated words for the current block"""
    global current_block_words, current_block_word_ids
    current_block_words = []
    current_block_word_ids = set()


def update_block_text_from_words(words_raw, fallback_text):
    """Append new words to the current block text based on unique word ids"""
    global current_block_text, current_block_words, current_block_word_ids
    added = False

    if words_raw:
        for w in words_raw:
            text = w.get("text", "").strip()
            if not text:
                continue
            start = w.get("start", 0)
            end = w.get("end", 0)
            key = f"{start}-{end}-{text}"
            if key not in current_block_word_ids:
                current_block_word_ids.add(key)
                current_block_words.append(text)
                added = True

    if added and current_block_words:
        current_block_text = " ".join(current_block_words).strip()
    elif fallback_text and not current_block_words:
        # Use fallback only if we don't have any words yet
        current_block_text = fallback_text


def get_consensus_speaker(vote_window, threshold=0.67):
    """Get consensus speaker from voting window"""
    if not vote_window:
        return None

    from collections import Counter

    votes = Counter(vote_window)

    if votes:
        most_common, count = votes.most_common(1)[0]
        vote_fraction = count / len(vote_window)
        if vote_fraction >= threshold:
            return most_common

    return None


def emit_transcription(speaker, text, is_partial=False):
    """Emit transcription with speaker label"""
    try:
        if not text or not text.strip():
            return

        # Check for duplicates
        if check_duplicate(text):
            print(f"⚠ Duplicate detected, skipping: {text[:50]}...")
            return

        # Mark as emitted
        normalized = normalize_text(text)
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        emitted_text_hashes.add(text_hash)

        # For finalized blocks, add to recent blocks
        if not is_partial:
            recent_emitted_blocks.append(text)
            if len(recent_emitted_blocks) > recent_blocks_max:
                recent_emitted_blocks.pop(0)

        # Format text if final
        if not is_partial:
            text = format_transcript(text)

        # Emit with speaker label
        socketio.emit(
            "transcription",
            {
                "speaker": speaker,
                "text": text,
                "is_partial": is_partial,
                "timestamp": time.time(),
            },
            namespace="/",
        )

    except Exception as e:
        print(f"Error emitting transcription: {e}")


async def transcription_worker(audio_mode="microphone"):
    """Main transcription worker with speaker diarization"""

    async def audio_gen():
        global is_recording
        try:
            frame_source = (
                system_audio_frames() if audio_mode == "system" else mic_frames()
            )

            async for frame in frame_source:
                if not is_recording:
                    break
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
        global is_recording, current_speaker, current_block_text, last_speaker_switch_time

        if not is_recording:
            return

        # Skip non-transcript messages
        if "transcript" not in data and "words" not in data:
            return

        end_of_turn = data.get("end_of_turn", False)
        transcript = data.get("transcript", "").strip()
        words_raw = data.get("words", [])

        if not transcript:
            return

        # Only detect speaker on FINALIZED transcripts for stability
        # Don't run diarization on partials to avoid mid-sentence false switches
        detected_speaker = None
        if end_of_turn and words_raw:
            start_time = words_raw[0].get("start", 0) if words_raw else 0
            end_time = words_raw[-1].get("end", 0) if words_raw else 0

            # Convert to seconds if in milliseconds
            if start_time > 1000:
                start_time = start_time / 1000.0
            if end_time > 1000:
                end_time = end_time / 1000.0

            # Get audio segment for speaker detection
            audio_seg = ring.slice(start_time, end_time)
            if audio_seg is not None and len(audio_seg) > 0:
                min_samples = int(SAMPLE_RATE * MIN_AUDIO_SECS)
                if len(audio_seg) >= min_samples:
                    detected_speaker = diarizer.assign(audio_seg, sr=SAMPLE_RATE)

        # DISABLE DIARIZATION FOR NOW - just use generic "Speaker" label
        # Focus on getting clean, correct transcription without duplication
        speaker_to_use = "Speaker"

        # Accumulate words for this turn
        update_block_text_from_words(words_raw, transcript)

        # Emit partial (grey) during transcription for real-time feedback ONLY
        if not end_of_turn and current_block_text:
            emit_transcription(speaker_to_use, current_block_text, is_partial=True)

        # On end_of_turn: emit final (green) and reset for next turn
        # Each turn becomes one clean finalized block
        elif end_of_turn and current_block_text:
            emit_transcription(speaker_to_use, current_block_text, is_partial=False)
            # Reset for next turn
            reset_block_words()

    await aai_stream(audio_gen(), on_result)


def run_transcription(audio_mode="microphone"):
    """Run transcription in a separate thread"""
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


@app.route("/")
def index():
    """Serve the main page"""
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    print("Client connected")
    try:
        devices = list_audio_devices()
        emit("connected", {"status": "Connected", "audio_devices": devices})
    except Exception as e:
        print(f"Error listing audio devices: {e}")
        emit("connected", {"status": "Connected"})


@socketio.on("list_audio_devices")
def handle_list_audio_devices():
    """List all available audio input devices"""
    try:
        devices = list_audio_devices()
        emit("audio_devices", {"devices": devices})
    except Exception as e:
        emit("error", {"message": f"Failed to list audio devices: {str(e)}"})


@socketio.on("start_recording")
def handle_start_recording(data=None):
    """Start transcription"""
    global is_recording, recording_thread, ring, emitted_text_hashes, recent_emitted_blocks
    global current_speaker, current_block_text, diarizer
    global last_speaker_switch_time, current_block_words, current_block_word_ids

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

    # Reset state
    ring = RingBuffer()
    diarizer.reset()
    emitted_text_hashes = set()
    recent_emitted_blocks = []
    current_speaker = None
    current_block_text = ""
    last_speaker_switch_time = 0
    reset_block_words()

    # Start transcription thread
    recording_thread = Thread(target=run_transcription, args=(audio_mode,), daemon=True)
    recording_thread.start()


@socketio.on("stop_recording")
def handle_stop_recording():
    """Stop transcription"""
    global is_recording, current_speaker, current_block_text

    print("Stop recording requested")

    # Finalize current block if exists (even if speaker is generic)
    if current_block_text:
        speaker_label = current_speaker if current_speaker else "Speaker"
        print(
            f"Finalizing last block for {speaker_label}: {current_block_text[:100]}..."
        )
        emit_transcription(speaker_label, current_block_text, is_partial=False)
        current_block_text = ""
        reset_block_words()

    is_recording = False
    emit("recording_status", {"is_recording": False}, broadcast=True)


if __name__ == "__main__":
    print("Starting Live Caption Server...")
    print("Open http://localhost:5000 in your browser")
    socketio.run(
        app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True
    )
