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
is_recording = False
recording_thread = None
emitted_text_hashes = set()  # Track emitted text to prevent duplicates
recent_emitted_blocks = []  # Track recent finalized blocks for containment checking
recent_blocks_max = 10  # Keep last 10 blocks


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


def emit_transcription(text, is_partial=False):
    """Emit transcription, checking for duplicates"""
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

        # Emit with simple "Speaker" label
        socketio.emit(
            "transcription",
            {
                "speaker": "Speaker",
                "text": text,
                "is_partial": is_partial,
                "timestamp": time.time(),
            },
            namespace="/",
        )

    except Exception as e:
        print(f"Error emitting transcription: {e}")


async def transcription_worker(audio_mode="microphone"):
    """Main transcription worker"""

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
        global is_recording
        if not is_recording:
            return

        # Skip non-transcript messages
        if "transcript" not in data and "words" not in data:
            return

        end_of_turn = data.get("end_of_turn", False)
        transcript = data.get("transcript", "").strip()

        if not transcript:
            return

        # Handle partial transcripts
        if not end_of_turn:
            emit_transcription(transcript, is_partial=True)
            return

        # Handle finalized transcripts
        if end_of_turn:
            emit_transcription(transcript, is_partial=False)

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
    emitted_text_hashes = set()
    recent_emitted_blocks = []

    # Start transcription thread
    recording_thread = Thread(target=run_transcription, args=(audio_mode,), daemon=True)
    recording_thread.start()


@socketio.on("stop_recording")
def handle_stop_recording():
    """Stop transcription"""
    global is_recording
    print("Stop recording requested")
    is_recording = False
    emit("recording_status", {"is_recording": False}, broadcast=True)


if __name__ == "__main__":
    print("Starting Live Caption Server...")
    print("Open http://localhost:5000 in your browser")
    socketio.run(
        app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True
    )
