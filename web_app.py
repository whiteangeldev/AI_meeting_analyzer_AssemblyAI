import asyncio
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from threading import Thread
import time
import hashlib
from dotenv import load_dotenv

from audio_capture import mic_frames
from ringbuffer import RingBuffer
from sentence_assembler import SentenceAssembler
from diarizer import SpeakerRegistry
from stream_aai import aai_stream
from config import SAMPLE_RATE

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
# Configure SocketIO to handle HTTP properly and prevent HTTPS upgrade attempts
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=True,  # Enable for debugging
    engineio_logger=True,  # Enable for debugging
)

# Global state
ring = RingBuffer()
assembler = SentenceAssembler()
registry = SpeakerRegistry()
is_recording = False
recording_thread = None
last_processed_turn = None  # Track last processed turn to avoid duplicates
recent_transcriptions = (
    set()
)  # Track recently emitted transcriptions to avoid duplicates (max 200)
recent_time_windows = (
    {}
)  # Track time windows where we've emitted (start_time -> end_time)


def is_sentence_complete(text, words_raw=None):
    """Determine if a sentence is complete based on context analysis"""
    if not text:
        return False

    import re

    text = text.strip()

    # If it already ends with punctuation, it's complete
    if text and text[-1] in ".!?":
        return True

    # Check if last word suggests completion
    if words_raw and len(words_raw) > 0:
        last_word = words_raw[-1].get("text", "").lower().strip()
        # Remove any trailing punctuation from the word
        last_word = re.sub(r"[.,!?;:]+$", "", last_word)

        # Words that typically end sentences (common sentence-ending words)
        sentence_ending_words = {
            "end",
            "ends",
            "ended",
            "ending",
            "finish",
            "finishes",
            "finished",
            "finishing",
            "complete",
            "completes",
            "completed",
            "completing",
            "done",
            "over",
            "through",
            "concluded",
            "concludes",
        }

        # Check if last word is a sentence-ending word
        if last_word in sentence_ending_words:
            return True

    # Check if sentence has minimum structure (at least 3 words suggests it might be complete)
    words = text.split()
    if len(words) < 3:
        return False

    # Check for natural sentence boundaries
    # Look for patterns that suggest completion:
    # - Ends with common sentence-ending patterns
    text_lower = text.lower()

    # Common sentence-ending phrases
    ending_phrases = [
        " and so on",
        " etc",
        " etc.",
        " and more",
        " and others",
        " in the end",
        " at last",
        " finally",
        " in conclusion",
    ]

    for phrase in ending_phrases:
        if text_lower.endswith(phrase):
            return True

    # Check if sentence has subject-verb structure (basic check)
    # Look for common verb patterns at the end
    last_few_words = " ".join(words[-3:]).lower()

    # Patterns that suggest a complete thought
    complete_patterns = [
        r"\b(is|are|was|were|has|have|had|will|would|can|could|should|may|might)\s+\w+$",
        r"\b\w+ed\s+\w+$",  # past tense verb + object
        r"\b\w+ing\s+\w+$",  # gerund + object
    ]

    for pattern in complete_patterns:
        if re.search(pattern, last_few_words):
            return True

    # If sentence is long enough (5+ words) and doesn't end with connecting words, likely complete
    connecting_words = {
        "and",
        "or",
        "but",
        "so",
        "because",
        "if",
        "when",
        "while",
        "as",
        "the",
        "a",
        "an",
    }
    return len(words) >= 5 and words[-1].lower() not in connecting_words


def format_transcript(text, words_raw=None):
    """Format transcript with proper capitalization and punctuation"""
    if not text:
        return text

    import re

    # Remove extra whitespace but preserve single spaces
    text = " ".join(text.split())

    # Ensure first letter is capitalized
    if text and text[0].islower():
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    # Only add period if sentence is actually complete and doesn't have ending punctuation
    if text and text[-1] not in ".!?" and is_sentence_complete(text, words_raw):
        text = f"{text}."
        # If not complete, leave it without punctuation (will be updated when complete)

    # Fix spacing issues while preserving commas and other punctuation
    # Add space after sentence-ending punctuation if missing
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
    # Fix multiple spaces (but preserve single spaces)
    text = re.sub(r"\s+", " ", text)
    # Fix spacing before punctuation (remove spaces before commas, periods, etc.)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    # Add space after commas and other punctuation if missing (but not if already there)
    text = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", text)
    # Ensure space after sentence-ending punctuation
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)

    return text.strip()


def emit_transcription(speaker, text, is_partial=False):
    """Emit transcription to all connected clients"""
    try:
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
        print(f"Emitted: {speaker} - {text[:50]}...")  # Debug
    except Exception as e:
        print(f"Error emitting transcription: {e}")


async def transcription_worker():
    """Main transcription worker that processes audio and sends results via WebSocket"""

    async def audio_gen():
        global is_recording
        async for frame in mic_frames():
            # Check if recording was stopped
            if not is_recording:
                print("Recording stopped, ending audio stream...")
                break
            ring.append(frame)
            yield frame

    async def on_result(data: dict):
        # Check if recording was stopped
        global is_recording
        if not is_recording:
            return

        # Universal Streaming API returns Turn objects with transcript and words
        # Debug: print first few messages to see structure
        if "transcript" not in data and "words" not in data:
            # Only log non-transcript messages occasionally to avoid spam
            if "message_type" in data or "error" in data:
                print(f"Non-transcript message: {data}")
            return

        # Check if this is end of turn (finalized transcript)
        end_of_turn = data.get("end_of_turn", False)
        transcript = data.get("transcript", "")
        words_raw = data.get("words", [])
        print(
            f"Received transcript: end_of_turn={end_of_turn}, transcript='{transcript[:50]}...', words={len(words_raw)}"
        )

        # Handle partial transcript updates (live transcription)
        if transcript and not end_of_turn:
            emit_transcription("...", transcript, is_partial=True)
            return

        # Handle finalized transcripts (end_of_turn = True)
        if end_of_turn and words_raw:
            # Get timing info
            start_time = words_raw[0].get("start", 0) if words_raw else 0
            end_time = words_raw[-1].get("end", 0) if words_raw else 0
            # Convert to seconds if in milliseconds
            if start_time > 1000:
                start_time = start_time / 1000.0
            if end_time > 1000:
                end_time = end_time / 1000.0

            # Round times to 0.5 second windows for deduplication
            start_window = round(start_time * 2) / 2
            end_window = round(end_time * 2) / 2

            # Check if we've already emitted something in this time window
            global recent_time_windows
            time_key = (start_window, end_window)
            if time_key in recent_time_windows:
                print(
                    f"Skipping duplicate turn (time window: {start_window:.1f}-{end_window:.1f}s)"
                )
                return

            # Also check for overlapping time windows (within 0.5 seconds)
            for (prev_start, prev_end), prev_text in recent_time_windows.items():
                if (
                    abs(prev_start - start_window) < 0.5
                    and abs(prev_end - end_window) < 0.5
                ):
                    # Same time window - skip this duplicate
                    print("Skipping duplicate turn (overlapping time window)")
                    return

            # Create a robust fingerprint using normalized word content
            # Normalize all words to lowercase and create a hash
            normalized_words = " ".join(
                [
                    w.get("text", "").lower().strip()
                    for w in words_raw
                    if w.get("text", "").strip()
                ]
            )
            word_hash = hashlib.md5(normalized_words.encode()).hexdigest()[:8]
            turn_fingerprint = (word_hash, start_window, end_window)

            # Skip if this is a duplicate of the last processed turn
            global last_processed_turn
            if last_processed_turn == turn_fingerprint:
                print(f"Skipping duplicate turn (hash: {word_hash})")
                return

            last_processed_turn = turn_fingerprint

            # Extract the final transcript text
            # The API sometimes sends both raw and formatted versions concatenated
            # We want only the formatted (final) version - the complete formatted text
            final_text = None
            import re

            if transcript and transcript.strip():
                transcript_clean = transcript.strip()

                # Strategy 1: Look for transition from lowercase to uppercase (raw -> formatted)
                # The formatted version usually starts with a capital letter after lowercase text
                # Split by spaces but preserve punctuation attached to words
                words_list = transcript_clean.split()
                if len(words_list) > 3:
                    # Find where formatted version starts (capital after lowercase)
                    for i in range(1, len(words_list)):
                        prev_word = words_list[i - 1].strip()
                        curr_word = words_list[i].strip()
                        # Remove punctuation for comparison but preserve it in output
                        prev_word_clean = re.sub(r"[^\w]", "", prev_word)
                        curr_word_clean = re.sub(r"[^\w]", "", curr_word)
                        # Check if we transition from lowercase to uppercase
                        if (
                            prev_word_clean
                            and curr_word_clean
                            and prev_word_clean[0].islower()
                            and curr_word_clean[0].isupper()
                        ):
                            # Take everything from this capitalized word onwards (entire formatted part)
                            # Join with spaces to preserve commas and other punctuation
                            final_text = " ".join(words_list[i:]).strip()
                            break

                # Strategy 2: Check if transcript is duplicated (same content twice)
                # If halves are very similar, take the second half (formatted version)
                if not final_text and len(words_list) > 5:
                    mid_point = len(words_list) // 2
                    first_half = " ".join(words_list[:mid_point]).lower()
                    second_half = " ".join(words_list[mid_point:]).lower()
                    # If halves are very similar (>80% similarity), take second half
                    if first_half and second_half:
                        common_chars = sum(
                            a == b for a, b in zip(first_half, second_half)
                        )
                        similarity = common_chars / max(
                            len(first_half), len(second_half)
                        )
                        if similarity > 0.8:
                            final_text = " ".join(words_list[mid_point:]).strip()

                # Strategy 3: Check if transcript starts with lowercase (raw) - take from first capital
                if not final_text:
                    # Find the first sentence that starts with capital
                    sentences = re.split(r"([.!?]\s+)", transcript_clean)
                    formatted_start_idx = -1
                    for i, part in enumerate(sentences):
                        if part.strip() and part.strip()[0].isupper():
                            formatted_start_idx = i
                            break

                    if formatted_start_idx >= 0:
                        # Take everything from the first capitalized sentence onwards
                        final_text = "".join(sentences[formatted_start_idx:]).strip()

                # Strategy 4: Fallback - use original transcript if it looks formatted
                if not final_text:
                    # If transcript starts with capital, assume it's already formatted
                    if transcript_clean[0].isupper():
                        final_text = transcript_clean
                    else:
                        # Try to find any capitalized sentence and use from there
                        match = re.search(r"([A-Z][^.!?]*[.!?])", transcript_clean)
                        if match:
                            # Find the start of this sentence
                            start_pos = transcript_clean.rfind(match.group(1))
                            if start_pos >= 0:
                                final_text = transcript_clean[start_pos:].strip()
                            else:
                                final_text = transcript_clean
                        else:
                            final_text = transcript_clean
            else:
                # No transcript field - build from words, preserving punctuation
                # Join words with spaces, but preserve punctuation that's part of the word text
                word_texts = []
                for w in words_raw:
                    word_text = w.get("text", "").strip()
                    if word_text:
                        word_texts.append(word_text)
                final_text = " ".join(word_texts)

            # Normalize and check for duplicates before emitting
            if final_text:
                normalized_text = " ".join(final_text.lower().split())
                text_hash = hashlib.md5(normalized_text.encode()).hexdigest()[:8]

                # Skip if we've recently emitted this exact text
                global recent_transcriptions
                if text_hash in recent_transcriptions:
                    print(f"Skipping duplicate transcript: {final_text[:50]}...")
                    return

                # Mark this time window as used
                recent_time_windows[time_key] = final_text
                # Keep only last 50 time windows
                if len(recent_time_windows) > 50:
                    # Remove oldest entries
                    keys_to_remove = list(recent_time_windows.keys())[:-50]
                    for key in keys_to_remove:
                        del recent_time_windows[key]

                recent_transcriptions.add(text_hash)
                # Limit size to prevent memory growth (keep last 200)
                if len(recent_transcriptions) > 200:
                    to_remove = list(recent_transcriptions)[:100]
                    recent_transcriptions -= set(to_remove)

                # Ensure proper formatting (capitalization, punctuation)
                # Pass words_raw to help determine if sentence is complete
                final_text = format_transcript(final_text, words_raw)

                # Get audio segment for speaker diarization
                audio_seg = ring.slice(start_time, end_time)
                stable = registry.assign(audio_seg, sr=SAMPLE_RATE)
                emit_transcription(stable, final_text, is_partial=False)

    await aai_stream(audio_gen(), on_result)


def run_transcription():
    """Run transcription in a separate thread"""
    global is_recording
    print("Transcription worker starting...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        print("Starting async transcription loop...")
        loop.run_until_complete(transcription_worker())
    except Exception as e:
        import traceback

        print(f"Transcription error: {e}")
        traceback.print_exc()
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
    emit("connected", {"status": "Connected to transcription server"})


@socketio.on("start_recording")
def handle_start_recording():
    """Start transcription"""
    global is_recording, recording_thread

    print("Start recording requested")
    if is_recording:
        emit("error", {"message": "Recording already in progress"})
        return

    is_recording = True
    emit("recording_status", {"is_recording": True}, broadcast=True)
    print("Recording started, initializing transcription...")

    # Reset state
    global ring, assembler, registry, last_processed_turn, recent_transcriptions, recent_time_windows
    ring = RingBuffer()
    assembler = SentenceAssembler()
    registry = SpeakerRegistry()
    last_processed_turn = None  # Reset duplicate tracking
    recent_transcriptions = set()  # Reset transcription tracking
    recent_time_windows = {}  # Reset time window tracking

    # Start transcription in background thread
    recording_thread = Thread(target=run_transcription, daemon=True)
    recording_thread.start()
    print("Transcription thread started")


@socketio.on("stop_recording")
def handle_stop_recording():
    """Stop transcription"""
    global is_recording
    print("Stop recording requested")
    is_recording = False
    emit("recording_status", {"is_recording": False}, broadcast=True)
    print("Recording stopped - transcription will end after current processing")


if __name__ == "__main__":
    print("Starting Live Caption Server...")
    print("Open http://localhost:5000 in your browser")
    print(
        "Note: Ignore any 'Bad request version' errors - these are harmless browser connection attempts"
    )
    socketio.run(
        app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True
    )
