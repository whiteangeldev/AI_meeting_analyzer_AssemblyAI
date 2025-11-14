import asyncio, numpy as np
from rich import print
from rich.console import Console
from dotenv import load_dotenv

from audio_capture import mic_frames
from ringbuffer import RingBuffer
from sentence_assembler import SentenceAssembler
from diarizer import SpeakerRegistry
from stream_aai import aai_stream
from config import SAMPLE_RATE

load_dotenv()

ring = RingBuffer()
assembler = SentenceAssembler()
registry = SpeakerRegistry()
console = Console()

# Track current partial transcript for live updates
current_partial = {"speaker": None, "text": "", "last_update": 0}


async def main():
    async def audio_gen():
        async for frame in mic_frames():
            ring.append(frame)
            yield frame

    async def on_result(data: dict):
        # Universal Streaming API returns Turn objects with transcript and words
        # Check if this is a Turn object (Universal Streaming API format)
        if "transcript" not in data and "words" not in data:
            return

        # Check if this is end of turn (finalized transcript)
        end_of_turn = data.get("end_of_turn", False)
        transcript = data.get("transcript", "")
        words_raw = data.get("words", [])

        # Handle partial transcript updates (live transcription)
        if transcript and not end_of_turn:
            # Update current partial transcript in place
            import time
            current_partial["text"] = transcript
            current_partial["last_update"] = time.time()
            # Clear line and show updated partial transcript
            console.print(f"\r[dim]... {transcript}[/dim]", end="")
            return

        # Handle finalized transcripts (end_of_turn = True)
        if end_of_turn and words_raw:
            # Clear the partial transcript line
            console.print("\r" + " " * 80 + "\r", end="")
            
            # Convert to our format: seconds + temp speaker
            words = []
            for w in words_raw:
                # Universal Streaming API word format
                text = w.get("text", "")
                # Times are in seconds, not milliseconds
                start = w.get("start", 0)
                end = w.get("end", 0)
                # Convert to seconds if in milliseconds (check if > 1000)
                if start > 1000:
                    start = start / 1000.0
                if end > 1000:
                    end = end / 1000.0
                spk = w.get("speaker", "A")
                words.append(
                    {
                        "text": text,
                        "start": start,
                        "end": end,
                        "speaker": spk,
                    }
                )

            assembler.update_voice_time()
            finals = assembler.add_words(words)
            for sent in finals:
                audio_seg = ring.slice(sent["start"], sent["end"])
                stable = registry.assign(audio_seg, sr=SAMPLE_RATE)
                print(f"[bold]{stable}[/]: {sent['text']}")

            # silence-based flush
            maybe = assembler.maybe_flush_on_silence()
            if maybe:
                audio_seg = ring.slice(maybe["start"], maybe["end"])
                stable = registry.assign(audio_seg, sr=SAMPLE_RATE)
                print(f"[bold]{stable}[/]: {maybe['text']}")

    await aai_stream(audio_gen(), on_result)


if __name__ == "__main__":
    asyncio.run(main())
