import asyncio
import argparse
from dotenv import load_dotenv
from audio_capture import mic_frames, system_audio_frames
from stream_aai import aai_stream

load_dotenv()


async def main(audio_mode="microphone"):
    """CLI transcription app"""

    async def audio_gen():
        frame_source = system_audio_frames() if audio_mode == "system" else mic_frames()
        async for frame in frame_source:
            yield frame

    async def on_result(data: dict):
        if "transcript" not in data:
            return

        end_of_turn = data.get("end_of_turn", False)
        transcript = data.get("transcript", "").strip()

        if not transcript:
            return

        # Show partial transcripts
        if not end_of_turn:
            print(f"\r... {transcript}", end="", flush=True)
            return

        # Show finalized transcripts
        print(f"\r{transcript}", flush=True)

    await aai_stream(audio_gen(), on_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live transcription")
    parser.add_argument(
        "--audio-mode",
        type=str,
        choices=["microphone", "system"],
        default="microphone",
        help="Audio input mode",
    )
    args = parser.parse_args()
    asyncio.run(main(audio_mode=args.audio_mode))
