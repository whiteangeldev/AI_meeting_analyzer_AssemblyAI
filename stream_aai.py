import os
import json
import asyncio
import numpy as np
import websockets
import urllib.parse
from dotenv import load_dotenv
from config import SAMPLE_RATE

load_dotenv()
AAI_KEY = os.getenv("ASSEMBLYAI_API_KEY")

PARAMS = {
    "sample_rate": str(SAMPLE_RATE),
    "format_turns": "true",
    "speech_model": "universal-streaming-english",
}
AAI_WS_URL = f"wss://streaming.assemblyai.com/v3/ws?{urllib.parse.urlencode(PARAMS)}"


def pcm16le_bytes(f32: np.ndarray) -> bytes:
    """Convert float32 audio to PCM16LE bytes"""
    s = np.clip(f32, -1.0, 1.0)
    i16 = (s * 32767.0).astype(np.int16)
    return i16.tobytes()


async def aai_stream(frame_gen, on_result):
    """Stream audio to AssemblyAI and handle results"""
    headers = [("Authorization", AAI_KEY)]

    try:
        async with websockets.connect(
            AAI_WS_URL,
            extra_headers=headers,
            ping_interval=5,
            ping_timeout=20,
        ) as ws:
            print("Connected to AssemblyAI")

            async def sender():
                try:
                    async for frame in frame_gen:
                        audio_bytes = pcm16le_bytes(frame)
                        await ws.send(audio_bytes)
                except Exception as e:
                    print(f"Sender error: {e}")

            async def receiver():
                try:
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            await on_result(data)
                        except json.JSONDecodeError:
                            pass
                except websockets.exceptions.ConnectionClosed:
                    raise
                except Exception as e:
                    print(f"Receiver error: {e}")
                    raise

            await asyncio.gather(sender(), receiver())
    except Exception as e:
        print(f"Connection error: {e}")
        raise
