import os, json, base64, asyncio, numpy as np, websockets, urllib.parse
from dotenv import load_dotenv
from config import SAMPLE_RATE

load_dotenv()
AAI_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Universal Streaming API: configuration goes in URL query params
PARAMS = {
    "sample_rate": str(SAMPLE_RATE),
    "format_turns": "true",  # URL params need to be strings
    "speech_model": "universal-streaming-english",
}
AAI_WS_URL = f"wss://streaming.assemblyai.com/v3/ws?{urllib.parse.urlencode(PARAMS)}"


def pcm16le_bytes(f32: np.ndarray) -> bytes:
    s = np.clip(f32, -1.0, 1.0)
    i16 = (s * 32767.0).astype(np.int16)
    return i16.tobytes()


async def aai_stream(frame_gen, on_result):
    # AssemblyAI uses raw API key, NOT "Bearer" prefix
    headers = [("Authorization", AAI_KEY)]

    async with websockets.connect(
        AAI_WS_URL,
        extra_headers=headers,
        ping_interval=5,
        ping_timeout=20,
    ) as ws:

        async def sender():
            try:
                async for frame in frame_gen:
                    # Universal Streaming API: send raw binary audio bytes, NOT JSON
                    audio_bytes = pcm16le_bytes(frame)
                    await ws.send(audio_bytes)
            except Exception as e:
                print(f"Sender error: {e}")
            finally:
                # 🔹 END session - close connection
                await ws.close()

        async def receiver():
            try:
                async for msg in ws:
                    # Responses are JSON, but audio is sent as binary
                    data = json.loads(msg)
                    await on_result(data)
            except websockets.exceptions.ConnectionClosed as e:
                print(f"Connection closed: code={e.code}, reason={e.reason}")
                raise
            except json.JSONDecodeError as e:
                print(f"Non-JSON message received: {msg[:100]}")

        await asyncio.gather(sender(), receiver())
