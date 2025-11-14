from audio_capture import mic_frames
import asyncio

async def test():
    async for f in mic_frames():
        print(len(f))
        break

asyncio.run(test())

