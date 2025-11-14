import asyncio, numpy as np, pyaudio
from config import SAMPLE_RATE, FRAME_SECS

CHUNK = int(SAMPLE_RATE * FRAME_SECS)


async def mic_frames():
    """
    Async generator yielding float32 mono frames of length FRAME_SECS.
    """
    pa = pyaudio.PyAudio()
    loop = asyncio.get_event_loop()
    # Increased queue size to prevent QueueFull exceptions
    q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=128)

    def callback(in_data, frame_count, time_info, status):
        # from int16 → float32 [-1, 1]
        audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        # Use put_nowait with a callback that handles QueueFull silently
        def put_audio():
            try:
                q.put_nowait(audio)
            except asyncio.QueueFull:
                # Drop oldest frame and add new one to prevent blocking
                try:
                    q.get_nowait()  # Remove oldest
                    q.put_nowait(audio)  # Add new
                except asyncio.QueueEmpty:
                    pass  # Queue was already empty, just skip
        try:
            loop.call_soon_threadsafe(put_audio)
        except Exception:
            pass  # Silently ignore any errors
        return (None, pyaudio.paContinue)

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=callback,
    )
    stream.start_stream()

    try:
        while True:
            frame = await q.get()
            yield frame
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
