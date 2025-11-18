import asyncio
import numpy as np
import pyaudio
from config import SAMPLE_RATE, FRAME_SECS
from typing import Optional

CHUNK = int(SAMPLE_RATE * FRAME_SECS)


def list_audio_devices():
    """List all available audio input devices"""
    pa = pyaudio.PyAudio()
    devices = []
    try:
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'rate': int(info['defaultSampleRate'])
                    })
            except Exception:
                pass
    finally:
        pa.terminate()
    return devices


def find_system_audio_device():
    """Find a system audio device (BlackHole, Soundflower, etc.)"""
    devices = list_audio_devices()
    system_audio_names = ['blackhole', 'soundflower', 'multi-output', 'loopback', 'black hole']
    
    for device in devices:
        name_lower = device['name'].lower()
        if any(keyword in name_lower for keyword in system_audio_names):
            return device['index']
    return None


async def mic_frames(device_index: Optional[int] = None):
    """Async generator yielding float32 mono frames"""
    pa = pyaudio.PyAudio()
    loop = asyncio.get_event_loop()
    q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=128)
    
    channels = [1]
    needs_stereo_to_mono = [False]

    def callback(in_data, frame_count, time_info, status):
        audio_int16 = np.frombuffer(in_data, dtype=np.int16)
        audio = audio_int16.astype(np.float32) / 32768.0
        
        # Convert stereo to mono if needed
        if needs_stereo_to_mono[0] and len(audio) >= 2:
            num_samples = len(audio) // channels[0]
            audio_reshaped = audio[:num_samples * channels[0]].reshape(num_samples, channels[0])
            audio = np.mean(audio_reshaped, axis=1)
        
        def put_audio():
            try:
                q.put_nowait(audio)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(audio)
                except asyncio.QueueEmpty:
                    pass
        
        try:
            loop.call_soon_threadsafe(put_audio)
        except Exception:
            pass
        return (None, pyaudio.paContinue)

    device_name = "default"
    try:
        if device_index is not None:
            try:
                device_info = pa.get_device_info_by_index(device_index)
                device_name = device_info['name']
                max_channels = device_info['maxInputChannels']
                
                if max_channels >= 2:
                    channels[0] = 2
                    needs_stereo_to_mono[0] = True
                    print(f"Using device: {device_name} ({channels[0]} channels -> mono)")
            except OSError:
                raise RuntimeError(f"Audio device index {device_index} does not exist")
        
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels[0],
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
            stream_callback=callback,
        )
        stream.start_stream()
        
        if not stream.is_active():
            raise RuntimeError(f"Audio stream failed to start for device: {device_name}")
        
        try:
            while True:
                frame = await q.get()
                yield frame
        finally:
            stream.stop_stream()
            stream.close()
    except OSError as e:
        raise RuntimeError(f"Failed to open audio device '{device_name}': {e}")
    finally:
        pa.terminate()


async def system_audio_frames():
    """Async generator for system audio (uses BlackHole, etc.)"""
    device_index = find_system_audio_device()
    if device_index is None:
        devices = list_audio_devices()
        device_list = "\n".join([f"  {d['index']}: {d['name']}" for d in devices])
        raise RuntimeError(
            f"System audio device not found.\n"
            f"Available devices:\n{device_list}\n\n"
            f"Please install BlackHole or configure a virtual audio device."
        )
    
    device_name = next((d['name'] for d in list_audio_devices() if d['index'] == device_index), "Unknown")
    print(f"Using system audio device: {device_name}")
    
    try:
        async for frame in mic_frames(device_index=device_index):
            yield frame
    except OSError as e:
        raise RuntimeError(f"Failed to open system audio device: {e}")
