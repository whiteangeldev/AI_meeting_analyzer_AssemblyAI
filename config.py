SAMPLE_RATE = 16000

# how we send audio to AssemblyAI
FRAME_SECS = 0.1  # 100ms frames for lower latency (was 0.32s)
# diarization / segmentation
MAX_BUFFER_SECS = 30  # keep last 30s audio
SILENCE_END_MS = 600  # finalize sentence after 600ms silence
PUNCT_TRIGGER = True

# diarization thresholds
SIM_THRESH = 0.85  # cosine similarity threshold for same speaker (increased for >85% accuracy)
EMA = 0.85  # smoothing for centroids (slightly lower for faster adaptation)
MAX_SPK = 8
MIN_AUDIO_SECS = 0.5  # minimum audio length for reliable diarization (500ms)
