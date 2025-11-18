# Why Transcript Accuracy is Lower with Multiple Speakers

## Root Causes

### 1. **Double Speaker Diarization**
- **Problem**: The system performs speaker diarization TWICE:
  1. AssemblyAI's Universal Streaming API already does speaker diarization (provides speaker labels in words)
  2. The code then does ANOTHER speaker diarization using resemblyzer on audio segments
- **Impact**: These two systems can conflict, causing incorrect speaker assignments
- **Location**: `web_app.py` line 293: `detected_speaker = registry.assign(audio_seg, sr=SAMPLE_RATE)`

### 2. **Audio Segment Quality Issues**
- **Problem**: When extracting audio segments for speaker detection:
  - `ring.slice(start_time, end_time)` may not perfectly align with speech boundaries
  - Short segments (<0.5s) are unreliable for speaker detection
  - Background noise from the other speaker can contaminate the audio segment
- **Impact**: Poor audio quality leads to incorrect speaker assignments
- **Location**: `web_app.py` line 275: `audio_seg = ring.slice(start_time, end_time)`

### 3. **Processing Logic Interference**
- **Problem**: The duplicate detection and speaker switching logic can:
  - Drop legitimate content when it thinks it's a duplicate
  - Fragment sentences when speaker detection is wrong
  - Keep wrong speaker assigned due to conservative switching
- **Impact**: Content loss and fragmentation
- **Location**: `web_app.py` lines 387-500 (duplicate detection logic)

### 4. **AssemblyAI API Behavior**
- **Problem**: Multi-speaker scenarios are inherently more challenging:
  - Overlapping speech reduces accuracy
  - Frequent speaker switches degrade performance
  - Speaker diarization adds complexity
- **Impact**: Lower baseline accuracy from the API itself

### 5. **Short Audio Segments**
- **Problem**: Many turns have short audio segments (<0.5s)
- **Impact**: These fall back to "current_active_speaker", which can be wrong
- **Location**: `web_app.py` lines 278-290

## Comparison: Single vs Multiple Speakers

### Single Speaker (Higher Accuracy)
- ✅ No speaker diarization needed
- ✅ No speaker switching logic interference
- ✅ No duplicate detection conflicts
- ✅ Cleaner audio (no background from other speaker)
- ✅ AssemblyAI can focus solely on transcription

### Multiple Speakers (Lower Accuracy)
- ❌ Double speaker diarization (API + resemblyzer)
- ❌ Speaker switching logic can fragment content
- ❌ Duplicate detection may drop legitimate content
- ❌ Audio contamination from other speaker
- ❌ AssemblyAI must do both transcription AND diarization

## Potential Solutions

### Option 1: Use AssemblyAI's Speaker Labels Directly
Instead of doing our own speaker diarization, trust AssemblyAI's speaker labels:
```python
# Use speaker from API words instead of resemblyzer
detected_speaker = words_raw[0].get("speaker", "A")  # Use API's speaker label
```

### Option 2: Improve Audio Segment Extraction
- Use longer audio segments for speaker detection
- Add padding around speech boundaries
- Filter out background noise

### Option 3: Simplify Processing Logic
- Reduce duplicate detection aggressiveness
- Simplify speaker switching logic
- Trust AssemblyAI's speaker assignments more

### Option 4: Use Separate Microphones
- Each speaker has their own microphone
- Eliminates audio contamination
- Much better speaker separation

## Recommended Approach

**Use AssemblyAI's built-in speaker diarization** instead of doing our own. The API already provides speaker labels, and doing our own diarization adds errors without significant benefit.

