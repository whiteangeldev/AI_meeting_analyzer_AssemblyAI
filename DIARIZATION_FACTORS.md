# Factors Affecting Speaker Diarization Accuracy

## 1. **Audio Window Quality** (Most Critical)

### Window Length

- **Too short (< 0.8s)**: Not enough phonemes for reliable embedding
- **Optimal (1.2-2.5s)**: Enough speech for stable embeddings
- **Too long (> 3s)**: May include multiple speakers or silence

**Current setting**: `min_window_sec=1.2s`, `fallback_window_sec=1.8s` ✅

### Window Alignment

- **Misaligned**: Cutting through words/sentences → poor embeddings
- **Well-aligned**: Using AssemblyAI word timestamps → better embeddings

**Current**: Uses `_word_time_bounds()` to align with actual speech ✅

### Audio Contamination

- **Background noise**: Reduces embedding quality
- **Overlapping speech**: Mixes speaker features → wrong assignment
- **Silence**: Empty segments produce unreliable embeddings

**Mitigation**: Padding (0.15s) and min window size help, but can't eliminate all noise

---

## 2. **Speaker Embedding Similarity** (Voice Distinctness)

### Voice Characteristics

- **Pitch difference**: Male vs female voices easier to separate
- **Timbre**: Unique vocal characteristics per person
- **Speaking style**: Rate, accent, clarity

**Your case**: Male (odd chunks) vs Female (even chunks) should be distinct ✅

### Similarity Scores

- **> 0.85**: Very confident match
- **0.72-0.85**: Good match (current threshold)
- **0.55-0.72**: Uncertain (current min acceptance)
- **< 0.55**: Too low, rejected

**Current settings**:

- `threshold=0.72` (register new speaker if below this)
- `MIN_DIARIZATION_SIM=0.55` (accept result if above this)

---

## 3. **Centroid Update Strategy** (Learning & Adaptation)

### History Buffer Size

- **Small (5-10)**: Adapts quickly but unstable
- **Medium (20)**: Good balance ✅ (current)
- **Large (50+)**: Very stable but slow to adapt

**Current**: `history_size=20` ✅

### Update Guard

- **High (0.15+)**: Conservative, only updates on high confidence
- **Medium (0.12)**: Balanced ✅ (current)
- **Low (< 0.10)**: Aggressive, may contaminate centroids

**Current**: `update_guard=0.12` ✅

### Update Frequency

- **Every match**: Can contaminate with noisy segments
- **Only high confidence**: More stable but slower adaptation

**Current**: Updates when `similarity >= (threshold - update_guard)` ✅

---

## 4. **Threshold Settings** (Decision Boundaries)

### Registration Threshold (`threshold`)

- **High (0.80+)**: Harder to register new speakers
- **Medium (0.72)**: Balanced ✅ (current)
- **Low (< 0.65)**: Creates too many false speakers

**Current**: `threshold=0.72` ✅

### Minimum Acceptance (`MIN_DIARIZATION_SIM`)

- **High (0.70+)**: Very conservative, may reject valid speakers
- **Medium (0.55)**: Balanced ✅ (current)
- **Low (< 0.50)**: Accepts unreliable assignments

**Current**: `MIN_DIARIZATION_SIM=0.55` ✅

---

## 5. **Timing & Synchronization**

### Ring Buffer Alignment

- **Misaligned**: Audio window doesn't match transcript timing
- **Well-aligned**: Uses word timestamps from AssemblyAI

**Current**: Uses `slice_audio_window()` with word timestamps ✅

### Buffer Size

- **Too small**: Audio may be trimmed before diarization
- **Adequate (30s)**: Covers typical turn lengths ✅

**Current**: `MAX_BUFFER_SECS=30` ✅

---

## 6. **Audio Preprocessing**

### Sample Rate

- **16kHz**: Standard for speech recognition ✅
- **Higher**: More detail but slower
- **Lower**: Faster but less accurate

**Current**: `SAMPLE_RATE=16000` ✅

### Normalization

- **Proper**: Embeddings normalized to unit vectors
- **Missing**: Similarity scores unreliable

**Current**: All embeddings normalized ✅

---

## 7. **Speaker Order & First Impressions**

### First Speaker Dominance

- **Problem**: First speaker gets more samples → stronger centroid
- **Solution**: Balanced history buffers help, but first speaker still has advantage

**Impact**: User1 (first registered) may dominate if User2 speaks less

### Speaker Balance

- **Unbalanced**: One speaker talks 80% → their centroid dominates
- **Balanced**: Both speakers talk equally → better separation

**Your case**: Check if one speaker talks significantly more

---

## 8. **Real-time vs Batch Processing**

### Real-time Constraints

- **Limited context**: Only sees past audio, not future
- **No lookahead**: Can't correct mistakes retroactively
- **Streaming**: Must make decisions immediately

**Impact**: Less accurate than batch processing with full audio

---

## 9. **API Speaker Labels (AssemblyAI)**

### When Available

- **Reliable**: AssemblyAI's diarization is often better
- **Used first**: We trust API labels when present ✅

**Current**: `resolve_api_speaker()` checked before Resemblyzer ✅

### When Missing

- **Fallback**: Use Resemblyzer diarization
- **Risk**: May be less accurate than API

---

## 10. **Duplicate Detection Interference**

### Over-aggressive Filtering

- **Problem**: Legitimate transcripts marked as duplicates
- **Impact**: Can't verify speaker assignments

**Current**: Fixed to 95% overlap threshold ✅

---

## Recommendations for Your Case

Based on your logs showing duplicate diarization calls and some misassignments:

1. **Fix duplicate processing**: The hash-based deduplication isn't working (still seeing duplicate logs)
2. **Check window alignment**: Verify word timestamps are accurate
3. **Monitor similarity scores**: Log shows some low scores (0.572, 0.664) - may need threshold tuning
4. **Balance speaker samples**: If one speaker talks more, their centroid dominates

### Quick Wins:

- Increase `MIN_DIARIZATION_SIM` to 0.60 for more conservative assignments
- Increase `history_size` to 30 for more stable centroids
- Add logging to show when API speaker labels are used vs Resemblyzer
