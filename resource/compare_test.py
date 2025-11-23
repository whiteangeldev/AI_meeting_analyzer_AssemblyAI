"""
compare_test.py
----------------
Evaluate diarization accuracy on the synthetic 8-chunk dataset plus the
concatenated `output.mp3`. Odd-numbered files belong to the first speaker
(`User1`), even-numbered files belong to the second speaker (`User2`).

Usage:
    python resource/compare_test.py
"""

# sourcery skip: use-named-expression

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import cosine

try:
    from pydub import AudioSegment  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "pydub is required for compare_test.py. Install it via 'pip install pydub'."
    ) from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR))

from config import SAMPLE_RATE
from diarizer import OnlineDiarizer

RESOURCE_DIR = Path(__file__).resolve().parent
CHUNK_FILES = [RESOURCE_DIR / f"{idx}.mp3" for idx in range(1, 9)]
OUTPUT_FILE = RESOURCE_DIR / "output.mp3"
EXPECTED_LABELS = {idx: (1 if idx % 2 else 2) for idx in range(1, 9)}


def audiosegment_to_array(segment: AudioSegment) -> np.ndarray:
    """
    Convert a pydub AudioSegment into a numpy float32 array resampled to SAMPLE_RATE.
    """
    mono = segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
    samples = np.array(mono.get_array_of_samples()).astype(np.float32)
    sample_width = mono.sample_width
    dtype_max = float(2 ** (8 * sample_width - 1))
    samples /= max(dtype_max, 1.0)
    return samples


def diarize_segment(
    segment: AudioSegment, diarizer: OnlineDiarizer, with_embedding: bool = True
) -> tuple[str | None, np.ndarray | None]:
    wav = audiosegment_to_array(segment)
    label, similarity = diarizer.diarize(wav, update_centroid=True)
    if not with_embedding:
        return label, None

    emb = diarizer.encoder.embed_utterance(wav)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return label, emb


def summarize_embedding(emb: np.ndarray | None) -> str:
    if emb is None:
        return "emb=NA"
    stats = {
        "mean": np.mean(emb),
        "std": np.std(emb),
        "min": np.min(emb),
        "max": np.max(emb),
    }
    head_vals = ", ".join(f"{v:+.2f}" for v in emb[:5])
    stats_str = ", ".join(f"{k}={v:+.3f}" for k, v in stats.items())
    return f"emb_stats[{stats_str}] head[{head_vals}]"


def similarity_report(emb: np.ndarray | None, diarizer: OnlineDiarizer) -> str:
    if emb is None or not diarizer.speakers:
        return "sims=NA"
    sims = {
        label: float(1.0 - cosine(centroid, emb))
        for label, centroid in diarizer.speakers.items()
    }
    sims_str = " ".join(f"{label}={value:+.3f}" for label, value in sims.items())
    return f"sims[{sims_str}]"


def evaluate_sequence(
    files: List[Path], diarizer: OnlineDiarizer, label_map: Dict[int, int], title: str
) -> None:
    print(f"\n=== {title} ===")
    diarizer.reset()
    results: List[bool] = []
    for idx, path in enumerate(files, start=1):
        expected = label_map[idx]
        audio = AudioSegment.from_file(path)
        predicted, emb = diarize_segment(audio, diarizer, with_embedding=True)
        is_match = predicted == f"User{expected}"
        results.append(is_match)
        status = "✅" if is_match else "❌"
        sim_str = similarity_report(emb, diarizer)
        emb_str = summarize_embedding(emb)
        print(
            f"{path.name:6s} -> expected=User{expected} predicted={predicted or 'None':>5s} "
            f"{status} | {sim_str} | {emb_str}"
        )

    correct = sum(results)
    total = len(results)
    accuracy = correct / max(total, 1)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")


def evaluate_output(
    diarizer: OnlineDiarizer,
    label_map: Dict[int, int],
    title: str = "output.mp3 segments",
) -> None:
    if not OUTPUT_FILE.exists():
        print(f"⚠ Skipping {title}: {OUTPUT_FILE} not found.")
        return

    combined = AudioSegment.from_file(OUTPUT_FILE)
    reference_segments = [AudioSegment.from_file(path) for path in CHUNK_FILES]
    ref_durations = [len(seg) for seg in reference_segments]
    total_ref = sum(ref_durations)
    if abs(len(combined) - total_ref) > 50:
        print(
            f"⚠ Warning: combined duration ({len(combined)} ms) "
            f"differs from reference ({total_ref} ms)."
        )

    diarizer.reset()
    cursor = 0
    results: List[bool] = []
    print(f"\n=== {title} ===")
    for idx, duration in enumerate(ref_durations, start=1):
        segment = combined[cursor : cursor + duration]
        cursor += duration
        predicted, emb = diarize_segment(segment, diarizer, with_embedding=True)
        expected = label_map[idx]
        is_match = predicted == f"User{expected}"
        results.append(is_match)
        status = "✅" if is_match else "❌"
        sim_str = similarity_report(emb, diarizer)
        emb_str = summarize_embedding(emb)
        print(
            f"Segment {idx:02d}: expected=User{expected} predicted={predicted or 'None':>5s} "
            f"{status} | {sim_str} | {emb_str}"
        )

    correct = sum(results)
    total = len(results)
    accuracy = correct / max(total, 1)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate diarization accuracy on the synthetic 8-chunk dataset."
    )
    parser.add_argument(
        "--skip-output",
        action="store_true",
        help="Only evaluate the individual chunk files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if missing_files := [path.name for path in CHUNK_FILES if not path.exists()]:
        raise FileNotFoundError(f"Missing chunk files: {missing_files}")

    diarizer = OnlineDiarizer(sample_rate=SAMPLE_RATE, threshold=0.72, max_speakers=3)

    evaluate_sequence(CHUNK_FILES, diarizer, EXPECTED_LABELS, "Individual chunk files")

    if not args.skip_output:
        evaluate_output(diarizer, EXPECTED_LABELS)

    print("\nDone.")


if __name__ == "__main__":
    main()
