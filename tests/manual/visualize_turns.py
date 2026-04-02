"""
Manual test script: transcribe a WAV file and visualise detected turns.

Usage:
    cd cltl-asr
    source venv/bin/activate
    python tests/manual/visualize_turns.py tests/resources/multi_turn_pauses.wav
    python tests/manual/visualize_turns.py tests/resources/multi_turn.wav --output turns.png

The script streams the audio through LocalParakeetRNNTStreamingASR in 100 ms
packets, collects all final StreamTranscription results, and plots:
  - the raw waveform
  - coloured spans for each detected turn (start / end sample positions)
  - a label per turn with the first 40 characters of the transcript

A PNG is saved alongside the audio file (or to --output) and also displayed
interactively if a display is available.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cltl.asr.api_streaming import StreamTranscription
from cltl.asr.parakeet_stream import LocalParakeetRNNTStreamingASR

PACKET_SECS = 0.1
COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _stream_audio(asr: LocalParakeetRNNTStreamingASR, audio: np.ndarray,
                  sample_rate: int) -> list[StreamTranscription]:
    packet_size = int(PACKET_SECS * sample_rate)
    finals = []

    for offset in range(0, len(audio), packet_size):
        packet = audio[offset: offset + packet_size]
        for transcript in asr.push_audio(packet):
            if transcript.is_final:
                finals.append(transcript)
                print(f"  [{transcript.start / sample_rate:6.2f}s – "
                      f"{transcript.end / sample_rate:6.2f}s]  {transcript.text}")

    last = asr.finish()
    if last.text.strip():
        finals.append(last)
        print(f"  [{last.start / sample_rate:6.2f}s – "
              f"{last.end / sample_rate:6.2f}s]  {last.text}")

    return finals


def _plot(audio: np.ndarray, sample_rate: int,
          turns: list[StreamTranscription], output_path: Path) -> None:
    time_axis = np.arange(len(audio)) / sample_rate

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time_axis, audio, color="steelblue", linewidth=0.4, alpha=0.7)

    for i, turn in enumerate(turns):
        colour = COLOURS[i % len(COLOURS)]
        start_sec = turn.start / sample_rate
        end_sec = (turn.end or turn.start) / sample_rate
        width = max(end_sec - start_sec, 0.05)

        ax.axvspan(start_sec, start_sec + width, alpha=0.25, color=colour)
        ax.axvline(start_sec, color=colour, linewidth=1.2, linestyle="--")
        ax.axvline(start_sec + width, color=colour, linewidth=1.2, linestyle=":")

        label = turn.text[:40] + ("…" if len(turn.text) > 40 else "")
        ax.text(
            start_sec + width / 2,
            ax.get_ylim()[1] * 0.85,
            f"T{i + 1}\n{label}",
            ha="center", va="top", fontsize=6.5, color=colour,
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colour, alpha=0.8),
        )

    legend_handles = [
        mpatches.Patch(color=COLOURS[i % len(COLOURS)],
                       label=f"T{i + 1}: {t.text[:50]}")
        for i, t in enumerate(turns)
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7,
              framealpha=0.9)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Streaming ASR turns — {output_path.stem}")
    ax.set_xlim(0, time_axis[-1])
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")

    try:
        plt.show()
    except Exception:
        pass  # headless environment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav_file", type=Path, help="Path to the input WAV file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: <wav_file>.png)")
    parser.add_argument("--device", default="cpu",
                        help="Torch device to use (default: cpu)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.wav_file.exists():
        print(f"Error: file not found: {args.wav_file}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or args.wav_file.with_suffix(".png")

    print(f"Loading model…")
    asr = LocalParakeetRNNTStreamingASR(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        device=args.device,
        compute_dtype=torch.float32,
        chunk_secs=0.5,
        left_context_secs=5.0,
        right_context_secs=2.0,
        turn_threshold_sec=1.0,
    )

    print(f"Reading {args.wav_file}…")
    audio, sample_rate = sf.read(str(args.wav_file), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != asr.sample_rate:
        print(f"Error: sample rate {sample_rate} Hz does not match model rate "
              f"{asr.sample_rate} Hz.", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming {len(audio) / sample_rate:.1f}s of audio…")
    turns = _stream_audio(asr, audio, sample_rate)

    print(f"\nDetected {len(turns)} turn(s).")
    _plot(audio, sample_rate, turns, output_path)


if __name__ == "__main__":
    main()
