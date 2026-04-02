"""
Integration tests for LocalParakeetRNNTStreamingASR.

Requires the full NeMo / Parakeet dependencies and the test audio files:
    tests/resources/multi_turn_pauses.wav
    tests/resources/multi_turn.wav

Run with:
    cd cltl-asr
    python -m unittest tests.test_parakeet_stream_integration -v
"""

import string
import unittest
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch

from cltl.asr.api_streaming import StreamTranscription
from cltl.asr.parakeet_stream import LocalParakeetRNNTStreamingASR


RESOURCES = Path(__file__).parent / "resources"

_asr_model: Optional[LocalParakeetRNNTStreamingASR] = None


def _get_asr_model() -> LocalParakeetRNNTStreamingASR:
    """Return a cached model instance, creating it on first call.

    Torch only allows thread-count configuration before parallel work starts,
    so we reuse one model across all test classes in this module.
    """
    global _asr_model
    if _asr_model is None:
        _asr_model = LocalParakeetRNNTStreamingASR(
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            device="cpu",
            compute_dtype=torch.float32,
            chunk_secs=0.5,
            left_context_secs=5.0,
            right_context_secs=2.0,
            turn_threshold_sec=1.0,
        )
    return _asr_model

PACKET_SECS = 0.1

# Timestamp tolerance: ± one chunk size (0.5 s).
TIMESTAMP_TOLERANCE_SECS = 0.5
# Minimum word-overlap fraction required to claim two transcripts match.
MIN_WORD_OVERLAP = 0.7


def _normalise(text: str) -> str:
    return text.strip().lower().strip(string.punctuation)


def _words(text: str) -> List[str]:
    return _normalise(text).split()


def _word_overlap(actual: str, expected: str) -> float:
    """Fraction of expected words that appear in the actual transcript."""
    expected_words = _words(expected)
    if not expected_words:
        return 1.0
    actual_words = set(_words(actual))
    matches = sum(1 for w in expected_words if w in actual_words)
    return matches / len(expected_words)


def _find_best_matching_turn(
    finals: List[StreamTranscription],
    expected_text: str,
    expected_start_sample: int,
    sample_rate: int,
) -> Optional[StreamTranscription]:
    """Return the final turn whose text best overlaps with expected_text and
    whose start timestamp is roughly within range."""
    tolerance_samples = int(TIMESTAMP_TOLERANCE_SECS * 3 * sample_rate)
    best, best_score = None, 0.0
    for turn in finals:
        if abs(turn.start - expected_start_sample) > tolerance_samples:
            continue
        score = _word_overlap(turn.text, expected_text)
        if score > best_score:
            best, best_score = turn, score
    return best


class _BaseStreamingASRTest(unittest.TestCase):
    """Shared scaffolding for streaming ASR integration tests."""

    AUDIO_FILE: Path = None  # subclasses must set this

    @classmethod
    def setUpClass(cls) -> None:
        if cls.AUDIO_FILE is None:
            raise unittest.SkipTest("Base class — no audio file configured")

        asr = _get_asr_model()
        asr.reset()

        audio, sr = sf.read(str(cls.AUDIO_FILE), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != asr.sample_rate:
            raise ValueError(
                f"Audio sample rate {sr} Hz does not match model rate "
                f"{asr.sample_rate} Hz. Resample the file first."
            )

        cls.sample_rate = sr
        cls.finals = cls._run_stream(asr, audio, sr)

    @classmethod
    def _run_stream(cls, asr, audio: np.ndarray, sample_rate: int) -> List[StreamTranscription]:
        packet_size = int(PACKET_SECS * sample_rate)
        raw_finals = []

        for offset in range(0, len(audio), packet_size):
            packet = audio[offset: offset + packet_size]
            for transcript in asr.push_audio(packet):
                if transcript.is_final:
                    raw_finals.append(transcript)

        last = asr.finish()
        if last.text.strip():
            raw_finals.append(last)

        print("Finals:", raw_finals)

        return raw_finals

    def _assert_turn(self, expected_start_sample: int, expected_end_sample: Optional[int],
                     expected_text: str, label: str) -> None:
        turn = _find_best_matching_turn(
            self.finals, expected_text, expected_start_sample, self.sample_rate
        )
        self.assertIsNotNone(
            turn,
            f"{label}: no turn found near sample {expected_start_sample} "
            f"({expected_start_sample / self.sample_rate:.1f}s) "
            f"matching '{expected_text[:60]}...'\n"
            f"All finals:\n" + "\n".join(
                f"  start={t.start} ({t.start / self.sample_rate:.1f}s)  "
                f"end={t.end} ({t.end / self.sample_rate:.1f}s)  {t.text!r}"
                for t in self.finals
            ),
        )

        tolerance_samples = int(TIMESTAMP_TOLERANCE_SECS * self.sample_rate)
        self.assertAlmostEqual(
            turn.start, expected_start_sample,
            delta=tolerance_samples,
            msg=(
                f"{label}: start sample {turn.start} ({turn.start / self.sample_rate:.1f}s), "
                f"expected {expected_start_sample} ({expected_start_sample / self.sample_rate:.1f}s) "
                f"± {tolerance_samples} samples ({TIMESTAMP_TOLERANCE_SECS}s)"
            ),
        )
        if expected_end_sample is not None:
            self.assertAlmostEqual(
                turn.end, expected_end_sample,
                delta=tolerance_samples,
                msg=(
                    f"{label}: end sample {turn.end} ({turn.end / self.sample_rate:.1f}s), "
                    f"expected {expected_end_sample} ({expected_end_sample / self.sample_rate:.1f}s) "
                    f"± {tolerance_samples} samples ({TIMESTAMP_TOLERANCE_SECS}s)"
                ),
            )

        overlap = _word_overlap(turn.text, expected_text)
        self.assertGreaterEqual(
            overlap, MIN_WORD_OVERLAP,
            msg=(
                f"{label}: word overlap {overlap:.0%} below {MIN_WORD_OVERLAP:.0%}\n"
                f"  actual:   {turn.text!r}\n"
                f"  expected: {expected_text!r}"
            ),
        )

    def _assert_expected_turn_count(self) -> None:
        self.assertEqual(
            len(self.finals), len(self.EXPECTED_TURNS),
            f"Expected {len(self.EXPECTED_TURNS)} substantive final turns, "
            f"got {len(self.finals)}:\n"
            + "\n".join(
                f"  [{i}] start={t.start} ({t.start / self.sample_rate:.1f}s)  "
                f"end={t.end} ({t.end / self.sample_rate:.1f}s)  {t.text!r}"
                for i, t in enumerate(self.finals)
            ),
        )


class TestLocalParakeetRNNTStreamingASR(_BaseStreamingASRTest):
    """End-to-end test against multi_turn_pauses.wav — a recording with clear
    pauses between turns and no filler sounds.

    Six finals produced (including one short filler 'Okay, so I' between turns 3 and 4):
        1.4 –  8.3 s  Turn 1: Veerle introduction part 1
       10.6 – 17.0 s  Turn 2: Veerle introduction part 2 ('Ik denk dat...')
       20.2 – 24.3 s  Turn 3: 'Daar kan ik...'
       32.5 – 35.8 s  Filler: 'Okay, so I'
       27.7 – 35.9 s  Turn 4: cats in Amersfoort
       39.8 – 56.9 s  Turn 5: school / robots

    Timestamp semantics:
    - start: sample position of the first chunk in which the model emits non-empty
             output for this turn.  Due to the right-context pre-fill on the first
             decode step (~2.5 s), onset is delayed relative to true speech onset.
    - end:   sample position of the last chunk consumed when the turn is finalised.
    Both values carry a tolerance of ± one chunk size (0.5 s / 8 000 samples).
    """

    AUDIO_FILE = RESOURCES / "multi_turn_pauses.wav"

    # (start_sample, end_sample, expected_text)
    EXPECTED_TURNS = [
        (
             47_360,  #  3.0 s
            170_240,  # 10.6 s
            "Hello, mijn naam is Veerle. Ik vind het leuk om spelletjes te spelen.",
        ),
        (
            225_280,  # 14.1 s
            340_480,  # 21.3 s
            "Ik denk dat ik heel erg goed ben in rekenen. Ik kan superveel rekensommen oplossen.",
        ),
        (
            410_880,  # 25.7 s
            487_680,  # 30.5 s
            "Daar kan ik wel 8 of 10 van tegelijk doen.",
        ),
        (
            519_680,  # 32.5 s — filler fragment between pauses
            573_440,  # 35.8 s
            "Okay, so I",
        ),
        (
            605_440,  # 37.8 s
            736_000,  # 46.0 s
            "Ik heb thuis twee katten en die wonen in Abenswoord, ze heten Fret en Tony.",
        ),
        (
            829_440,  # 51.8 s
          1_113_600,  # 69.6 s
            "Bij basisschool staat in Nijmegen. De rollen zijn heel erg leuk. "
            "Ik heb heel erg veel zin om met de robots te werken. "
            "Want ik ben blij om technologie te zien. "
            "En ik vind de robot er ergens schattig uitzien.",
        ),
    ]

    def test_exactly_six_finals_produced(self):
        self._assert_expected_turn_count()

    def test_turn_1_veerle_introduction_part_1(self):
        start, end, text = self.EXPECTED_TURNS[0]
        self._assert_turn(start, end, text, "Turn 1 (Veerle introduction part 1)")

    def test_turn_2_veerle_introduction_part_2(self):
        start, end, text = self.EXPECTED_TURNS[1]
        self._assert_turn(start, end, text, "Turn 2 (Veerle introduction part 2)")

    def test_turn_3_daar_kan_ik(self):
        start, end, text = self.EXPECTED_TURNS[2]
        self._assert_turn(start, end, text, "Turn 3 ('Daar kan ik...')")

    def test_turn_4_filler_okay_so_i(self):
        start, end, text = self.EXPECTED_TURNS[3]
        self._assert_turn(start, end, text, "Turn 4 (filler: 'Okay, so I')")

    def test_turn_5_cats_in_amersfoort(self):
        start, end, text = self.EXPECTED_TURNS[4]
        self._assert_turn(start, end, text, "Turn 5 (cats in Amersfoort)")

    def test_turn_6_school_robots(self):
        start, end, text = self.EXPECTED_TURNS[5]
        self._assert_turn(start, end, text, "Turn 6 (school / robots)")


class TestLocalParakeetRNNTStreamingASRWithFillers(_BaseStreamingASRTest):
    """End-to-end test against multi_turn.wav — the same recording with filler
    sounds (e.g. 'um') inserted between turns.

    The fillers bridge the gap between Turns 1 and 2, causing the streamer to
    merge them into a single final.  The two 'Um' filler finals are included
    as-is, giving six finals total.

    Six finals at the following approximate speech boundaries (16 000 Hz):
        3.0 – 21.2 s  Turn 1+2: Veerle introduction (merged by fillers)
       23.7 – 29.0 s  Turn 3: 'Daar kan ik...'
       30.9 – 32.9 s  Filler: 'Um'
       34.9 – 44.0 s  Turn 4: cats in Amersfoort
       46.0 – 47.9 s  Filler: 'Um'
       51.4 – 69.6 s  Turn 5: school / robots
    """

    AUDIO_FILE = RESOURCES / "multi_turn.wav"

    # (start_sample, end_sample, expected_text)
    EXPECTED_TURNS = [
        (
             47_360,  #  3.0 s
            339_200,  # 21.2 s
            "Hallo, mijn naam is Veerle. Ik vind het leuk om spelletjes te spelen. "
            "Ik denk dat ik heel erg goed ben in rekenen. Ik kan superveel rekensen oplossen. En",
        ),
        (
            378_880,  # 23.7 s
            463_360,  # 29.0 s
            "Daar kan ik wel acht of tien van tegelijk doen.",
        ),
        (
            495_360,  # 30.9 s — filler between turns 2 and 3
            526_080,  # 32.9 s
            "Um",
        ),
        (
            558_080,  # 34.9 s
            704_000,  # 44.0 s
            "Ik heb thuis twee katten en die wonen in Abensfort. Zeven, Fretant Tony.",
        ),
        (
            736_000,  # 46.0 s — filler between turns 3 and 4
            766_720,  # 47.9 s
            "Um",
        ),
        (
            821_760,  # 51.4 s
          1_113_472,  # 69.6 s
            "Bij basisschool staat in Nijmegen. De rollen zijn heel erg leuk. "
            "Ik heb heel erg veel zin om met de robots te werken. "
            "Want ik ben blij om technologie te zien. "
            "En ik vind de robot er ergens schattig uitzien. De juffemeesters zijn",
        ),
    ]

    def test_exactly_six_finals_produced(self):
        self._assert_expected_turn_count()

    def test_turn_1_veerle_introduction_merged(self):
        start, end, text = self.EXPECTED_TURNS[0]
        self._assert_turn(start, end, text, "Turn 1+2 (Veerle introduction, merged by fillers)")

    def test_turn_2_daar_kan_ik(self):
        start, end, text = self.EXPECTED_TURNS[1]
        self._assert_turn(start, end, text, "Turn 2 ('Daar kan ik...')")

    def test_turn_3_filler_um(self):
        start, end, text = self.EXPECTED_TURNS[2]
        self._assert_turn(start, end, text, "Turn 3 (filler: 'Um')")

    def test_turn_4_cats_in_amersfoort(self):
        start, end, text = self.EXPECTED_TURNS[3]
        self._assert_turn(start, end, text, "Turn 4 (cats in Amersfoort)")

    def test_turn_5_filler_um(self):
        start, end, text = self.EXPECTED_TURNS[4]
        self._assert_turn(start, end, text, "Turn 5 (filler: 'Um')")

    def test_turn_6_school_robots(self):
        start, end, text = self.EXPECTED_TURNS[5]
        self._assert_turn(start, end, text, "Turn 6 (school / robots)")


if __name__ == "__main__":
    unittest.main()
