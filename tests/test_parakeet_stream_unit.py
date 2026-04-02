"""
Unit tests for LocalParakeetRNNTStreamingASR pure-logic functions.

All external dependencies (torch, numpy, nemo) are stubbed with lightweight
pure-Python replacements so the tests run on a plain Python 3.11 interpreter
with no third-party packages installed.

Run with:
    python3 -m unittest tests.test_parakeet_stream_unit -v
(from the cltl-asr root, with PYTHONPATH=src)
"""

import array
import math
import string
import sys
import types
import unittest
from collections import deque
from unittest.mock import MagicMock, patch


# ===========================================================================
# Step 1: Inject stubs BEFORE the module under test is imported
# ===========================================================================

def _make_numpy_stub():
    """Pure-Python numpy stub covering the numpy surface used by parakeet_stream."""
    np = types.ModuleType("numpy")

    class ndarray:
        def __init__(self, data, dtype=None):
            self._data  = list(data)
            self.dtype  = dtype or "float64"
            self.shape  = (len(self._data),)
            self.ndim   = 1
            self.size   = len(self._data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            result = self._data[idx]
            if isinstance(result, list):
                a = ndarray(result, self.dtype)
                return a
            return result

        def mean(self, axis=None):
            if axis == 1:
                # Each element is itself a list (stereo row)
                means = [sum(row) / len(row) for row in self._data]
                return ndarray(means, self.dtype)
            return sum(self._data) / len(self._data)

        def astype(self, dtype):
            if dtype == np.int16 or dtype == "int16":
                converted = [int(max(-32768, min(32767, round(x)))) for x in self._data]
            elif dtype == np.float32 or dtype == "float32":
                converted = [float(x) for x in self._data]
            else:
                converted = list(self._data)
            a = ndarray(converted, dtype)
            return a

        def ravel(self):
            flat = []
            for item in self._data:
                if isinstance(item, (list, ndarray)):
                    flat.extend(item if isinstance(item, list) else item._data)
                else:
                    flat.append(item)
            a = ndarray(flat, self.dtype)
            return a

        def flatten(self):
            return self.ravel()

        def tolist(self):
            return [int(x) if self.dtype in ("int16", "int32") else float(x)
                    for x in self._data]

        @property
        def nbytes(self):
            return self.size * 2  # approximate

    # dtype sentinels
    np.int16   = "int16"
    np.int32   = "int32"
    np.float32 = "float32"
    np.float64 = "float64"

    np.ndarray = ndarray

    def _concatenate(arrays, axis=None):
        flat = []
        for a in arrays:
            if isinstance(a, ndarray):
                flat.extend(a._data)
            else:
                flat.extend(list(a))
        if not flat:
            return ndarray([], "float32")
        dtype = arrays[0].dtype if arrays else "float32"
        return ndarray(flat, dtype)

    np.concatenate = _concatenate

    def _array(data, dtype=None):
        dtype = dtype or "float64"
        if isinstance(data, ndarray):
            return ndarray(data._data, dtype)
        if isinstance(data, list):
            # Handle nested lists (2-D)
            if data and isinstance(data[0], list):
                a = ndarray(data, dtype)
                a.ndim  = 2
                a.shape = (len(data), len(data[0]))
                return a
        return ndarray(list(data) if not isinstance(data, ndarray) else data._data, dtype)

    np.array   = _array
    np.zeros   = lambda shape, dtype=None: ndarray([0] * (shape if isinstance(shape, int) else shape[0]), dtype or "float32")
    np.atleast_1d = lambda x: x if isinstance(x, ndarray) else ndarray([x])
    np.ascontiguousarray = lambda a, dtype=None: a  # already contiguous in stub

    return np


def _make_torch_stub(np_stub):
    """Pure-Python torch stub."""
    torch = types.ModuleType("torch")

    class Tensor:
        def __init__(self, data=None, dtype=None):
            if data is None:
                self._data  = []
                self._dtype = dtype or torch.float32
            elif isinstance(data, np_stub.ndarray):
                self._data  = list(data._data)
                self._dtype = dtype or torch.float32
            elif isinstance(data, list):
                self._data  = list(data)
                self._dtype = dtype or torch.float32
            else:
                self._data  = [data]
                self._dtype = dtype or torch.float32

        @property
        def dtype(self):
            return self._dtype

        def numel(self):
            return len(self._data)

        def flatten(self):
            flat = []
            for item in self._data:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            t = Tensor(dtype=self._dtype)
            t._data = flat
            return t

        def detach(self):
            return self

        def cpu(self):
            return self

        def to(self, dtype_or_device):
            if dtype_or_device == "cpu":
                return self
            t = Tensor(dtype=dtype_or_device)
            if dtype_or_device == torch.float32:
                t._data = [float(x) for x in self._data]
            elif dtype_or_device == torch.int16:
                t._data = [int(max(-32768, min(32767, round(x)))) for x in self._data]
            else:
                t._data = list(self._data)
            return t

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                t = Tensor(dtype=self._dtype)
                t._data = self._data[idx]
                return t
            return self._data[idx]

        def item(self):
            return self._data[0] if self._data else 0

        def tolist(self):
            return list(self._data)

        def clone(self):
            t = Tensor(dtype=self._dtype)
            t._data = list(self._data)
            return t

        def __truediv__(self, scalar):
            t = Tensor(dtype=self._dtype)
            t._data = [x / scalar for x in self._data]
            return t

        def __len__(self):
            return len(self._data)

        def unsqueeze(self, dim):
            # Returns a 1-element batch wrapper; keeps _data as-is for numel()
            return self


    torch.float32 = "float32"
    torch.int16   = "int16"
    torch.long    = "long"
    torch.bool    = "bool_"

    torch.Tensor  = Tensor

    torch.empty   = lambda size=0, dtype=None: Tensor(dtype=dtype)
    torch.cat     = lambda tensors, dim=0: _torch_cat(tensors)
    torch.tensor  = lambda data, dtype=None: Tensor(data if isinstance(data, list) else [data], dtype=dtype)
    def _from_numpy(arr):
        dtype = torch.int16 if getattr(arr, "dtype", None) in ("int16", np_stub.int16) else torch.float32
        t = Tensor(dtype=dtype)
        t._data = list(arr._data) if isinstance(arr, np_stub.ndarray) else list(arr)
        return t

    torch.from_numpy = _from_numpy
    torch.device  = lambda x: x
    torch.where   = lambda cond, a, b: a if cond else b
    torch.inference_mode = lambda: (lambda f: f)
    torch.set_num_threads         = lambda n: None
    torch.set_num_interop_threads = lambda n: None

    # Type hint sentinel used in function signatures
    torch.dtype = type("dtype", (), {})

    def _torch_cat(tensors):
        result = Tensor(dtype=tensors[0].dtype if tensors else torch.float32)
        data = []
        for t in tensors:
            data.extend(t._data)
        result._data = data
        return result

    return torch


def _make_nemo_stubs():
    """Stub all nemo submodules that parakeet_stream.py imports."""
    stubs = {}
    for name in [
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "nemo.collections.asr.models",
        "nemo.collections.asr.parts",
        "nemo.collections.asr.parts.submodules",
        "nemo.collections.asr.parts.submodules.rnnt_decoding",
        "nemo.collections.asr.parts.utils",
        "nemo.collections.asr.parts.utils.rnnt_utils",
        "nemo.collections.asr.parts.utils.streaming_utils",
    ]:
        stubs[name] = types.ModuleType(name)

    stubs["nemo.collections.asr.models"].ASRModel               = MagicMock()
    stubs["nemo.collections.asr.models"].EncDecHybridRNNTCTCModel = MagicMock()
    stubs["nemo.collections.asr.models"].EncDecRNNTModel         = MagicMock()
    stubs["nemo.collections.asr.parts.submodules.rnnt_decoding"].RNNTDecodingConfig = MagicMock()
    stubs["nemo.collections.asr.parts.utils.rnnt_utils"].BatchedHyps = MagicMock()

    class _ContextSize:
        def __init__(self, left=0, chunk=0, right=0):
            self.left  = left
            self.chunk = chunk
            self.right = right

        def total(self):
            return self.left + self.chunk + self.right

        def subsample(self, factor):
            return _ContextSize(self.left // factor, self.chunk // factor, self.right // factor)

    class _StreamingBatchedAudioBuffer:
        def __init__(self, batch_size, context_samples, dtype, device):
            self.samples            = None
            self.context_size       = context_samples
            self.context_size_batch = context_samples

        def add_audio_batch_(self, **kwargs):
            pass

    stubs["nemo.collections.asr.parts.utils.streaming_utils"].ContextSize = _ContextSize
    stubs["nemo.collections.asr.parts.utils.streaming_utils"].StreamingBatchedAudioBuffer = _StreamingBatchedAudioBuffer

    return stubs


def _install_all_stubs():
    """Only inject stubs for packages that are not already importable."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        _np = _make_numpy_stub()
        sys.modules.setdefault("numpy", _np)

    try:
        import torch  # noqa: F401
    except ImportError:
        import numpy as _np_real
        _torch = _make_torch_stub(_np_real)
        sys.modules.setdefault("torch", _torch)

    _nemo = _make_nemo_stubs()
    for name, mod in _nemo.items():
        sys.modules.setdefault(name, mod)


_install_all_stubs()

# ---------------------------------------------------------------------------
# Safe to import the module under test now
# ---------------------------------------------------------------------------
from cltl.asr.parakeet_stream import LocalParakeetRNNTStreamingASR  # noqa: E402


def make_divisible_by(num: int, factor: int) -> int:
    """Local copy for testing; mirrors the inlined logic in _init_context_sizes."""
    return (num // factor) * factor
from cltl.asr.api_streaming import StreamTranscription  # noqa: E402

import numpy as np   # noqa: E402 (our stub)
import torch         # noqa: E402 (our stub)

from nemo.collections.asr.parts.utils.streaming_utils import ContextSize  # noqa: E402


# ===========================================================================
# Step 2: Factory – build an instance without running __init__
# ===========================================================================

def _make_asr(*, turn_threshold_chunks: int = 3, vad=None, vad_threshold: int = None):
    asr = object.__new__(LocalParakeetRNNTStreamingASR)

    asr.sample_rate   = 16_000
    asr.device        = "cpu"
    asr.compute_dtype = torch.float32

    asr.context_samples = ContextSize(left=80_000, chunk=8_000, right=32_000)

    asr._turn_threshold_chunks    = turn_threshold_chunks
    asr._vad                      = vad
    asr._vad_silence_threshold    = vad_threshold

    buffer_mock = MagicMock()
    buffer_mock.samples = None  # _collect_recent_audio checks this
    asr.buffer = buffer_mock

    asr.pending_audio                    = torch.empty(0, dtype=torch.float32)
    asr.partial_transcripts              = deque([], maxlen=turn_threshold_chunks)
    asr.current_batched_hyps             = None
    asr.state                            = None
    asr.started                          = False
    asr.closed                           = False
    asr._consecutive_silence_samples     = 0
    asr._last_encoder_output             = None
    asr._last_encoder_output_len         = None
    asr._last_encoder_context            = None
    asr._last_encoder_context_batch      = None
    asr._total_samples_consumed      = 0
    asr._transcript_onset_sample     = None

    asr.model             = MagicMock()
    asr.decoding_computer = MagicMock()

    return asr


# ===========================================================================
# Tests
# ===========================================================================

class TestMakeDivisibleBy(unittest.TestCase):
    def test_already_divisible(self):
        self.assertEqual(make_divisible_by(8, 4), 8)

    def test_truncates_to_lower_multiple(self):
        self.assertEqual(make_divisible_by(9,  4), 8)
        self.assertEqual(make_divisible_by(11, 4), 8)
        self.assertEqual(make_divisible_by(15, 4), 12)

    def test_zero_input(self):
        self.assertEqual(make_divisible_by(0, 4), 0)

    def test_factor_one(self):
        self.assertEqual(make_divisible_by(7, 1), 7)


class TestToMonoFloat32(unittest.TestCase):
    """Tests for _to_mono_float32, which replaced the separate _concat_to_mono
    and _normalize_input methods."""

    def test_mono_int16_frames_scaled_to_float32(self):
        asr    = _make_asr()
        frame  = np.array([32767, -32768, 0], dtype=np.int16)
        result = asr._to_mono_float32([frame])
        self.assertAlmostEqual(float(result[0]),  32767 / 32768.0,  places=4)
        self.assertAlmostEqual(float(result[1]), -32768 / 32768.0,  places=4)
        self.assertAlmostEqual(float(result[2]),  0.0,              places=4)

    def test_mono_float32_values_unchanged(self):
        asr    = _make_asr()
        frame  = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        result = asr._to_mono_float32([frame])
        self.assertAlmostEqual(float(result[0]),  0.5, places=4)
        self.assertAlmostEqual(float(result[1]), -0.5, places=4)

    def test_multiple_frames_concatenated(self):
        asr    = _make_asr()
        f1     = np.array([0.1, 0.2], dtype=np.float32)
        f2     = np.array([0.3, 0.4], dtype=np.float32)
        result = asr._to_mono_float32([f1, f2])
        self.assertEqual(result.numel(), 4)

    def test_2d_mono_frame_flattened(self):
        asr    = _make_asr()
        frame  = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        result = asr._to_mono_float32([frame])
        self.assertEqual(result.numel(), 3)

    @unittest.skipIf(
        not hasattr(np, "__version__"),
        "Stereo test requires real numpy (stub always flattens to 1-D)"
    )
    def test_stereo_averaged_to_mono(self):
        """Stereo frame [1000, -1000] averages to 0 per sample."""
        asr    = _make_asr()
        frame  = np.array([[1000, -1000], [2000, -2000]], dtype=np.int16)
        result = asr._to_mono_float32([frame])
        self.assertEqual(result.numel(), 2)
        self.assertAlmostEqual(float(result[0]), 0.0, places=4)
        self.assertAlmostEqual(float(result[1]), 0.0, places=4)


class TestDetectSilence(unittest.TestCase):
    def _asr_with_vad(self):
        vad = MagicMock()
        return _make_asr(vad=vad, vad_threshold=1000), vad

    def test_no_vad_leaves_counter_at_zero(self):
        asr   = _make_asr()
        frame = np.zeros(480, dtype=np.int16)
        asr._detect_silence([frame])
        self.assertEqual(asr._consecutive_silence_samples, 0)

    def test_silent_frames_accumulate(self):
        asr, vad = self._asr_with_vad()
        vad.is_vad.return_value = False
        frame = np.zeros(480, dtype=np.int16)
        asr._detect_silence([frame, frame])
        self.assertEqual(asr._consecutive_silence_samples, 960)

    def test_speech_frame_resets_counter(self):
        asr, vad = self._asr_with_vad()
        asr._consecutive_silence_samples = 500
        vad.is_vad.return_value = True
        frame = np.zeros(480, dtype=np.int16)
        asr._detect_silence([frame])
        self.assertEqual(asr._consecutive_silence_samples, 0)

    def test_speech_then_silence_accumulates_only_silence(self):
        asr, vad = self._asr_with_vad()
        vad.is_vad.side_effect = [True, False]
        frame = np.zeros(480, dtype=np.int16)
        asr._detect_silence([frame, frame])
        self.assertEqual(asr._consecutive_silence_samples, 480)


class TestIsRightContextSilent(unittest.TestCase):
    def test_no_vad_always_returns_false(self):
        asr = _make_asr()
        asr._consecutive_silence_samples = 999_999
        self.assertFalse(asr._is_right_context_silent())

    def test_below_threshold_returns_false(self):
        asr = _make_asr(vad=MagicMock(), vad_threshold=1000)
        asr._consecutive_silence_samples = 999
        self.assertFalse(asr._is_right_context_silent())

    def test_at_threshold_returns_true(self):
        asr = _make_asr(vad=MagicMock(), vad_threshold=1000)
        asr._consecutive_silence_samples = 1000
        self.assertTrue(asr._is_right_context_silent())

    def test_above_threshold_returns_true(self):
        asr = _make_asr(vad=MagicMock(), vad_threshold=1000)
        asr._consecutive_silence_samples = 9999
        self.assertTrue(asr._is_right_context_silent())


class TestSamplePositionTracking(unittest.TestCase):
    def test_initial_position_is_zero(self):
        asr = _make_asr()
        self.assertEqual(asr.get_current_sample_position(), 0)

    def test_returns_current_consumed_counter(self):
        asr = _make_asr()
        asr._total_samples_consumed = 8_000
        self.assertEqual(asr.get_current_sample_position(), 8_000)

    def test_full_reset_clears_counters(self):
        asr = _make_asr()
        asr._total_samples_consumed  = 16_000
        asr._transcript_onset_sample = 8_000

        with patch("cltl.asr.parakeet_stream.StreamingBatchedAudioBuffer"):
            asr.reset(keep_recent=False)

        self.assertEqual(asr._total_samples_consumed, 0)
        self.assertIsNone(asr._transcript_onset_sample)

    def test_keep_recent_resets_onset_but_preserves_total(self):
        """After keep_recent reset, onset is cleared so the next turn tracks fresh."""
        asr = _make_asr()
        asr._total_samples_consumed  = 48_000
        asr._transcript_onset_sample = 16_000

        with patch("cltl.asr.parakeet_stream.StreamingBatchedAudioBuffer"), \
             patch.object(asr, "_collect_recent_audio", return_value=torch.empty(0)):
            asr.reset(keep_recent=True)

        self.assertEqual(asr._total_samples_consumed, 48_000)
        self.assertIsNone(asr._transcript_onset_sample)

    def test_speech_end_returns_total_samples_consumed(self):
        """_speech_end() returns the end of the finalising chunk."""
        asr = _make_asr()
        asr._total_samples_consumed = 80_000

        self.assertEqual(asr._speech_end(), 80_000)

    def test_turn_start_returns_onset_when_set(self):
        """_turn_start() returns onset sample when the transcript has started."""
        asr = _make_asr()
        asr._transcript_onset_sample = 24_000
        asr._total_samples_consumed  = 48_000

        self.assertEqual(asr._turn_start(), 24_000)

    def test_turn_start_falls_back_to_consumed_when_onset_not_set(self):
        """_turn_start() falls back to total consumed if onset is not yet set."""
        asr = _make_asr()
        asr._transcript_onset_sample = None
        asr._total_samples_consumed  = 48_000

        self.assertEqual(asr._turn_start(), 48_000)

    def test_keep_recent_does_not_reset_total(self):
        asr = _make_asr()
        asr._total_samples_consumed = 32_000

        with patch("cltl.asr.parakeet_stream.StreamingBatchedAudioBuffer"), \
             patch.object(asr, "_collect_recent_audio", return_value=torch.empty(0)):
            asr.reset(keep_recent=True)

        self.assertEqual(asr._total_samples_consumed, 32_000)

    def test_onset_is_none_at_stream_start(self):
        """At stream start, onset is None (no transcript emitted yet)."""
        asr = _make_asr()
        self.assertIsNone(asr._transcript_onset_sample)


class TestSpeculativeTextComparison(unittest.TestCase):

    @staticmethod
    def _normalise(text: str) -> str:
        return text.strip().strip(string.punctuation)

    def test_identical_strings_match(self):
        self.assertEqual(self._normalise("hello"), self._normalise("hello"))

    def test_trailing_punctuation_stripped(self):
        self.assertEqual(self._normalise("hello world"), self._normalise("hello world."))
        self.assertEqual(self._normalise("hello world!"), self._normalise("hello world?"))

    def test_leading_whitespace_stripped(self):
        self.assertEqual(self._normalise("  hello"), self._normalise("hello"))

    def test_different_texts_not_equal(self):
        self.assertNotEqual(self._normalise("hello"), self._normalise("goodbye"))

    def test_only_punctuation_normalises_to_empty(self):
        self.assertEqual(self._normalise("..."), "")

    def test_finalize_appends_result_and_returns_true_when_texts_match(self):
        asr = _make_asr()
        asr._total_samples_consumed  = 48_000
        asr._transcript_onset_sample = 8_000
        asr._speculative_finish      = MagicMock(return_value="hello world.")

        results   = []
        finalized = asr._try_speculative_finalize("hello world.", results)

        self.assertTrue(finalized)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_final)
        self.assertEqual(results[0].start, 8_000)
        # end = total_consumed at finalisation
        self.assertEqual(results[0].end, 48_000)

    def test_finalize_returns_false_and_no_result_on_mismatch(self):
        asr = _make_asr()
        asr._speculative_finish = MagicMock(return_value="something else")

        results   = []
        finalized = asr._try_speculative_finalize("hello world", results)

        self.assertFalse(finalized)
        self.assertEqual(len(results), 0)

    def test_finalize_calls_reset_with_keep_recent_on_match(self):
        asr = _make_asr()
        asr._transcript_onset_sample = 0
        asr._speculative_finish = MagicMock(return_value="hello.")
        asr.reset               = MagicMock()

        asr._try_speculative_finalize("hello.", [])

        asr.reset.assert_called_once_with(keep_recent=True)


class TestTryFinalize(unittest.TestCase):
    """Tests for _try_finalize — the turn-threshold path."""

    def _full_deque_asr(self, current: str) -> object:
        """Return an ASR instance whose deque is full with `current` as every entry."""
        asr = _make_asr(turn_threshold_chunks=3)
        asr._transcript_onset_sample = 0
        for _ in range(3):
            asr.partial_transcripts.append(current)
        return asr

    def test_turn_threshold_calls_speculative_finish_before_forcing(self):
        """When the deque is full, speculative finish is consulted first."""
        asr = self._full_deque_asr("hello world")
        asr._speculative_finish = MagicMock(return_value="hello world.")
        asr.reset = MagicMock()

        results = []
        finalized = asr._try_finalize("hello world", results)

        asr._speculative_finish.assert_called_once()
        self.assertTrue(finalized)

    def test_turn_threshold_emits_final_and_resets_when_speculative_agrees(self):
        """Speculative finish agrees → final emitted, reset called."""
        asr = self._full_deque_asr("stable text")
        asr._speculative_finish = MagicMock(return_value="stable text.")
        asr.reset = MagicMock()

        results = []
        asr._try_finalize("stable text", results)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_final)
        asr.reset.assert_called_once_with(keep_recent=True)

    def test_turn_threshold_forces_final_and_resets_when_speculative_disagrees(self):
        """Speculative finish disagrees (still changing) → forced final still emitted and reset called."""
        asr = self._full_deque_asr("partial text")
        # speculative returns something longer — transcript still growing
        asr._speculative_finish = MagicMock(return_value="partial text with more words")
        asr.reset = MagicMock()

        results = []
        finalized = asr._try_finalize("partial text", results)

        self.assertTrue(finalized)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_final)
        asr.reset.assert_called_once_with(keep_recent=True)

    def test_turn_threshold_does_not_fire_when_deque_not_full(self):
        asr = _make_asr(turn_threshold_chunks=3)
        asr.partial_transcripts.append("hello")  # only 1 of 3
        asr._speculative_finish = MagicMock(return_value="something else")

        results = []
        finalized = asr._try_finalize("hello", results)

        self.assertFalse(finalized)
        self.assertEqual(results, [])

    def test_turn_threshold_does_not_fire_when_transcript_changed(self):
        asr = self._full_deque_asr("old text")
        asr._speculative_finish = MagicMock(return_value="something else")

        results = []
        # current is different from deque[0] — should not trigger threshold
        finalized = asr._try_finalize("new text", results)

        self.assertFalse(finalized)


class TestPushAudioGuards(unittest.TestCase):
    def test_raises_when_stream_is_closed(self):
        asr        = _make_asr()
        asr.closed = True
        with self.assertRaises(RuntimeError):
            asr.push_audio(np.zeros(100, dtype=np.int16))

    def test_raises_on_mismatched_sample_rate(self):
        asr = _make_asr()
        with self.assertRaises(ValueError):
            asr.push_audio(np.zeros(100, dtype=np.int16), sampling_rate=8_000)

    def test_returns_empty_list_when_insufficient_audio(self):
        """Less than chunk+right samples → no decode step → no transcripts."""
        asr   = _make_asr()
        tiny  = np.zeros(10, dtype=np.float32)
        result = asr.push_audio(tiny)
        self.assertEqual(result, [])

    def test_bare_ndarray_accepted_without_error(self):
        asr   = _make_asr()
        tiny  = np.zeros(10, dtype=np.float32)
        # Should not raise; returns [] because not enough audio
        self.assertIsInstance(asr.push_audio(tiny), list)


if __name__ == "__main__":
    unittest.main()
