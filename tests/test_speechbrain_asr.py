import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf
from importlib_resources import path

from cltl.asr.speechbrain_asr import SpeechbrainASR


class TestSpeechbrainASR(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.mkdtemp()
        cls.asr = SpeechbrainASR("speechbrain/asr-transformer-transformerlm-librispeech", cls.tempdir, cls.tempdir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tempdir)
        del cls.asr

    def test_resampling_stereo_to_mono(self):
        resampled = self.asr._resample(np.stack((np.full((16,), 101, dtype=np.int16), np.zeros((16,), dtype=np.int16)), axis=1), 16000)
        self.assertEqual(resampled.shape, (16,))
        self.assertTrue(all(resampled == 50))

    def test_resampling_stereo_to_mono_max_volume(self):
        resampled = self.asr._resample(np.full((16,), 32767, dtype=np.int16), 16000)
        self.assertEqual(resampled.shape, (16,))
        self.assertTrue(all(resampled == 32767))

    def test_resampling_single_channel_is_squeezed(self):
        resampled = self.asr._resample(np.ones((16, 1), dtype=np.int16), 16000)
        self.assertEqual(resampled.shape, (16,))

    def test_resampling_single_channel(self):
        resampled = self.asr._resample(np.ones((16,), dtype=np.int16), 16000)
        self.assertEqual(resampled.shape, (16,))

    def test_speech_to_text(self):
        with path("resources", "test.wav") as wav:
            speech_array, sampling_rate = sf.read(wav, dtype=np.int16)

        transcript = self.asr.speech_to_text(speech_array, sampling_rate)
        self.assertEqual("IT'S HEALTHIER TO COOK WITHOUT SUGAR", transcript.upper())