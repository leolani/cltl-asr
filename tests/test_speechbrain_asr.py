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

    def test_speech_to_text(self):
        with path("resources", "test.wav") as wav:
            speech_array, sampling_rate = sf.read(wav, dtype=np.int16)

        transcript = self.asr.speech_to_text(speech_array, sampling_rate)
        self.assertEqual("IT'S HEALTHIER TO COOK WITHOUT SUGAR", transcript.upper())