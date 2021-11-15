import os
import shutil
import tempfile
import time

import numpy as np
from speechbrain.pretrained import EncoderDecoderASR

from cltl.asr.api import ASR
from cltl.asr.util import store_wav


class SpeechbrainASR(ASR):
    def __init__(self, model_id: str, storage: str = None, model_dir: str = None):
        self.processor = EncoderDecoderASR.from_hparams(source=model_id, savedir=model_dir)
        self._storage = storage if storage else tempfile.mkdtemp()
        self._clean_storage = storage is None

    def clean(self):
        shutil.rmtree(self._storage)

    def speech_to_text(self, audio: np.array, sampling_rate: int) -> str:
        wav_file = str(os.path.join(self._storage, f"asr-{time.time()}.wav"))
        try:
            store_wav(audio, sampling_rate, wav_file)

            return self.processor.transcribe_file(wav_file)
        finally:
            if self._clean_storage:
                os.remove(wav_file)
