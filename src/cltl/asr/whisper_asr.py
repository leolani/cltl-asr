import logging
import numpy as np
import os
import shutil
import tempfile
import time
import whisper
from cltl.combot.infra.time_util import timestamp_now

from cltl.asr.api import ASR
from cltl.asr.util import store_wav, sanitize_whisper_result

logger = logging.getLogger(__name__)


class WhisperASR(ASR):
    def __init__(self, model_id: str = "base", language: str = 'en', storage: str = None):
        self._model = whisper.load_model(model_id)
        self._language = language
        self._storage = storage if storage else tempfile.mkdtemp()
        self._clean_storage = storage is None

    def clean(self):
        shutil.rmtree(self._storage)

    def speech_to_text(self, audio: np.array, sampling_rate: int) -> str:
        wav_file = str(os.path.join(self._storage, f"asr-{timestamp_now()}.wav"))
        try:
            store_wav(audio, sampling_rate, wav_file)

            start = time.time()
            transcription = self._model.transcribe(wav_file, fp16=False, language=self._language, task='transcribe')

            audio_duration = audio.shape[0] / sampling_rate
            transcription_text = sanitize_whisper_result(audio_duration, transcription['text'])

            logger.debug("Transcribed audio (%s sec) in %s to %s",
                         audio_duration, time.time() - start, transcription)

            return transcription_text
        finally:
            if self._clean_storage:
                os.remove(wav_file)
