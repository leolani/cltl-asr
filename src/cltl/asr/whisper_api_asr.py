import logging
import os
import shutil
import tempfile

import numpy as np
import time
from cltl.combot.infra.time_util import timestamp_now
from cltl.asr.util import sanitize_whisper_result
from openai import OpenAI

from cltl.asr.api import ASR
from cltl.asr.util import store_wav

logger = logging.getLogger(__name__)


class WhisperApiASR(ASR):
    def __init__(self, api_key: str, model_id: str = "whisper-1", language: str = 'en', storage: str = None):
        self._model_id = model_id
        self._language = language
        self._storage = storage if storage else tempfile.mkdtemp()
        self._clean_storage = storage is None

        self._openai = OpenAI(api_key=api_key)
        self._model = model_id

    def clean(self):
        shutil.rmtree(self._storage)

    def speech_to_text(self, audio: np.array, sampling_rate: int) -> str:
        wav_file = str(os.path.abspath(os.path.join(self._storage, f"asr-{timestamp_now()}.wav")))
        try:
            store_wav(audio, sampling_rate, wav_file)

            start = time.time()

            with open(wav_file, "rb") as audio_file:
                response = self._openai.audio.transcriptions.create(
                    model=self._model_id,
                    file=audio_file,
                    language=self._language
                )

            transcription = response.text.strip()

            audio_duration = audio.shape[0] / sampling_rate
            transcription = sanitize_whisper_result(audio_duration, transcription)

            logger.debug("Transcribed audio (%s sec) in %s to %s",
                         audio_duration, time.time() - start, transcription)

            return transcription
        finally:
            if self._clean_storage:
                os.remove(wav_file)
