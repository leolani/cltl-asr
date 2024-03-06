import logging
import os
import shutil
import tempfile
import time

import numpy as np
import requests
from cltl.combot.infra.time_util import timestamp_now

from cltl.asr.api import ASR
from cltl.asr.util import store_wav

logger = logging.getLogger(__name__)


class WhisperCppASR(ASR):
    def __init__(self, url: str, model_id: str = "base", language: str = 'en', storage: str = None):
        self._url = url
        self._model_id = model_id
        self._language = language
        self._storage = storage if storage else tempfile.mkdtemp()
        self._clean_storage = storage is None

    def clean(self):
        shutil.rmtree(self._storage)

    def speech_to_text(self, audio: np.array, sampling_rate: int) -> str:
        wav_file = str(os.path.abspath(os.path.join(self._storage, f"asr-{timestamp_now()}.wav")))
        try:
            store_wav(audio, sampling_rate, wav_file)

            start = time.time()

            with open(wav_file, 'rb') as tmp_file:
                form = {
                    'file': (wav_file.split('/')[-1], tmp_file, 'audio/wav'),
                    'response_format': 'json',
                    'language': self._language
                }
                response = requests.post(self._url, files=form)
                if not response.ok:
                    raise ValueError('Failed to transcribe audio for %s: %s (%s)', form, response.text, response.status_code)

            transcription = response.json()['text'].strip()

            logger.debug("Transcribed audio (%s sec) in %s to %s",
                         audio.shape[0]/sampling_rate, time.time() - start, transcription)

            return transcription
        finally:
            if self._clean_storage:
                os.remove(wav_file)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    with open('/Users/thomasbaier/automatic/robot/spot-woz-parent/spot-woz/py-app/storage/audio/debug/asr/asr-1709730396913.wav', 'rb') as audio:
        files = {'file': ('asr-1709730396913.wav', audio, 'audio/wav')}
        start = time.time()
        transcription = requests.post('http://127.0.0.1:8989/inference', files=files)

    print(transcription.json())
    logger.debug("Transcribed audio in %s to %s",
                 time.time() - start, transcription.json()['text'].strip())

    print(transcription.json()['text'].strip())