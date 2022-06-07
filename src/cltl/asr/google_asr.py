from typing import List

import numpy as np
from google.cloud import speech_v1 as speech

from cltl.asr.api import ASR


class GoogleASR(ASR):
    MAX_ALTERNATIVES = 10

    def __init__(self, language: str, sampling_rate: int, internal_language: str = None, hints: List[str] = ()):
        self._sampling_rate = sampling_rate
        self._language = language
        self._hints = hints

        self._client = speech.SpeechClient()

    @property
    def sampling_rate(self):
        return self._sampling_rate

    def speech_to_text(self, audio: np.array, sampling_rate: int) -> str:
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1)

        request_audio = speech.RecognitionAudio(content=self._resample(audio, sampling_rate).tobytes())
        results = self._client.recognize(audio=request_audio, config=self._get_config(sampling_rate)).results
        segments = [self._get_highest_confidence(result.alternatives) for result in results]

        return " ".join(segments).strip()

    def _get_highest_confidence(self, alternatives):
        sorted_alternatives = list(sorted(alternatives, key=lambda alt: alt.confidence, reverse=True))

        return sorted_alternatives[0].transcript if sorted_alternatives else ""

    def _get_config(self, sample_rate):
        return speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            # Tip: use en-GB for better understanding of 'academic English'
            language_code=self._language,
            max_alternatives=self.MAX_ALTERNATIVES,
            enable_automatic_punctuation=True,
            # Particular words or phrases the Speech Recognition should be extra sensitive to
            speech_contexts=[speech.SpeechContext(phrases=self._hints)])

    def _resample(self, audio, sampling_rate):
        if not audio.dtype == np.int16:
            raise ValueError(f"Invalid sample depth {audio.dtype}, expected np.int16")

        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Unsupported sampling rate: {sampling_rate}, expected {self.sampling_rate}")

        if audio.ndim == 1 or audio.shape[1] == 1:
            return audio.squeeze()

        if audio.ndim > 2 or audio.shape[1] > 2:
            raise ValueError(f"audio must have at most two channels, shape was {audio.shape}")

        return audio.mean(axis=1, dtype=np.int16).ravel()