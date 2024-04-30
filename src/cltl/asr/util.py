import logging
import re

import numpy as np
import soundfile
import sounddevice as sd

logger = logging.getLogger(__name__)


def store_wav(frames, sampling_rate, save=None):
    if not isinstance(frames, np.ndarray):
        audio = np.concatenate(frames)
    else:
        audio = frames
    if save:
        soundfile.write(save, audio, sampling_rate)
    else:
        sd.play(audio, sampling_rate)
        sd.wait()


def sanitize_whisper_result(audio_duration, transcription: str):
    if ((audio_duration < 1 and len(transcription) > 25)
            or "TV GELDERLAND" in transcription.upper()
            or "ondertitel" in transcription.lower()
            or transcription.startswith("*") or transcription.startswith("[") or transcription.startswith("(")
            or {token for token in re.split('[^a-zA-Z]+', transcription) if token} == {"MUZIEK"}):
        logger.debug("Sanitized %s", transcription)
        return ""

    return transcription
