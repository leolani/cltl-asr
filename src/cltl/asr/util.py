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


def sanitize_whisper_result(audio_duration: float, transcription: str):
    # * 6 syllables per sec * 6 letters per syllable (which should be very fast)
    #   Set a minimum audio duration to avoid edge cases
    #   https: // en.wikipedia.org / wiki / Speech_tempo
    # * Remove subtitle related transcript errors
    # * Remove meta-information related transcript errors
    # * Remove background music
    if ((audio_duration <= 1 and len(transcription) > 30)
            or (audio_duration > 1 and len(transcription) > 6 * 5 * audio_duration)
            or "TV GELDERLAND" in transcription.upper()
            or "ondertitel" in transcription.lower()
            or transcription.startswith("*") or transcription.startswith("[") or transcription.startswith("(")
            or {token for token in re.split('[^a-zA-Z]+', transcription) if token} == {"MUZIEK"}):
        logger.debug("Sanitized %s", transcription)
        return ""

    return transcription
