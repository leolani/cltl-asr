import abc
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class StreamTranscription:
    text: str
    is_final: bool
    start: int
    end: int = None
    speaker: str = None


class BufferedASR(abc.ABC):
    def push_audio(self, audio_frames: Iterable[np.ndarray], sampling_rate: int = None) -> Iterable[StreamTranscription]:
        """
        Transcribe the provided audio sample to text.

        Parameters
        ----------
        audio_frames : Iterable[np.ndarray]
            Stream of audio frames to be subscribed.

            The provided np.ndarray elements must be either of shape (n), (n, 1) for mono
            or (n, 2) for stereo input, where n is the number of samples
            contained in an audio frame.

            Implementations may support only specific frame formats.

        sampling_rate : int
            The sampling rate of the audio frames

        Returns
        -------
        Iterable[StreamTranscription]
            Text transcripts of the audio input. The elements may represent a
            partial transcription of the head of the input stream.

        Raises
        ------
        ValueError
            If the format of the provided audio_frames is not supported.
        """
        raise NotImplementedError()

    def get_current_sample_position(self) -> int:
        """
        Get the current sample position in the audio stream.

        This represents the total number of samples that have been consumed
        by the ASR since the stream started or was last fully reset. This
        position is used for timestamp tracking and transcript filtering.

        Returns
        -------
        int
            Current sample position (number of samples consumed)

        Notes
        -----
        The default implementation returns 0, which disables position-based
        transcript filtering. Implementations should override this method
        to provide accurate sample position tracking.
        """
        return 0


class StreamingASR(abc.ABC):
    def speech_to_text(self,
                   audio_frames: Iterable[np.ndarray],
                   sampling_rate: int = None,
                   blocking: bool = True,
                   timeout: int = 0) -> Iterable[StreamTranscription]:
        """
        Transcribe the provided audio sample to text.

        Parameters
        ----------
        audio_frames : Iterable[np.ndarray]
            Stream of audio frames to be subscribed.

            The provided np.ndarray elements must be either of shape (n), (n, 1) for mono
            or (n, 2) for stereo input, where n is the number of samples
            contained in an audio frame.

            Implementations may support only specific frame formats.

        sampling_rate : int
            The sampling rate of the audio frames

        blocking : bool
            If True, the method blocks until voice activity is detected.

        timeout : float
            Maximum duration of audio frames accepted for voice activity detection
            in seconds.

        Returns
        -------
        Iterable[StreamTranscription]
            Text transcripts of the audio input. The elements in the stream may
            represent a partial transcription of the head of the input stream.

        Raises
        ------
        ValueError
            If the format of the provided audio_frames is not supported.

        ASRTimeout
            If no voice activity was detected within the specified timeout.
        """
        raise NotImplementedError()

    def get_current_sample_position(self) -> int:
        """
        Get the current sample position in the audio stream.

        This represents the total number of samples that have been consumed
        by the ASR since the stream started or was last fully reset. This
        position is used for timestamp tracking and transcript filtering.

        Returns
        -------
        int
            Current sample position (number of samples consumed)

        Notes
        -----
        The default implementation returns 0, which disables position-based
        transcript filtering. Implementations should override this method
        to provide accurate sample position tracking.
        """
        return 0