import logging
import uuid
from typing import Callable

import numpy as np
from cltl.backend.api.storage import STORAGE_SCHEME
from cltl.backend.source.client_source import ClientAudioSource
from cltl.backend.spi.audio import AudioSource
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from cltl_service.emissordata.client import EmissorDataClient
from cltl_service.vad.schema import VadMentionEvent
from emissor.representation.container import Index, TemporalRuler
from emissor.representation.scenario import Modality, TextSignal

from cltl.asr.api import ASR
from cltl_service.asr.schema import AsrTextSignalEvent

logger = logging.getLogger(__name__)


CONTENT_TYPE_SEPARATOR = ';'


class AsrService:
    @classmethod
    def from_config(cls, asr: ASR, emissor_data: EmissorDataClient,
                    event_bus: EventBus, resource_manager: ResourceManager, config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.asr")
        buffer = config.get_int("buffer") if "buffer" in config else 0
        gap_timeout = config.get_int("gap_timeout") / 1000 if "gap_timeout" in config else 0

        def audio_loader(url, offset, length) -> AudioSource:
            return ClientAudioSource.from_config(config_manager, url, offset, length)

        return cls(config.get("vad_topic"), config.get("asr_topic"), asr, gap_timeout, buffer,
                   emissor_data, audio_loader, event_bus, resource_manager)

    def __init__(self, vad_topic: str, asr_topic: str, asr: ASR, gap_timeout: float, buffer: int,
                 emissor_data: EmissorDataClient, audio_loader: Callable[[str, int, int], AudioSource],
                 event_bus: EventBus, resource_manager: ResourceManager):
        """
        Service to create TextSignals from voice activity detections.

        Parameters
        ----------
        vad_topic: str
            Input topic for voice activity events
        asr_topic: str
            Output topic for text signal events
        asr: ASR
            ASR implementation
        gap_timeout: float
            Allow merging of voice activity events if the contiunation of an utterance by the speaker is expected
            by :py:class:`~cltl.asr.api.ASR`. This is signaled by the transcript ending in
            :py:const:`~cltl.asr.api.ASR.GAP_INDICATOR`. If set to 0, continuation signals will be ignored.
        buffer: int
            Number of events buffered during event processing. If set to 0, events that arrive during processing of will
            be dropped. Note: if a positive gap_timeout is used and buffer is set to 0, internally still a buffer will
            be created to ensure continuation events are not lost. Content of this buffer will be dropped if currently
            no continuation is expected and subsequent invocations of the process method are instantaneous.
        emissor_data: EmissorDataClient
            client to retrieve emissor data
        audio_loader: Callable[[str, int, int], AudioSource]
            Callable that provides an AudioSource to access raw audio referenced in VAD events
        event_bus: EventBus
            Event bus of the application
        resource_manager: ResourceManager
            ResourceManager of the application
        """
        self._asr = asr
        self._emissor_data = emissor_data
        self._audio_loader = audio_loader
        self._event_bus = event_bus
        self._resource_manager = resource_manager
        self._vad_topic = vad_topic
        self._asr_topic = asr_topic

        self._gap_timeout = gap_timeout
        self._buffer = buffer
        self._transcript = []
        self._mentions_transcript = []

        self._last_event = timestamp_now()

        self._topic_worker = None

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        # If gap_timeout is configured, still add a buffer to catch continuation events
        buffer_size = self._buffer if self._gap_timeout == 0 else max(self._buffer, 4)
        self._topic_worker = TopicWorker([self._vad_topic], self._event_bus, provides=[self._asr_topic],
                                         resource_manager=self._resource_manager, processor=self._process,
                                         buffer_size=buffer_size, name=self.__class__.__name__,
                                         interval=self._gap_timeout)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event[VadMentionEvent]):
        if self._gap_timeout > 0 and self._buffer == 0:
            # Manually drop events that arrived during processing, but only if we don't expect continuation
            # We consider 10ms as instantaneous invocation
            event_buffered_during_execution = timestamp_now() - self._last_event < 10
            if event_buffered_during_execution and not self._transcript:
                self._last_event = timestamp_now()
                return

        transcript = None
        if event is not None:
            transcript = self._transcribe(event)
            self._mentions_transcript.append(event.payload.mentions[0])
            if transcript:
                self._transcript.append(transcript)

        if (event is None and not self._transcript) or (event is not None and transcript is None):
            # Ignore scheduled invocations if there is no transcript waiting for continuation and empty VAD detections
            pass
        elif self._transcript and transcript == "":
            logger.debug("Ignore empty transcript while waiting for continuation of %s (%s)", self._transcript[-1], event.id)
        elif self._gap_timeout and event is not None and self._transcript and self._transcript[-1].endswith(ASR.GAP_INDICATOR):
            # Ignore empty transcripts while waiting for continuation
            logger.debug("Partially transcribed event %s to %s", event.id, self._transcript[-1])
        else:
            # Full (potentially empty) utterance or gap timeout reached
            asr_event = self._create_payload()
            self._event_bus.publish(self._asr_topic, Event.for_payload(asr_event))
            logger.info("Transcribed event %s to %s %s", event.id, asr_event.signal.text,
                        f"({self._transcript})" if len(self._transcript) > 1 else "")

            self._transcript = []
            self._mentions_transcript = []

        self._last_event = timestamp_now()

    def _transcribe(self, event: Event[VadMentionEvent]):
        payload = event.payload
        # Ignore empty VAD events
        if not payload.mentions or not payload.mentions[0].segment:
            logger.info("No speech recognized in event %s", event.id)
            return None

        segment: Index = payload.mentions[0].segment[0]
        # Ignore empty audio
        if segment.stop == segment.start:
            logger.info("No speech recognized in event %s", event.id)
            return None

        url = f"{STORAGE_SCHEME}:{Modality.AUDIO.name.lower()}/{segment.container_id}"

        with self._audio_loader(url, segment.start, segment.stop - segment.start) as source:
            return self._asr.speech_to_text(np.concatenate(tuple(source.audio)), source.rate)

    def _create_payload(self):
        scenario_id = self._emissor_data.get_scenario_for_id(self._mentions_transcript[0].id)
        signal_id = str(uuid.uuid4())
        transcript = " ".join(self._strip(part) for part in self._transcript)
        segments = [segment for mention in self._mentions_transcript for segment in mention.segment]

        signal = TextSignal(signal_id, Index.from_range(signal_id, 0, len(transcript)), list(transcript), Modality.TEXT,
                            TemporalRuler(scenario_id, timestamp_now(), timestamp_now()), [], [], transcript)

        return AsrTextSignalEvent.create_asr(signal, 1.0, segments)

    def _strip(self, text):
        text = text[len(ASR.GAP_INDICATOR):] if text.startswith(ASR.GAP_INDICATOR) else text
        text = text[:-len(ASR.GAP_INDICATOR)] if text.endswith(ASR.GAP_INDICATOR) else text

        return text.strip()