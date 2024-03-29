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

        def audio_loader(url, offset, length) -> AudioSource:
            return ClientAudioSource.from_config(config_manager, url, offset, length)

        return cls(config.get("vad_topic"), config.get("asr_topic"), asr, emissor_data, audio_loader,
                   event_bus, resource_manager)

    def __init__(self, vad_topic: str, asr_topic: str, asr: ASR, emissor_data: EmissorDataClient,
                 audio_loader: Callable[[str, int, int], AudioSource],
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._asr = asr
        self._emissor_data = emissor_data
        self._audio_loader = audio_loader
        self._event_bus = event_bus
        self._resource_manager = resource_manager
        self._vad_topic = vad_topic
        self._asr_topic = asr_topic

        self._topic_worker = None

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._vad_topic], self._event_bus, provides=[self._asr_topic],
                                         resource_manager=self._resource_manager, processor=self._process,
                                         buffer_size=0, name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event[VadMentionEvent]):
        payload = event.payload
        segment: Index = payload.mentions[0].segment[0]

        # Ignore empty audio
        if segment.stop == segment.start:
            return

        url = f"{STORAGE_SCHEME}:{Modality.AUDIO.name.lower()}/{segment.container_id}"

        with self._audio_loader(url, segment.start, segment.stop - segment.start) as source:
            transcript = self._asr.speech_to_text(np.concatenate(tuple(source.audio)), source.rate)

        if transcript and self._accept_transcript(transcript):
            asr_event = self._create_payload(transcript, payload)
            self._event_bus.publish(self._asr_topic, Event.for_payload(asr_event))

    def _create_payload(self, transcript, payload):
        signal_id = str(uuid.uuid4())
        scenario_id = self._emissor_data.get_scenario_for_id(payload.mentions[0].id)
        signal = TextSignal(signal_id, Index.from_range(signal_id, 0, len(transcript)), list(transcript), Modality.TEXT,
                            TemporalRuler(scenario_id, timestamp_now(), timestamp_now()), [], [], transcript)

        return AsrTextSignalEvent.create_asr(signal, 1.0, payload.mentions[0].segment)

    def _accept_transcript(self, transcript):
        return (len(transcript) > 1 or transcript.lower() in ['i'])
