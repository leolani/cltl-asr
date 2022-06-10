from dataclasses import dataclass

from cltl.combot.event.emissor import TextSignalEvent, ConversationalAgent
from emissor.representation.container import Index
from emissor.representation.scenario import Modality, TextSignal


@dataclass
class AsrTextSignalEvent(TextSignalEvent):
    confidence: float
    audio_segment: Index

    @classmethod
    def create_asr(cls, signal: TextSignal, confidence: float, audio_segment: Index):
        TextSignalEvent.add_agent_annotation(signal, ConversationalAgent.SPEAKER)

        return cls(cls.__name__, Modality.TEXT, signal, confidence, audio_segment)