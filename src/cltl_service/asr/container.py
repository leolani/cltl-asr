import logging
from typing import Optional

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton
from cltl_service.asr.service import AsrService

logger = logging.getLogger(__name__)


class ASRContainer(InfraContainer):
    """Container for the ASR service.

    Requires ``EmissorStorageContainer`` in the MRO to provide ``emissor_data_client``.
    ASR implementation modules are imported lazily based on the configured implementation,
    keeping heavy ML dependencies (torch, transformers, etc.) out of the import graph
    for unconfigured backends.
    """

    @property
    @singleton
    def asr_service(self) -> Optional[AsrService]:
        config = self.config_manager.get_config("cltl.asr")
        sampling_rate = config.get_int("sampling_rate")
        implementation = config.get("implementation")

        asr = self._create_asr_implementation(implementation, sampling_rate)

        if asr is None:
            logger.warning("No ASR implementation configured")
            # @singleton cannot handle None
            return False

        return AsrService.from_config(asr, self.event_bus, self.resource_manager, self.config_manager)

    def _create_asr_implementation(self, implementation: str, sampling_rate: int):
        if implementation == "google":
            from cltl.asr.google_asr import GoogleASR
            impl_config = self.config_manager.get_config("cltl.asr.google")
            return GoogleASR(impl_config.get("language"), impl_config.get_int("sampling_rate"),
                             hints=impl_config.get("hints", multi=True))
        elif implementation == "whisper":
            from cltl.asr.whisper_asr import WhisperASR
            impl_config = self.config_manager.get_config("cltl.asr.whisper")
            return WhisperASR(impl_config.get("model"), impl_config.get("language"))
        elif implementation == "speechbrain":
            from cltl.asr.speechbrain_asr import SpeechbrainASR
            impl_config = self.config_manager.get_config("cltl.asr.speechbrain")
            return SpeechbrainASR(impl_config.get("model"))
        elif implementation == "wav2vec":
            from cltl.asr.wav2vec_asr import Wav2Vec2ASR
            impl_config = self.config_manager.get_config("cltl.asr.wav2vec")
            return Wav2Vec2ASR(impl_config.get("model"), sampling_rate=sampling_rate)
        elif not implementation:
            return None
        else:
            raise ValueError("Unsupported implementation " + implementation)

    def start(self):
        logger.info("Start ASR")
        super().start()
        if self.asr_service:
            self.asr_service.start()

    def stop(self):
        if self.asr_service:
            self.asr_service.stop()
            logger.info("Stop ASR")
        super().stop()
