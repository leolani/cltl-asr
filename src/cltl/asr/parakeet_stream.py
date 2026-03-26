import copy
import time
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCModel, EncDecRNNTModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.rnnt_utils import BatchedHyps
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer


import torch


def make_divisible_by(num: int, factor: int) -> int:
    return (num // factor) * factor


@dataclass
class StreamResult:
    text: str
    is_final: bool


class LocalParakeetRNNTStreamer:
    """
    Local single-stream streaming ASR loop for Parakeet TDT / RNNT-style NeMo models.

    Intended model:
        nvidia/parakeet-tdt-0.6b-v3

    Notes:
    - This is for RNNT / hybrid-RNNT models, not pure CTC checkpoints.
    - Input audio must be mono and already resampled to the model sample rate.
    - `push_audio()` returns zero or more partial hypotheses.
    - `finish()` flushes the tail and returns the final hypothesis.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cpu",
        compute_dtype: torch.dtype = torch.float32,
        chunk_secs: float = 2.0,
        left_context_secs: float = 10.0,
        right_context_secs: float = 2.0,
    ):
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype

        model = ASRModel.from_pretrained(model_name=model_name, map_location=self.device)
        if not isinstance(model, (EncDecRNNTModel, EncDecHybridRNNTCTCModel)):
            raise TypeError(
                f"{model_name} is not an RNNT / Hybrid RNNT model. "
                "Use this class with Parakeet TDT / RNNT style checkpoints."
            )

        self.model = model.eval().to(self.device)
        self.model.freeze()
        self.model.to(self.compute_dtype)

        # Match the official example's decoding constraints.
        decoding_cfg = RNNTDecodingConfig()
        decoding_cfg.strategy = "greedy_batch"
        decoding_cfg.greedy.loop_labels = True
        decoding_cfg.greedy.preserve_alignments = False
        decoding_cfg.fused_batch_size = -1
        decoding_cfg.beam.return_best_hypothesis = True
        decoding_cfg.tdt_include_token_duration = False

        if isinstance(self.model, EncDecRNNTModel):
            self.model.change_decoding_strategy(decoding_cfg)
        else:
            # Hybrid model, but decode through RNNT.
            self.model.change_decoding_strategy(decoding_cfg, decoder_type="rnnt")

        self.decoding_computer = self.model.decoding.decoding.decoding_computer

        preproc = copy.deepcopy(self.model._cfg.preprocessor)
        # Same streaming assumptions as the official example.
        if str(preproc.normalize) != "per_feature":
            raise ValueError(
                "This loop follows NeMo's streaming RNNT example, which expects "
                "models trained with per_feature normalization."
            )

        self.sample_rate = int(preproc.sample_rate)
        feature_stride_sec = float(preproc.window_stride)
        self.encoder_subsampling_factor = int(self.model.encoder.subsampling_factor)

        features_per_sec = 1.0 / feature_stride_sec
        self.features_frame2audio_samples = make_divisible_by(
            int(self.sample_rate * feature_stride_sec),
            factor=self.encoder_subsampling_factor,
        )
        self.encoder_frame2audio_samples = (
            self.features_frame2audio_samples * self.encoder_subsampling_factor
        )

        self.context_encoder_frames = ContextSize(
            left=int(left_context_secs * features_per_sec / self.encoder_subsampling_factor),
            chunk=int(chunk_secs * features_per_sec / self.encoder_subsampling_factor),
            right=int(right_context_secs * features_per_sec / self.encoder_subsampling_factor),
        )
        self.context_samples = ContextSize(
            left=self.context_encoder_frames.left
            * self.encoder_subsampling_factor
            * self.features_frame2audio_samples,
            chunk=self.context_encoder_frames.chunk
            * self.encoder_subsampling_factor
            * self.features_frame2audio_samples,
            right=self.context_encoder_frames.right
            * self.encoder_subsampling_factor
            * self.features_frame2audio_samples,
        )

        self.reset()

    def reset(self) -> None:
        self.buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=self.context_samples,
            dtype=torch.float32,
            device=self.device,
        )
        self.pending_audio = torch.empty(0, dtype=torch.float32)
        self.current_batched_hyps: Optional[BatchedHyps] = None
        self.state = None
        self.started = False
        self.closed = False

    def _normalize_input(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        audio = audio.detach().cpu().flatten()

        if audio.dtype == torch.int16:
            audio = audio.to(torch.float32) / 32768.0
        else:
            audio = audio.to(torch.float32)

        return audio

    def profile_encoder_once(self, samples, lengths):
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=False,
                profile_memory=False,
        ) as prof:
            with torch.inference_mode():
                processed_signal, processed_signal_len = self.model.preprocessor(
                    input_signal=samples,
                    length=lengths,
                )
                encoder_output, encoder_output_len = self.model.encoder(
                    audio_signal=processed_signal,
                    length=processed_signal_len,
                )

        print(
            prof.key_averages().table(
                sort_by="self_cpu_time_total",
                row_limit=30,
            )
        )

    def _decode_text(self) -> str:
        if self.current_batched_hyps is None:
            return ""

        length = int(self.current_batched_hyps.current_lengths[0].item())
        if length <= 0:
            return ""

        token_ids = (
            self.current_batched_hyps.transcript[0, :length]
            .detach()
            .cpu()
            .tolist()
        )
        return self.model.tokenizer.ids_to_text(token_ids)

    @torch.inference_mode()
    def _run_step(self, new_audio: torch.Tensor, is_final: bool) -> StreamResult:
        """
        Adds one piece of new audio to the streaming buffer and runs one decode step.

        Important:
        - For normal streaming steps:
            * first step adds (chunk + right_context)
            * later steps add (chunk)
        - For the final flush:
            * add any remaining tail with is_final=True
            * if there is no remaining tail, pass zero new audio with is_final=True
              to promote buffered right-context into the final chunk.
        """
        new_audio = new_audio.to(self.device)
        audio_batch = new_audio.unsqueeze(0)  # [1, T]
        audio_lengths = torch.tensor([new_audio.numel()], dtype=torch.long, device=self.device)
        is_last_chunk_batch = torch.tensor([is_final], dtype=torch.bool, device=self.device)

        from time import perf_counter

        t0 = perf_counter()

        self.buffer.add_audio_batch_(
            audio_batch=audio_batch,
            audio_lengths=audio_lengths,
            is_last_chunk=is_final,
            is_last_chunk_batch=is_last_chunk_batch,
        )
        t1 = perf_counter()

        samples = self.buffer.samples
        lengths = self.buffer.context_size_batch.total()

        processed_signal, processed_signal_len = self.model.preprocessor(
            input_signal=samples,
            length=lengths,
        )
        t2 = perf_counter()

        encoder_output, encoder_output_len = self.model.encoder(
            audio_signal=processed_signal,
            length=processed_signal_len,
        )
        t3 = perf_counter()

        print(
            f"buf={samples.shape[1] / self.sample_rate:6.2f}s "
            f"(L/C/R={self.context_samples.left / self.sample_rate:.2f}/"
            f"{self.context_samples.chunk / self.sample_rate:.2f}/"
            f"{self.context_samples.right / self.sample_rate:.2f}s) | "
            f"in_len={int(lengths[0].item()):7d} smp | "
            f"feat_len={int(processed_signal_len[0].item()):5d} frm | "
            f"enc_len={int(encoder_output_len[0].item()):5d} frm | "
            f"buffer={t1 - t0:6.3f}s preproc={t2 - t1:6.3f}s encoder={t3 - t2:6.3f}s total={t3 - t0:6.3f}s"
        )

        if samples.shape[1] / self.sample_rate > 7.4:
            self.profile_encoder_once(samples, lengths)

        # NeMo example converts [B, C, T] -> [B, T, C]
        encoder_output = encoder_output.transpose(1, 2)

        encoder_context = self.buffer.context_size.subsample(
            factor=self.encoder_frame2audio_samples
        )
        encoder_context_batch = self.buffer.context_size_batch.subsample(
            factor=self.encoder_frame2audio_samples
        )

        # Drop left context from encoder frames and decode only current chunk.
        encoder_output = encoder_output[:, encoder_context.left:]

        out_len = torch.where(
            is_last_chunk_batch,
            encoder_output_len - encoder_context_batch.left,
            encoder_context_batch.chunk,
        )

        chunk_batched_hyps, _, self.state = self.decoding_computer(
            x=encoder_output,
            out_len=out_len,
            prev_batched_state=self.state,
            multi_biasing_ids=None,
        )

        if self.current_batched_hyps is None:
            self.current_batched_hyps = chunk_batched_hyps
        else:
            self.current_batched_hyps.merge_(chunk_batched_hyps)

        self.started = True

        return StreamResult(text=self._decode_text(), is_final=is_final)

    def push_audio(self, audio: Union[np.ndarray, torch.Tensor]) -> List[StreamResult]:
        """
        Feed more mono audio samples. Returns zero or more partial results.

        The caller can feed arbitrary chunk sizes (e.g. 20 ms, 100 ms, 500 ms).
        Internally, decoding happens only when enough audio has accumulated.
        """
        if self.closed:
            raise RuntimeError("Stream is already closed. Call reset() for a new stream.")

        audio = self._normalize_input(audio)
        if audio.numel() > 0:
            self.pending_audio = torch.cat([self.pending_audio, audio], dim=0)

        results: List[StreamResult] = []

        while True:
            # Match the official script:
            # first decode step needs chunk + right_context,
            # subsequent non-final steps need one new chunk.
            needed = (
                self.context_samples.chunk + self.context_samples.right
                if not self.started
                else self.context_samples.chunk
            )

            if self.pending_audio.numel() < needed:
                break

            step_audio = self.pending_audio[:needed]
            self.pending_audio = self.pending_audio[needed:]

            results.append(self._run_step(step_audio, is_final=False))

        return results

    def finish(self) -> StreamResult:
        """
        Flush the tail and close the stream.
        """
        if self.closed:
            return StreamResult(text=self._decode_text(), is_final=True)

        self.closed = True

        # Case 1: no audio ever arrived
        if not self.started and self.pending_audio.numel() == 0:
            return StreamResult(text="", is_final=True)

        # Case 2: some tail remains -> flush it as final
        if self.pending_audio.numel() > 0:
            tail = self.pending_audio
            self.pending_audio = torch.empty(0, dtype=torch.float32)
            return self._run_step(tail, is_final=True)

        # Case 3: no tail remains, but buffered right-context still has to be promoted
        # into the final chunk. The NeMo buffer utilities support a zero-length final add.
        empty = torch.empty(0, dtype=torch.float32)
        return self._run_step(empty, is_final=True)