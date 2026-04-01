import collections
import copy
import string
from collections import deque
from typing import List, Optional, Union, Iterable

import numpy as np
import torch
from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCModel, EncDecRNNTModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.rnnt_utils import BatchedHyps
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer

from cltl.asr.api_streaming import BufferedASR, StreamTranscription


def make_divisible_by(num: int, factor: int) -> int:
    return (num // factor) * factor


class LocalParakeetRNNTStreamingASR(BufferedASR):
    """
    Local single-stream streaming ASR loop for Parakeet TDT / RNNT-style NeMo models.

    Intended model:
        nvidia/parakeet-tdt-0.6b-v3

    Notes:
    - This is for RNNT / hybrid-RNNT models, not pure CTC checkpoints.
    - Input audio must be mono and already resampled to the model sample rate.
    - `push_audio()` returns zero or more partial hypotheses.
    - `finish()` flushes the tail and returns the final hypothesis.
    - StreamTranscription.start and .end fields contain sample positions (not seconds)
      in the input audio stream, enabling accurate timestamp tracking.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cpu",
        compute_dtype: torch.dtype = torch.float32,
        chunk_secs: float = 0.5,
        left_context_secs: float = 5.0,
        right_context_secs: float = 2.0,
        turn_threshold_sec: float = 1.0,
        vad=None,
    ):
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype

        model = ASRModel.from_pretrained(model_name=model_name, map_location=self.device)
        if not isinstance(model, (EncDecRNNTModel, EncDecHybridRNNTCTCModel)):
            raise TypeError(
                f"{model_name} is not an RNNT / Hybrid RNNT model. "
                "Use this class with Parakeet TDT / RNNT style checkpoints."
            )

        if device.startswith("cpu"):
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

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

        self._turn_threshold_chunks = int(turn_threshold_sec // chunk_secs + 1)

        self._vad = vad
        self._vad_silence_threshold = self.context_samples.right if vad else None

        self.reset()

    def reset(self, keep_recent: bool = False) -> None:
        self.buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=self.context_samples,
            dtype=torch.float32,
            device=self.device,
        )
        self.pending_audio = torch.empty(0, dtype=torch.float32) if not keep_recent else self._collect_recent_audio()
        # TODO Two parameters for threshold
        self.partial_transcripts = deque([], maxlen=self._turn_threshold_chunks)
        self.current_batched_hyps: Optional[BatchedHyps] = None
        self.state = None
        self.started = False
        self.closed = False
        self._consecutive_silence_samples = 0

        self._last_encoder_output = None
        self._last_encoder_output_len = None
        self._last_encoder_context = None
        self._last_encoder_context_batch = None

        # Sample position tracking: maintain absolute positions across resets
        # when keep_recent=True (turn boundaries), reset on full reset
        if keep_recent:
            self._current_transcript_start_sample = self._total_samples_consumed
        else:
            self._total_samples_consumed = 0
            self._current_transcript_start_sample = 0

    def get_current_sample_position(self) -> int:
        """
        Get the current sample position in the audio stream.

        This represents the total number of samples that have been consumed
        by the ASR since the stream started or was last fully reset.

        Returns:
            Current sample position (number of samples)
        """
        return self._total_samples_consumed

    def _collect_recent_audio(self) -> torch.Tensor:
        """
        Return the most recent raw audio we still have access to, combining:
        - the current sliding buffer
        - any not-yet-consumed pending audio
        """
        pieces = []

        if self.buffer.samples is not None and self.buffer.samples.numel() > 0:
            pieces.append(self.buffer.samples[0].detach().cpu())

        if self.pending_audio.numel() > 0:
            pieces.append(self.pending_audio.detach().cpu())

        complete = torch.cat(pieces, dim=0) if pieces else torch.empty(0, dtype=torch.float32)

        return complete[-self.context_samples.right:].clone()

    def _normalize_input(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        audio = audio.detach().cpu().flatten()

        if audio.dtype == torch.int16:
            audio = audio.to(torch.float32) / 32768.0
        else:
            audio = audio.to(torch.float32)

        return audio

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
    def _speculative_finish(self) -> str:
        """Decode right-context frames speculatively without mutating stream state.

        Reuses the encoder output cached by the most recent `_run_step` and
        re-runs only the decoder with an extended `out_len` that includes the
        right-context frames.  Decoder state is saved before and restored after,
        so the stream can continue as if this call never happened.
        """
        if self._last_encoder_output is None:
            return self._decode_text()

        out_len = (self._last_encoder_output_len - self._last_encoder_context_batch.left)
        if out_len.item() <= 0:
            return self._decode_text()

        # RNNT hidden states are contiguous; deepcopy safely clones all tensors.
        saved_state = copy.deepcopy(self.state)
        saved_hyps = copy.deepcopy(self.current_batched_hyps)

        try:
            encoder_output = self._last_encoder_output[:, self._last_encoder_context.left:]

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

            return self._decode_text()
        finally:
            self.state = saved_state
            self.current_batched_hyps = saved_hyps

    @torch.inference_mode()
    def _run_step(self, new_audio: torch.Tensor, is_final: bool) -> str:
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

        self.buffer.add_audio_batch_(
            audio_batch=audio_batch,
            audio_lengths=audio_lengths,
            is_last_chunk=is_final,
            is_last_chunk_batch=is_last_chunk_batch,
        )

        encoder_output, encoder_output_len = self.model(
            input_signal=self.buffer.samples,
            input_signal_length=self.buffer.context_size_batch.total(),
        )

        # NeMo example converts [B, C, T] -> [B, T, C]
        encoder_output = encoder_output.transpose(1, 2)

        encoder_context = self.buffer.context_size.subsample(
            factor=self.encoder_frame2audio_samples
        )
        encoder_context_batch = self.buffer.context_size_batch.subsample(
            factor=self.encoder_frame2audio_samples
        )

        # Cache pre-slice encoder output for speculative finish
        self._last_encoder_output = encoder_output
        self._last_encoder_output_len = encoder_output_len
        self._last_encoder_context = encoder_context
        self._last_encoder_context_batch = encoder_context_batch

        # Drop left context from encoder frames and decode only current chunk.
        chunk_encoder_output = encoder_output[:, encoder_context.left:]

        out_len = torch.where(
            is_last_chunk_batch,
            encoder_output_len - encoder_context_batch.left,
            encoder_context_batch.chunk,
        )

        chunk_batched_hyps, _, self.state = self.decoding_computer(
            x=chunk_encoder_output,
            out_len=out_len,
            prev_batched_state=self.state,
            multi_biasing_ids=None,
        )

        if self.current_batched_hyps is None:
            self.current_batched_hyps = chunk_batched_hyps
        else:
            self.current_batched_hyps.merge_(chunk_batched_hyps)

        self.started = True

        return self._decode_text()

    def _try_speculative_finalize(self, current: str, results: List[StreamTranscription]) -> bool:
        """Speculatively decode right context; finalize if transcript is stable."""
        speculative_text = self._speculative_finish()
        print("XXX", current, speculative_text)

        if current.strip().strip(string.punctuation) != speculative_text.strip().strip(string.punctuation):
            return False

        # Finalize with correct sample positions
        # The speculative finish has decoded up to the current consumed position
        results.append(StreamTranscription(
            speculative_text.strip(),
            is_final=True,
            start=self._current_transcript_start_sample,
            end=self._total_samples_consumed
        ))
        self.reset(keep_recent=True)

        return True

    def _is_right_context_silent(self) -> bool:
        """Check if consecutive silence has reached the right-context threshold."""
        return (self._vad and self._vad_silence_threshold
                and self._consecutive_silence_samples >= self._vad_silence_threshold)

    def push_audio(self, audio_frames: Iterable[np.ndarray], sampling_rate: int = None) -> Iterable[StreamTranscription]:
        """
        Feed more mono audio samples. Returns zero or more partial results.

        The caller can feed arbitrary chunk sizes (e.g. 20 ms, 100 ms, 500 ms).
        Internally, decoding happens only when enough audio has accumulated.
        """
        if self.closed:
            raise RuntimeError("Stream is already closed. Call reset() for a new stream.")

        if sampling_rate and sampling_rate != self.sample_rate:
            raise ValueError(f"Sampling rate {sampling_rate} is not supported (expected {self.sample_rate}).")

        audio_frames = (audio_frames,) if isinstance(audio_frames, np.ndarray) else audio_frames

        self._detect_silence(audio_frames)
        audio = self._normalize_input(self._concat_to_mono(audio_frames))
        if audio.numel() > 0:
            self.pending_audio = torch.cat([self.pending_audio, audio], dim=0)

        results = None
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

            results = []

            # Track sample position before consuming audio
            chunk_end_sample = self._total_samples_consumed + needed

            step_audio = self.pending_audio[:needed]
            self.pending_audio = self.pending_audio[needed:]

            # Update sample position counter
            self._total_samples_consumed += needed

            current = self._run_step(step_audio, is_final=False)

            finalized = False
            if current.strip() and (
                    current.strip().endswith((".", "?", "!"))
                    or self._is_right_context_silent()):
                finalized = self._try_speculative_finalize(current, results)
            elif (len(self.partial_transcripts) == self.partial_transcripts.maxlen
                        and current.strip() and current == self.partial_transcripts[0]):
                # Turn threshold reached - finalize transcript with correct sample positions
                results.append(StreamTranscription(
                    current,
                    is_final=True,
                    start=self._current_transcript_start_sample,
                    end=self._total_samples_consumed
                ))
                finalized = True

            if not finalized:
                self.partial_transcripts.append(current)

        # Return intermediate results
        if results is not None and len(self.partial_transcripts):
            # Partial transcript - only has start position, no end yet
            results.append(StreamTranscription(
                self.partial_transcripts[-1],
                is_final=False,
                start=self._current_transcript_start_sample
            ))

        return results if results else []

    def _detect_silence(self, audio_frames):
        if self._vad is None:
            return

        for frame in audio_frames:
            if self._vad.is_vad(frame, self.sample_rate):
                self._consecutive_silence_samples = 0
            else:
                self._consecutive_silence_samples += frame.size

    def finish(self) -> StreamTranscription:
        """
        Flush the tail and close the stream.
        """
        if self.closed:
            # Already closed, return current state
            return StreamTranscription(
                text=self._decode_text(),
                is_final=True,
                start=self._current_transcript_start_sample,
                end=self._total_samples_consumed
            )

        self.closed = True

        # Case 1: no audio ever arrived
        if not self.started and self.pending_audio.numel() == 0:
            return StreamTranscription(text="", is_final=True, start=0, end=0)

        # Case 2: some tail remains -> flush it as final
        if self.pending_audio.numel() > 0:
            tail = self.pending_audio
            tail_samples = tail.numel()
            self.pending_audio = torch.empty(0, dtype=torch.float32)

            # Process tail and update position
            transcript_text = self._run_step(tail, is_final=True)
            final_end = self._total_samples_consumed + tail_samples

            return StreamTranscription(
                transcript_text,
                is_final=True,
                start=self._current_transcript_start_sample,
                end=final_end
            )

        # Case 3: no tail remains, but buffered right-context still has to be promoted
        # into the final chunk. The NeMo buffer utilities support a zero-length final add.
        empty = torch.empty(0, dtype=torch.float32)
        transcript_text = self._run_step(empty, is_final=True)

        return StreamTranscription(
            transcript_text,
            is_final=True,
            start=self._current_transcript_start_sample,
            end=self._total_samples_consumed
        )

    def _concat_to_mono(self, audio_frames):
        audio = np.concatenate(audio_frames)
        is_mono = audio.ndim == 1 or audio.shape[1] == 1

        return audio if is_mono else audio.mean(axis=1).astype(np.int16).ravel()
