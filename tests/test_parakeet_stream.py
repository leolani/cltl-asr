import time

import soundfile as sf
import torch

from cltl.asr.parakeet_stream import LocalParakeetRNNTStreamer

streamer = LocalParakeetRNNTStreamer(
    model_name="nvidia/parakeet-tdt-0.6b-v3",
    device="cpu",
    compute_dtype=torch.float32,
    chunk_secs=0.1,
    left_context_secs=0.1,
    right_context_secs=0.1,
)

audio, sr = sf.read("resources/mic_sample2.wav", dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)

if sr != streamer.sample_rate:
    raise ValueError(f"Resample first: file is {sr} Hz, model expects {streamer.sample_rate} Hz")

# Simulate a live source feeding 100 ms packets.
packet = int(0.1 * sr)

start = time.time()
for i in range(0, len(audio), packet):
    for result in streamer.push_audio(audio[i : i + packet]):
        print("PARTIAL:", result.text, round(i / sr, 2), round(time.time() - start, 2))

final_result = streamer.finish()
print("FINAL:", final_result.text)