from __future__ import annotations
import io, wave, tempfile, numpy as np, torch
from typing import List
from torch.hub import load as torch_hub_load

from .config import TARGET_SR, VAD_THRESHOLD, MIN_SILENCE_MS, SPEECH_PAD_MS, MAX_SPEECH_MS

vad_model, vad_utils = torch_hub_load("snakers4/silero-vad", "silero_vad")
(_, _, _, VADIterator, _) = vad_utils

# Derived constants
SAMPLE_RATE = TARGET_SR
WINDOW_SAMPLES = 512          # 32 ms frame

# Helper: float32 → int16 PCM bytes
def _f32_to_pcm16(frames: np.ndarray) -> bytes:
    return np.clip(frames * 32768, -32768, 32767).astype(np.int16).tobytes()

class StreamingVAD:
    """
    Feed successive 20–40 ms PCM frames (16 kHz, int16 mono).
    Emits temp-file *paths* when a full utterance is detected.
    """

    def __init__(self):
        self.vad = VADIterator(
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )
        self.buffer = bytearray()
        self.speech_ms = 0


    def _flush(self) -> List[str]:
        if not self.buffer:
            return []
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(self.buffer)
        self.buffer.clear()
        self.speech_ms = 0
        self.vad.reset_states()
        return [tmp.name]

    def feed(self, frame_bytes: bytes) -> List[str]:
        out: List[str] = []

        pcm_f32 = np.frombuffer(frame_bytes, np.int16).astype("float32") / 32768
        for start in range(0, len(pcm_f32), WINDOW_SAMPLES):
            window = pcm_f32[start:start + WINDOW_SAMPLES]
            if len(window) < WINDOW_SAMPLES:
                break  # wait for full 32 ms window

            voice_event = self.vad(window, return_seconds=False)
            self.buffer.extend(_f32_to_pcm16(window))
            self.speech_ms += 32

            # Flush on trailing-silence event or max-length guard
            if voice_event and voice_event.get("end"):
                out.extend(self._flush())
            elif self.speech_ms >= MAX_SPEECH_MS:
                out.extend(self._flush())

        return out
