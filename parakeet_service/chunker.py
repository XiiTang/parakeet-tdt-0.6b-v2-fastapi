"""
VAD-aware audio chunkers for ASR processing.
Split audio on speech boundaries, cap chunks at MAX_CHUNK_MS.
Returns List[Tuple[pathlib.Path, float]] of (chunk path, offset seconds).
"""

from __future__ import annotations

import pathlib
import tempfile
import wave
from typing import List, Tuple

import numpy as np
import soundfile as sf
from torch.hub import load as torch_hub_load

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
MAX_CHUNK_MS = 60_000        # Hard cap per chunk
MIN_SILENCE_MS = 300         # Minimum silence to consider as cut point
SPEECH_PAD_MS = 120          # Padding around speech
VAD_THRESHOLD = 0.60         # VAD probability threshold
STRIPE_SEC = 2               # Read audio in N-second stripes
STRIPE_FRAMES = SAMPLE_RATE * STRIPE_SEC
# Avoid returning ultra-short chunks that break NeMo normalization
MIN_CHUNK_SAMPLES = 1024     # ~64 ms @16k

# Load Silero VAD model once
_vad_model, _vad_utils = torch_hub_load("snakers4/silero-vad", "silero_vad")
_, _, _, VADIterator, _ = _vad_utils


def _flush(buf: bytearray) -> pathlib.Path:
    """Write audio buffer to a temporary WAV file."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(buf))
    return pathlib.Path(tmp.name)


def _new_vad():
    """Create a new VAD iterator."""
    return VADIterator(_vad_model, sampling_rate=SAMPLE_RATE, threshold=VAD_THRESHOLD,
                       min_silence_duration_ms=MIN_SILENCE_MS, speech_pad_ms=SPEECH_PAD_MS)


def vad_chunk_lowmem(path: pathlib.Path) -> List[Tuple[pathlib.Path, float]]:
    """Low-memory VAD chunking using librosa for format conversion."""
    import librosa
    
    with sf.SoundFile(path) as snd:
        duration = len(snd) / snd.samplerate
    
    vad_iter = _new_vad()
    chunks, buf, chunk_ms, chunk_start, total = [], bytearray(), 0.0, 0, 0
    
    for stripe_start in range(0, int(duration * SAMPLE_RATE), STRIPE_FRAMES):
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, offset=stripe_start/SAMPLE_RATE,
                                duration=STRIPE_SEC, mono=True)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        for i in range(0, len(audio_int16), 512):
            window = audio_int16[i:i+512]
            if len(window) == 0:
                break

            # Pad final window for VAD, but keep real length for timestamps
            vad_window = window
            if len(window) < 512:
                vad_window = np.pad(window, (0, 512 - len(window)))

            evt = vad_iter(vad_window.astype(np.float32) / 32768)
            buf.extend(window.tobytes())

            step_ms = len(window) * 1000.0 / SAMPLE_RATE
            chunk_ms += step_ms
            total += len(window)
            
            if (evt and evt.get("end")) or chunk_ms >= MAX_CHUNK_MS:
                if len(buf) // 2 >= MIN_CHUNK_SAMPLES:
                    chunks.append((_flush(buf), chunk_start / SAMPLE_RATE))
                    buf.clear(); chunk_ms = 0.0; chunk_start = total
                
    if len(buf) // 2 >= MIN_CHUNK_SAMPLES:
        chunks.append((_flush(buf), chunk_start / SAMPLE_RATE))
    return chunks


def vad_chunk_streaming(path: pathlib.Path) -> List[Tuple[pathlib.Path, float]]:
    """Streaming VAD chunking using soundfile for efficient I/O."""
    vad_iter = _new_vad()
    chunks, buf, chunk_ms, chunk_start, total = [], bytearray(), 0.0, 0, 0
    
    with sf.SoundFile(path) as snd:
        while True:
            audio = snd.read(frames=STRIPE_FRAMES, dtype="int16", always_2d=False)
            if len(audio) == 0:
                break
            audio_f32 = audio.astype("float32") / 32768
            
            for i in range(0, len(audio_f32), 512):
                window = audio_f32[i:i+512]
                if len(window) == 0:
                    break

                vad_window = window
                if len(window) < 512:
                    vad_window = np.pad(window, (0, 512 - len(window)))

                evt = vad_iter(vad_window)
                buf.extend(audio[i:i+len(window)].tobytes())

                step_ms = len(window) * 1000.0 / SAMPLE_RATE
                chunk_ms += step_ms
                total += len(window)
                
                if (evt and evt.get("end")) or chunk_ms >= MAX_CHUNK_MS:
                    if len(buf) // 2 >= MIN_CHUNK_SAMPLES:
                        chunks.append((_flush(buf), chunk_start / SAMPLE_RATE))
                        buf.clear(); chunk_ms = 0.0; chunk_start = total
    
    if len(buf) // 2 >= MIN_CHUNK_SAMPLES:
        chunks.append((_flush(buf), chunk_start / SAMPLE_RATE))
    return chunks
