"""
Offline VAD-aware splitters
───────────────────────────
* vad_chunk_lowmem     – Low-memory chunker for non-streaming processing
  - target 60-s chunks (±10 s)
  - never cut mid-utterance (trailing silence ≥ 300 ms)
  - processes audio incrementally to minimize memory usage
  - returns List[Tuple[pathlib.Path, float]] of (temp .wav file, offset in seconds)

* vad_chunk_streaming  – Low-RAM streamer for streaming use cases
  - processes audio in small stripes (2 seconds at a time)
  - uses VADIterator to split on speech boundaries
  - returns List[Tuple[pathlib.Path, float]] of (temp .wav file, offset in seconds)
"""

from __future__ import annotations
import tempfile, wave, pathlib, numpy as np
from typing import List, Tuple
import soundfile as sf


from torch.hub import load as torch_hub_load

vad_model, vad_utils = torch_hub_load("snakers4/silero-vad", "silero_vad")
get_speech_ts, _, _, VADIterator, _ = vad_utils 

SAMPLE_RATE        = 16_000
TARGET_SEC         = 60
MAX_SEC            = 70          # never exceed this in one chunk
TRAIL_SIL_MS       = 300         # keep ≥300 ms silence at cut point
THRESH             = 0.60        # stricter prob threshold

def vad_chunk_lowmem(path: pathlib.Path) -> List[Tuple[pathlib.Path, float]]:
    """Low-memory VAD chunking for non-streaming processing
    
    Returns:
        List of (chunk_path, offset_seconds) tuples
    """
    import librosa
    
    # Get audio file info
    with sf.SoundFile(path) as snd:
        file_sr = snd.samplerate
        duration = len(snd) / file_sr
        
    # Initialize VAD iterator
    vad_iter = VADIterator(
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=THRESH,
        min_silence_duration_ms=TRAIL_SIL_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )
    
    # Buffer for current chunk
    current_chunk = bytearray()
    chunks = []  # List of (path, offset_seconds)
    speech_ms = 0
    chunk_start_sample = 0  # Track start of current chunk in samples
    total_samples_processed = 0
    
    # Process audio in chunks
    for chunk_start in range(0, int(duration * SAMPLE_RATE), STRIPE_FRAMES):
        # Load audio segment with librosa for format conversion
        y, _ = librosa.load(
            path, 
            sr=SAMPLE_RATE, 
            offset=chunk_start/SAMPLE_RATE,
            duration=STRIPE_SEC,
            mono=True
        )
        
        # Convert to int16
        audio_int16 = (y * 32767).astype(np.int16)
        
        # Process in 512-sample windows
        for i in range(0, len(audio_int16), 512):
            window = audio_int16[i:i+512]
            if len(window) < 512:
                break
                
            # Convert to float for VAD
            window_f32 = window.astype(np.float32) / 32768
            evt = vad_iter(window_f32)
            
            # Add to current chunk
            current_chunk.extend(window.tobytes())
            speech_ms += 32
            total_samples_processed += 512
            
            # Check if we should finalize chunk
            if (evt and evt.get("end")) or speech_ms >= MAX_CHUNK_MS:
                if current_chunk:
                    offset_seconds = chunk_start_sample / SAMPLE_RATE
                    chunks.append((_flush(current_chunk), offset_seconds))
                    current_chunk.clear()
                    speech_ms = 0
                    chunk_start_sample = total_samples_processed
    
    # Finalize last chunk
    if current_chunk:
        offset_seconds = chunk_start_sample / SAMPLE_RATE
        chunks.append((_flush(current_chunk), offset_seconds))
    
    return chunks

# Constants for both chunkers
STRIPE_SEC        = 2                         # read 2-second stripes
STRIPE_FRAMES     = SAMPLE_RATE * STRIPE_SEC
MAX_CHUNK_MS      = 60_000                    # hard 60s cap
SPEECH_PAD_MS     = 120                       # same as live VAD
TARGET_SEC        = 60
MAX_SEC           = 70
TRAIL_SIL_MS      = 300
THRESH            = 0.60

def _flush(buf: bytearray) -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(buf))
    return pathlib.Path(tmp.name)

def vad_chunk_streaming(path: pathlib.Path) -> List[Tuple[pathlib.Path, float]]:
    """
    Stream the file in small stripes and split on VADIterator boundaries.
    Uses the SAME PyTorch Silero model, but keeps only a few seconds in RAM.
    
    Returns:
        List of (chunk_path, offset_seconds) tuples
    """
    vad_iter = VADIterator(
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=THRESH,
        min_silence_duration_ms=TRAIL_SIL_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )

    paths, buf = [], bytearray()
    speech_ms  = 0
    chunk_start_sample = 0  # Track start of current chunk in samples
    total_samples_processed = 0

    with sf.SoundFile(path) as snd:
        while True:
            audio = snd.read(frames=STRIPE_FRAMES, dtype="int16", always_2d=False)
            if not len(audio):
                break

            # Normalise to float32 [-1,1] for VADIterator
            audio_f32 = audio.astype("float32") / 32768

            # Feed 512-sample windows
            for start in range(0, len(audio_f32), 512):
                window = audio_f32[start:start+512]
                if len(window) < 512:
                    break
                evt = vad_iter(window)
                buf.extend(audio[start:start+512].tobytes())
                speech_ms += 32
                total_samples_processed += 512

                if (evt and evt.get("end")) or speech_ms >= MAX_CHUNK_MS:
                    offset_seconds = chunk_start_sample / SAMPLE_RATE
                    paths.append((_flush(buf), offset_seconds))
                    buf.clear()
                    speech_ms = 0
                    chunk_start_sample = total_samples_processed

    if buf:
        offset_seconds = chunk_start_sample / SAMPLE_RATE
        paths.append((_flush(buf), offset_seconds))
    return paths
