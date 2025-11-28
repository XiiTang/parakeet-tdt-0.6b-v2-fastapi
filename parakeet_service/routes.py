from __future__ import annotations
import asyncio
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status, Request, Form

from .audio import ensure_mono_16k, schedule_cleanup
from .model import _to_builtin
from .schemas import TranscriptionResponse, TranscriptionSegment, TranscriptionWord
from .config import logger

from parakeet_service.model import reset_fast_path
from parakeet_service.chunker import vad_chunk_lowmem, vad_chunk_streaming


router = APIRouter(tags=["speech"])


@router.get("/healthz", summary="Liveness/readiness probe")
def health():
    return {"status": "ok"}


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
)
@router.post(
    "/audio/transcriptions",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
)
@router.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file (OpenAI-compatible endpoint)",
)
async def transcribe_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., media_type="audio/*"),
    # OpenAI Whisper API compatible parameters
    model: str = Form("parakeet-tdt-0.6b-v2", description="Model name (ignored, uses parakeet)"),
    language: str = Form("en", description="Language code"),
    prompt: str = Form("", description="Initial prompt for transcription"),
    response_format: str = Form("verbose_json", description="Response format: json, text, srt, verbose_json, vtt"),
    timestamp_granularities: List[str] = Form(default=[], description="Timestamp granularities: word, segment"),
    # Legacy parameters for backward compatibility
    include_timestamps: bool = Form(
        False, description="Return char/word/segment offsets (legacy parameter)",
    ),
    should_chunk: bool = Form(True,
        description="If true (default), split long audio into "
                    "~60s VAD-aligned chunks for batching"),
):
    # Determine if timestamps are needed based on response_format or timestamp_granularities
    need_timestamps = (
        include_timestamps 
        or response_format == "verbose_json" 
        or len(timestamp_granularities) > 0
    )
    
    logger.info(f"Transcription request: model={model}, language={language}, response_format={response_format}, "
                f"timestamp_granularities={timestamp_granularities}, need_timestamps={need_timestamps}")
    
    # Create temp file with appropriate extension
    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
    
    # Stream upload directly to processing with cancellation handling
    try:
        # Use FFmpeg for MP3 files to fix header issues
        # Create temp MP3 file if needed
        mp3_tmp_path = None
        if suffix.lower() == ".mp3":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp:
                mp3_tmp_path = Path(mp3_tmp.name)
            # Write entire MP3 to temporary file
            with open(mp3_tmp_path, "wb") as f:
                while True:
                    try:
                        chunk = await file.read(8192)
                    except asyncio.CancelledError:
                        logger.warning("File upload cancelled during MP3 saving")
                        raise
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Update FFmpeg command to read from file
            ffmpeg_cmd = [
                "ffmpeg", "-v", "error", "-nostdin", "-y",
                "-i", str(mp3_tmp_path),
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                "-f", "wav", str(tmp_path)
            ]
        else:
            ffmpeg_cmd = None
            # For non-MP3, stream directly to file
            with open(tmp_path, "wb") as f:
                while True:
                    try:
                        chunk = await file.read(8192)
                    except asyncio.CancelledError:
                        logger.warning("File upload cancelled during processing")
                        raise
                    if not chunk:
                        break
                    f.write(chunk)
        
        # Run FFmpeg if processing MP3
        if ffmpeg_cmd:
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.DEVNULL,  # We don't need stdout
                stderr=asyncio.subprocess.PIPE
            )
            
            # Read stderr in real-time
            stderr_lines = []
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                stderr_lines.append(line_str)
                logger.debug(f"FFmpeg: {line_str}")
            
            # Wait for process to finish
            return_code = await process.wait()
            stderr_str = "\n".join(stderr_lines)
            
            if return_code != 0:
                logger.error(f"FFmpeg failed with return code {return_code}")
                logger.error(f"FFmpeg error output: {stderr_str}")
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Invalid audio format: {stderr_str[:200]}"
                )
            else:
                logger.debug(f"FFmpeg completed successfully")
    except asyncio.CancelledError:
        # Clean up temporary files if processing was cancelled
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise
    except BrokenPipeError:
        logger.error("FFmpeg process terminated unexpectedly")
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio processing failed due to FFmpeg crash"
        )
    finally:
        await file.close()

    # Process audio to ensure mono 16kHz
    original, to_model = ensure_mono_16k(tmp_path)

    if should_chunk:
        # Use low-memory chunker for non-streaming requests
        # Returns List[Tuple[pathlib.Path, float]] - (chunk_path, offset_seconds)
        chunk_data = vad_chunk_lowmem(to_model) or [(to_model, 0.0)]
        chunk_paths = [p for p, _ in chunk_data]
        chunk_offsets = [offset for _, offset in chunk_data]
    else:
        chunk_paths = [to_model]
        chunk_offsets = [0.0]

    logger.info("transcribe(): sending %d chunks to ASR (offsets: %s)", 
                len(chunk_paths), [f"{o:.1f}s" for o in chunk_offsets])

    # Clean up all temporary files
    cleanup_files = [original, to_model] + chunk_paths
    if mp3_tmp_path:
        cleanup_files.append(mp3_tmp_path)
    schedule_cleanup(background_tasks, *cleanup_files)

    # 2 – run ASR
    asr_model = request.app.state.asr_model

    try:
        outs = asr_model.transcribe(
            [str(p) for p in chunk_paths],
            batch_size=2,
            timestamps=need_timestamps,
        )
        if (
          not need_timestamps                     # switch back to model fast-path if timestamps turned off
          and getattr(asr_model.cfg.decoding, "compute_timestamps", False)
        ):
          reset_fast_path(asr_model)                    
    except RuntimeError as exc:
        logger.exception("ASR failed")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc

    if isinstance(outs, tuple):
      outs = outs[0]
    texts = []
    merged = defaultdict(list)

    for idx, h in enumerate(outs):
        texts.append(getattr(h, "text", str(h)))
        if need_timestamps:
            # Get the time offset for this chunk (in seconds)
            chunk_offset = chunk_offsets[idx] if idx < len(chunk_offsets) else 0.0
            
            for k, v in _to_builtin(getattr(h, "timestamp", {})).items():
                # Add chunk offset to each timestamp
                for item in v:
                    if isinstance(item, dict):
                        if 'start' in item:
                            item['start'] = item['start'] + chunk_offset
                        if 'end' in item:
                            item['end'] = item['end'] + chunk_offset
                merged[k].extend(v)

    merged_text = " ".join(texts).strip()
    
    # Convert NeMo timestamps to OpenAI Whisper API format
    segments: Optional[List[TranscriptionSegment]] = None
    words: Optional[List[TranscriptionWord]] = None
    
    if need_timestamps and merged:
        # Build segments from NeMo's 'segment' or fall back to 'word' data
        if 'segment' in merged and merged['segment']:
            segments = []
            for i, seg in enumerate(merged['segment']):
                segments.append(TranscriptionSegment(
                    id=i,
                    seek=0,
                    start=float(seg.get('start', 0)),
                    end=float(seg.get('end', 0)),
                    text=seg.get('segment', '').strip(),
                    tokens=[],
                    temperature=0.0,
                    avg_logprob=0.0,
                    compression_ratio=1.0,
                    no_speech_prob=0.0,
                ))
        elif 'word' in merged and merged['word']:
            # If no segments, create a single segment from all words
            word_list = merged['word']
            if word_list:
                full_text = ' '.join(w.get('word', '') for w in word_list).strip()
                segments = [TranscriptionSegment(
                    id=0,
                    seek=0,
                    start=float(word_list[0].get('start', 0)),
                    end=float(word_list[-1].get('end', 0)),
                    text=full_text,
                    tokens=[],
                    temperature=0.0,
                    avg_logprob=0.0,
                    compression_ratio=1.0,
                    no_speech_prob=0.0,
                )]
        
        # Build words list
        if 'word' in merged and merged['word']:
            words = []
            for w in merged['word']:
                words.append(TranscriptionWord(
                    word=w.get('word', ''),
                    start=float(w.get('start', 0)),
                    end=float(w.get('end', 0)),
                ))

    return TranscriptionResponse(
        text=merged_text,
        task="transcribe",
        language="en",
        duration=None,
        segments=segments,
        words=words,
    )

@router.get("/debug/cfg")
def show_cfg(request: Request):
    from omegaconf import OmegaConf
    model = request.app.state.asr_model         
    yaml_str = OmegaConf.to_yaml(model.cfg, resolve=True) 
    return yaml_str