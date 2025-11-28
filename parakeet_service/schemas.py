from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TranscriptionWord(BaseModel):
    """Word-level timestamp info (OpenAI Whisper API compatible)."""
    word: str = Field(..., description="The word text.")
    start: float = Field(..., description="Start time in seconds.")
    end: float = Field(..., description="End time in seconds.")


class TranscriptionSegment(BaseModel):
    """Segment-level timestamp info (OpenAI Whisper API compatible)."""
    id: int = Field(..., description="Segment ID.")
    seek: int = Field(0, description="Seek offset.")
    start: float = Field(..., description="Start time in seconds.")
    end: float = Field(..., description="End time in seconds.")
    text: str = Field(..., description="Segment text.")
    tokens: List[int] = Field(default_factory=list, description="Token IDs.")
    temperature: float = Field(0.0, description="Sampling temperature.")
    avg_logprob: float = Field(0.0, description="Average log probability.")
    compression_ratio: float = Field(1.0, description="Compression ratio.")
    no_speech_prob: float = Field(0.0, description="No speech probability.")


class TranscriptionResponse(BaseModel):
    """OpenAI Whisper API verbose_json compatible response."""
    text: str = Field(..., description="Full transcription text.")
    task: str = Field("transcribe", description="Task type.")
    language: str = Field("en", description="Detected language.")
    duration: Optional[float] = Field(None, description="Audio duration in seconds.")
    segments: Optional[List[TranscriptionSegment]] = Field(
        None,
        description="Segment-level timestamps (OpenAI Whisper API compatible).",
    )
    words: Optional[List[TranscriptionWord]] = Field(
        None,
        description="Word-level timestamps (OpenAI Whisper API compatible).",
    )