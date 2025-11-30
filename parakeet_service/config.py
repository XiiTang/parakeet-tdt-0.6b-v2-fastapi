"""Configuration module - all settings loaded from .env file.

To change configuration, edit the .env file in the project root.
This module only reads environment variables and provides defaults.
"""
import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# Model Configuration
# =============================================================================
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "fp16")  # fp16 or fp32
DEVICE = os.getenv("DEVICE", "cuda")  # cuda or cpu
ENABLE_CUDA_COMPILE = os.getenv("ENABLE_CUDA_COMPILE", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}  # Enable torch.compile CUDA optimizations

# =============================================================================
# Batch Processing
# =============================================================================
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Increase for faster processing

# =============================================================================
# Audio Processing
# =============================================================================
TARGET_SR = int(os.getenv("TARGET_SR", "16000"))  # Model's native sample rate

# =============================================================================
# VAD (Voice Activity Detection) Configuration
# =============================================================================
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))  # Speech detection sensitivity
MIN_SILENCE_MS = int(os.getenv("MIN_SILENCE_MS", "500"))  # Silence to trigger chunk split
SPEECH_PAD_MS = int(os.getenv("SPEECH_PAD_MS", "120"))  # Padding around detected speech
MAX_CHUNK_MS = int(os.getenv("MAX_CHUNK_MS", "120000"))  # Max chunk length (2 min default)
MAX_SPEECH_MS = int(os.getenv("MAX_SPEECH_MS", "8000"))  # Streaming hard stop (8 sec)

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger("parakeet_service")
