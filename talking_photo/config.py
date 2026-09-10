"""Application configuration constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = PROJECT_ROOT / "temp"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

APP_NAME = "AI Talking Photo"
MAX_TEXT_LENGTH = 3000
MIN_IMAGE_SIZE = 256
DEFAULT_VOICE = "台灣女聲"
DEFAULT_RATE = 0.95
LOW_VRAM_THRESHOLD_GB = 2.5
