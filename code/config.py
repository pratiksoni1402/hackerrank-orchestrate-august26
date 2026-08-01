"""
Configuration module — reads API keys from environment variables and defines constants.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# --- API Keys ---
OPENAI_API_KEY = os.environ.get("OPEN_AI_API_KEY", "")

# --- Model Configuration ---
# gpt-4o-mini: ~$0.15/M input, $0.60/M output — very cost efficient
ROUTING_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"          # supports vision at same price
WHISPER_MODEL = "whisper-1"           # $0.006/minute of audio

# --- Paths ---
DATASET_DIR = _project_root / "dataset"
MEDIA_DIR = DATASET_DIR / "media"
IMAGES_DIR = MEDIA_DIR / "images"
AUDIO_DIR = MEDIA_DIR / "audio"
OUTPUT_PATH = DATASET_DIR / "output.csv"

# --- Cache ---
CACHE_DIR = _project_root / "code" / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# --- Constants ---
ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_MESSAGE_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown"
}

OUTPUT_COLUMNS = [
    "message_id", "action", "message_type",
    "reason", "confidence", "evidence_message_ids"
]

# --- Rate Limiting ---
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds, exponential backoff
BATCH_SIZE = 5        # concurrent API calls
