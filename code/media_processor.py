"""
Media processor — extracts content from images (vision) and voice notes (ASR).
Uses OpenAI gpt-4o-mini for vision and Whisper for audio transcription.
Results are cached to disk to avoid repeated API calls.
"""

import json
import base64
import time
from pathlib import Path

from openai import OpenAI

from config import (
    OPENAI_API_KEY, VISION_MODEL, WHISPER_MODEL,
    DATASET_DIR, CACHE_DIR, MAX_RETRIES, RETRY_DELAY_BASE
)


class MediaProcessor:
    """Processes images and voice notes, caching results to disk."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self._image_cache_path = CACHE_DIR / "image_descriptions.json"
        self._voice_cache_path = CACHE_DIR / "voice_transcriptions.json"
        self._image_cache = self._load_cache(self._image_cache_path)
        self._voice_cache = self._load_cache(self._voice_cache_path)

    @staticmethod
    def _load_cache(path: Path) -> dict:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache: dict, path: Path):
        with open(path, "w") as f:
            json.dump(cache, f, indent=2)

    def _call_with_retry(self, fn, *args, **kwargs):
        """Call a function with exponential backoff retry."""
        for attempt in range(MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  ⚠ API call failed after {MAX_RETRIES} retries: {e}")
                    return None
                wait = RETRY_DELAY_BASE ** (attempt + 1)
                print(f"  ⚠ Retry {attempt + 1}/{MAX_RETRIES} in {wait}s: {e}")
                time.sleep(wait)

    def describe_image(self, image_id: str, file_path: str) -> dict:
        """Use gpt-4o-mini vision to describe an image.

        Returns:
            dict with keys: description, extracted_text, category, risk_signals
        """
        if image_id in self._image_cache:
            return self._image_cache[image_id]

        full_path = DATASET_DIR / file_path
        if not full_path.exists():
            result = {
                "description": "Image file not found",
                "extracted_text": "",
                "category": "unknown",
                "risk_signals": []
            }
            self._image_cache[image_id] = result
            self._save_cache(self._image_cache, self._image_cache_path)
            return result

        # Read and encode image
        with open(full_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine MIME type
        suffix = full_path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif"}
        mime_type = mime_map.get(suffix, "image/jpeg")

        def _call():
            response = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You analyze WhatsApp images for a message notification router. "
                            "Provide a JSON response with these fields:\n"
                            '- "description": 1-2 sentence description of what the image shows\n'
                            '- "extracted_text": any readable text in the image (OCR), empty string if none\n'
                            '- "category": one of: promotion_poster, event_notice, document, screenshot, '
                            'photo, meme, scam_content, safety_warning, receipt, other\n'
                            '- "risk_signals": list of any suspicious elements (phishing URLs, fake logos, '
                            "urgency pressure, OTP requests). Empty list if none.\n"
                            "Respond ONLY with valid JSON, no markdown fences."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this WhatsApp image:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}",
                                    "detail": "low"  # low detail = cheaper
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.1
            )
            return response.choices[0].message.content

        raw = self._call_with_retry(_call)
        if raw is None:
            result = {
                "description": "Failed to analyze image",
                "extracted_text": "",
                "category": "unknown",
                "risk_signals": []
            }
        else:
            try:
                # Clean potential markdown fences
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1]
                    cleaned = cleaned.rsplit("```", 1)[0]
                result = json.loads(cleaned)
            except (json.JSONDecodeError, IndexError):
                result = {
                    "description": raw[:200],
                    "extracted_text": "",
                    "category": "unknown",
                    "risk_signals": []
                }

        self._image_cache[image_id] = result
        self._save_cache(self._image_cache, self._image_cache_path)
        return result

    def transcribe_voice_note(self, voice_note_id: str, file_path: str) -> dict:
        """Use Whisper to transcribe a voice note.

        Handles mislabeled file extensions by detecting actual format.

        Returns:
            dict with keys: transcription, duration_hint
        """
        if voice_note_id in self._voice_cache:
            # Re-process if previous attempt failed
            cached = self._voice_cache[voice_note_id]
            if "Failed" not in cached.get("transcription", ""):
                return cached

        full_path = DATASET_DIR / file_path
        if not full_path.exists():
            result = {"transcription": "Voice note file not found", "duration_hint": "unknown"}
            self._voice_cache[voice_note_id] = result
            self._save_cache(self._voice_cache, self._voice_cache_path)
            return result

        # Detect actual file format from magic bytes
        actual_ext = self._detect_audio_format(full_path)

        def _call():
            import shutil
            import tempfile

            # If extension matches actual format, send directly
            if full_path.suffix.lower() == actual_ext:
                with open(full_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=audio_file,
                        language="en"
                    )
                return response.text

            # Otherwise create a temp copy with correct extension
            tmp_dir = CACHE_DIR / "tmp_audio"
            tmp_dir.mkdir(exist_ok=True)
            tmp_path = tmp_dir / f"{voice_note_id}{actual_ext}"
            shutil.copy2(full_path, tmp_path)
            try:
                with open(tmp_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=audio_file,
                        language="en"
                    )
                return response.text
            finally:
                tmp_path.unlink(missing_ok=True)

        transcription = self._call_with_retry(_call)
        if transcription is None:
            transcription = "Failed to transcribe voice note"

        result = {
            "transcription": transcription,
            "duration_hint": "short"  # All hackathon voice notes are short
        }

        self._voice_cache[voice_note_id] = result
        self._save_cache(self._voice_cache, self._voice_cache_path)
        return result

    @staticmethod
    def _detect_audio_format(file_path: Path) -> str:
        """Detect actual audio format from file magic bytes."""
        with open(file_path, "rb") as f:
            header = f.read(12)

        # Check magic bytes
        if header[:3] == b'ID3' or (header[:2] == b'\xff\xfb'):
            return ".mp3"
        elif header[:4] == b'RIFF' and header[8:12] == b'WAVE':
            return ".wav"
        elif header[:4] == b'fLaC':
            return ".flac"
        elif header[:4] == b'OggS':
            return ".ogg"
        elif header[4:8] == b'ftyp':
            # M4A / MP4 container
            return ".m4a"
        else:
            return ".mp3"  # fallback

    def process_images(self, data_loader, console=None) -> tuple[dict, dict]:
        """Pre-process all images upfront. Returns results and stats."""
        results = {}
        img_start = time.time()
        img_cache_hits = 0
        img_total = len(data_loader.images_df)
        
        for _, row in data_loader.images_df.iterrows():
            image_id = row["image_id"]
            file_path = row["file_path"]
            
            if image_id in self._image_cache:
                img_cache_hits += 1
                
            result = self.describe_image(image_id, file_path)
            results[image_id] = {"type": "image", **result}
            
            if console:
                category = result.get('category', 'unknown')
                formatted_category = category.replace("_", " ").title()
                console.print(f"  [green]✓[/green] [white]{image_id}: {formatted_category}[/white]")
            
        img_time_ms = int((time.time() - img_start) * 1000)
        img_stats = {
            "total": img_total,
            "cache_hits": img_cache_hits,
            "time_ms": img_time_ms
        }
        return results, img_stats

    def process_voice_notes(self, data_loader, console=None) -> tuple[dict, dict]:
        """Pre-process all voice notes upfront. Returns results and stats."""
        results = {}
        vn_start = time.time()
        vn_cache_hits = 0
        vn_total = len(data_loader.voice_notes_df)
        
        for _, row in data_loader.voice_notes_df.iterrows():
            vn_id = row["voice_note_id"]
            file_path = row["file_path"]
            
            if vn_id in self._voice_cache:
                vn_cache_hits += 1
                
            result = self.transcribe_voice_note(vn_id, file_path)
            results[vn_id] = {"type": "voice", **result}
            
            if console:
                console.print(f"  [green]✓[/green] [white]{vn_id}: Transcribed[/white]")
            
        vn_stats = {
            "total": vn_total,
            "cache_hits": vn_cache_hits
        }
        return results, vn_stats
