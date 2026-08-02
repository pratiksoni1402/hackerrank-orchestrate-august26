"""
Media processor — extracts content from images (vision) and voice notes (ASR).
Uses OpenAI gpt-4o-mini for vision and Whisper for audio transcription.
Results are cached to disk to avoid repeated API calls.
Supports async parallel processing for all media items.
"""

import asyncio
import json
import base64
import shutil
import time
from pathlib import Path

from openai import OpenAI, AsyncOpenAI

from config import (
    OPENAI_API_KEY, VISION_MODEL, WHISPER_MODEL,
    DATASET_DIR, CACHE_DIR, MAX_RETRIES, RETRY_DELAY_BASE,
    MAX_CONCURRENT_MEDIA
)


class MediaProcessor:
    """Processes images and voice notes, caching results to disk.
    Supports both sync and async (parallel) processing."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_MEDIA)
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

    # ─── Async Image Processing ─────────────────────────────────

    async def describe_image_async(self, image_id: str, file_path: str) -> dict:
        """Use gpt-4o-mini vision to describe an image (async).

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

        # Detect actual image format from magic bytes (files may have wrong extension)
        mime_type = self._detect_image_format(full_path)

        if mime_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            # Transcode unsupported formats (e.g. AVIF) to JPEG in memory
            import io
            from PIL import Image
            import pillow_avif
            try:
                with Image.open(full_path) as img:
                    img = img.convert("RGB")
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    mime_type = "image/jpeg"
            except Exception as e:
                result = {
                    "description": f"Image in unsupported format ({mime_type}) and transcoding failed: {e}",
                    "extracted_text": "",
                    "category": "unknown",
                    "risk_signals": []
                }
                self._image_cache[image_id] = result
                self._save_cache(self._image_cache, self._image_cache_path)
                return result
        else:
            # Read and encode supported image natively
            with open(full_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

        result = None
        async with self.semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    response = await self.async_client.chat.completions.create(
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
                    raw = response.choices[0].message.content
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        print(f"  ⚠ Vision API failed after {MAX_RETRIES} retries: {e}")
                        raw = None
                    else:
                        wait = RETRY_DELAY_BASE ** (attempt + 1)
                        print(f"  ⚠ Vision retry {attempt + 1}/{MAX_RETRIES} in {wait}s: {e}")
                        await asyncio.sleep(wait)

        if raw is None:
            result = {
                "description": "Failed to analyze image",
                "extracted_text": "",
                "category": "unknown",
                "risk_signals": []
            }
        else:
            try:
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

    # ─── Async Voice Note Processing ────────────────────────────

    async def transcribe_voice_note_async(self, voice_note_id: str, file_path: str) -> dict:
        """Use Whisper to transcribe a voice note (async).

        Returns:
            dict with keys: transcription, duration_hint
        """
        if voice_note_id in self._voice_cache:
            cached = self._voice_cache[voice_note_id]
            if "Failed" not in cached.get("transcription", ""):
                return cached

        full_path = DATASET_DIR / file_path
        if not full_path.exists():
            result = {"transcription": "Voice note file not found", "duration_hint": "unknown"}
            self._voice_cache[voice_note_id] = result
            self._save_cache(self._voice_cache, self._voice_cache_path)
            return result

        actual_ext = self._detect_audio_format(full_path)
        transcription = None

        async with self.semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    if full_path.suffix.lower() == actual_ext:
                        with open(full_path, "rb") as audio_file:
                            response = await self.async_client.audio.transcriptions.create(
                                model=WHISPER_MODEL,
                                file=audio_file,
                                language="en"
                            )
                        transcription = response.text
                    else:
                        # Create temp copy with correct extension
                        tmp_dir = CACHE_DIR / "tmp_audio"
                        tmp_dir.mkdir(exist_ok=True)
                        tmp_path = tmp_dir / f"{voice_note_id}{actual_ext}"
                        shutil.copy2(full_path, tmp_path)
                        try:
                            with open(tmp_path, "rb") as audio_file:
                                response = await self.async_client.audio.transcriptions.create(
                                    model=WHISPER_MODEL,
                                    file=audio_file,
                                    language="en"
                                )
                            transcription = response.text
                        finally:
                            tmp_path.unlink(missing_ok=True)
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        print(f"  ⚠ Whisper failed after {MAX_RETRIES} retries: {e}")
                        transcription = "Failed to transcribe voice note"
                    else:
                        wait = RETRY_DELAY_BASE ** (attempt + 1)
                        print(f"  ⚠ Whisper retry {attempt + 1}/{MAX_RETRIES} in {wait}s: {e}")
                        await asyncio.sleep(wait)

        if transcription is None:
            transcription = "Failed to transcribe voice note"

        result = {
            "transcription": transcription,
            "duration_hint": "short"
        }

        self._voice_cache[voice_note_id] = result
        self._save_cache(self._voice_cache, self._voice_cache_path)
        return result

    # ─── Async Batch Processing ─────────────────────────────────

    async def process_images_async(self, data_loader, console=None) -> tuple[dict, dict]:
        """Pre-process all images in parallel. Returns results and stats."""
        results = {}
        img_start = time.time()
        img_cache_hits = 0
        img_skipped = 0
        img_skipped_details = []  # list of (image_id, reason)
        img_total = len(data_loader.images_df)

        # Separate cached from uncached
        uncached_items = []
        for _, row in data_loader.images_df.iterrows():
            image_id = row["image_id"]
            file_path = row["file_path"]
            if image_id in self._image_cache:
                img_cache_hits += 1
                cached = self._image_cache[image_id]
                results[image_id] = {"type": "image", **cached}
                desc = cached.get("description", "").lower()
                if "unsupported format" in desc:
                    img_skipped += 1
                    img_skipped_details.append((image_id, cached["description"]))
                elif "not found" in desc:
                    img_skipped += 1
                    img_skipped_details.append((image_id, "File not found"))
                elif "failed to analyze" in desc:
                    img_skipped += 1
                    img_skipped_details.append((image_id, "Analysis failed"))
                if console:
                    category = cached.get('category', 'unknown')
                    formatted_category = category.replace("_", " ").title()
                    console.print(f"  [green]✓[/green] [white]{image_id}: {formatted_category}[/white] [dim](cached)[/dim]")
            else:
                uncached_items.append((image_id, file_path))

        # Process uncached in parallel
        if uncached_items:
            async def process_one(img_id, fp):
                result = await self.describe_image_async(img_id, fp)
                return img_id, result

            tasks = [process_one(img_id, fp) for img_id, fp in uncached_items]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for item in completed:
                if isinstance(item, Exception):
                    print(f"  ⚠ Image processing error: {item}")
                    continue
                img_id, result = item
                results[img_id] = {"type": "image", **result}
                desc = result.get("description", "").lower()
                if "unsupported format" in desc:
                    img_skipped += 1
                    img_skipped_details.append((img_id, result["description"]))
                elif "not found" in desc:
                    img_skipped += 1
                    img_skipped_details.append((img_id, "File not found"))
                elif "failed to analyze" in desc:
                    img_skipped += 1
                    img_skipped_details.append((img_id, "Analysis failed"))
                if console:
                    category = result.get('category', 'unknown')
                    formatted_category = category.replace("_", " ").title()
                    console.print(f"  [green]✓[/green] [white]{img_id}: {formatted_category}[/white]")

        img_time_ms = int((time.time() - img_start) * 1000)
        img_stats = {
            "total": img_total,
            "cache_hits": img_cache_hits,
            "skipped": img_skipped,
            "skipped_details": img_skipped_details,
            "time_ms": img_time_ms
        }
        return results, img_stats

    async def process_voice_notes_async(self, data_loader, console=None) -> tuple[dict, dict]:
        """Pre-process all voice notes in parallel. Returns results and stats."""
        results = {}
        vn_start = time.time()
        vn_cache_hits = 0
        vn_skipped = 0
        vn_skipped_details = []  # list of (voice_note_id, reason)
        vn_total = len(data_loader.voice_notes_df)

        # Separate cached from uncached
        uncached_items = []
        for _, row in data_loader.voice_notes_df.iterrows():
            vn_id = row["voice_note_id"]
            file_path = row["file_path"]
            if vn_id in self._voice_cache:
                cached = self._voice_cache[vn_id]
                if "Failed" not in cached.get("transcription", ""):
                    vn_cache_hits += 1
                    results[vn_id] = {"type": "voice", **cached}
                    if console:
                        console.print(f"  [green]✓[/green] [white]{vn_id}: Transcribed[/white] [dim](cached)[/dim]")
                    continue
                else:
                    vn_skipped += 1
                    vn_skipped_details.append((vn_id, cached.get("transcription", "Unknown error")))
            uncached_items.append((vn_id, file_path))

        # Process uncached in parallel
        if uncached_items:
            async def process_one(vid, fp):
                result = await self.transcribe_voice_note_async(vid, fp)
                return vid, result

            tasks = [process_one(vid, fp) for vid, fp in uncached_items]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for item in completed:
                if isinstance(item, Exception):
                    print(f"  ⚠ Voice note processing error: {item}")
                    vn_skipped += 1
                    vn_skipped_details.append(("unknown", str(item)[:80]))
                    continue
                vn_id, result = item
                results[vn_id] = {"type": "voice", **result}
                if "Failed" in result.get("transcription", ""):
                    vn_skipped += 1
                    vn_skipped_details.append((vn_id, result.get("transcription", "Unknown error")))
                if console:
                    console.print(f"  [green]✓[/green] [white]{vn_id}: Transcribed[/white]")

        vn_time_ms = int((time.time() - vn_start) * 1000)
        vn_stats = {
            "total": vn_total,
            "cache_hits": vn_cache_hits,
            "skipped": vn_skipped,
            "skipped_details": vn_skipped_details,
            "time_ms": vn_time_ms
        }
        return results, vn_stats

    # ─── Sync Methods (kept for backward compatibility) ─────────

    def describe_image(self, image_id: str, file_path: str) -> dict:
        """Use gpt-4o-mini vision to describe an image (sync).

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

        # Detect actual image format from magic bytes (files may have wrong extension)
        mime_type = self._detect_image_format(full_path)

        # AVIF and other unsupported formats can't be sent to OpenAI Vision
        if mime_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            result = {
                "description": f"Image in unsupported format ({mime_type}), cannot analyze",
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
                                    "detail": "low"
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
        """Use Whisper to transcribe a voice note (sync).

        Returns:
            dict with keys: transcription, duration_hint
        """
        if voice_note_id in self._voice_cache:
            cached = self._voice_cache[voice_note_id]
            if "Failed" not in cached.get("transcription", ""):
                return cached

        full_path = DATASET_DIR / file_path
        if not full_path.exists():
            result = {"transcription": "Voice note file not found", "duration_hint": "unknown"}
            self._voice_cache[voice_note_id] = result
            self._save_cache(self._voice_cache, self._voice_cache_path)
            return result

        actual_ext = self._detect_audio_format(full_path)

        def _call():
            if full_path.suffix.lower() == actual_ext:
                with open(full_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=audio_file,
                        language="en"
                    )
                return response.text

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
            "duration_hint": "short"
        }

        self._voice_cache[voice_note_id] = result
        self._save_cache(self._voice_cache, self._voice_cache_path)
        return result

    def process_images(self, data_loader, console=None) -> tuple[dict, dict]:
        """Pre-process all images sequentially (sync). Returns results and stats."""
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
        """Pre-process all voice notes sequentially (sync). Returns results and stats."""
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

    # ─── Utilities ──────────────────────────────────────────────

    @staticmethod
    def _detect_image_format(file_path: Path) -> str:
        """Detect actual image format from file magic bytes.

        Many dataset images have .jpg extension but are actually PNG, WebP, or AVIF.
        Returns the correct MIME type.
        """
        with open(file_path, "rb") as f:
            header = f.read(16)

        if header[:2] == b'\xff\xd8':
            return "image/jpeg"
        elif header[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        elif header[:4] == b'GIF8':
            return "image/gif"
        elif header[:4] == b'RIFF' and len(header) >= 12 and header[8:12] == b'WEBP':
            return "image/webp"
        elif len(header) >= 8 and header[4:8] == b'ftyp':
            # Could be AVIF, HEIF, or MP4 container
            return "image/avif"
        elif header[:4] in (b'II\x2a\x00', b'MM\x00\x2a'):
            return "image/tiff"
        elif header[:2] == b'BM':
            return "image/bmp"
        else:
            return "image/jpeg"  # fallback

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
