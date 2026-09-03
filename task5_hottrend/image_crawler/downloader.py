from __future__ import annotations

import io
from pathlib import Path

import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..shared.models import ImageCandidate, SearchResult, TrendPackageItem
from ..shared.utils import stable_id, valid_http_url


MAX_IMAGE_DOWNLOAD_BYTES = 18_000_000
MAX_SIDE = 1800


class Downloader:
    def __init__(self, output_dir: Path, *, timeout: int = 30, max_bytes: int = MAX_IMAGE_DOWNLOAD_BYTES):
        self.output_dir = output_dir
        self.image_dir = output_dir / "downloaded_images"
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        retry = Retry(
            total=4,
            connect=3,
            read=3,
            status=4,
            backoff_factor=0.7,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": "PinterestHotTrendDownloader/1.0", "Accept": "image/*,*/*;q=0.5"})

    def result_to_candidate(self, result: SearchResult, trend: TrendPackageItem) -> ImageCandidate:
        return ImageCandidate(
            image_id=stable_id(result.image_url),
            query=result.query,
            trend_id=result.trend_id,
            trend=result.trend,
            image_url=result.image_url,
            pin_url=result.pin_url,
            pin_id=result.pin_id,
            title=result.title,
            source=result.source,
            trend_strength=trend.trend_strength,
            semantic_fit=trend.semantic_fit,
        )

    def download(self, candidate: ImageCandidate) -> ImageCandidate:
        if not valid_http_url(candidate.image_url):
            candidate.download_error = "Invalid image URL"
            return candidate
        path = self.image_dir / f"{candidate.image_id}.jpg"
        if path.exists():
            try:
                with Image.open(path) as image:
                    candidate.width, candidate.height = image.size
                    candidate.local_path = str(path)
                    return candidate
            except Exception:
                try:
                    path.unlink()
                except OSError:
                    pass

        try:
            with self.session.get(candidate.image_url, stream=True, timeout=self.timeout) as response:
                if not 200 <= response.status_code < 300:
                    candidate.download_error = f"HTTP {response.status_code}"
                    return candidate
                content_type = (response.headers.get("content-type") or "").lower()
                if content_type and "image" not in content_type:
                    candidate.download_error = f"Unexpected content-type: {content_type}"
                    return candidate
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_content(128 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > self.max_bytes:
                        candidate.download_error = "Image exceeds byte limit"
                        return candidate
                    chunks.append(chunk)
            with Image.open(io.BytesIO(b"".join(chunks))) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            longest = max(width, height)
            if longest > MAX_SIDE:
                scale = MAX_SIDE / float(longest)
                image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
            image.save(path, "JPEG", quality=90, optimize=True)
            candidate.width, candidate.height = image.size
            candidate.local_path = str(path)
            return candidate
        except Exception as exc:
            candidate.download_error = str(exc)
            return candidate

