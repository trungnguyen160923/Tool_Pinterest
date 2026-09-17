from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..shared.cache import JsonCache
from ..shared.models import ImageCandidate, VisionResult
from ..shared.product_policy import ProductPolicy, infer_product_policy
from ..shared.utils import clamp, clamp01, env, fingerprint, truncate_text


LOG = logging.getLogger("pinterest.vision")


class ProductVisionFilter:
    def __init__(
        self,
        *,
        niche: str,
        model: str = "gemini-2.5-flash",
        backend: str = "auto",
        batch_size: int = 5,
        cache: JsonCache | None = None,
        refresh_cache: bool = False,
        mode: str = "auto",
        product_focus: str = "auto",
        product_policy: ProductPolicy | None = None,
        crawl_purpose: str = "product",
    ):
        self.niche = niche
        self.model = model
        self.backend = backend
        self.batch_size = max(1, min(8, int(batch_size)))
        self.cache = cache
        self.refresh_cache = refresh_cache
        self.mode = mode
        self.crawl_purpose = (crawl_purpose or "product").strip().lower().replace("-", "_")
        self.product_policy = product_policy or infer_product_policy(niche, product_focus)
        self.client = None if mode == "off" else self._build_client()

    def _build_client(self) -> Any:
        try:
            from google import genai
        except Exception as exc:
            if self.mode == "required":
                raise
            LOG.warning("google-genai unavailable; Vision fallback will be used: %s", exc)
            return None
        api_key = env("GEMINI_API_KEY") or env("GOOGLE_API_KEY")
        project = env("GOOGLE_CLOUD_PROJECT")
        location = env("GOOGLE_CLOUD_LOCATION", "us-central1")
        use_enterprise = env("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in {"1", "true", "yes"}
        http_options = {"timeout": 60000}
        try:
            if self.backend == "api-key" or (self.backend == "auto" and api_key and not use_enterprise):
                return genai.Client(api_key=api_key, http_options=http_options)
            if self.backend in {"enterprise", "auto"} and project:
                return genai.Client(vertexai=True, project=project, location=location, http_options=http_options)
            if api_key:
                return genai.Client(api_key=api_key, http_options=http_options)
        except Exception:
            if self.mode == "required":
                raise
        return None

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                parsed = json.loads(match.group(0))
            else:
                list_match = re.search(r"\[.*\]", text, re.S)
                if not list_match:
                    raise
                parsed = json.loads(list_match.group(0))
        if isinstance(parsed, list):
            return {"results": parsed}
        if isinstance(parsed, dict):
            if isinstance(parsed.get("results"), list):
                return parsed
            if parsed.get("image_id"):
                return {"results": [parsed]}
        raise ValueError("Gemini Vision response must be an object or a list of result objects.")

    def _fallback(self, candidate: ImageCandidate, reason: str = "Vision disabled/unavailable.") -> VisionResult:
        is_off = self.mode == "off"
        return VisionResult(
            image_id=candidate.image_id,
            accepted=is_off,
            product_present=is_off,
            product_role="PRIMARY" if is_off else "UNCERTAIN",
            product_confidence=0.85 if is_off else 0.0,
            product_visibility=85.0 if is_off else 0.0,
            trend_relevance=75.0 if is_off else 0.0,
            commercial_quality=70.0 if is_off else 0.0,
            aesthetic="unverified" if is_off else "",
            detected_product="unverified_artwork" if is_off else "",
            reason=reason,
            confidence=0.85 if is_off else 0.0,
            main_subject="pattern" if is_off else "unknown",
            target_product_type="printable_inspiration" if is_off else "",
            is_single_product=is_off,
            is_physical_product=is_off,
            is_floor_textile=is_off,
            flat_artwork_score=0.80 if is_off else 0.0,
            printability_score=0.75 if is_off else 0.0,
            reject_reason_code="" if is_off else "VISION_UNAVAILABLE",
        )

    def _prompt(self, batch: list[ImageCandidate]) -> str:
        payload = [
            {
                "image_id": item.image_id,
                "trend_id": item.trend_id,
                "trend": item.trend,
                "query": item.query,
                "title": item.title,
                "source": item.source,
            }
            for item in batch
        ]
        if self.crawl_purpose == "inspiration":
            return f"""
You are a strict TREND ARTWORK / VISUAL INSPIRATION gate for a Pinterest crawler.

Target downstream product: {self.product_policy.display_name}
Trend/niche context: {self.niche}

For every image, decide whether it is a useful visual source for creating a new printable rug/blanket artwork.
Accept images that have clear motifs, pattern direction, color palette, illustration style, composition, or texture that can transfer to a print product.
Reject images that are mostly screenshots, memes, text blocks, watermarks, brand logos, celebrity/IP characters, collage boards, blurry thumbnails, room-only photos, or images where the trend idea is not visually inspectable.

Return only JSON:
{{
  "results": [
    {{
      "image_id": "exact id",
      "accepted": true,
      "product_present": true,
      "product_role": "PRIMARY|SECONDARY|INCIDENTAL|ABSENT|UNCERTAIN",
      "main_subject": "artwork|pattern|motif|room|text|logo|collage|product|other|unknown",
      "target_product_type": "printable_inspiration|pattern|artwork|motif|texture|not_usable|unknown",
      "is_single_product": false,
      "is_physical_product": false,
      "is_floor_textile": false,
      "is_collage": false,
      "is_doormat": false,
      "is_bath_mat": false,
      "is_wall_tapestry": false,
      "motifs": ["short visual motif"],
      "source_role": "artwork_source|style_reference|product_reference|extraction_required|reject",
      "is_lifestyle_scene": false,
      "foreground_coverage": 0.9,
      "background_complexity": 0.1,
      "flat_artwork_score": 0.9,
      "printability_score": 0.9,
      "requires_extraction": false,
      "reject_reason_code": "",
      "product_confidence": 0.95,
      "product_visibility": 85,
      "trend_relevance": 80,
      "commercial_quality": 75,
      "aesthetic": "short printable style and palette",
      "detected_product": "usable visual source description",
      "reason": "visible evidence only",
      "confidence": 0.95
    }}
  ]
}}

For inspiration mode:
- product_present means a usable visual pattern/artwork is present.
- product_role PRIMARY means the motif/artwork/pattern is the main subject.
- product_visibility means motif/artwork clarity.
- commercial_quality means print/ecommerce suitability.
- Use reject_reason_code for strong rejections:
  REJECT_HANDS_OR_NAILS (hands, nails, fingers, manicure),
  REJECT_PHONE_WALLPAPER (phone frame, lockscreen clock, battery, status bar),
  REJECT_3D_ROOM_SCENE (photo of 3D interior room with perspective tilt & furniture obscuring floor),
  REJECT_TEXT_BLOCK (text quotes, word art, meme text),
  REJECT_WATERMARK (copyright stamps, watermark across art),
  REJECT_LOGO (brand logos),
  REJECT_BLURRY (low resolution or illegible),
  REJECT_COLLAGE (multi-image grid/moodboard).
- flat_artwork_score: 1.0 = completely flat 2D graphic, vector, top-down seamless repeat; 0.0 = angled 3D photo or room interior.
- printability_score: 1.0 = ready for POD direct printing onto rug/blanket; 0.0 = cluttered, dirty, occluded.
- source_role: artwork_source can be printed directly; style_reference is only for palette/style; extraction_required needs foreground separation; reject is unusable.
- A 3D room photo or angled lifestyle photo is NEVER artwork_source.

Image metadata:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

        niche_rules = f"""
Product policy:
- Target product names/keywords: {self.product_policy.target_hint()}.
- Preferred target_product_type values: {self.product_policy.accepted_type_hint()}.
- Explicit non-target product types: {self.product_policy.excluded_type_hint() or "none"}.
- Use target_product_type="target_product" only when the target niche product is visibly present but no more specific product type fits.
- main_subject is the primary visible subject, for example: target_product, room, pet, person, furniture, collage, pattern_sheet, exterior, other, unknown.
- PRIMARY means the target product itself is one of the main visual subjects and its material/shape/color/pattern is inspectable.
- SECONDARY means a clear target product is visible but the room, furniture, person, or another object is the main subject.
""".strip()
        if self.product_policy.require_floor_textile:
            niche_rules = """
Rug-specific rules:
- Analyze the image first; do not apply business policy yourself.
- target_product_type should be one of: area_rug, runner_rug, shag_rug, floor_carpet, doormat, bath_mat, wall_tapestry, upholstery, blanket, pattern_sheet, collage, not_rug, unknown.
- main_subject is the primary visible subject: rug, room, pet, person, furniture, collage, pattern_sheet, exterior, other.
- If a cat/dog/person appears only as a pattern printed on the rug, keep main_subject="rug" and put it in motifs. Do not set main_subject to cat/dog/person for printed motifs.
- If a pet/person is physically the main subject and the rug is just background, set main_subject="pet" or "person" and product_role="INCIDENTAL" or "SECONDARY".
- Set is_doormat, is_bath_mat, is_wall_tapestry, is_collage explicitly.
- PRIMARY means the rug itself is one of the main visual subjects and its pattern/shape/color is inspectable.
- SECONDARY means a clear floor rug is visible but room/furniture/pet/person is the main subject.
""".strip()

        return f"""
You are a strict PRODUCT PRESENCE gate for a Pinterest hot trend image crawler.

Target niche: {self.niche}

For every image, first decide whether a real visible product from the target niche is present.
Do not accept an image just because its room, motif, color, or style could inspire the product.
{niche_rules}

Return only JSON:
{{
  "results": [
    {{
      "image_id": "exact id",
      "accepted": true,
      "product_present": true,
      "product_role": "PRIMARY|SECONDARY|INCIDENTAL|ABSENT|UNCERTAIN",
      "main_subject": "target_product|room|pet|person|furniture|collage|pattern_sheet|exterior|other|unknown",
      "target_product_type": "specific_snake_case_type|target_product|pattern_sheet|collage|not_target_product|unknown",
      "is_single_product": true,
      "is_physical_product": true,
      "is_floor_textile": true,
      "is_collage": false,
      "is_doormat": false,
      "is_bath_mat": false,
      "is_wall_tapestry": false,
      "motifs": ["cat", "pumpkin"],
      "reject_reason_code": "",
      "product_confidence": 0.95,
      "product_visibility": 80,
      "trend_relevance": 75,
      "commercial_quality": 70,
      "aesthetic": "short visible aesthetic",
      "detected_product": "visible product description",
      "reason": "visible evidence only",
      "confidence": 0.95
    }}
  ]
}}

Score fields product_visibility, trend_relevance, commercial_quality are 0..100.
Confidence fields are 0..1.
accepted means your vision analysis believes the target product is present with PRIMARY or SECONDARY role.
Do not reject printed motifs. Policy filtering happens later.

Image metadata:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    def _call_batch(self, batch: list[ImageCandidate]) -> dict[str, VisionResult]:
        if self.client is None:
            raise RuntimeError("Gemini Vision client is not configured.")
        from google.genai import types

        parts: list[Any] = [types.Part.from_text(text=self._prompt(batch))]
        for index, candidate in enumerate(batch, start=1):
            parts.append(types.Part.from_text(text=f"IMAGE {index}: {candidate.image_id}"))
            parts.append(
                types.Part.from_bytes(
                    data=Path(candidate.local_path).read_bytes(),
                    mime_type="image/jpeg",
                )
            )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        raw = self._parse_json(getattr(response, "text", "") or "")
        output: dict[str, VisionResult] = {}
        for item in raw.get("results") or []:
            if not isinstance(item, dict):
                continue
            image_id = str(item.get("image_id") or "")
            if not image_id:
                continue
            role = str(item.get("product_role") or "UNCERTAIN").upper()
            product_present = bool(item.get("product_present"))
            accepted = bool(item.get("accepted")) and product_present and role in {"PRIMARY", "SECONDARY"}
            motifs = item.get("motifs")
            if not isinstance(motifs, list):
                motifs = []
            source_role = truncate_text(item.get("source_role"), 40).lower()
            is_lifestyle_scene = bool(item.get("is_lifestyle_scene"))
            foreground_coverage = clamp(item.get("foreground_coverage"), 0.0, 1.0)
            background_complexity = clamp(item.get("background_complexity"), 0.0, 1.0)
            flat_artwork_score = clamp(item.get("flat_artwork_score"), 0.0, 1.0)
            printability_score = clamp(item.get("printability_score"), 0.0, 1.0)
            requires_extraction = bool(item.get("requires_extraction"))
            if not source_role:
                if requires_extraction:
                    source_role = "extraction_required"
                elif is_lifestyle_scene or background_complexity >= 0.65:
                    source_role = "style_reference"
                elif flat_artwork_score >= 0.75 and foreground_coverage >= 0.55:
                    source_role = "artwork_source"
                else:
                    source_role = "unknown"
            output[image_id] = VisionResult(
                image_id=image_id,
                accepted=accepted,
                product_present=product_present,
                product_role=role,
                product_confidence=clamp01(item.get("product_confidence")),
                product_visibility=clamp(item.get("product_visibility")),
                trend_relevance=clamp(item.get("trend_relevance")),
                commercial_quality=clamp(item.get("commercial_quality")),
                aesthetic=truncate_text(item.get("aesthetic"), 220),
                detected_product=truncate_text(item.get("detected_product"), 220),
                reason=truncate_text(item.get("reason"), 650),
                confidence=clamp01(item.get("confidence"), 0.5),
                main_subject=truncate_text(item.get("main_subject"), 80).lower(),
                target_product_type=truncate_text(item.get("target_product_type"), 80).lower(),
                is_single_product=bool(item.get("is_single_product")),
                is_physical_product=bool(item.get("is_physical_product")),
                is_floor_textile=bool(item.get("is_floor_textile")),
                is_collage=bool(item.get("is_collage")),
                is_doormat=bool(item.get("is_doormat")),
                is_bath_mat=bool(item.get("is_bath_mat")),
                is_wall_tapestry=bool(item.get("is_wall_tapestry")),
                motifs=[truncate_text(value, 60).lower() for value in motifs if str(value).strip()],
                reject_reason_code=truncate_text(item.get("reject_reason_code"), 120).upper(),
                source_role=source_role,
                is_lifestyle_scene=is_lifestyle_scene,
                foreground_coverage=foreground_coverage,
                background_complexity=background_complexity,
                flat_artwork_score=flat_artwork_score,
                printability_score=printability_score,
                requires_extraction=requires_extraction,
            )
        return output

    def analyze(self, candidates: list[ImageCandidate]) -> dict[str, VisionResult]:
        results: dict[str, VisionResult] = {}
        pending: list[ImageCandidate] = []
        cache_keys: dict[str, str] = {}

        for candidate in candidates:
            key = fingerprint(
                {
                    "kind": "product-vision-v3-structured",
                    "niche": self.niche,
                    "product_policy": self.product_policy.policy_id,
                    "crawl_purpose": self.crawl_purpose,
                    "model": self.model,
                    "image_id": candidate.image_id,
                    "dhash": candidate.dhash,
                    "trend": candidate.trend,
                    "query": candidate.query,
                }
            )
            cache_keys[candidate.image_id] = key
            cached = None if self.refresh_cache or not self.cache else self.cache.get(key)
            if isinstance(cached, dict):
                results[candidate.image_id] = VisionResult(**cached)
            else:
                pending.append(candidate)

        if self.client is None:
            if self.mode == "required":
                raise RuntimeError("Gemini Vision is required but not configured.")
            for candidate in pending:
                results[candidate.image_id] = self._fallback(candidate)
            return results

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            LOG.info("Vision batch %d-%d / %d", start + 1, start + len(batch), len(pending))
            try:
                batch_results = self._call_batch(batch)
            except Exception as exc:
                LOG.warning("Vision batch failed; falling back per image: %s", exc)
                batch_results = {}
                for candidate in batch:
                    try:
                        batch_results.update(self._call_batch([candidate]))
                    except Exception as one_exc:
                        batch_results[candidate.image_id] = self._fallback(candidate, str(one_exc))

            for candidate in batch:
                result = batch_results.get(candidate.image_id) or self._fallback(candidate, "Vision omitted this image.")
                results[candidate.image_id] = result
                if self.cache and not result.error:
                    self.cache.set(cache_keys[candidate.image_id], result.__dict__)
            LOG.info("Vision batch %d-%d / %d completed.", start + 1, start + len(batch), len(pending))

        if self.cache:
            self.cache.save()
        return results
