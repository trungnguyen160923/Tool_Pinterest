from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from .models import SimilarityResult


LOG = logging.getLogger("task6_image_similarity.gemini")
BASE_DIR = Path(__file__).resolve().parents[1]


def load_env_file(path: Path = BASE_DIR / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def build_client(backend: str = "auto") -> Any:
    load_env_file()
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("google-genai is not installed. Run: pip install google-genai") from exc

    api_key = env("GEMINI_API_KEY") or env("GOOGLE_API_KEY")
    project = env("GOOGLE_CLOUD_PROJECT")
    location = env("GOOGLE_CLOUD_LOCATION", "us-central1")
    use_enterprise = env("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in {"1", "true", "yes"}

    if backend == "api-key" or (backend == "auto" and api_key and not use_enterprise):
        return genai.Client(api_key=api_key)
    if backend in {"enterprise", "auto"} and project:
        return genai.Client(vertexai=True, project=project, location=location)
    if api_key:
        return genai.Client(api_key=api_key)
    raise RuntimeError("Gemini credentials not found. Set GEMINI_API_KEY/GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT.")


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def parse_json(text: str) -> dict[str, Any]:
    text = strip_json_fence(text)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"results": value}
    raise json.JSONDecodeError("Could not parse JSON object from Gemini response", text, 0)


def save_raw_response(raw_output_dir: Path | None, batch_index: int, text: str) -> None:
    if raw_output_dir is None:
        return
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    (raw_output_dir / f"gemini_raw_batch_{batch_index:03d}.txt").write_text(text, encoding="utf-8")


def image_part(path: Path):
    from google.genai import types

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def prompt_for_batch(candidates: list[SimilarityResult]) -> str:
    payload = [
        {
            "image_id": item.image_id,
            "filename": item.filename,
            "local_score": item.score,
            "trend": (item.metadata or {}).get("trend", ""),
            "query": (item.metadata or {}).get("query", ""),
        }
        for item in candidates
    ]
    return f"""
You are reranking image search results by direct visual similarity to the QUERY image.

Judge what a human would see, not keyword/theme similarity. Prioritize same scene, same main objects, same composition, same camera angle, same blanket/fabric appearance, same room/furniture, and same crop. Penalize images that merely share a broad topic such as blanket, Halloween, bedroom, living room, or fall decor.

Scoring guide:
- 95-100: exact image or near-identical duplicate.
- 85-94: same scene/product with small crop, resize, rotation, lighting, or compression changes.
- 70-84: clearly same main subject and very similar composition, but not the same source image.
- 50-69: same product category or visual style, noticeably different image.
- 0-49: weak or unrelated visual match.

Return one result for every candidate. Use the exact image_id values. Keep reason short, plain, and without quotation marks.

Candidate metadata:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def response_schema():
    from google.genai import types

    return types.Schema(
        type=types.Type.OBJECT,
        required=["results"],
        properties={
            "results": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    required=["image_id", "gemini_score", "match_level", "reason"],
                    properties={
                        "image_id": types.Schema(type=types.Type.STRING),
                        "gemini_score": types.Schema(type=types.Type.NUMBER, minimum=0, maximum=100),
                        "match_level": types.Schema(
                            type=types.Type.STRING,
                            enum=["exact", "near_duplicate", "strong", "medium", "weak", "unrelated"],
                        ),
                        "reason": types.Schema(type=types.Type.STRING),
                    },
                ),
            ),
        },
    )


def clamp_score(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except Exception:
        return 0.0


def local_fallback(item: SimilarityResult, reason: str) -> SimilarityResult:
    metadata = dict(item.metadata or {})
    metadata.setdefault("local_score", item.score)
    metadata.setdefault("gemini_score", "")
    metadata.setdefault("gemini_match_level", "fallback")
    metadata.setdefault("gemini_reason", reason)
    item.metadata = metadata
    return item


def rerank_with_gemini(
    *,
    query_path: Path,
    results: list[SimilarityResult],
    model: str,
    backend: str = "auto",
    batch_size: int = 5,
    raw_output_dir: Path | None = None,
) -> list[SimilarityResult]:
    if not results:
        return []

    from google.genai import types

    client = build_client(backend)
    batch_size = max(1, min(8, int(batch_size)))
    scored_by_id: dict[str, dict[str, Any]] = {}
    failed_ids: set[str] = set()

    for batch_index, start in enumerate(range(0, len(results), batch_size), start=1):
        batch = results[start : start + batch_size]
        LOG.info("Gemini reranking candidates %s-%s/%s", start + 1, start + len(batch), len(results))
        parts: list[Any] = [types.Part.from_text(text=prompt_for_batch(batch))]
        parts.append(types.Part.from_text(text="QUERY IMAGE"))
        parts.append(image_part(query_path))
        for index, item in enumerate(batch, start=1):
            parts.append(types.Part.from_text(text=f"CANDIDATE {index}: {item.image_id} / {item.filename}"))
            parts.append(image_part(Path(item.path)))

        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=response_schema(),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        raw_text = getattr(response, "text", "") or "{}"
        try:
            payload = parse_json(raw_text)
        except json.JSONDecodeError as exc:
            save_raw_response(raw_output_dir, batch_index, raw_text)
            LOG.warning(
                "Gemini returned malformed JSON for batch %s; keeping local scores for this batch. Raw response saved in %s. Error: %s",
                batch_index,
                raw_output_dir or "not saved",
                exc,
            )
            failed_ids.update(item.image_id for item in batch)
            continue

        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            image_id = str(item.get("image_id") or "")
            if not image_id:
                continue
            scored_by_id[image_id] = {
                "gemini_score": clamp_score(item.get("gemini_score")),
                "gemini_match_level": str(item.get("match_level") or "").strip(),
                "gemini_reason": str(item.get("reason") or "").strip(),
            }

    reranked: list[SimilarityResult] = []
    for item in results:
        gemini = scored_by_id.get(item.image_id)
        if not gemini:
            reason = "Gemini rerank unavailable for this candidate; kept local score."
            if item.image_id in failed_ids:
                reason = "Gemini returned malformed JSON for this batch; kept local score."
            reranked.append(local_fallback(item, reason))
            continue
        metadata = dict(item.metadata or {})
        metadata["local_score"] = item.score
        metadata.update(gemini)
        item.score = round(float(gemini["gemini_score"]), 2)
        item.metadata = metadata
        reranked.append(item)

    reranked.sort(key=lambda item: item.score, reverse=True)
    for rank, item in enumerate(reranked, start=1):
        item.rank = rank
    return reranked