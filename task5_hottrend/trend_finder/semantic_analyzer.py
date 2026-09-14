from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..shared.cache import JsonCache
from ..shared.models import QuerySpec, TrendPackageItem
from ..shared.utils import clamp, clamp01, env, fingerprint, normalize_text, stable_id, truncate_text, unique
from .models import TrendCandidate


LOG = logging.getLogger("task5_hottrend.semantic")


RELATIONSHIPS = {
    "DIRECT_PRODUCT",
    "DESIGN_INSPIRATION",
    "CONTEXTUAL_USE",
    "AUDIENCE_ADJACENT",
    "IRRELEVANT",
}

VISUAL_TREND_TERMS = {
    "art", "artwork", "aesthetic", "botanical", "floral", "flower", "garden",
    "pattern", "print", "textile", "fabric", "color", "colour", "palette",
    "geometric", "checkerboard", "checkered", "striped", "plaid", "gingham",
    "vintage", "retro", "boho", "minimal", "modern", "abstract", "illustration",
    "halloween", "pumpkin", "christmas", "holiday", "autumn", "fall", "spring",
    "summer", "winter", "beach", "coastal", "tropical", "celestial", "animal",
    "fruit", "mushroom", "butterfly", "cottagecore", "farmhouse", "decor",
}


class GeminiSemanticAnalyzer:
    def __init__(
        self,
        *,
        niche: str,
        model: str = "gemini-2.5-flash",
        backend: str = "auto",
        batch_size: int = 20,
        cache: JsonCache | None = None,
        refresh_cache: bool = False,
    ):
        self.niche = niche.strip()
        self.model = model
        self.backend = backend
        self.batch_size = max(1, min(50, int(batch_size)))
        self.cache = cache
        self.refresh_cache = refresh_cache
        self.client = self._build_client()

    def _build_client(self) -> Any:
        try:
            from google import genai
        except Exception as exc:
            LOG.warning("google-genai unavailable; using heuristic semantic fallback: %s", exc)
            return None

        api_key = env("GEMINI_API_KEY") or env("GOOGLE_API_KEY")
        project = env("GOOGLE_CLOUD_PROJECT")
        location = env("GOOGLE_CLOUD_LOCATION", "us-central1")
        use_enterprise = env("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in {"1", "true", "yes"}

        try:
            if self.backend == "api-key" or (self.backend == "auto" and api_key and not use_enterprise):
                return genai.Client(api_key=api_key)
            if self.backend in {"enterprise", "auto"} and project:
                return genai.Client(vertexai=True, project=project, location=location)
            if api_key:
                return genai.Client(api_key=api_key)
        except Exception as exc:
            LOG.warning("Could not initialize Gemini; using heuristic fallback: %s", exc)
        return None

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise
            return json.loads(match.group(0))

    def _heuristic_item(self, candidate: TrendCandidate) -> TrendPackageItem:
        text_norm = normalize_text(candidate.name)
        home_terms = {
            "home", "decor", "interior", "room", "living", "bedroom", "floor",
            "style", "vintage", "retro", "modern", "boho", "minimal", "pattern",
            "color", "textile", "cozy", "farmhouse", "mid century",
        }
        visual = any(term in text_norm for term in VISUAL_TREND_TERMS)
        contextual = any(term in text_norm for term in home_terms)

        if contextual or visual:
            relationship = "DESIGN_INSPIRATION"
            semantic_fit = 72.0 if visual else 68.0
        else:
            # Do not reject an unfamiliar category at the keyword stage. The
            # crawler and image printability gate can judge its actual visuals.
            relationship = "AUDIENCE_ADJACENT"
            semantic_fit = 50.0

        trend = candidate.name
        queries = [QuerySpec(query=trend, intent="trend", priority=1)]
        return TrendPackageItem(
            trend_id="trend_" + candidate.candidate_id[:12],
            trend=trend,
            trend_strength=candidate.strength,
            relationship=relationship,
            semantic_fit=semantic_fit,
            queries=queries[:5],
            reason="Heuristic fallback; retain the trend for image-level visual and printability review.",
            sources=[candidate.source],
            source_metrics=candidate.metrics,
            tags=[],
        )

    def _prompt(self, candidates: list[TrendCandidate]) -> str:
        payload = [
            {
                "candidate_id": item.candidate_id,
                "name": item.name,
                "source": item.source,
                "rank": item.rank,
                "strength": item.strength,
                "metrics": item.metrics,
            }
            for item in candidates
        ]
        return f"""
You are the visual trend intelligence layer for a Pinterest-to-print design tool.

The eventual product format is not a restriction. Any visually rich trend can inspire a rug, blanket, or other textile print.
Reference context only: {self.niche or "textile print"}

For each Pinterest trend candidate:
1. Decide whether it has useful visual inspiration value.
2. Classify relationship as one of:
   DIRECT_PRODUCT, DESIGN_INSPIRATION, CONTEXTUAL_USE, AUDIENCE_ADJACENT, IRRELEVANT.
3. Score semantic_fit on 0..100 using this rubric:
   - visual_relevance: does the trend imply motifs, colors, patterns, textures, or a visual aesthetic?
   - printability_potential: can those visual elements become a flat, repeatable, or full-bleed print?
   - design_transferability: can the visual language be reinterpreted into original artwork?
   Do not score down a trend merely because it comes from beauty, fashion, sports, food styling, or another category.
   Reject only when the trend is clearly non-visual, text-first, service-oriented, or has no usable visual language.
Important:
- Do not accept random pop culture/person names unless there is a clear visual/product reason.
- Keep borderline design aesthetics if they can transfer into product style, color, motif, texture, or room context.
- Seasonal themes are valid when they have clear visual motifs.
- Beauty, fashion, sports, and lifestyle trends are valid sources of visual inspiration when their imagery can transfer to print.
- Do not require any product name to appear in the trend keyword.
- Do not translate, rewrite, or generate search queries. The original Pinterest Trends API keyword is the crawl query.

Return only JSON:
{{
  "items": [
    {{
      "candidate_id": "exact id",
      "trend": "clean label",
      "relationship": "DIRECT_PRODUCT|DESIGN_INSPIRATION|CONTEXTUAL_USE|AUDIENCE_ADJACENT|IRRELEVANT",
      "semantic_fit": 86,
      "reject": false,
      "visual_relevance": 90,
      "printability_potential": 86,
      "design_transferability": 86,
      "reason": "short reason",
      "tags": ["short tags"]
    }}
  ]
}}

Candidates:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    def _call_gemini(self, candidates: list[TrendCandidate]) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Gemini client is not configured.")
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=self._prompt(candidates),
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return self._parse_json(getattr(response, "text", "") or "")

    def analyze(self, candidates: list[TrendCandidate]) -> tuple[list[TrendPackageItem], list[dict[str, Any]]]:
        by_id = {item.candidate_id: item for item in candidates}
        accepted: list[TrendPackageItem] = []
        rejected: list[dict[str, Any]] = []
        pending: list[TrendCandidate] = []

        for candidate in candidates:
            hard_reject = self._hard_reject_reason(candidate.name)
            if hard_reject:
                rejected.append({
                    "candidate_id": candidate.candidate_id,
                    "trend": candidate.name,
                    "reject": True,
                    "reason": hard_reject,
                    "filter": "deterministic_non_visual_filter",
                })
                continue
            key = fingerprint(
                {
                    "kind": "trend-semantic-v3",
                    "niche": self.niche,
                    "model": self.model,
                    "candidate": {
                        "id": candidate.candidate_id,
                        "name": candidate.name,
                        "source": candidate.source,
                        "strength": candidate.strength,
                    },
                }
            )
            cached = None if self.refresh_cache or not self.cache else self.cache.get(key)
            if isinstance(cached, dict):
                item, reject = self._item_from_raw(candidate, cached)
                if reject:
                    rejected.append(cached)
                else:
                    accepted.append(item)
                continue
            pending.append(candidate)

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            raw_items: list[dict[str, Any]]
            try:
                raw = self._call_gemini(batch)
                raw_items = [item for item in raw.get("items", []) if isinstance(item, dict)]
            except Exception as exc:
                LOG.warning("Gemini semantic batch failed; using heuristic fallback: %s", exc)
                raw_items = []
                for candidate in batch:
                    fallback = self._heuristic_item(candidate)
                    raw_items.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "trend": fallback.trend,
                            "relationship": fallback.relationship,
                            "semantic_fit": fallback.semantic_fit,
                            "reject": fallback.relationship == "IRRELEVANT",
                            "reason": fallback.reason,
                            "tags": fallback.tags,
                            "queries": [query.__dict__ for query in fallback.queries],
                        }
                    )

            seen_ids: set[str] = set()
            for raw_item in raw_items:
                candidate_id = str(raw_item.get("candidate_id") or "").strip()
                candidate = by_id.get(candidate_id)
                if candidate is None or candidate_id in seen_ids:
                    continue
                seen_ids.add(candidate_id)
                item, reject = self._item_from_raw(candidate, raw_item)
                cache_key = fingerprint(
                    {
                        "kind": "trend-semantic-v3",
                        "niche": self.niche,
                        "model": self.model,
                        "candidate": {
                            "id": candidate.candidate_id,
                            "name": candidate.name,
                            "source": candidate.source,
                            "strength": candidate.strength,
                        },
                    }
                )
                if self.cache:
                    self.cache.set(cache_key, raw_item)
                if reject:
                    rejected.append(raw_item)
                else:
                    accepted.append(item)

            for candidate in batch:
                if candidate.candidate_id in seen_ids:
                    continue
                item = self._heuristic_item(candidate)
                if item.relationship == "IRRELEVANT":
                    rejected.append({"candidate_id": candidate.candidate_id, "trend": candidate.name, "reject": True})
                else:
                    accepted.append(item)

        if self.cache:
            self.cache.save()

        accepted.sort(key=lambda item: (item.semantic_fit, item.trend_strength), reverse=True)
        for index, item in enumerate(accepted, start=1):
            item.trend_id = f"trend_{index:03d}"
        return accepted, rejected

    @staticmethod
    def _hard_reject_reason(name: str) -> str:
        # Keyword-level rejection is intentionally disabled. A source category
        # is not a reliable proxy for the quality of the images it returns.
        return ""

    def _item_from_raw(self, candidate: TrendCandidate, raw: dict[str, Any]) -> tuple[TrendPackageItem, bool]:
        relationship = str(raw.get("relationship") or "IRRELEVANT").strip().upper()
        if relationship not in RELATIONSHIPS:
            relationship = "IRRELEVANT"
        semantic_fit = clamp(raw.get("semantic_fit"), default=0.0)
        reject = bool(raw.get("reject")) or relationship == "IRRELEVANT" or semantic_fit <= 0

        queries = [QuerySpec(query=candidate.name, intent="trend", priority=1)]

        tags = raw.get("tags") or raw.get("semantic_tags") or []
        if not isinstance(tags, list):
            tags = []

        item = TrendPackageItem(
            trend_id="trend_" + stable_id(candidate.candidate_id, raw.get("trend"), length=8),
            trend=truncate_text(candidate.name, 180),
            trend_strength=candidate.strength,
            relationship=relationship,
            semantic_fit=semantic_fit,
            queries=queries,
            reason=truncate_text(raw.get("reason"), 500),
            sources=[candidate.source],
            source_metrics=candidate.metrics,
            tags=unique(tags)[:12],
        )
        return item, reject
