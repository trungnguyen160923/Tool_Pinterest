from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..shared.cache import JsonCache
from ..shared.models import QuerySpec, TrendPackageItem
from ..shared.utils import clamp, clamp01, env, fingerprint, normalize_text, stable_id, truncate_text, unique
from .models import TrendCandidate


LOG = logging.getLogger("pinterest.semantic")


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


NON_TEXTILE_STOPWORDS_REGEX = re.compile(
    r"\b("
    r"nails?|nail[\s_-]*art|nail[\s_-]*tech|press[\s_-]*on[\s_-]*nails?|acrylic[\s_-]*nails?|gel[\s_-]*nails?|manicure|pedicure|"
    r"hair|hair[\s_-]*styles?|hair[\s_-]*cuts?|hair[\s_-]*color|braids?|updo|"
    r"makeup|make[\s_-]*up|lipsticks?|eye[\s_-]*shadow|mascara|lip[\s_-]*gloss|skin[\s_-]*care|eye[\s_-]*lashes?|eye[\s_-]*brows?|"
    r"wallpapers?|lock[\s_-]*screens?|phone[\s_-]*cases?|iphone[\s_-]*wallpapers?|widgets?|home[\s_-]*screens?|"
    r"outfits?|ootd|shoes|sneakers|heels|dresses|tattoos?|piercings?|jewelry|"
    r"quotes?|memes?|workout|gym|diet"
    r")\b",
    re.IGNORECASE,
)


def build_smart_queries(trend: str) -> list[QuerySpec]:
    clean = trend.strip()
    return [
        QuerySpec(query=f"{clean} surface pattern design", intent="surface_pattern", priority=1),
        QuerySpec(query=f"{clean} textile pattern flat", intent="textile_flat", priority=2),
        QuerySpec(query=f"{clean} seamless pattern vector", intent="seamless_vector", priority=3),
        QuerySpec(query=f"{clean} rug design illustration", intent="rug_illustration", priority=4),
        QuerySpec(query=f"{clean} blanket pattern design", intent="blanket_pattern", priority=5),
        QuerySpec(query=f"{clean} pattern", intent="pattern", priority=6),
        QuerySpec(query=clean, intent="trend_raw", priority=7),
    ]


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
        queries = build_smart_queries(trend)
        return TrendPackageItem(
            trend_id="trend_" + candidate.candidate_id[:12],
            trend=trend,
            trend_strength=candidate.strength,
            relationship=relationship,
            semantic_fit=semantic_fit,
            queries=queries[:6],
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
You are the visual trend intelligence layer for a Print-on-Demand (POD) Home Textiles tool (Rugs, Blankets, Throws, Surface Prints).

Target product domain: Home Decor, Floor Coverings & Bedding Textiles (Rug & Blanket prints).
Reference context / niche: {self.niche or "home decor textiles"}

For each Pinterest trend candidate:
1. Decide whether it provides strong visual, motif, or pattern inspiration for home textiles (rugs, blankets, surface pattern design).
2. Classify relationship as one of:
   DIRECT_PRODUCT, DESIGN_INSPIRATION, CONTEXTUAL_USE, AUDIENCE_ADJACENT, IRRELEVANT.
3. Score semantic_fit on 0..100:
   - visual_relevance: does the trend have rich aesthetic motifs, color palettes, textures, or repeat patterns?
   - printability_potential: can these visual elements be translated into a flat 2D surface pattern or direct print for rugs/blankets?
   - design_transferability: can this motif be printed onto large textiles (area rugs, fleece/sherpa blankets)?

REJECT explicitly (set reject=true and relationship=IRRELEVANT):
- Beauty, nails, hair, makeup, skincare, cosmetics, manicures.
- Phone wallpapers, lockscreens, device themes, tech icons.
- Personal fashion outfits (OOTD, shoes, apparel styling).
- Text-only memes, quotes, celebrity gossip, workout routines.
- Trends without distinct visual motifs or surface patterns.

PRIORITIZE:
- Surface patterns (floral, botanical, geometric, checkerboard, plaid, abstract, boho, vintage, celestial, cottagecore, seasonal/holiday decor).
- Textile prints, folk art, retro illustrations, tapestry designs.

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
        text = name.strip().lower()
        if not text:
            return "empty_trend_keyword"
        match = NON_TEXTILE_STOPWORDS_REGEX.search(text)
        if match:
            return f"Matched non-home-textile stopword: '{match.group(0)}'"
        return ""

    def _item_from_raw(self, candidate: TrendCandidate, raw: dict[str, Any]) -> tuple[TrendPackageItem, bool]:
        relationship = str(raw.get("relationship") or "IRRELEVANT").strip().upper()
        if relationship not in RELATIONSHIPS:
            relationship = "IRRELEVANT"
        semantic_fit = clamp(raw.get("semantic_fit"), default=0.0)
        reject = bool(raw.get("reject")) or relationship == "IRRELEVANT" or semantic_fit <= 0

        queries = build_smart_queries(candidate.name)

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
