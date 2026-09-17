from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import re


def normalized(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def text_tokens(value: str) -> set[str]:
    return {token for token in normalized(value).split("_") if token}


@dataclass(frozen=True)
class ProductPolicy:
    policy_id: str
    display_name: str
    target_keywords: frozenset[str]
    accepted_types: frozenset[str]
    excluded_types: frozenset[str] = field(default_factory=frozenset)
    require_floor_textile: bool = False
    require_physical_product: bool = True
    reject_collage: bool = True

    def target_hint(self) -> str:
        return ", ".join(sorted(self.target_keywords))

    def accepted_type_hint(self) -> str:
        return ", ".join(sorted(self.accepted_types))

    def excluded_type_hint(self) -> str:
        return ", ".join(sorted(self.excluded_types))

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "display_name": self.display_name,
            "target_keywords": sorted(self.target_keywords),
            "accepted_types": sorted(self.accepted_types),
            "excluded_types": sorted(self.excluded_types),
            "require_floor_textile": self.require_floor_textile,
            "require_physical_product": self.require_physical_product,
            "reject_collage": self.reject_collage,
        }


def policy_from_dict(payload: dict[str, Any]) -> ProductPolicy:
    return ProductPolicy(
        policy_id=normalized(str(payload.get("policy_id") or "dynamic-product")).replace("_", "-"),
        display_name=str(payload.get("display_name") or "Dynamic target product"),
        target_keywords=frozenset(normalized_list(payload.get("target_keywords"))),
        accepted_types=frozenset(normalized_list(payload.get("accepted_types")) or ["target_product"]),
        excluded_types=frozenset(normalized_list(payload.get("excluded_types"))),
        require_floor_textile=bool(payload.get("require_floor_textile", False)),
        require_physical_product=bool(payload.get("require_physical_product", True)),
        reject_collage=bool(payload.get("reject_collage", True)),
    )


def normalized_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = normalized(str(item))
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


AREA_RUG_TYPES = frozenset(
    {
        "area_rug",
        "runner_rug",
        "shag_rug",
        "floor_rug",
        "wool_rug",
        "washable_rug",
        "floor_carpet",
        "target_product",
    }
)

NON_RUG_TYPES = frozenset(
    {
        "doormat",
        "bath_mat",
        "wall_tapestry",
        "upholstery",
        "blanket",
        "throw_blanket",
        "quilt",
        "comforter",
        "duvet",
        "pillow",
        "pillow_cover",
        "pattern_sheet",
        "collage",
        "not_target_product",
        "not_rug",
    }
)

BLANKET_TYPES = frozenset(
    {
        "blanket",
        "throw_blanket",
        "quilt",
        "comforter",
        "duvet",
        "bedspread",
        "bedding_blanket",
        "target_product",
    }
)

NON_BLANKET_TYPES = frozenset(
    {
        "area_rug",
        "runner_rug",
        "shag_rug",
        "floor_rug",
        "floor_carpet",
        "doormat",
        "bath_mat",
        "wall_tapestry",
        "upholstery",
        "pillow",
        "pillow_cover",
        "pattern_sheet",
        "collage",
        "not_target_product",
        "not_blanket",
    }
)

PILLOW_TYPES = frozenset(
    {
        "pillow",
        "throw_pillow",
        "cushion",
        "pillow_cover",
        "cushion_cover",
        "target_product",
    }
)

NON_PILLOW_TYPES = frozenset(
    {
        "area_rug",
        "runner_rug",
        "shag_rug",
        "floor_rug",
        "floor_carpet",
        "blanket",
        "throw_blanket",
        "quilt",
        "comforter",
        "duvet",
        "wall_tapestry",
        "upholstery",
        "pattern_sheet",
        "collage",
        "not_target_product",
        "not_pillow",
    }
)


def infer_product_policy(niche: str, product_focus: str = "auto") -> ProductPolicy:
    focus = normalized(product_focus)
    niche_norm = normalized(niche)
    tokens = text_tokens(niche)
    blanket_tokens = {"blanket", "blankets", "throw", "quilt", "comforter", "duvet", "bedspread"}
    rug_tokens = {"rug", "rugs", "carpet", "carpets", "tham", "thảm"}
    pillow_tokens = {"pillow", "pillows", "cushion", "cushions"}

    if focus == "blanket" or (focus == "auto" and tokens & blanket_tokens):
        return ProductPolicy(
            policy_id="blanket",
            display_name="Blanket / throw blanket",
            target_keywords=frozenset({"blanket", "blankets", "throw blanket", "quilt", "comforter", "duvet", "bedspread"}),
            accepted_types=BLANKET_TYPES,
            excluded_types=NON_BLANKET_TYPES,
        )

    if focus == "pillow" or (focus == "auto" and tokens & pillow_tokens):
        return ProductPolicy(
            policy_id="pillow",
            display_name="Pillow / cushion",
            target_keywords=frozenset({"pillow", "pillows", "cushion", "cushions", "pillow cover", "cushion cover"}),
            accepted_types=PILLOW_TYPES,
            excluded_types=NON_PILLOW_TYPES,
        )

    if focus in {"area_rug", "any_floor_covering"} or (focus == "auto" and tokens & rug_tokens):
        return ProductPolicy(
            policy_id="area-rug" if focus != "any_floor_covering" else "any-floor-covering",
            display_name="Area rug / floor textile",
            target_keywords=frozenset({"rug", "rugs", "carpet", "carpets", "area rug", "runner rug", "floor rug"}),
            accepted_types=AREA_RUG_TYPES,
            excluded_types=NON_RUG_TYPES if focus != "any_floor_covering" else NON_RUG_TYPES - {"doormat", "bath_mat"},
            require_floor_textile=True,
        )

    keywords = frozenset(token for token in tokens if len(token) > 1) or frozenset({niche_norm or "product"})
    return ProductPolicy(
        policy_id="generic",
        display_name=f"Target product: {niche}",
        target_keywords=keywords,
        accepted_types=frozenset({"target_product", niche_norm}),
        excluded_types=frozenset({"pattern_sheet", "collage", "not_target_product"}),
    )


def generate_product_policy(
    *,
    niche: str,
    product_focus: str = "auto",
    model: str = "gemini-2.5-flash",
    backend: str = "auto",
    output_path: Path | None = None,
) -> ProductPolicy:
    target = product_target_text(niche, product_focus)
    fallback = infer_product_policy(target, product_focus)
    if output_path and output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8-sig"))
            return policy_from_dict(payload)
        except Exception:
            pass
    try:
        policy = generate_product_policy_with_gemini(target=target, niche=niche, model=model, backend=backend)
    except Exception:
        policy = fallback
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return policy


def product_target_text(niche: str, product_focus: str) -> str:
    focus = product_focus.strip()
    if focus and normalized(focus) not in {"auto", "area_rug", "any_floor_covering"}:
        return focus
    return niche.strip() or focus or "target product"


def generate_product_policy_with_gemini(*, target: str, niche: str, model: str, backend: str) -> ProductPolicy:
    from google import genai

    try:
        from ..shared.utils import env
    except Exception:
        from .utils import env

    api_key = env("GEMINI_API_KEY") or env("GOOGLE_API_KEY")
    project = env("GOOGLE_CLOUD_PROJECT")
    location = env("GOOGLE_CLOUD_LOCATION", "us-central1")
    use_enterprise = env("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in {"1", "true", "yes"}
    if backend == "api-key" or (backend == "auto" and api_key and not use_enterprise):
        client = genai.Client(api_key=api_key)
    elif backend in {"enterprise", "auto"} and project:
        client = genai.Client(vertexai=True, project=project, location=location)
    elif api_key:
        client = genai.Client(api_key=api_key)
    else:
        raise RuntimeError("Gemini credentials are not configured.")

    from google.genai import types

    prompt = f"""
Create a strict image-crawling product policy for a generic ecommerce workflow.

Target product: {target}
Search niche/query context: {niche}

Return JSON only:
{{
  "policy_id": "short-kebab-case-id",
  "display_name": "human readable product family",
  "target_keywords": ["common search/product words"],
  "accepted_types": ["snake_case_target_product_types"],
  "excluded_types": ["snake_case_non_target_product_types"],
  "require_floor_textile": false,
  "require_physical_product": true,
  "reject_collage": true
}}

Rules:
- Do not assume the target is a rug unless the target product is actually a rug/carpet.
- accepted_types should include "target_product" plus specific variants.
- excluded_types should include visually adjacent but wrong products.
- For leather bags, include handbags/tote/shoulder/crossbody/satchel variants and exclude wallets, shoes, jackets, backpacks unless target says backpack.
- For blankets, accept blankets/throws/quilts/duvets and exclude rugs/carpets.
- Keep lists concise: 4-12 items each.
""".strip()
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
    )
    payload = parse_policy_json(getattr(response, "text", "") or "")
    return policy_from_dict(payload)


def parse_policy_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.I).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Product policy response was not a JSON object.")
    return data
