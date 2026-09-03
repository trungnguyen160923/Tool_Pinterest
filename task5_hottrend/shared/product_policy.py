from __future__ import annotations

from dataclasses import dataclass, field


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

    if focus in {"area_rug", "any_floor_covering"} or tokens & {"rug", "rugs", "carpet", "carpets", "tham", "thảm"}:
        return ProductPolicy(
            policy_id="area-rug" if focus != "any_floor_covering" else "any-floor-covering",
            display_name="Area rug / floor textile",
            target_keywords=frozenset({"rug", "rugs", "carpet", "carpets", "area rug", "runner rug", "floor rug"}),
            accepted_types=AREA_RUG_TYPES,
            excluded_types=NON_RUG_TYPES if focus != "any_floor_covering" else NON_RUG_TYPES - {"doormat", "bath_mat"},
            require_floor_textile=True,
        )

    if tokens & {"blanket", "blankets", "throw", "quilt", "comforter", "duvet", "bedspread"}:
        return ProductPolicy(
            policy_id="blanket",
            display_name="Blanket / throw blanket",
            target_keywords=frozenset({"blanket", "blankets", "throw blanket", "quilt", "comforter", "duvet", "bedspread"}),
            accepted_types=BLANKET_TYPES,
            excluded_types=NON_BLANKET_TYPES,
        )

    if tokens & {"pillow", "pillows", "cushion", "cushions"}:
        return ProductPolicy(
            policy_id="pillow",
            display_name="Pillow / cushion",
            target_keywords=frozenset({"pillow", "pillows", "cushion", "cushions", "pillow cover", "cushion cover"}),
            accepted_types=PILLOW_TYPES,
            excluded_types=NON_PILLOW_TYPES,
        )

    keywords = frozenset(token for token in tokens if len(token) > 1) or frozenset({niche_norm or "product"})
    return ProductPolicy(
        policy_id="generic",
        display_name=f"Target product: {niche}",
        target_keywords=keywords,
        accepted_types=frozenset({"target_product", niche_norm}),
        excluded_types=frozenset({"pattern_sheet", "collage", "not_target_product"}),
    )
