from __future__ import annotations

from dataclasses import dataclass

from ..shared.models import ImageCandidate, RankedImage, VisionResult
from ..shared.product_policy import ProductPolicy, infer_product_policy, normalized


@dataclass(frozen=True)
class PolicyConfig:
    niche: str = ""
    product_focus: str = "auto"
    accepted_roles: frozenset[str] = frozenset({"PRIMARY"})
    min_product_visibility: float = 75.0
    min_trend_relevance: float = 70.0
    min_score: float = 20.0
    product_policy: ProductPolicy | None = None


def target_matches_policy(vision: VisionResult, policy: ProductPolicy) -> bool:
    product_type = normalized(vision.target_product_type)
    detected = normalized(vision.detected_product)
    main_subject = normalized(vision.main_subject)
    searchable = f"{product_type} {detected} {main_subject}"
    keyword_matches = any(normalized(keyword) in searchable for keyword in policy.target_keywords)
    return product_type in policy.accepted_types or keyword_matches


def policy_reject_reason(candidate: ImageCandidate, vision: VisionResult, policy: PolicyConfig) -> str:
    role = str(vision.product_role or "").upper()
    product_type = normalized(vision.target_product_type)
    main_subject = normalized(vision.main_subject)
    product_policy = policy.product_policy or infer_product_policy(policy.niche, policy.product_focus)
    has_structured_analysis = bool(
        product_type
        or main_subject
        or vision.is_physical_product
        or vision.is_floor_textile
        or vision.is_collage
        or vision.is_doormat
        or vision.is_bath_mat
        or vision.is_wall_tapestry
        or vision.reject_reason_code
    )

    if not vision.product_present or not vision.accepted:
        return vision.reject_reason_code or "REJECT_NOT_TARGET_PRODUCT"
    if role not in policy.accepted_roles:
        return "REJECT_ROLE_NOT_ACCEPTED"
    if vision.product_visibility < policy.min_product_visibility:
        return "REJECT_LOW_VISIBILITY"
    if vision.trend_relevance < policy.min_trend_relevance:
        return "REJECT_LOW_TREND_RELEVANCE"

    if has_structured_analysis:
        if product_policy.reject_collage and vision.is_collage:
            return "REJECT_COLLAGE"
        if product_policy.require_physical_product and not vision.is_physical_product:
            return "REJECT_NOT_PHYSICAL_PRODUCT"
        if product_type in product_policy.excluded_types:
            return f"REJECT_PRODUCT_TYPE_{product_type.upper()}"
        if product_type in {"pattern_sheet", "collage"}:
            return "REJECT_NOT_SINGLE_PRODUCT"
        if product_policy.require_floor_textile:
            if vision.is_doormat and "doormat" in product_policy.excluded_types:
                return "REJECT_DOORMAT"
            if vision.is_bath_mat and "bath_mat" in product_policy.excluded_types:
                return "REJECT_BATH_MAT"
            if vision.is_wall_tapestry and "wall_tapestry" in product_policy.excluded_types:
                return "REJECT_WALL_TAPESTRY"
            if not vision.is_floor_textile:
                return "REJECT_NOT_FLOOR_TEXTILE"
        if product_type and product_type != "unknown" and not target_matches_policy(vision, product_policy):
            return f"REJECT_NOT_{product_policy.policy_id.upper().replace('-', '_')}"

        # Motifs printed on the product are allowed. Only reject when the main subject is not the product.
        if main_subject in {"pet", "person", "exterior", "poster", "pattern_sheet"} and role != "PRIMARY":
            return f"REJECT_MAIN_SUBJECT_{main_subject.upper()}"

    return ""


def score_image(candidate: ImageCandidate, vision: VisionResult, is_direct_printable: bool = False) -> float:
    score = (
        candidate.trend_strength * 0.20
        + candidate.semantic_fit * 0.12
        + vision.product_visibility * 0.18
        + vision.trend_relevance * 0.16
        + vision.commercial_quality * 0.14
        + vision.product_confidence * 100.0 * 0.05
        + vision.confidence * 100.0 * 0.05
        + (vision.flat_artwork_score * 100.0 * 0.05)
        + (vision.printability_score * 100.0 * 0.05)
    )
    if is_direct_printable:
        score += 8.0
    if vision.product_role == "SECONDARY":
        score *= 0.95
    if vision.product_role not in {"PRIMARY", "SECONDARY", "UNVERIFIED"}:
        score *= 0.45
    return round(max(0.0, min(100.0, score)), 2)


def inspiration_reject_reason(candidate: ImageCandidate, vision: VisionResult, policy: PolicyConfig) -> str:
    role = str(vision.product_role or "").upper()
    main_subject = normalized(vision.main_subject)
    product_type = normalized(vision.target_product_type)
    if not vision.accepted or not vision.product_present:
        return vision.reject_reason_code or "REJECT_NOT_PRINTABLE_INSPIRATION"
    if role not in policy.accepted_roles and role != "UNVERIFIED":
        return "REJECT_INSPIRATION_ROLE_NOT_ACCEPTED"
    if vision.product_visibility < policy.min_product_visibility:
        return "REJECT_LOW_MOTIF_CLARITY"
    if vision.trend_relevance < policy.min_trend_relevance:
        return "REJECT_LOW_TREND_RELEVANCE"
    if vision.is_collage or main_subject == "collage" or product_type == "collage":
        return "REJECT_COLLAGE"
    if vision.reject_reason_code:
        return vision.reject_reason_code
    if main_subject in {"text", "logo"} or product_type in {"not_usable", "unknown"}:
        return f"REJECT_MAIN_SUBJECT_{main_subject.upper() or product_type.upper()}"
    return ""


def rank_images(
    *,
    candidates: list[ImageCandidate],
    vision_results: dict[str, VisionResult],
    top_images: int,
    min_score: float = 20.0,
    accepted_roles: set[str] | None = None,
    min_product_visibility: float = 0.0,
    min_trend_relevance: float = 0.0,
    niche: str = "",
    product_focus: str = "auto",
    product_policy: ProductPolicy | None = None,
    crawl_purpose: str = "product",
) -> tuple[list[RankedImage], list[dict]]:
    accepted_roles = accepted_roles or {"PRIMARY", "SECONDARY", "UNVERIFIED"}
    product_policy = product_policy or infer_product_policy(niche, product_focus)
    policy = PolicyConfig(
        niche=niche,
        product_focus=product_focus,
        accepted_roles=frozenset(accepted_roles),
        min_product_visibility=min_product_visibility,
        min_trend_relevance=min_trend_relevance,
        min_score=min_score,
        product_policy=product_policy,
    )
    ranked: list[RankedImage] = []
    rejected: list[dict] = []
    inspiration_mode = (crawl_purpose or "product").strip().lower().replace("-", "_") == "inspiration"
    for candidate in candidates:
        vision = vision_results.get(candidate.image_id)
        if vision is None:
            rejected.append({"image_id": candidate.image_id, "reason": "missing_vision_result"})
            continue

        is_direct_printable = bool(
            vision.flat_artwork_score >= 0.70
            and vision.printability_score >= 0.65
            and not vision.is_lifestyle_scene
            and not vision.requires_extraction
            and not vision.is_collage
            and not bool(vision.reject_reason_code)
        )
        if vision.flat_artwork_score >= 0.75:
            classification = "Flat Pattern"
        elif vision.flat_artwork_score >= 0.55 or vision.printability_score >= 0.65:
            classification = "Printable Artwork"
        elif vision.printability_score >= 0.45:
            classification = "Texture/Inspiration"
        else:
            classification = "3D Scene/Photo"

        score = score_image(candidate, vision, is_direct_printable=is_direct_printable)
        reject_reason = (
            inspiration_reject_reason(candidate, vision, policy)
            if inspiration_mode
            else policy_reject_reason(candidate, vision, policy)
        )
        if not reject_reason and score < min_score:
            reject_reason = "REJECT_LOW_SCORE"
        if reject_reason:
            vision.policy_reject_reason = reject_reason
            rejected.append(
                {
                    "image_id": candidate.image_id,
                    "image_url": candidate.image_url,
                    "pin_url": candidate.pin_url,
                    "query": candidate.query,
                    "trend": candidate.trend,
                    "reason": reject_reason,
                    "vision_reason": vision.reason,
                    "product_present": vision.product_present,
                    "product_role": vision.product_role,
                    "product_visibility": vision.product_visibility,
                    "trend_relevance": vision.trend_relevance,
                    "main_subject": vision.main_subject,
                    "target_product_type": vision.target_product_type,
                    "is_collage": vision.is_collage,
                    "is_doormat": vision.is_doormat,
                    "is_bath_mat": vision.is_bath_mat,
                    "is_wall_tapestry": vision.is_wall_tapestry,
                    "motifs": vision.motifs,
                    "detected_product": vision.detected_product,
                    "source_role": vision.source_role,
                    "classification": classification,
                    "is_direct_printable": is_direct_printable,
                }
            )
            continue
        ranked.append(
            RankedImage(
                rank=0,
                image_id=candidate.image_id,
                image_score=score,
                trend_id=candidate.trend_id,
                trend=candidate.trend,
                query=candidate.query,
                image_url=candidate.image_url,
                local_path=candidate.local_path,
                pin_url=candidate.pin_url,
                pin_id=candidate.pin_id,
                width=candidate.width,
                height=candidate.height,
                product_role=vision.product_role,
                product_confidence=vision.product_confidence,
                product_visibility=vision.product_visibility,
                trend_relevance=vision.trend_relevance,
                commercial_quality=vision.commercial_quality,
                trend_strength=candidate.trend_strength,
                semantic_fit=candidate.semantic_fit,
                source=candidate.source,
                aesthetic=vision.aesthetic,
                detected_product=vision.detected_product,
                reason=vision.reason,
                main_subject=vision.main_subject,
                target_product_type=vision.target_product_type,
                motifs=vision.motifs,
                source_role=vision.source_role,
                is_lifestyle_scene=vision.is_lifestyle_scene,
                foreground_coverage=vision.foreground_coverage,
                background_complexity=vision.background_complexity,
                flat_artwork_score=vision.flat_artwork_score,
                printability_score=vision.printability_score,
                requires_extraction=vision.requires_extraction,
                classification=classification,
                is_direct_printable=is_direct_printable,
            )
        )
    ranked.sort(key=lambda item: item.image_score, reverse=True)
    selected = ranked[:top_images]
    for item in ranked[top_images:]:
        rejected.append(
            {
                "image_id": item.image_id,
                "image_url": item.image_url,
                "pin_url": item.pin_url,
                "query": item.query,
                "trend": item.trend,
                "reason": "NOT_SELECTED_TOP_LIMIT",
                "vision_reason": item.reason,
                "product_role": item.product_role,
                "product_visibility": item.product_visibility,
                "trend_relevance": item.trend_relevance,
                "main_subject": item.main_subject,
                "target_product_type": item.target_product_type,
                "motifs": item.motifs,
                "detected_product": item.detected_product,
                "image_score": item.image_score,
            }
        )
    for index, item in enumerate(selected, start=1):
        item.rank = index
    return selected, rejected
