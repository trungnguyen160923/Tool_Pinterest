from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from .config import ProductTarget
from .crawler import CandidateImage
from .mockup_profile import MockupProductProfile
from .product_asset import create_gemini_client, extract_response_text, image_part, is_transient_gemini_error, parse_json_relaxed


@dataclass(frozen=True)
class PrintabilityDecision:
    stage: str
    source_path: Path
    accepted: bool
    reason: str
    metrics: dict[str, object]
    assessment: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_candidate(candidate: CandidateImage, target: ProductTarget, *, backend: str, model: str) -> PrintabilityDecision:
    role = _source_role(candidate)
    try:
        with Image.open(candidate.path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
    except Exception as exc:
        return PrintabilityDecision("candidate", candidate.path, False, f"unreadable image: {exc}", {}, {})

    metrics = {
        "width": width,
        "height": height,
        "long_edge": max(width, height),
        "source_role": role,
    }
    if role not in {"artwork_source", "style_reference", "extraction_required"}:
        return PrintabilityDecision(
            "candidate",
            candidate.path,
            False,
            f"source_role={role} is not reusable artwork.",
            metrics,
            {},
        )
    if max(width, height) < 512:
        return PrintabilityDecision("candidate", candidate.path, False, "reference resolution is below 512 px on the long edge.", metrics, {})

    try:
        assessment = _vision_assessment(image, candidate_prompt(target), backend=backend, model=model)
    except Exception as exc:
        return PrintabilityDecision("candidate", candidate.path, False, f"candidate vision assessment failed: {exc}", metrics, {})

    score = _percentage(assessment.get("visual_reusability_score"))
    usable_motif = _bool(assessment.get("has_usable_motif"))
    # This is a reference, not the print file. Text, watermarks, and a room
    # scene are generation constraints, not automatic disqualifiers here.
    accepted = score >= 70 and usable_motif
    reason = str(assessment.get("reason") or "candidate meets the visual-reference rubric.")
    if not accepted:
        reason = f"candidate rejected: {reason}"
    return PrintabilityDecision("candidate", candidate.path, accepted, reason, metrics | {"visual_reusability_score": score}, assessment)


def assess_final_artwork(
    print_path: Path,
    native_artwork_path: Path,
    target: ProductTarget,
    *,
    backend: str,
    model: str,
    require_repeat_seams: bool,
) -> PrintabilityDecision:
    try:
        with Image.open(print_path) as opened:
            image = opened.convert("RGB")
        with Image.open(native_artwork_path) as opened:
            native_size = opened.size
    except Exception as exc:
        return PrintabilityDecision("final", print_path, False, f"unreadable artwork: {exc}", {}, {})

    width, height = image.size
    native_long_edge = max(native_size)
    contrast = float(ImageStat.Stat(image.convert("L").resize((128, 128))).stddev[0])
    metrics = {
        "width": width,
        "height": height,
        "native_width": native_size[0],
        "native_height": native_size[1],
        "native_long_edge": native_long_edge,
        "luminance_stddev": round(contrast, 2),
    }
    if (width, height) != (target.width_px, target.height_px):
        return PrintabilityDecision("final", print_path, False, "print canvas does not match the target dimensions.", metrics, {})
    if native_long_edge < 1024:
        return PrintabilityDecision("final", print_path, False, "native artwork resolution is below 1024 px on the long edge.", metrics, {})
    if contrast < 10.0:
        return PrintabilityDecision("final", print_path, False, "artwork has insufficient visual information.", metrics, {})

    try:
        assessment = _vision_assessment(
            image,
            final_prompt(target, require_repeat_seams=require_repeat_seams),
            backend=backend,
            model=model,
        )
    except Exception as exc:
        return PrintabilityDecision("final", print_path, False, f"final vision assessment failed: {exc}", metrics, {})

    coverage = _percentage(assessment.get("canvas_coverage_percent"))
    disallowed = any(
        _bool(assessment.get(key))
        for key in ("has_text_or_logo", "has_watermark", "has_mockup_or_product_edges", "has_perspective_or_room_scene", "has_visible_artifacts")
    )
    accepted = (
        _bool(assessment.get("is_print_ready"))
        and coverage >= 95
        and str(assessment.get("critical_crop_risk") or "").strip().lower() == "low"
        and str(assessment.get("motif_scale") or "").strip().lower() == "printable"
        and (not require_repeat_seams or _bool(assessment.get("repeat_seams_ok")))
        and not disallowed
    )
    reason = str(assessment.get("reason") or "artwork meets the final print rubric.")
    if not accepted:
        reason = f"final artwork rejected: {reason}"
    return PrintabilityDecision("final", print_path, accepted, reason, metrics | {"canvas_coverage_percent": coverage}, assessment)


def assess_generated_artwork(
    artwork_path: Path,
    target: ProductTarget,
    *,
    backend: str,
    model: str,
) -> PrintabilityDecision:
    """Reject a generation that is still a photo before expensive print fitting."""
    try:
        with Image.open(artwork_path) as opened:
            image = opened.convert("RGB")
    except Exception as exc:
        return PrintabilityDecision("generated", artwork_path, False, f"unreadable generated artwork: {exc}", {}, {})

    width, height = image.size
    metrics = {
        "width": width,
        "height": height,
        "long_edge": max(width, height),
    }
    try:
        assessment = _vision_assessment(image, generated_prompt(target), backend=backend, model=model)
    except Exception as exc:
        return PrintabilityDecision("generated", artwork_path, False, f"generated artwork assessment failed: {exc}", metrics, {})

    rejected = any(
        _bool(assessment.get(key))
        for key in ("is_photo", "has_perspective", "has_shadows", "has_product_edges", "has_room_scene", "has_mockup")
    )
    accepted = _bool(assessment.get("is_flat_artwork")) and not rejected
    reason = str(assessment.get("reason") or "generated artwork passed the flat-artwork check.")
    if not accepted:
        reason = f"generated artwork rejected: {reason}"
    return PrintabilityDecision("generated", artwork_path, accepted, reason, metrics, assessment)


def assess_product_mockup(
    product_reference_path: Path,
    mockup_path: Path,
    target: ProductTarget,
    profile: MockupProductProfile,
    *,
    backend: str,
    model: str,
) -> PrintabilityDecision:
    """Verify that a background replacement preserved the intended product shape."""
    try:
        with Image.open(product_reference_path) as opened:
            source = opened.convert("RGBA")
            reference = Image.new("RGB", source.size, "white")
            reference.paste(source, mask=source.getchannel("A"))
        with Image.open(mockup_path) as opened:
            mockup = opened.convert("RGB")
    except Exception as exc:
        return PrintabilityDecision("mockup", mockup_path, False, f"unreadable mockup: {exc}", {}, {})

    metrics = {
        "reference_width": reference.width,
        "reference_height": reference.height,
        "mockup_width": mockup.width,
        "mockup_height": mockup.height,
    }
    try:
        assessment = _vision_pair_assessment(
            reference,
            mockup,
            mockup_prompt(target, profile),
            backend=backend,
            model=model,
        )
    except Exception as exc:
        return PrintabilityDecision("mockup", mockup_path, False, f"mockup quality assessment failed: {exc}", metrics, {})

    score = _percentage(assessment.get("overall_realism_score"))
    accepted = (
        _bool(assessment.get("product_type_preserved"))
        and _bool(assessment.get("material_behavior_correct"))
        and _bool(assessment.get("context_is_valid"))
        and _bool(assessment.get("scale_is_plausible"))
        and _bool(assessment.get("silhouette_preserved"))
        and not _bool(assessment.get("unintended_shape_distortion"))
        and _bool(assessment.get("support_surface_consistent"))
        and _bool(assessment.get("contact_shadow_realistic"))
        and _bool(assessment.get("lighting_coherent"))
        and score >= 80
    )
    reason = str(assessment.get("reason") or "mockup passed shape and realism checks.")
    if not accepted:
        reason = f"mockup rejected: {reason}"
    return PrintabilityDecision("mockup", mockup_path, accepted, reason, metrics | {"overall_realism_score": score}, assessment)


def assess_template_mockup(
    mockup: Image.Image,
    source_path: Path,
    target: ProductTarget,
    *,
    backend: str,
    model: str,
) -> PrintabilityDecision:
    """Judge the final local composite, not merely its blank AI template."""
    image = mockup.convert("RGB")
    metrics = {"mockup_width": image.width, "mockup_height": image.height}
    try:
        assessment = _vision_assessment(
            image,
            template_mockup_prompt(target),
            backend=backend,
            model=model,
        )
    except Exception as exc:
        return PrintabilityDecision("template_mockup", source_path, False, f"final mockup quality assessment failed: {exc}", metrics, {})

    score = _percentage(assessment.get("listing_realism_score"))
    accepted = (
        _bool(assessment.get("product_type_correct"))
        and _bool(assessment.get("full_size_scale_plausible"))
        and _bool(assessment.get("fabric_material_believable"))
        and _bool(assessment.get("fold_geometry_consistent"))
        and _bool(assessment.get("occlusion_and_contact_believable"))
        and _bool(assessment.get("lighting_coherent"))
        and not _bool(assessment.get("looks_like_flat_overlay"))
        and not _bool(assessment.get("looks_like_wrong_product"))
        and score >= 84
    )
    reason = str(assessment.get("reason") or "template mockup passed final realism QA.")
    if not accepted:
        reason = f"final mockup rejected: {reason}"
    return PrintabilityDecision("template_mockup", source_path, accepted, reason, metrics | {"listing_realism_score": score}, assessment)


def assess_direct_ai_mockup(
    print_path: Path,
    mockup_path: Path,
    target: ProductTarget,
    *,
    pose_name: str = "",
    pose_requirement: str = "",
    require_matching_pillowcases: bool = False,
    backend: str,
    model: str,
) -> PrintabilityDecision:
    """Check both product realism and that direct AI retained the supplied print artwork."""
    try:
        with Image.open(print_path) as opened:
            reference = opened.convert("RGB")
        with Image.open(mockup_path) as opened:
            mockup = opened.convert("RGB")
    except Exception as exc:
        return PrintabilityDecision("direct_ai_mockup", mockup_path, False, f"unreadable direct AI mockup: {exc}", {}, {})
    metrics = {
        "reference_width": reference.width,
        "reference_height": reference.height,
        "mockup_width": mockup.width,
        "mockup_height": mockup.height,
    }
    try:
        assessment = _vision_pair_assessment(
            reference,
            mockup,
            direct_ai_mockup_prompt(
                target,
                pose_name=pose_name,
                pose_requirement=pose_requirement,
                require_matching_pillowcases=require_matching_pillowcases,
            ),
            backend=backend,
            model=model,
        )
    except Exception as exc:
        return PrintabilityDecision("direct_ai_mockup", mockup_path, False, f"direct AI mockup quality assessment failed: {exc}", metrics, {})
    score = _percentage(assessment.get("listing_realism_score"))
    clean_hems = _bool(assessment.get("clean_straight_hems_no_scallops", assessment.get("clean_hems_and_no_ruffles", True)))
    pillowcases_clean = _bool(assessment.get("pillowcases_clean_and_uncluttered", True))
    no_quilting = _bool(assessment.get("no_comforter_quilting_grids", True))
    typography_clean = _bool(assessment.get("typography_crisp_and_legible", True))

    is_blanket = target.name == "blanket"
    is_rug = target.name == "rug"

    accepted = (
        _bool(assessment.get("artwork_identity_preserved"))
        and _bool(assessment.get("product_type_correct"))
        and (not is_rug or _bool(assessment.get("rug_shape_correct")))
        and _bool(assessment.get("full_size_scale_plausible"))
        and _bool(assessment.get("fabric_material_believable"))
        and _bool(assessment.get("fold_geometry_consistent"))
        and _bool(assessment.get("occlusion_and_contact_believable"))
        and _bool(assessment.get("lighting_coherent"))
        and not _bool(assessment.get("looks_like_flat_overlay"))
        and not _bool(assessment.get("looks_like_wrong_product"))
        and (not require_matching_pillowcases or (_bool(assessment.get("matching_pillowcases_present")) and pillowcases_clean))
        and (not is_blanket or (clean_hems and no_quilting))
        and typography_clean
        and score >= 84
    )
    reason = str(assessment.get("reason") or "direct AI mockup passed artwork fidelity and realism QA.")
    if not accepted:
        reason = f"direct AI mockup rejected: {reason}"
    return PrintabilityDecision("direct_ai_mockup", mockup_path, accepted, reason, metrics | {"listing_realism_score": score}, assessment)


def candidate_prompt(target: ProductTarget) -> str:
    return f"""
Evaluate whether this Pinterest image is a reusable visual reference for a new {target.name} print design.
Reject images without a reusable motif, palette, pattern, illustration, or texture. A product photo or lifestyle reference is acceptable when its surface, palette, texture, or motif can be extracted into new artwork; the downstream artwork prompt removes the product photography.
Flag text, watermarks, room scenes, perspective, and product photography in their dedicated fields, but do not reject them by themselves when a reusable motif is clearly present. They will become exclusions for artwork generation.
Do not assess copyright ownership. Assess only visual suitability.
Return JSON only:
{{
  "visual_reusability_score": 0,
  "has_usable_motif": false,
  "has_text_or_logo": false,
  "has_watermark": false,
  "is_mockup_or_product_photo": false,
  "has_perspective_or_room_scene": false,
  "motifs": ["short motif"],
  "palette": ["color or material cue"],
  "style": "short visual style",
  "composition": "short layout guidance",
  "exclude_from_artwork": ["text", "room scene"],
  "reason": "short reason"
}}
Use an integer score from 0 to 100, not a decimal fraction.
""".strip()


def final_prompt(target: ProductTarget, *, require_repeat_seams: bool) -> str:
    seam_rule = "Also verify all edges can repeat without an obvious seam." if require_repeat_seams else "repeat_seams_ok may be true."
    return f"""
Evaluate this final {target.name} print design. It must be clean, flat, full-bleed printable artwork, not a product photo or mockup.
Reject text, logos, watermarks, borders, room scenes, perspective, shadows, product edges, obvious AI artifacts, or a composition likely to lose its main motif at the edges.
{seam_rule}
Return JSON only:
{{
  "is_print_ready": false,
  "canvas_coverage_percent": 0,
  "critical_crop_risk": "low | medium | high",
  "motif_scale": "printable | too_small | unclear",
  "has_text_or_logo": false,
  "has_watermark": false,
  "has_mockup_or_product_edges": false,
  "has_perspective_or_room_scene": false,
  "has_visible_artifacts": false,
  "repeat_seams_ok": true,
  "reason": "short reason"
}}
Use an integer coverage percentage from 0 to 100, not a decimal fraction.
""".strip()


def generated_prompt(target: ProductTarget) -> str:
    return f"""
Evaluate this generated {target.name} artwork before it enters print production.
It must be a newly created 2D graphic design: flat, full-bleed, and suitable for textile printing.
Reject it if it is still a photograph of a craft, room, product, object, or physical surface.
Reject perspective, camera angle, cast shadows, highlights caused by lighting, product edges, mockups,
frames, borders, props, furniture, text, logos, and watermarks.
The design may contain illustrated shadows or intentional texture, but it must read as artwork rather than a photo.
Return JSON only:
{{
  "is_flat_artwork": false,
  "is_photo": false,
  "has_perspective": false,
  "has_shadows": false,
  "has_product_edges": false,
  "has_room_scene": false,
  "has_mockup": false,
  "reason": "short reason"
}}
""".strip()


def mockup_prompt(target: ProductTarget, profile: MockupProductProfile) -> str:
    return f"""
Compare the ORIGINAL_PRODUCT_REFERENCE with the BACKGROUND_MOCKUP for a {target.name}.
{profile.prompt_contract()}
The product may have any intended silhouette: rectangle, round, oval, runner, scalloped, or die-cut.
Allow only the pose described by the product profile. Reject unintended tapering, warping, missing edges, or a shape that no longer matches the original product.
Reject any presentation that changes the product type, material behavior, context, or physical scale. Evaluate the final ecommerce image, not just the background scene.
Return JSON only:
{{
  "product_type_preserved": false,
  "material_behavior_correct": false,
  "context_is_valid": false,
  "scale_is_plausible": false,
  "silhouette_preserved": false,
  "unintended_shape_distortion": false,
  "support_surface_consistent": false,
  "contact_shadow_realistic": false,
  "lighting_coherent": false,
  "overall_realism_score": 0,
  "reason": "short reason"
}}
Use an integer score from 0 to 100, not a decimal fraction.
""".strip()


def template_mockup_prompt(target: ProductTarget) -> str:
    product_rule = (
        "It must read as a full-size throw blanket, not a bedspread, chair cover, towel, scarf, placemat, or wall hanging."
        if target.name.strip().lower() == "blanket"
        else "It must read as the intended full-size product with physically plausible scale and material."
    )
    return f"""
Evaluate this final ecommerce lifestyle mockup of a {target.name}. The product artwork was composited locally onto an AI-generated blank textile template.
{product_rule}
Judge the visible final result, not the intended pipeline. Reject a print that appears as a flat poster, sticker, screen, or 2D overlay pasted over furniture.
The artwork may be bold and high contrast; that alone is not a defect. Require believable textile thickness, fold geometry, edge binding, occlusion where fabric passes behind furniture, and lighting/shadow that follows the fabric surface.
Return JSON only:
{{
  "product_type_correct": false,
  "full_size_scale_plausible": false,
  "fabric_material_believable": false,
  "fold_geometry_consistent": false,
  "occlusion_and_contact_believable": false,
  "lighting_coherent": false,
  "looks_like_flat_overlay": false,
  "looks_like_wrong_product": false,
  "listing_realism_score": 0,
  "reason": "short concrete reason"
}}
Use an integer score from 0 to 100. Be strict: 84 means a credible ecommerce listing image, not merely an attractive AI picture.
""".strip()


def direct_ai_mockup_prompt(
    target: ProductTarget,
    *,
    pose_name: str = "",
    pose_requirement: str = "",
    require_matching_pillowcases: bool = False,
) -> str:
    pillowcase_rule = (
        "This is a bedroom-set shot: require exactly two matching pillowcases using the same recognizable print identity as the blanket. "
        "The two pillowcases must be distinct, separate, and aesthetic, with motifs scaled naturally to pillow proportions without squishing or visual clutter. "
        "Reject mockups with plain white pillows, missing pillowcases, merged/deformed pillows, or cluttered/garbled pillowcase prints."
        if require_matching_pillowcases
        else "Matching pillowcases are not required for this pose."
    )
    rug_shape_rule = (
        f"Required rug silhouette: {target.rug_shape}. Reject a different silhouette."
        if target.name == "rug"
        else "Rug silhouette is not applicable."
    )
    blanket_edge_rule = (
        "For blankets: require clean, continuous straight modern sewn hems. The bottom hem hanging across the foot of the bed must form a level, continuous, straight horizontal line parallel to the floor between corners. Reject wavy scalloped edges, die-cut tabs protruding around badge/motif shapes, drooping tongues or flaps of fabric extended to complete illustrations, notched or concave cut-out hem corners, ruffled frills, lace trims, or comforter/duvet box quilting grids. Drape must be soft, fluid, and natural, not stiff origami/cardboard folds. Reject mockups where large folded-over flaps or oversized turned-back sections excessively conceal or hide the printed artwork on the sofa or bed. For sofa showcase shots, the blanket must realistically conform to the 3D sofa contours (backrest, horizontal seat cushion depth, and front waterfall drape)—reject flat 2D cardboard billboard overlays or vertical posters stretched flat across the sofa that erase the seat cushion. For casual armrest drape shots, the blanket is draped over one armrest and seat with fluid, natural folds. For folded/detail shots, the blanket must be neatly folded into thick, rounded layers on furniture with authentic cloth thickness—reject mockups where the blanket is stretched flat across the sofa like an unfolded flat sheet, curtain, poster, or backdrop. Any typography from the artwork must be rendered crisply and legibly without garbled, distorted, or scrambled nonsense characters."
        if target.name == "blanket"
        else ""
    )
    return f"""
Compare REFERENCE_PRINT_ARTWORK with the DIRECT_AI_LIFESTYLE_MOCKUP for a {target.name}.
The reference is a flat textile design. The mockup is allowed to bend, fold, crop, and repeat that design naturally across the product surface, but must preserve its recognizable motifs, palette, and visual identity.
Required listing pose: {pose_name or "a credible product showcase"}. {pose_requirement or "Show the product naturally and clearly."}
{pillowcase_rule}
{rug_shape_rule}
{blanket_edge_rule}

Evaluation criteria:
1. Pillowcase Quality (if required for bedroom set):
   - 'matching_pillowcases_present': must be true if exactly two matching pillowcases are present on the bed.
   - 'pillowcases_clean_and_uncluttered': must be true ONLY if motifs are scaled naturally to pillow proportions with 1-3 prominent hero motifs and clean margins. Must be FALSE if the pillows have tiny squished micro-repeats, crowded cluttered badges, deformed pillow shapes, or unreadable garbled text. (If matching pillowcases are not required, set true).
2. Blanket Edges & Drape (for blankets):
   - 'clean_straight_hems_no_scallops': must be true ONLY if the blanket perimeter hems (especially across the foot of the bed or sofa) are intact, continuous geometric lines without notched/concave cutouts. Must be FALSE if the hem has protruding tabs, contouring lobes around motifs/badges, sagging tongues drooping downward, wavy scallops, die-cut tabs at the bottom/sides, notched or concave cut-out hem corners, or ruffled frills. (If not blanket, set true).
   - 'no_comforter_quilting_grids': must be true if the blanket is smooth unquilted fleece/woven textile. Must be FALSE if there are duvet/comforter box, grid, or diamond quilting stitches. (If not blanket, set true).
3. Typography Fidelity:
   - 'typography_crisp_and_legible': must be true if any visible lettering/slogans from the artwork are sharp, legible, and ungarbled. Must be FALSE if text is distorted, scrambled, or rendered as nonsense pseudo-characters. If reference has no text, set true.
4. Overall Realism:
   - Judge whether the final image is a credible ecommerce photograph: correct product type and full-size scale, realistic textile thickness/weave, natural soft folds without rigid cardboard lines, furniture occlusion, contact shadows, and coherent lighting.
   - Reject artwork that was substituted with a different design, turned into a generic unrelated pattern, or appears as a flat overlay, poster, or sticker.

Return JSON only:
{{
  "artwork_identity_preserved": false,
  "product_type_correct": false,
  "rug_shape_correct": false,
  "full_size_scale_plausible": false,
  "fabric_material_believable": false,
  "fold_geometry_consistent": false,
  "occlusion_and_contact_believable": false,
  "lighting_coherent": false,
  "looks_like_flat_overlay": false,
  "looks_like_wrong_product": false,
  "matching_pillowcases_present": false,
  "pillowcases_clean_and_uncluttered": false,
  "clean_straight_hems_no_scallops": false,
  "clean_hems_and_no_ruffles": false,
  "no_comforter_quilting_grids": false,
  "typography_crisp_and_legible": false,
  "listing_realism_score": 0,
  "reason": "short concrete reason"
}}
Use an integer score from 0 to 100. Be strict: 84 means a credible ecommerce listing image, not merely a pleasant AI illustration.
""".strip()


def _vision_assessment(image: Image.Image, prompt: str, *, backend: str, model: str) -> dict[str, object]:
    from google.genai import types

    client = create_gemini_client(backend)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[image_part(image), types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
            )
            return parse_json_relaxed(extract_response_text(response))
        except Exception as exc:
            last_error = exc
            if attempt >= 3 or not is_transient_gemini_error(exc):
                break
            time.sleep(float(attempt) * 2.0)
    raise RuntimeError(str(last_error or "vision assessment failed"))


def _vision_pair_assessment(
    first: Image.Image,
    second: Image.Image,
    prompt: str,
    *,
    backend: str,
    model: str,
) -> dict[str, object]:
    from google.genai import types

    client = create_gemini_client(backend)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[
                    types.Part.from_text(text="ORIGINAL_PRODUCT_REFERENCE"),
                    image_part(first),
                    types.Part.from_text(text="BACKGROUND_MOCKUP"),
                    image_part(second),
                    types.Part.from_text(text=prompt),
                ])],
                config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
            )
            return parse_json_relaxed(extract_response_text(response))
        except Exception as exc:
            last_error = exc
            if attempt >= 3 or not is_transient_gemini_error(exc):
                break
            time.sleep(float(attempt) * 2.0)
    raise RuntimeError(str(last_error or "mockup quality assessment failed"))


def reference_design_brief(assessment: dict[str, object]) -> dict[str, object]:
    """Keep only image-derived design cues, never source-photo details."""
    def values(name: str) -> list[str]:
        raw = assessment.get(name) or []
        if not isinstance(raw, list):
            return []
        return [str(value).strip() for value in raw if str(value).strip()][:8]

    exclusions = values("exclude_from_artwork")
    for field, label in (
        ("has_text_or_logo", "text, logos, and typography"),
        ("has_watermark", "watermarks"),
        ("is_mockup_or_product_photo", "product photography and product edges"),
        ("has_perspective_or_room_scene", "room scenes, furniture, camera perspective, and cast shadows"),
    ):
        if _bool(assessment.get(field)):
            exclusions.append(label)

    return {
        "motifs": values("motifs"),
        "palette": values("palette"),
        "style": str(assessment.get("style") or "").strip()[:240],
        "composition": str(assessment.get("composition") or "").strip()[:240],
        "exclude": list(dict.fromkeys(exclusions))[:10],
    }


def _source_role(candidate: CandidateImage) -> str:
    role = str(candidate.source_role or "").strip().lower()
    if role in {"", "unknown"} and isinstance(candidate.metadata, dict):
        role = str(candidate.metadata.get("source_role") or "").strip().lower()
    return role or "unknown"


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "yes", "1"}


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _percentage(value: object) -> float:
    number = _number(value)
    if 0.0 <= number <= 1.0:
        return number * 100.0
    return number
