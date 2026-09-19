from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .config import ProductTarget
from .printability import assess_direct_ai_mockup, assess_template_mockup
from .product_asset import (
    create_gemini_client,
    extract_image_bytes,
    extract_response_text,
    image_part,
    is_transient_gemini_error,
    parse_json_relaxed,
)


MARKER_RGB = (0, 174, 220)


@dataclass(frozen=True)
class TemplatePose:
    name: str
    scene: str
    placement: str
    avoid: str
    display_rule: str = ""


@dataclass(frozen=True)
class TemplateMockupRecord:
    print_path: Path
    template_path: Path | None
    mask_path: Path | None
    mockup_path: Path | None
    model: str
    pose: str
    status: str
    notes: str
    metrics: dict[str, object]
    render_mode: str = "template_composite"
    variant: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_template_mockup(
    print_path: Path,
    output_dir: Path,
    target: ProductTarget,
    *,
    backend: str,
    model: str,
    quality_model: str,
    attempts: int = 3,
    pose: TemplatePose | None = None,
    variant: int = 1,
    progress: Any = None,
    **kwargs: Any,
) -> TemplateMockupRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = print_path.stem.replace("_rgb", "")
    suffix = f"_v{max(1, variant):02d}"
    template_path = output_dir / "templates" / f"{stem}{suffix}_template.png"
    mask_path = output_dir / "template_masks" / f"{stem}{suffix}_mask.png"
    mockup_path = output_dir / "lifestyle_mockups" / f"{stem}{suffix}_lifestyle.png"
    client = create_gemini_client(backend)
    pose = pose or template_pose_for_index(target, 1)
    last_error = ""
    correction = ""

    for attempt in range(1, max(1, attempts) + 1):
        try:
            template = generate_template(client, target, model, pose, correction=correction)
            pose_assessment = assess_template_pose(client, template, target, pose, quality_model)
            validate_template_pose(pose_assessment)
            mask, metrics = extract_marker_mask(template)
            validate_template_mask(metrics, template.size)
            composite = composite_print_on_template(print_path, template, mask, target)
            mockup_quality = assess_template_mockup(
                composite,
                mockup_path,
                target,
                backend=backend,
                model=quality_model,
            )
            metrics["mockup_quality"] = mockup_quality.to_dict()
            if not mockup_quality.accepted:
                raise RuntimeError(f"final mockup QA rejected: {mockup_quality.reason}")
            template_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mockup_path.parent.mkdir(parents=True, exist_ok=True)
            template.save(template_path)
            mask.save(mask_path)
            composite.save(mockup_path)
            metrics["generation_attempt"] = attempt
            metrics["pose_assessment"] = pose_assessment
            return TemplateMockupRecord(
                print_path,
                template_path,
                mask_path,
                mockup_path,
                model,
                pose.name,
                "ok",
                "AI generated a blank product template; artwork was composited locally.",
                metrics,
                variant=variant,
            )
        except Exception as exc:
            last_error = str(exc)
            if "template pose qa rejected:" in last_error.lower() or "final mockup qa rejected:" in last_error.lower():
                correction = (
                    "The prior template/composite was rejected by visual QA for this reason: "
                    f"{last_error.split(':', 1)[-1].strip()}. Correct that specific issue while keeping "
                    "a full-size loose throw blanket, visible textile folds, edge thickness, coherent lighting, "
                    "furniture occlusion, and a physically plausible pose."
                )
            if attempt < max(1, attempts) and is_transient_gemini_error(exc):
                time.sleep(float(attempt) * 2.0)
                continue
            if attempt < max(1, attempts) and (
                "marker mask" in last_error.lower()
                or "template pose qa" in last_error.lower()
                or "final mockup qa" in last_error.lower()
            ):
                continue
            break
    return TemplateMockupRecord(print_path, None, None, None, model, pose.name, "failed", last_error, {}, variant=variant)


def build_direct_ai_mockup(
    print_path: Path,
    output_dir: Path,
    target: ProductTarget,
    *,
    backend: str,
    model: str,
    quality_model: str,
    attempts: int = 3,
    pose: TemplatePose | None = None,
    variant: int = 1,
    progress: Any = None,
    **kwargs: Any,
) -> TemplateMockupRecord:
    """Have the image model render real cloth geometry rather than compositing a flat print."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = print_path.stem.replace("_rgb", "")
    suffix = f"_v{max(1, variant):02d}"
    mockup_path = output_dir / "lifestyle_mockups" / f"{stem}{suffix}_lifestyle.png"
    client = create_gemini_client(backend)
    pose = pose or template_pose_for_index(target, 1)
    correction = ""
    last_error = ""
    try:
        with Image.open(print_path) as opened:
            artwork = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        return TemplateMockupRecord(print_path, None, None, None, model, pose.name, "failed", f"unreadable print artwork: {exc}", {}, "direct_ai", variant)

    for attempt in range(1, max(1, attempts) + 1):
        candidate_path = output_dir / "direct_ai_candidates" / f"{stem}{suffix}_attempt_{attempt}.png"
        try:
            generated = generate_direct_ai_lifestyle(client, artwork, target, model, pose, correction=correction)
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            generated.save(candidate_path)
            quality = assess_direct_ai_mockup(
                print_path,
                candidate_path,
                target,
                pose_name=pose.name,
                pose_requirement=pose.display_rule or pose.placement,
                require_matching_pillowcases=pose.name == "bed_full_showcase",
                backend=backend,
                model=quality_model,
            )
            if not quality.accepted:
                raise RuntimeError(f"direct AI mockup QA rejected: {quality.reason}")
            mockup_path.parent.mkdir(parents=True, exist_ok=True)
            generated.save(mockup_path)
            return TemplateMockupRecord(
                print_path,
                None,
                None,
                mockup_path,
                model,
                pose.name,
                "ok",
                "Gemini rendered the final lifestyle image directly from the approved print artwork.",
                {"generation_attempt": attempt, "mockup_quality": quality.to_dict()},
                "direct_ai",
                variant,
            )
        except Exception as exc:
            last_error = str(exc)
            if "direct ai mockup qa rejected:" in last_error.lower():
                qa_detail = last_error.split(":", 1)[-1].strip()
                correction = (
                    f"The previous render failed final QA: {qa_detail}. Keep the reference artwork recognizable, "
                    "ensure clean modern sewn straight continuous linear hems with strictly no ruffled or scalloped edges, "
                    "never contour or tab fabric edges around badges/motifs, avoid comforter box quilting, "
                    "ensure the blanket realistically conforms to 3D furniture depth and cushion geometry without flattening into a 2D billboard, "
                    "ensure the two pillowcases (if applicable) are distinct, balanced, and uncluttered with 1-3 well-scaled legible hero motifs, "
                    "and correct cloth geometry, folds, scale, and lighting."
                )
            if attempt < max(1, attempts) and is_transient_gemini_error(exc):
                time.sleep(float(attempt) * 2.0)
                continue
            if attempt < max(1, attempts) and "direct ai mockup qa" in last_error.lower():
                continue
            break
    return TemplateMockupRecord(print_path, None, None, None, model, pose.name, "failed", last_error, {}, "direct_ai", variant)


def generate_direct_ai_lifestyle(
    client: Any,
    artwork: Image.Image,
    target: ProductTarget,
    model: str,
    pose: TemplatePose,
    *,
    correction: str = "",
) -> Image.Image:
    from google.genai import types

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=0.35,
        image_config=types.ImageConfig(aspect_ratio="1:1", image_size="2K", output_mime_type="image/png"),
    )
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[image_part(artwork, max_side=1536, max_bytes=3_500_000), types.Part.from_text(text=direct_ai_lifestyle_prompt(target, pose, correction))],
            )
        ],
        config=config,
    )
    image_bytes, _ = extract_image_bytes(response)
    if not image_bytes:
        raise RuntimeError("direct AI lifestyle generation returned no image.")
    with Image.open(io.BytesIO(image_bytes)) as generated:
        return generated.convert("RGB")


def direct_ai_lifestyle_prompt(target: ProductTarget, pose: TemplatePose, correction: str) -> str:
    product = target.name.strip().lower()
    if product == "blanket":
        product_rule = (
            "Create one full-size premium soft throw blanket. Treat the attached image as its exact print artwork reference, "
            "not as a flat image to paste over furniture. Faithfully reproduce its motifs, illustrations, color palette, and pattern language "
            "across the fabric with natural cloth folds. The full printed surface must face outward toward the camera so that motifs and typography are prominently showcased. "
            "If the reference artwork includes typography, lettering, or text slogans, "
            "render any visible text cleanly, sharply, and legibly without garbled, distorted, or scrambled characters. "
            "CRITICAL HEM & EDGE SPECIFICATION (NO PROTRUDING TABS / FLAPS / NOTCHES): The blanket is a STRICT RECTANGLE manufactured by cutting printed roll fabric with a straight rotary blade and sewing a straight linear hem on all sides. "
            "The physical perimeter edges must remain strictly intact geometric lines; never contour, curve, scallop, tab, notch, or cut out the fabric edge around individual badges, motifs, or illustrations, and never bite out concave chunks or empty gaps from the corners. "
            "ZERO PROTRUSIONS / ZERO TABS / ZERO NOTCHES: Do NOT dip, stretch, extend, or notch the fabric. If a motif, character, pumpkin, cat, skull, or framed badge falls along the edge, it MUST BE CUT CLEANLY IN HALF by the straight hemline, exactly like real cut-and-sewn cloth. "
            "Strictly no wavy scalloped cutouts, no die-cut tabs contouring around motifs, no sagging tongues or conical flaps of fabric drooping lower than the rest of the hem, no ruffled frills, no lettuce edges, no lace trims, and no decorative fringe. "
            "No large folded-over flaps or turned-back sections that conceal or hide the printed artwork. "
            "Textile Material & Drape: Soft, continuous plush fleece or woven fabric with natural weight that drapes smoothly and fluidly under gravity. "
            "Strictly no quilted comforter/duvet grid stitching, no puffy quilt squares, and no stiff cardboard or origami folds."
        )
        avoid = (
            "No scalloped borders, no die-cut or tabbed fabric edges, no motif-shaped hem protrusions, no drooping fabric flaps or tongues, "
            "no notched, concave, or cut-out hem corners, no large folded-over flaps or turned-back sections concealing the printed artwork, "
            "no flat 2D cardboard pillows, no paper placards or stickers, no ruffled edges or frills, no lace trims, "
            "no comforter/duvet box quilting stitches, no stiff origami folds, no bedspread skirt, "
            "no chair cover, towel, scarf, placemat, rug, wall hanging, no garbled or distorted lettering, "
            "no superimposed photographer watermarks, brand logos, or UI overlays."
        )
    elif product == "rug":
        shape = target.rug_shape.strip().lower() if target.rug_shape else "rectangle"
        product_rule = (
            f"Create one full-size {shape} floor rug. Treat the attached image as its exact print artwork reference, "
            "not as a flat image pasted on the floor. Preserve its motifs, palette, and visual identity on the rug surface."
        )
        avoid = (
            f"No rectangular rug when the required silhouette is {shape}, no blanket, bath mat, doormat, wall hanging, "
            "no superimposed photographer watermarks, brand logos, or UI overlays."
        )
    else:
        product_rule = "Create one full-size product using the attached image as its faithful print artwork reference."
        avoid = pose.avoid
    if pose.name == "bed_full_showcase":
        coordinated_products = (
            "Coordinated Bedroom Set showcase: show the full blanket covering the bed plus exactly two matching printed pillowcases "
            "propped neatly side-by-side at the head of the bed against the headboard (with plush white sleeping pillows visible behind them). "
            "Blanket drape & hems: The blanket drapes smoothly over the mattress with gentle natural cloth waves, showcasing the full printed pattern across the bed, a neat turned-down top cuff of white bedding near the pillows, and falls naturally over the foot and lower side edges with clean straight sewn hems (strictly no large side fold-overs or flipped edges concealing the printed pattern). "
            "The bottom hem hanging at the foot of the bed MUST FORM A LEVEL, CONTINUOUS, RULER-STRAIGHT HORIZONTAL LINE parallel to the floor between the bed's left and right corners—strictly no die-cut contouring, no scalloped tabs or tongues protruding around motifs or badges, no ruffled frills, and no comforter box quilting. Any artwork badge or motif near the edge must be cleanly sliced in half by the straight horizontal hemline. "
            "Pillowcase styling & scale: The two pillowcases must be distinct, separate, and authentic 3D bed pillows. They are visibly plump, soft, and fluffy decorative pillow shams propped at a natural, slightly reclined angle in front of white sleeping pillows against the headboard, with realistic fabric creasing, rounded contours, and sewn perimeter seams (never flat 2D cardboard cutouts, stickers, or rigid placards). "
            "Rather than repeating the entire dense pattern into tiny micro-icons, each pillowcase should feature 1 to 2 prominent, well-scaled hero motifs from the artwork "
            "(e.g. one clean decorative illustration, character, or badge per pillowcase) with clean negative space and readable typography in an uncluttered, balanced layout. "
            "Keep any lettering crisp, readable, and elegant. Absolutely no crowded micro-repeats, squished icons, or garbled text on the pillows."
        )
    else:
        coordinated_products = ""
    return f"""
Use case: final ecommerce lifestyle product photograph.
{product_rule}
Scene: {pose.scene}.
Placement: {pose.placement}.
Photorealism requirements: The textile must have real cloth geometry, natural gravity, visible thickness and edge binding, broad folds plus fine weave, physically correct occlusion behind furniture, soft contact shadows, and lighting that follows the folded surface. It must look like an actual camera photograph, not a 2D collage, poster, sticker, rendering, or graphic illustration.
Coordinated products: {coordinated_products or "None."}
Listing-shot requirement: {pose.display_rule or pose.placement}
Composition: Follow the requested listing-shot framing. Make the product full-size and plausible, with the artwork readable wherever the pose is intended to show it.
Constraints: {avoid} {pose.avoid}
Retry correction: {correction or "None. Render one credible, realistic product photograph."}
""".strip()


def template_pose_for_index(target: ProductTarget, index: int) -> TemplatePose:
    product = target.name.strip().lower()
    if product == "blanket":
        poses = (
            TemplatePose(
                "held_open_showcase",
                "a bright, tasteful home interior with a neutral sofa softly out of focus in the background",
                "a smiling adult holds the two upper corners of one full-size blanket open in front of their body, showing the entire front printable surface nearly flat to camera",
                "do not crop the blanket, do not turn it into a flag, curtain, wall hanging, or rigid poster",
                "HERO LISTING SHOT: the blanket fills 70-85 percent of the frame and the print is fully readable. Natural hands may hold the corners; the person is secondary to the product.",
            ),
            TemplatePose(
                "sofa_front_showcase",
                "a bright, warm, modern living room with a stylish neutral upholstered three-seat sofa, photographed at a natural eye-level perspective at a subtle 15-20 degree three-quarter angle with soft window lighting and realistic depth of field",
                "the blanket conforms strictly to the 3D sofa contours with physical cloth geometry: draped over the backrest, forming a distinct horizontal shelf across the seat cushions showing clear cushion depth, then cascading over the front seat edge in a soft vertical waterfall drape towards the floor; the full printed surface faces outward toward the camera with the entire design prominently visible without folded-over flaps; the perimeter edges remain an intact, continuous, unbroken rectangle parallel to the floor without cut-out notches, scallops, or empty bites",
                "do not create a flat 2D graphic overlay, poster, banner, sticker, or rigid billboard stretched flat across the sofa without seat cushion depth; strictly no large folded-over flaps or turned-back sections concealing the artwork; strictly no notched, concave, or bitten-off hem corners; no ruler-straight floating bottom hem without gravity, no garbled typography, no tilted extreme Dutch angles",
                "SOFA LISTING SHOT: showcase the full-size blanket on the sofa with authentic 3D depth and fabric weight. The printed artwork motifs and typography remain centered, crisp, and fully readable while warping organically and subtly across the cushion curves; motifs on the horizontal seat surface recede naturally in 3D perspective foreshortening; soft ambient shadows under the seat lip and at the bottom hem establish physical contact and realism.",
            ),
            TemplatePose(
                "bed_full_showcase",
                "a bright, elegant, modern bedroom with a beautifully made bed and stylish upholstered headboard, photographed from a natural eye-level perspective from the foot of the bed with soft natural window lighting and realistic depth of field",
                "the blanket covers the entire mattress with gentle natural cloth waves and a soft turned-back top cuff, draping naturally over the foot and side edges with continuous straight sewn hems (the bottom hem hanging at the foot is a clean straight horizontal line parallel to the floor, never scalloped, notched, or tabbed around artwork motifs); exactly two matching printed plump 3D pillowcases sit propped neatly in front of white sleeping pillows against the headboard",
                "do not leave plain white pillows without printed shams, do not make flat 2D cardboard pillows or stickers, do not merge or squish pillows into cluttered blobs, no scalloped cuts or ruffled frills, no die-cut tabs following badge shapes, no comforter box quilting stitches, no stiff origami folds, no garbled typography, no tilted or extreme diagonal camera angles, no photographer watermarks",
                "BEDROOM SET LISTING SHOT: showcase a coordinated set of one full blanket across the bed plus exactly two matching pillowcases. Pillowcases must feature clean, balanced, well-scaled motifs without clutter; blanket must have clean straight sewn hems and a realistic soft drape.",
            ),
            TemplatePose(
                "folded_detail_showcase",
                "a refined living room with a neutral sofa, armchair, or upholstered bench in soft window light",
                "the blanket is neatly folded into two or three thick, fluffy, cozy layers resting on one end of the furniture (e.g. on the sofa armrest or seat corner), with one corner soft-draped down naturally to reveal plush fabric thickness, soft cloth folds, hem stitching, and a readable cropped portion of the print",
                "strictly do not spread or hang the blanket flat across the sofa, do not make it an unfolded flat sheet, curtain, backdrop, or banner covering the cushions, do not make it a towel, scarf, tiny decorative textile, or a flat 2D poster; do not try to show the whole design flat",
                "DETAIL LISTING SHOT: emphasize realistic plush fleece material, edge binding, and thick rounded cloth folds. The folds naturally crop the artwork so only a portion of the motifs and badges are visible across the folded layers, exactly like a genuine high-end ecommerce listing photograph.",
            ),
            TemplatePose(
                "sofa_casual_drape",
                "a stylish, cozy modern living room with a neutral three-seat sofa, photographed at a natural eye-level perspective from a slight three-quarter angle with warm natural window sunlight and realistic depth of field",
                "the blanket is artfully and casually draped over one side of the sofa: resting over the upper backrest, cascading over one armrest and spreading loosely across the adjacent seat cushion with one corner hanging down toward the floor in soft, natural, fluid cloth folds; the printed design remains clearly visible across the drape without heavy reversed flaps; the sofa cushions and room setting remain naturally visible around it",
                "do not spread the blanket flat like a bedsheet or tablecloth across the entire sofa; strictly no flat 2D poster, banner, sticker, or rigid billboard; strictly no oversized blank flaps hiding the print, no ruler-straight floating bottom hem, no garbled typography, no messy squished blobs",
                "CASUAL LIFESTYLE SHOT: showcase the throw blanket draped effortlessly and authentically in a high-end interior scene. The main printed motifs and typography remain clearly recognizable and prominent across the drape, with natural fabric thickness, soft contact shadows, and realistic textile weight.",
            ),
        )
    else:
        poses = (
            TemplatePose("living_room_center", "a bright living room photographed at standing eye level", "the rug lies centered under a coffee table with all edges visible", "no blanket, bath mat, doormat, or second rug"),
            TemplatePose("bedroom_bedside", "a calm bedroom photographed at a natural three-quarter angle", "the rug sits beside the bed with a clear floor-plane perspective", "no blanket, bath mat, doormat, or second rug"),
            TemplatePose("entryway_runner", "a practical entryway photographed at standing eye level", "the rug is a runner on the floor with clear edges and natural scale", "no blanket, bath mat, or second rug"),
            TemplatePose("reading_corner", "a sunlit reading corner with one chair and floor lamp", "the rug lies flat in front of the chair with a visible perspective plane", "no blanket, bath mat, doormat, or second rug"),
            TemplatePose("dining_room", "a modest dining room photographed from a natural angle", "the rug lies under a small dining table with its outer edges clearly visible", "no blanket, bath mat, doormat, or second rug"),
        )
    return poses[(max(1, index) - 1) % len(poses)]


def generate_template(
    client: Any,
    target: ProductTarget,
    model: str,
    pose: TemplatePose,
    *,
    correction: str = "",
) -> Image.Image:
    from google.genai import types

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=0.42,
        image_config=types.ImageConfig(
            aspect_ratio="1:1",
            image_size="2K",
            output_mime_type="image/png",
        ),
    )
    response = client.models.generate_content(model=model, contents=[template_prompt(target, pose, correction=correction)], config=config)
    image_bytes, _ = extract_image_bytes(response)
    if not image_bytes:
        raise RuntimeError("AI template generation returned no image.")
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def template_prompt(target: ProductTarget, pose: TemplatePose, *, correction: str = "") -> str:
    product = target.name.strip().lower()
    if product == "blanket":
        subject = "one full-size soft woven throw blanket with clearly visible textile folds"
        avoid = f"no table, no placemat, no coaster, no rug, no bath mat, no second blanket; {pose.avoid}"
    else:
        subject = "one full-size floor rug with a clear visible silhouette"
        avoid = pose.avoid
    return f"""
Use case: product-mockup
Asset type: reusable blank product mockup template
Primary request: Create a photorealistic ecommerce lifestyle scene containing {subject}.
Scene/backdrop: {pose.scene}.
Subject: The entire printable surface of the single product must be one uniform saturated cyan-blue color close to RGB {MARKER_RGB}, with no pattern, no text, no logo, no border, and no decorative motif.
Composition/framing: {pose.placement}. Product is the hero and occupies about 18-35 percent of the image; retain generous room context. Keep all product edges clearly visible; do not crop the product.
Materials/textures: preserve a matte woven textile surface, visible edge binding and believable thickness. Give the cyan placeholder several broad, readable folds plus subtle fine weave, physically coherent occlusion where it passes behind furniture, soft contact shadows, and highlights that follow its folds. Never make it glossy, paper-like, rigid, or a flat rectangular sheet.
Constraints: Generate a blank template only. The cyan product surface must be easy to isolate from the scene. {avoid}.
Retry correction: {correction or "None. Follow the requested pose exactly."}
Avoid: text, watermarks, frames, collage layout, additional cyan objects.
""".strip()


def assess_template_pose(
    client: Any,
    template: Image.Image,
    target: ProductTarget,
    pose: TemplatePose,
    model: str,
) -> dict[str, object]:
    from google.genai import types

    prompt = f"""
Evaluate this blank ecommerce product template for a {target.name}.
Required pose: {pose.name}. {pose.placement}.
The intended product surface is a single cyan-blue placeholder textile. Scene: {pose.scene}.
Return JSON only:
{{
  "correct_product_type": false,
  "single_product": false,
  "pose_matches": false,
  "pose_is_natural": false,
  "cyan_surface_isolatable": false,
  "has_forbidden_presentation": false,
  "template_realism_score": 0,
  "reason": "short reason"
}}
Use an integer score from 0 to 100. A blanket that covers an entire chair like a fitted cover, or a blanket that covers an entire bed like a bedspread, fails pose_is_natural and pose_matches.
""".strip()
    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[image_part(template), types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
            )
            return parse_json_relaxed(extract_response_text(response))
        except Exception as exc:
            last_error = exc
            if attempt >= 2 or not is_transient_gemini_error(exc):
                break
            time.sleep(float(attempt) * 2.0)
    raise RuntimeError(f"template pose QA failed: {last_error or 'unknown error'}")


def validate_template_pose(assessment: dict[str, object]) -> None:
    def enabled(name: str) -> bool:
        return assessment.get(name) is True or str(assessment.get(name) or "").strip().lower() in {"true", "yes", "1"}

    try:
        score = float(assessment.get("template_realism_score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    # The model's description of a pose is often overly literal (for example, a throw
    # draped on both sides of an armchair). Product identity and physical plausibility
    # matter more than matching every wording detail of a requested lifestyle pose.
    accepted = (
        enabled("correct_product_type")
        and enabled("single_product")
        and enabled("pose_is_natural")
        and enabled("cyan_surface_isolatable")
        and not enabled("has_forbidden_presentation")
        and score >= 75
    )
    if not accepted:
        reason = str(assessment.get("reason") or "template pose did not pass QA")
        raise RuntimeError(f"template pose QA rejected: {reason}")


def extract_marker_mask(template: Image.Image) -> tuple[Image.Image, dict[str, object]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to isolate the AI template marker mask.") from exc

    rgb = np.asarray(template.convert("RGB"), dtype=np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    strict_marker = ((b - r > 55) & (g - r > 30) & (b > 105) & (g > 70)).astype(np.uint8) * 255
    broad_marker = ((b - r > 25) & (g - r > 5) & (b > 55) & (g > 40)).astype(np.uint8) * 255
    marker = cv2.morphologyEx(strict_marker, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(marker, connectivity=8)
    if count <= 1:
        raise RuntimeError("template marker mask was not found")
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    selected = (labels == label).astype(np.uint8) * 255
    x, y, w, h, _ = stats[label]
    pad_x = max(24, round(w * 0.16))
    pad_y = max(24, round(h * 0.16))
    region = np.zeros_like(selected)
    region[max(0, y - pad_y) : min(selected.shape[0], y + h + pad_y), max(0, x - pad_x) : min(selected.shape[1], x + w + pad_x)] = 255
    broad_count, broad_labels, broad_stats, _ = cv2.connectedComponentsWithStats(broad_marker, connectivity=8)
    min_component_area = max(80, round(selected.size * 0.00008))
    included_components = 1
    for component in range(1, broad_count):
        area = int(broad_stats[component, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        component_mask = (broad_labels == component).astype(np.uint8) * 255
        if np.any((component_mask > 0) & (region > 0)):
            selected = cv2.bitwise_or(selected, component_mask)
            included_components += 1
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8), iterations=2)
    selected = cv2.GaussianBlur(selected, (0, 0), sigmaX=1.2)
    ys, xs = np.where(selected > 32)
    if not len(xs):
        raise RuntimeError("template marker mask is empty")
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    height, width = selected.shape
    remaining_marker = (broad_marker > 0) & (selected <= 32) & (region > 0)
    marker_pixels = max(1, int(np.count_nonzero(broad_marker & region)))
    metrics: dict[str, object] = {
        "mask_coverage": float((selected > 32).mean()),
        "mask_bbox": [left, top, right + 1, bottom + 1],
        "mask_bbox_width_ratio": (right - left + 1) / width,
        "mask_bbox_height_ratio": (bottom - top + 1) / height,
        "included_marker_components": included_components,
        "marker_residual_ratio": float(np.count_nonzero(remaining_marker)) / marker_pixels,
    }
    return Image.fromarray(selected, mode="L"), metrics


def validate_template_mask(metrics: dict[str, object], size: tuple[int, int]) -> None:
    coverage = float(metrics["mask_coverage"])
    width_ratio = float(metrics["mask_bbox_width_ratio"])
    height_ratio = float(metrics["mask_bbox_height_ratio"])
    if not 0.04 <= coverage <= 0.60:
        raise RuntimeError(f"template marker mask coverage {coverage:.3f} is outside 0.04-0.60")
    if width_ratio < 0.18 or height_ratio < 0.18:
        raise RuntimeError("template marker mask is too small to represent a full-size product")
    if float(metrics.get("marker_residual_ratio", 0.0)) > 0.012:
        raise RuntimeError("template marker mask leaves visible placeholder pixels outside the product mask")


def composite_print_on_template(
    print_path: Path,
    template: Image.Image,
    mask: Image.Image,
    target: ProductTarget,
) -> Image.Image:
    with Image.open(print_path) as opened:
        artwork = ImageOps.exif_transpose(opened).convert("RGB")
    bbox = mask.getbbox()
    if bbox is None:
        raise RuntimeError("template marker mask is empty")
    artwork_layer = warp_artwork(artwork, mask, bbox, target)
    if target.name == "blanket":
        artwork_layer = displace_blanket_artwork(artwork_layer, template, mask)
    fold_shade, weave_detail, edge_occlusion = fabric_material_maps(template, mask)
    art_np = np.asarray(artwork_layer.convert("RGB"), dtype=np.float32)
    shaded = apply_fabric_material(art_np, fold_shade, weave_detail, edge_occlusion)
    printed = Image.fromarray(shaded, mode="RGB")
    result = template.convert("RGBA")
    printed.putalpha(mask)
    result.alpha_composite(printed)
    return result.convert("RGB")


def warp_artwork(artwork: Image.Image, mask: Image.Image, bbox: tuple[int, int, int, int], target: ProductTarget) -> Image.Image:
    canvas = Image.new("RGB", mask.size, (0, 0, 0))
    if target.name != "rug":
        canvas.paste(ImageOps.fit(artwork, (bbox[2] - bbox[0], bbox[3] - bbox[1]), Image.Resampling.LANCZOS), bbox[:2])
        return canvas
    destination = rug_quad(mask)
    if destination is None:
        canvas.paste(ImageOps.fit(artwork, (bbox[2] - bbox[0], bbox[3] - bbox[1]), Image.Resampling.LANCZOS), bbox[:2])
        return canvas
    try:
        import cv2
    except ImportError:
        return canvas
    art = np.asarray(artwork, dtype=np.uint8)
    src = np.float32([[0, 0], [art.shape[1] - 1, 0], [art.shape[1] - 1, art.shape[0] - 1], [0, art.shape[0] - 1]])
    matrix = cv2.getPerspectiveTransform(src, destination.astype(np.float32))
    warped = cv2.warpPerspective(art, matrix, mask.size, flags=cv2.INTER_LANCZOS4)
    return Image.fromarray(warped, mode="RGB")


def rug_quad(mask: Image.Image) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    binary = (np.asarray(mask) > 64).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    approximation = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
    points = approximation.reshape(-1, 2) if len(approximation) == 4 else cv2.boxPoints(cv2.minAreaRect(contour))
    if len(points) != 4:
        return None
    return order_quad(np.asarray(points, dtype=np.float32))


def order_quad(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def template_shade(template: Image.Image, mask: Image.Image) -> np.ndarray:
    fold_shade, _, _ = fabric_material_maps(template, mask)
    return fold_shade


def fabric_material_maps(template: Image.Image, mask: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive fold, weave, and edge-light maps from the blank textile template."""
    try:
        import cv2
    except ImportError:
        rgb = np.asarray(template.convert("RGB"), dtype=np.float32)
        luminance = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
        inside = np.asarray(mask) > 64
        median = float(np.median(luminance[inside])) if np.any(inside) else 128.0
        fold = np.clip(luminance / max(1.0, median), 0.68, 1.28)
        return fold, np.zeros_like(fold), np.zeros_like(fold)

    rgb = np.asarray(template.convert("RGB"), dtype=np.float32)
    luminance = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    inside = np.asarray(mask) > 64
    low_frequency = cv2.GaussianBlur(luminance, (0, 0), sigmaX=24)
    median = float(np.median(low_frequency[inside])) if np.any(inside) else 128.0
    broad_shade = low_frequency / max(1.0, median)
    fold_base = cv2.GaussianBlur(luminance, (0, 0), sigmaX=4.0)
    fold_detail = (fold_base - low_frequency) / max(1.0, median)
    # A blanket's broad folds carry substantially more visual information than weave.
    # Keep the multiplier bounded so the approved print's palette is never destroyed.
    fold_shade = np.clip(broad_shade + fold_detail * 1.35, 0.52, 1.32)

    weave_base = cv2.GaussianBlur(luminance, (0, 0), sigmaX=2.2)
    weave_detail = np.clip((luminance - weave_base) / 42.0, -1.0, 1.0)

    binary = (np.asarray(mask) > 64).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    edge_occlusion = np.clip((8.0 - distance) / 8.0, 0.0, 1.0)
    return fold_shade, weave_detail, edge_occlusion


def apply_fabric_material(
    artwork: np.ndarray,
    fold_shade: np.ndarray,
    weave_detail: np.ndarray,
    edge_occlusion: np.ndarray,
) -> np.ndarray:
    """Transfer textile shading and microtexture without recoloring the approved artwork."""
    weave_factor = 1.0 + weave_detail * 0.11
    edge_factor = 1.0 - edge_occlusion * 0.13
    material_factor = fold_shade * weave_factor * edge_factor
    shaded = artwork * material_factor[..., None]
    # Slightly reduce saturation in deep folds. This prevents saturated patterns from
    # looking like a flat screen print pasted over a dark piece of furniture.
    luma = shaded[..., 0] * 0.2126 + shaded[..., 1] * 0.7152 + shaded[..., 2] * 0.0722
    fold_depth = np.clip((0.92 - fold_shade) / 0.35, 0.0, 1.0)[..., None]
    shaded = shaded * (1.0 - fold_depth * 0.055) + luma[..., None] * fold_depth * 0.055
    return np.clip(shaded, 0, 255).astype(np.uint8)


def displace_blanket_artwork(artwork: Image.Image, template: Image.Image, mask: Image.Image) -> Image.Image:
    """Warp the print along visible blanket folds from the blank textile template."""
    try:
        import cv2
    except ImportError:
        return artwork
    rgb = np.asarray(template.convert("RGB"), dtype=np.float32)
    luminance = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    # Isolate mid-scale fold ridges. The gradients define a flow field that bends
    # the approved print in the same direction as the template's fabric surface.
    mid = cv2.GaussianBlur(luminance, (0, 0), sigmaX=5.0)
    broad = cv2.GaussianBlur(luminance, (0, 0), sigmaX=28.0)
    fold_signal = mid - broad
    fold_signal = cv2.GaussianBlur(fold_signal, (0, 0), sigmaX=5.0)
    grad_x = cv2.Sobel(fold_signal, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(fold_signal, cv2.CV_32F, 0, 1, ksize=3)
    grad_x = cv2.GaussianBlur(grad_x, (0, 0), sigmaX=4.0)
    grad_y = cv2.GaussianBlur(grad_y, (0, 0), sigmaX=4.0)
    binary = (np.asarray(mask) > 64).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    # Leave the edge stable so the printed shape keeps a clean, believable binding.
    interior_weight = np.clip(distance / 14.0, 0.0, 1.0)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    # Generated cyan templates can contain chroma noise that looks like a fold to a
    # gradient detector. Limit geometric motion to a few pixels; lighting conveys
    # the larger folds without turning the print into melted artwork.
    scale = min(2.0, max(0.45, min(template.size) / 1500.0))
    normalizer = np.percentile(magnitude[binary > 0], 92) if np.any(binary) else 1.0
    normalizer = max(1e-3, float(normalizer))
    grad_x = np.clip(grad_x / normalizer, -1.0, 1.0) * scale * interior_weight
    grad_y = np.clip(grad_y / normalizer, -1.0, 1.0) * scale * interior_weight
    height, width = luminance.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = np.clip(grid_x + grad_x, 0, width - 1)
    map_y = np.clip(grid_y + grad_y, 0, height - 1)
    warped = cv2.remap(np.asarray(artwork.convert("RGB")), map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(warped, mode="RGB")
