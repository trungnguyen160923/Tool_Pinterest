from __future__ import annotations

from dataclasses import dataclass

from .config import ProductTarget


@dataclass(frozen=True)
class MockupProductProfile:
    product_type: str
    scene_requirement: str
    material_requirement: str
    allowed_pose: str
    forbidden_presentations: tuple[str, ...]

    def prompt_contract(self) -> str:
        forbidden = ", ".join(self.forbidden_presentations)
        return (
            f"INTENDED PRODUCT TYPE: {self.product_type}. "
            f"SCENE REQUIREMENT: {self.scene_requirement}. "
            f"MATERIAL REQUIREMENT: {self.material_requirement}. "
            f"ALLOWED POSE: {self.allowed_pose}. "
            f"FORBIDDEN PRESENTATIONS: {forbidden}."
        )


def mockup_profile_for_target(target: ProductTarget) -> MockupProductProfile:
    name = target.name.strip().lower()
    if name == "blanket":
        return MockupProductProfile(
            product_type="a full-size soft woven throw blanket",
            scene_requirement="place a large full-size blanket across most of a bed, sofa, or armchair in a believable home setting; it must visibly cover a substantial furniture surface, never sit as a small object on a table",
            material_requirement="it must read as a soft textile blanket with plausible weave, thickness, and scale",
            allowed_pose="neatly folded or naturally draped over furniture with a soft hanging edge; the artwork must remain recognizably preserved",
            forbidden_presentations=("coaster", "placemat", "table runner", "bath mat", "rug", "doormat", "small table object", "coffee table", "dining table", "window hanging", "wall decor", "garden flag"),
        )
    if name == "custom":
        return MockupProductProfile(
            product_type="the exact custom textile product shown in the reference",
            scene_requirement="use a setting that is physically plausible for the original product silhouette",
            material_requirement="preserve the original material behavior and scale",
            allowed_pose="preserve the reference product identity and silhouette unless a physically plausible pose is required",
            forbidden_presentations=("unrelated product type", "window hanging", "garden flag"),
        )
    shape = target.rug_shape.strip().lower() if target.rug_shape else "rectangle"
    return MockupProductProfile(
        product_type=f"a full-size {shape} floor rug",
        scene_requirement="place it flat on a believable floor plane in an entryway, living room, or bedroom",
        material_requirement="it must read as a woven floor rug with plausible thickness and scale",
        allowed_pose=f"flat on the floor with a perspective-consistent {shape} silhouette and clearly visible outer edges",
        forbidden_presentations=("bath mat", "placemat", "table runner", "window hanging", "wall decor", "garden flag"),
    )
