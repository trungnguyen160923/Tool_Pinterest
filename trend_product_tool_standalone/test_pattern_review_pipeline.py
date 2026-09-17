from __future__ import annotations

import json
import py_compile
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from pinterest.trend_finder.semantic_analyzer import (
    NON_TEXTILE_STOPWORDS_REGEX,
    build_smart_queries,
    GeminiSemanticAnalyzer,
)
from pinterest.image_crawler.ranker import (
    score_image,
    inspiration_reject_reason,
    PolicyConfig,
)
from pinterest.shared.models import (
    ImageCandidate,
    VisionResult,
    RankedImage,
)
from trend_tool.config import PipelineConfig, ProductTarget, product_preset
from trend_tool.pipeline import (
    CandidateReviewItem,
    CandidateReviewPackage,
    run_production_from_candidates,
)


class TestTextilePatternAndReviewPipeline(unittest.TestCase):
    def test_syntax_compilation(self):
        """Ensure all modified python modules compile cleanly."""
        root = Path(__file__).resolve().parent
        files_to_check = [
            root / "app.py",
            root / "trend_tool" / "pipeline.py",
            root / "trend_tool" / "config.py",
            root / "trend_tool" / "task5_adapter.py",
            root / "pinterest" / "trend_finder" / "semantic_analyzer.py",
            root / "pinterest" / "image_crawler" / "hot_image_crawler.py",
            root / "pinterest" / "image_crawler" / "vision_filter.py",
            root / "pinterest" / "image_crawler" / "ranker.py",
            root / "pinterest" / "shared" / "models.py",
        ]
        for f in files_to_check:
            self.assertTrue(f.exists(), f"File missing: {f}")
            # py_compile will raise PyCompileError if there is any syntax error
            py_compile.compile(str(f), doraise=True)

    def test_non_textile_stopwords_filtering(self):
        """Ensure beauty, nails, hairstyles, phone wallpapers, and outfits are rejected."""
        reject_cases = [
            "french tip acrylic nails",
            "almond nails design",
            "gel nail art aesthetic",
            "short manicure inspiration",
            "messy bun hairstyle for girls",
            "butterfly haircut layers",
            "soft glam makeup tutorial",
            "dewy skincare routine",
            "cute iphone wallpaper aesthetic",
            "pastel lockscreen wallpaper",
            "phone cases collage",
            "fall ootd outfit ideas",
            "streetwear sneakers styling",
            "motivational quotes wallpaper",
        ]
        for term in reject_cases:
            match = NON_TEXTILE_STOPWORDS_REGEX.search(term)
            self.assertIsNotNone(
                match,
                f"Expected stopword match for: '{term}'",
            )
            self.assertTrue(
                bool(GeminiSemanticAnalyzer._hard_reject_reason(term)),
                f"Expected hard reject reason for: '{term}'",
            )

        allow_cases = [
            "checkerboard pattern",
            "boho floral area rug",
            "coastal grandmother decor",
            "vintage botanical tapestry",
            "moroccan geometric tile print",
            "cottagecore plaid blanket",
            "abstract celestial surface pattern",
            "mid century modern textile print",
        ]
        for term in allow_cases:
            match = NON_TEXTILE_STOPWORDS_REGEX.search(term)
            self.assertIsNone(
                match,
                f"False positive reject for valid home textile/decor term: '{term}'",
            )
            self.assertEqual(
                GeminiSemanticAnalyzer._hard_reject_reason(term),
                "",
                f"Expected empty reject reason for valid term: '{term}'",
            )

    def test_build_smart_queries(self):
        """Verify queries expand with surface pattern and textile modifiers."""
        queries = build_smart_queries("sage green gingham")
        query_texts = [q.query for q in queries]
        self.assertIn("sage green gingham surface pattern design", query_texts)
        self.assertIn("sage green gingham textile pattern flat", query_texts)
        self.assertIn("sage green gingham seamless pattern vector", query_texts)
        self.assertIn("sage green gingham rug design illustration", query_texts)
        self.assertIn("sage green gingham blanket pattern design", query_texts)

    def test_ranker_scoring_and_classification(self):
        """Verify 2D flat artwork receives bonus, direct-printable flag, and correct classification."""
        candidate = ImageCandidate(
            image_id="img_test_123",
            query="checkerboard surface pattern design",
            trend_id="trend_001",
            trend="checkerboard",
            image_url="https://example.com/art.jpg",
            local_path="",
            trend_strength=80.0,
            semantic_fit=85.0,
        )
        flat_vision = VisionResult(
            image_id="img_test_123",
            accepted=True,
            product_present=True,
            product_role="PRIMARY",
            product_confidence=0.95,
            product_visibility=95.0,
            trend_relevance=90.0,
            commercial_quality=92.0,
            aesthetic="clean_pattern",
            detected_product="flat_pattern",
            reason="Clear 2D surface pattern",
            confidence=0.95,
            flat_artwork_score=0.92,
            printability_score=0.88,
            is_lifestyle_scene=False,
            requires_extraction=False,
            is_collage=False,
            target_product_type="flat_pattern",
            main_subject="geometric_pattern",
        )
        score = score_image(candidate, flat_vision, is_direct_printable=True)
        # Score includes the +8.0 bonus
        self.assertGreater(score, 75.0)

        # Policy reject reason test: should be empty for valid flat artwork
        policy = PolicyConfig()
        reject = inspiration_reject_reason(candidate, flat_vision, policy)
        self.assertEqual(reject, "")

        # Test rejection of phone wallpaper
        rejected_vision = VisionResult(
            image_id="img_wallpaper_456",
            accepted=False,
            product_present=False,
            product_role="NONE",
            product_confidence=0.1,
            product_visibility=0.0,
            trend_relevance=20.0,
            commercial_quality=30.0,
            aesthetic="wallpaper",
            detected_product="phone_screen",
            reason="Phone wallpaper detected",
            confidence=0.90,
            reject_reason_code="REJECT_PHONE_WALLPAPER",
        )
        reject_wp = inspiration_reject_reason(candidate, rejected_vision, policy)
        self.assertEqual(reject_wp, "REJECT_PHONE_WALLPAPER")

    def test_direct_production_from_candidates(self):
        """Test Step 2: run_production_from_candidates end-to-end with direct design mode."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="test_prod_"))
        try:
            # Create a test input pattern image
            test_img_path = tmp_dir / "input_pattern.png"
            img = Image.new("RGB", (600, 800), color=(180, 100, 120))
            img.save(test_img_path)

            target = ProductTarget(
                name="rug",
                width_px=400,
                height_px=640,
                dpi=150,
            )
            config = PipelineConfig(
                target=target,
                output_root=tmp_dir,
                design_mode="direct",
                export_cmyk=True,
                enhancement_mode="task2_local",
                task4_mockup_engine="off",
            )

            item = CandidateReviewItem(
                image_id="cand_001",
                local_path=str(test_img_path.resolve()),
                image_url="",
                pin_url="https://pinterest.com/pin/123",
                pin_id="123",
                trend="checkerboard",
                query="checkerboard surface pattern design",
                image_score=90.0,
                flat_artwork_score=0.95,
                printability_score=0.90,
                classification="Flat Pattern",
                is_direct_printable=True,
                recommended=True,
                width=600,
                height=800,
                reason="High quality repeating pattern",
                motifs=["checkers"],
                source_role="PRIMARY",
            )

            result = run_production_from_candidates(
                selected_items=[item],
                config=config,
                run_dir=tmp_dir,
            )

            self.assertTrue(result.report_path.exists(), "report.html was not generated")
            self.assertTrue((tmp_dir / "stage_manifest.json").exists(), "stage_manifest.json was not generated")
            self.assertTrue((tmp_dir / "production_config.json").exists(), "production_config.json was not generated")

            # Check that final prints were exported
            rgb_pngs = [p for p in result.final_images if p.suffix.lower() == ".png"]
            cmyk_jpgs = [p for p in result.final_images if p.suffix.lower() == ".jpg"]
            self.assertGreaterEqual(len(rgb_pngs), 1, "Expected at least 1 final RGB PNG")
            self.assertGreaterEqual(len(cmyk_jpgs), 1, "Expected at least 1 final CMYK JPG")

            with Image.open(rgb_pngs[0]) as final_img:
                self.assertEqual(final_img.size, (400, 640), "Final PNG dimensions must match target")

            manifest_data = json.loads((tmp_dir / "stage_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_data.get("status"), "completed")
            self.assertEqual(manifest_data.get("design_mode"), "direct")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
