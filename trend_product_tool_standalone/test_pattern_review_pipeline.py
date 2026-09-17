from __future__ import annotations

import dataclasses
import json
import py_compile
import shutil
import subprocess
import sys
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
    rank_images,
)
from pinterest.image_crawler.vision_filter import ProductVisionFilter
from pinterest.shared.models import (
    ImageCandidate,
    VisionResult,
    RankedImage,
)
from trend_tool.config import PipelineConfig, ProductTarget, product_preset
from trend_tool.comparison import build_comparison_rows
from trend_tool.pipeline import (
    CandidateReviewItem,
    CandidateReviewPackage,
    run_production_from_candidates,
)
from trend_tool.config import restore_pipeline_config


class TestTextilePatternAndReviewPipeline(unittest.TestCase):
    def test_syntax_compilation(self):
        """Ensure all modified python modules compile cleanly."""
        root = Path(__file__).resolve().parent
        files_to_check = [
            root / "app.py",
            root / "trend_tool" / "pipeline.py",
            root / "trend_tool" / "config.py",
            root / "trend_tool" / "comparison.py",
            root / "trend_tool" / "task5_adapter.py",
            root / "pinterest" / "trend_finder" / "semantic_analyzer.py",
            root / "pinterest" / "image_crawler" / "hot_image_crawler.py",
            root / "pinterest" / "image_crawler" / "vision_filter.py",
            root / "pinterest" / "image_crawler" / "ranker.py",
            root / "pinterest" / "shared" / "models.py",
        ]
        for f in files_to_check:
            self.assertTrue(f.exists(), f"File missing: {f}")
            py_compile.compile(str(f), doraise=True)

    def test_cli_standalone_execution(self):
        """Ensure hot_image_crawler.py executes standalone without NameError or missing imports."""
        root = Path(__file__).resolve().parent
        crawler_script = root / "pinterest" / "image_crawler" / "hot_image_crawler.py"
        res = subprocess.run(
            [sys.executable, str(crawler_script), "--help"],
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        self.assertEqual(res.returncode, 0, f"hot_image_crawler CLI failed: {res.stderr}")
        self.assertIn("--crawl-purpose", res.stdout)

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
            # Compound words without spaces:
            "nailart ideas",
            "autumn nailart",
            "homescreen aesthetic",
            "cute phonecase",
            "pink lipgloss",
            "press-on-nails",
            "press on nails",
            "skincare routine",
            "eyeshadow palette",
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

    def test_vision_fallback_values(self):
        """Verify vision fallback assigns sensible printability scores when vision is off."""
        candidate = ImageCandidate(
            image_id="img_fb_1",
            query="floral pattern",
            trend_id="trend_001",
            trend="floral",
            image_url="https://example.com/test.jpg",
            trend_strength=70.0,
            semantic_fit=80.0,
        )
        filter_off = ProductVisionFilter(niche="rug", mode="off")
        fallback_res = filter_off._fallback(candidate)
        self.assertTrue(fallback_res.accepted)
        self.assertGreaterEqual(fallback_res.flat_artwork_score, 0.70)
        self.assertGreaterEqual(fallback_res.printability_score, 0.65)

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
        """Test Step 2: run_production_from_candidates end-to-end with direct design mode and comparison row linking."""
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

            # Verify comparison rows correctly identify final print path (not None/missing)
            rows = build_comparison_rows(tmp_dir, manifest_data)
            self.assertEqual(len(rows), 1)
            self.assertIsNotNone(rows[0].final_print_path, "Comparison row final_print_path must not be None")
            self.assertTrue(rows[0].final_print_path.exists(), "Linked final print file must exist")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_direct_production_non_rgb_formats(self):
        """Test production pipeline with Grayscale, Palette, and RGBA images."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="test_formats_"))
        try:
            target = ProductTarget(name="blanket", width_px=500, height_px=550, dpi=100)
            config = PipelineConfig(
                target=target,
                output_root=tmp_dir,
                design_mode="direct",
                export_cmyk=True,
                enhancement_mode="task2_local",
                task4_mockup_engine="off",
            )
            for mode, ext in [("L", "gray.png"), ("RGBA", "rgba.png"), ("P", "pal.png")]:
                img_path = tmp_dir / ext
                if mode == "L":
                    Image.new(mode, (200, 250), color=150).save(img_path)
                elif mode == "P":
                    im = Image.new("P", (200, 250))
                    im.putpalette([i % 256 for i in range(768)])
                    im.save(img_path)
                else:
                    Image.new(mode, (200, 250), color=(100, 150, 200, 255)).save(img_path)

                item = CandidateReviewItem(
                    image_id=f"cand_{mode}",
                    local_path=str(img_path.resolve()),
                    image_url="",
                    pin_url="",
                    pin_id="",
                    trend="blanket pattern",
                    query="blanket pattern",
                    image_score=85.0,
                    flat_artwork_score=0.90,
                    printability_score=0.85,
                    classification="Flat Pattern",
                    is_direct_printable=True,
                    recommended=True,
                    width=200,
                    height=250,
                    reason="",
                    motifs=[],
                    source_role="PRIMARY",
                )
                run_sub = tmp_dir / f"run_{mode}"
                res = run_production_from_candidates([item], config, run_dir=run_sub)
                self.assertGreaterEqual(len(res.final_images), 2)
                png_file = next(p for p in res.final_images if p.suffix.lower() == ".png")
                with Image.open(png_file) as final_img:
                    self.assertEqual(final_img.size, (500, 550))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_candidate_review_item_safety_and_config_restoration(self):
        """Verify CandidateReviewItem ignores unknown fields and restore_pipeline_config restores full config."""
        raw_dict = {
            "image_id": "safe_001",
            "local_path": "some/path.png",
            "image_url": "https://example.com/img.png",
            "pin_url": "https://pinterest.com/pin/1",
            "pin_id": "pin1",
            "trend": "coastal",
            "query": "coastal surface pattern",
            "image_score": 88.0,
            "flat_artwork_score": 0.90,
            "printability_score": 0.85,
            "classification": "Flat Pattern",
            "is_direct_printable": True,
            "recommended": True,
            "width": 800,
            "height": 1000,
            "reason": "great",
            "motifs": ["shells"],
            "source_role": "PRIMARY",
            "unknown_extra_field": "should_be_ignored",
            "another_field": 12345,
        }
        valid_fields = {f.name for f in dataclasses.fields(CandidateReviewItem)}
        clean_item = {k: v for k, v in raw_dict.items() if k in valid_fields}
        item = CandidateReviewItem(**clean_item)
        self.assertEqual(item.image_id, "safe_001")

        raw_cfg = {
            "target": {"name": "blanket", "width_px": 10000, "height_px": 11000, "dpi": 300},
            "output_root": str(Path("dummy_root")),
            "design_mode": "direct",
            "artwork_image_size": "4K",
            "enhancement_mode": "task2_local",
            "task4_mockup_engine": "direct_ai",
            "task4_ai_limit": 10,
            "unknown_cfg_key": "ignore_me",
        }
        restored = restore_pipeline_config(raw_cfg, fallback_root=Path("fallback"))
        self.assertEqual(restored.target.name, "blanket")
        self.assertEqual(restored.target.width_px, 10000)
        self.assertEqual(restored.design_mode, "direct")
        self.assertEqual(restored.artwork_image_size, "4K")
        self.assertEqual(restored.task4_ai_limit, 10)

    def test_empty_candidates_handling(self):
        """Verify production with empty candidates gracefully fails with manifest and report."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="test_empty_"))
        try:
            config = PipelineConfig(
                target=ProductTarget(name="rug", width_px=400, height_px=640),
                output_root=tmp_dir,
            )
            with self.assertRaises(RuntimeError):
                run_production_from_candidates([], config, run_dir=tmp_dir)

            self.assertTrue((tmp_dir / "stage_manifest.json").exists())
            manifest = json.loads((tmp_dir / "stage_manifest.json").read_text())
            self.assertEqual(manifest.get("status"), "failed")
            self.assertEqual(manifest.get("reason"), "no_valid_production_sources")
            self.assertTrue((tmp_dir / "report.html").exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
