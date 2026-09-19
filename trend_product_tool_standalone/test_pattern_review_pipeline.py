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
    fork_selected_candidates_to_new_run,
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

    def test_bedroom_set_mockup_prompt_and_qa_alignment(self):
        """Verify direct AI lifestyle prompt and QA evaluator are aligned for bedroom set blankets."""
        from trend_tool.template_mockup import direct_ai_lifestyle_prompt, template_pose_for_index
        from trend_tool.printability import direct_ai_mockup_prompt

        target = ProductTarget(name="blanket", width_px=10000, height_px=11000, dpi=300)
        pose = template_pose_for_index(target, 3)
        self.assertEqual(pose.name, "bed_full_showcase")

        prompt = direct_ai_lifestyle_prompt(target, pose, "")
        # Composition checks:
        self.assertIn("Bedroom Set", prompt)
        self.assertIn("two matching printed pillowcases", prompt)
        # Blanket hem & drape checks:
        self.assertIn("clean", prompt.lower())
        self.assertIn("straight", prompt.lower())
        self.assertIn("sewn", prompt.lower())
        self.assertIn("strictly no wavy scalloped cutouts", prompt.lower())
        self.assertIn("no comforter/duvet box quilting stitches", prompt.lower())
        # Pillowcase aesthetic and scaling checks:
        self.assertIn("rather than repeating the entire dense pattern into tiny micro-icons", prompt.lower())
        self.assertIn("hero motifs", prompt.lower())
        self.assertIn("uncluttered", prompt.lower())

        # Direct AI Mockup QA Prompt alignment checks:
        qa_prompt = direct_ai_mockup_prompt(target, pose_name=pose.name, pose_requirement=pose.placement, require_matching_pillowcases=True)
        self.assertIn("clean_straight_hems_no_scallops", qa_prompt)
        self.assertIn("matching_pillowcases_present", qa_prompt)
        self.assertIn("pillowcases_clean_and_uncluttered", qa_prompt)
        self.assertIn("no_comforter_quilting_grids", qa_prompt)
        self.assertIn("typography_crisp_and_legible", qa_prompt)
        self.assertIn("clean, continuous straight modern sewn hems", qa_prompt.lower())
        self.assertIn("reject wavy scalloped edges", qa_prompt.lower())
        self.assertIn("motifs scaled naturally to pillow proportions without squishing or visual clutter", qa_prompt.lower())

    def test_direct_ai_qa_assessment_logic(self):
        """Verify assess_direct_ai_mockup decision logic accepts clean renders and rejects ruffles/clutter."""
        from unittest.mock import patch
        from trend_tool.printability import assess_direct_ai_mockup

        target = ProductTarget(name="blanket", width_px=10000, height_px=11000, dpi=300)
        mock_img = Image.new("RGB", (100, 100), color="white")

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "ref.png"
            mockup_path = Path(tmp) / "mockup.png"
            mock_img.save(ref_path)
            mock_img.save(mockup_path)

            base_assessment = {
                "artwork_identity_preserved": True,
                "product_type_correct": True,
                "full_size_scale_plausible": True,
                "fabric_material_believable": True,
                "fold_geometry_consistent": True,
                "occlusion_and_contact_believable": True,
                "lighting_coherent": True,
                "looks_like_flat_overlay": False,
                "looks_like_wrong_product": False,
                "matching_pillowcases_present": True,
                "pillowcases_clean_and_uncluttered": True,
                "clean_straight_hems_no_scallops": True,
                "no_comforter_quilting_grids": True,
                "typography_crisp_and_legible": True,
                "listing_realism_score": 92,
                "reason": "Passed all quality criteria.",
            }

            with patch("trend_tool.printability._vision_pair_assessment", return_value=base_assessment):
                decision = assess_direct_ai_mockup(
                    ref_path,
                    mockup_path,
                    target,
                    pose_name="bed_full_showcase",
                    require_matching_pillowcases=True,
                    backend="dummy",
                    model="dummy",
                )
                self.assertTrue(decision.accepted)

            # Rejection case 1: clean_straight_hems_no_scallops is False
            ruffled_assessment = dict(base_assessment)
            ruffled_assessment["clean_straight_hems_no_scallops"] = False
            ruffled_assessment["reason"] = "Ruffled scalloped edges detected on blanket hem."
            with patch("trend_tool.printability._vision_pair_assessment", return_value=ruffled_assessment):
                decision = assess_direct_ai_mockup(
                    ref_path,
                    mockup_path,
                    target,
                    pose_name="bed_full_showcase",
                    require_matching_pillowcases=True,
                    backend="dummy",
                    model="dummy",
                )
                self.assertFalse(decision.accepted)
                self.assertIn("Ruffled scalloped edges", decision.reason)

            # Rejection case 2: matching_pillowcases_present is False
            missing_pillows = dict(base_assessment)
            missing_pillows["matching_pillowcases_present"] = False
            missing_pillows["reason"] = "Missing matching pillowcases."
            with patch("trend_tool.printability._vision_pair_assessment", return_value=missing_pillows):
                decision = assess_direct_ai_mockup(
                    ref_path,
                    mockup_path,
                    target,
                    pose_name="bed_full_showcase",
                    require_matching_pillowcases=True,
                    backend="dummy",
                    model="dummy",
                )
                self.assertFalse(decision.accepted)

            # Rejection case 3: pillowcases_clean_and_uncluttered is False (cluttered/squished micro-repeats)
            cluttered_pillows = dict(base_assessment)
            cluttered_pillows["pillowcases_clean_and_uncluttered"] = False
            cluttered_pillows["reason"] = "Pillowcases have cluttered micro-repeats."
            with patch("trend_tool.printability._vision_pair_assessment", return_value=cluttered_pillows):
                decision = assess_direct_ai_mockup(
                    ref_path,
                    mockup_path,
                    target,
                    pose_name="bed_full_showcase",
                    require_matching_pillowcases=True,
                    backend="dummy",
                    model="dummy",
                )
                self.assertFalse(decision.accepted)

            # Rejection case 4: no_comforter_quilting_grids is False
            quilted_blanket = dict(base_assessment)
            quilted_blanket["no_comforter_quilting_grids"] = False
            quilted_blanket["reason"] = "Quilted duvet grid stitching visible."
            with patch("trend_tool.printability._vision_pair_assessment", return_value=quilted_blanket):
                decision = assess_direct_ai_mockup(
                    ref_path,
                    mockup_path,
                    target,
                    pose_name="bed_full_showcase",
                    require_matching_pillowcases=True,
                    backend="dummy",
                    model="dummy",
                )
                self.assertFalse(decision.accepted)

            # Rejection case 5: typography_crisp_and_legible is False
            garbled_text = dict(base_assessment)
            garbled_text["typography_crisp_and_legible"] = False
            garbled_text["reason"] = "Garbled pseudo-letters on blanket."
            with patch("trend_tool.printability._vision_pair_assessment", return_value=garbled_text):
                decision = assess_direct_ai_mockup(
                    ref_path,
                    mockup_path,
                    target,
                    pose_name="bed_full_showcase",
                    require_matching_pillowcases=True,
                    backend="dummy",
                    model="dummy",
                )
                self.assertFalse(decision.accepted)

            # Non-blanket target (e.g. rug) regression test:
            # Rug should pass even if blanket-specific clean_straight_hems_no_scallops is False
            rug_target = ProductTarget(name="rug", rug_shape="rectangle", width_px=4000, height_px=6400, dpi=150)
            rug_assessment = {
                "artwork_identity_preserved": True,
                "product_type_correct": True,
                "rug_shape_correct": True,
                "full_size_scale_plausible": True,
                "fabric_material_believable": True,
                "fold_geometry_consistent": True,
                "occlusion_and_contact_believable": True,
                "lighting_coherent": True,
                "looks_like_flat_overlay": False,
                "looks_like_wrong_product": False,
                "clean_straight_hems_no_scallops": False,
                "clean_hems_and_no_ruffles": False,
                "listing_realism_score": 90,
                "reason": "Realistic rectangular rug.",
            }
            with patch("trend_tool.printability._vision_pair_assessment", return_value=rug_assessment):
                rug_decision = assess_direct_ai_mockup(
                    ref_path,
                    mockup_path,
                    rug_target,
                    backend="dummy",
                    model="dummy",
                )
                self.assertTrue(rug_decision.accepted)

    def test_candidate_index_preservation_and_manifest_merge(self):
        """Verify that selecting candidates 1 and 5 preserves indices blanket_001 and blanket_005,
        and merges into existing stage_manifest.json rather than wiping existing products."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="test_preserve_idx_"))
        try:
            target = ProductTarget(name="blanket", width_px=300, height_px=330, dpi=100)
            config = PipelineConfig(
                target=target,
                output_root=tmp_dir,
                design_mode="direct",
                export_cmyk=False,
                enhancement_mode="task2_local",
                task4_mockup_engine="off",
            )

            # Create test pattern images for candidate 1 and candidate 5
            img1_path = tmp_dir / "cand1_img.png"
            img5_path = tmp_dir / "cand5_img.png"
            Image.new("RGB", (200, 220), color=(100, 150, 200)).save(img1_path)
            Image.new("RGB", (200, 220), color=(200, 100, 150)).save(img5_path)

            # Create candidate_review.json where cand1 is index 1, cand5 is index 5
            review_manifest = {
                "status": "ready_for_review",
                "run_dir": str(tmp_dir),
                "target_product": "blanket",
                "candidates": [
                    {"image_id": "cand_01", "local_path": str(img1_path), "candidate_index": 1},
                    {"image_id": "cand_02", "local_path": str(tmp_dir / "dummy2.png"), "candidate_index": 2},
                    {"image_id": "cand_03", "local_path": str(tmp_dir / "dummy3.png"), "candidate_index": 3},
                    {"image_id": "cand_04", "local_path": str(tmp_dir / "dummy4.png"), "candidate_index": 4},
                    {"image_id": "cand_05", "local_path": str(img5_path), "candidate_index": 5},
                ],
            }
            (tmp_dir / "candidate_review.json").write_text(json.dumps(review_manifest), encoding="utf-8")

            # Pre-seed stage_manifest.json with existing records for blanket_002
            existing_manifest = {
                "status": "completed",
                "workflow_mode": "trend_to_product",
                "design_mode": "direct",
                "selected_candidates_count": 1,
                "final_images_count": 1,
                "mockups_count": 0,
                "design_records": [
                    {
                        "source_path": str(tmp_dir / "dummy2.png"),
                        "output_path": str(tmp_dir / "artwork_designs" / "blanket_002_design.png"),
                        "mode": "direct",
                    }
                ],
                "enhancement_records": [],
                "product_render_records": [],
                "product_asset_records": [],
                "product_cutout_records": [],
                "artwork_generation_records": [],
                "rug_shape_records": [],
                "ai_background_final_records": [],
                "template_mockup_records": [],
                "mockup_quality_records": [],
            }
            (tmp_dir / "stage_manifest.json").write_text(json.dumps(existing_manifest), encoding="utf-8")

            # Now run production with ONLY Candidate 1 and Candidate 5
            item1 = CandidateReviewItem(
                image_id="cand_01",
                local_path=str(img1_path),
                image_url="",
                pin_url="",
                pin_id="",
                trend="",
                query="",
                image_score=90.0,
                flat_artwork_score=1.0,
                printability_score=1.0,
                classification="Flat Pattern",
                is_direct_printable=True,
                recommended=True,
                width=200,
                height=220,
                reason="",
                motifs=[],
                source_role="PRIMARY",
                candidate_index=1,
            )
            item5 = CandidateReviewItem(
                image_id="cand_05",
                local_path=str(img5_path),
                image_url="",
                pin_url="",
                pin_id="",
                trend="",
                query="",
                image_score=95.0,
                flat_artwork_score=1.0,
                printability_score=1.0,
                classification="Flat Pattern",
                is_direct_printable=True,
                recommended=True,
                width=200,
                height=220,
                reason="",
                motifs=[],
                source_role="PRIMARY",
                candidate_index=5,
            )

            res = run_production_from_candidates(
                selected_items=[item1, item5],
                config=config,
                run_dir=tmp_dir,
            )

            # Check that files on disk have blanket_001 and blanket_005 (NOT blanket_002!)
            designs = sorted(p.name for p in (tmp_dir / "artwork_designs").glob("*.png"))
            self.assertIn("blanket_001_design.png", designs)
            self.assertIn("blanket_005_design.png", designs)
            # Candidate 5 must NOT have overwritten blanket_002
            self.assertNotIn("blanket_002_design.png", [p.name for p in (tmp_dir / "final_print").glob("*blanket_002*")])

            # Check that stage_manifest.json merged blanket_002 from existing_manifest
            # along with newly produced blanket_001 and blanket_005
            manifest_after = json.loads((tmp_dir / "stage_manifest.json").read_text(encoding="utf-8"))
            design_bases = [
                Path(r["output_path"]).stem for r in manifest_after["design_records"]
            ]
            self.assertIn("blanket_001_design", design_bases)
            self.assertIn("blanket_002_design", design_bases)
            self.assertIn("blanket_005_design", design_bases)
            self.assertEqual(len(manifest_after["design_records"]), 3)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_fork_selected_candidates_to_new_run(self):
        """Verify that fork_selected_candidates_to_new_run clones ONLY the selected candidates
        into a new self-contained run directory, leaving the source run directory untouched."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="test_fork_"))
        try:
            source_run = tmp_dir / "run_source"
            source_crawl = source_run / "task5_crawl" / "downloaded_images"
            source_crawl.mkdir(parents=True, exist_ok=True)

            img_a = source_crawl / "cand_a.png"
            img_b = source_crawl / "cand_b.png"
            img_c = source_crawl / "cand_c.png"
            Image.new("RGB", (100, 100), color="red").save(img_a)
            Image.new("RGB", (100, 100), color="green").save(img_b)
            Image.new("RGB", (100, 100), color="blue").save(img_c)

            item_a = CandidateReviewItem(
                image_id="cand_a",
                local_path=str(img_a),
                image_url="",
                pin_url="",
                pin_id="",
                trend="",
                query="",
                image_score=90.0,
                flat_artwork_score=1.0,
                printability_score=1.0,
                classification="Flat Pattern",
                is_direct_printable=True,
                recommended=True,
                width=100,
                height=100,
                reason="",
                motifs=[],
                source_role="PRIMARY",
                candidate_index=1,
            )
            item_c = CandidateReviewItem(
                image_id="cand_c",
                local_path=str(img_c),
                image_url="",
                pin_url="",
                pin_id="",
                trend="",
                query="",
                image_score=85.0,
                flat_artwork_score=1.0,
                printability_score=1.0,
                classification="Flat Pattern",
                is_direct_printable=True,
                recommended=False,
                width=100,
                height=100,
                reason="",
                motifs=[],
                source_role="PRIMARY",
                candidate_index=3,
            )

            target = ProductTarget(name="blanket", width_px=200, height_px=220, dpi=100)
            config = PipelineConfig(
                target=target,
                output_root=tmp_dir,
                design_mode="direct",
                export_cmyk=False,
                enhancement_mode="task2_local",
                task4_mockup_engine="off",
            )

            # Fork ONLY items A and C to a new run
            new_items, new_run = fork_selected_candidates_to_new_run(
                selected_items=[item_a, item_c],
                source_run_dir=source_run,
                output_root=tmp_dir,
                config=config,
            )

            self.assertTrue(new_run.exists(), "New run directory was not created")
            self.assertNotEqual(source_run, new_run)

            # Check that only A and C were copied, B was NOT copied
            new_crawl = new_run / "task5_crawl" / "downloaded_images"
            self.assertTrue((new_crawl / "cand_a.png").exists())
            self.assertTrue((new_crawl / "cand_c.png").exists())
            self.assertFalse((new_crawl / "cand_b.png").exists())

            # Check new candidate_review.json
            new_rev = json.loads((new_run / "candidate_review.json").read_text(encoding="utf-8"))
            self.assertEqual(len(new_rev["candidates"]), 2)
            self.assertEqual(new_rev["forked_from"], source_run.name)
            self.assertEqual(new_items[0].candidate_index, 1)
            self.assertEqual(new_items[1].candidate_index, 2)

            # Run production in new_run
            res = run_production_from_candidates(new_items, config, run_dir=new_run)
            self.assertTrue(res.report_path.exists())
            manifest = json.loads((new_run / "stage_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["design_records"]), 2)

            # Check that source_run has NO final_print or artwork_designs (remained pristine)
            self.assertFalse((source_run / "final_print").exists())
            self.assertFalse((source_run / "artwork_designs").exists())

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

