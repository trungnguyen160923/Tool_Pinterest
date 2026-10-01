"""HTTP API for the trend-to-product pipeline.

Runs are intentionally asynchronous: Pinterest crawling, image generation, and
print rendering can take several minutes.  The API keeps job state in memory;
the actual assets and manifests remain in ``output/run_*`` as durable records.
"""

from __future__ import annotations

import mimetypes
import os
import json
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from trend_tool.config import PipelineConfig, ProductTarget, product_preset
from trend_tool.pipeline import (
    CandidateReviewItem,
    PipelineCancelled,
    PipelineResult,
    run_crawl_and_review_stage,
    run_pipeline,
    run_production_from_candidates,
)
from trend_tool.settings import TOOL_ROOT, env, env_float, env_int, load_tool_env, task5_token_path_from_env


load_tool_env()
OUTPUT_ROOT = (TOOL_ROOT / "output").resolve()

app = FastAPI(
    title="Trend Product API",
    version="1.0.0",
    description="Create print-ready trend product designs without using the Streamlit UI.",
)


class CreateJobRequest(BaseModel):
    niche: str = Field(min_length=1, max_length=300, description="Pinterest Trends niche, sent unchanged to Pinterest.")
    product: Literal["rug", "blanket", "custom"] = "rug"
    width_px: int | None = Field(default=None, ge=512, le=20_000)
    height_px: int | None = Field(default=None, ge=512, le=20_000)
    dpi: int = Field(default=300, ge=72, le=600)
    desired_output_count: int = Field(default=5, ge=1, le=20)
    trend_region: str = Field(default_factory=lambda: env("PINTEREST_REGION", "US"), min_length=2, max_length=10)
    trend_type: Literal["growing", "monthly", "yearly", "seasonal"] = "growing"
    design_mode: Literal["ai-artwork", "direct", "product-design", "pattern-repeat"] = "ai-artwork"
    artwork_image_size: Literal["1K", "2K", "4K"] = "2K"
    mockup_engine: Literal["direct_ai", "template_ai", "blender_3d"] = "direct_ai"
    ai_background_variants: int = Field(default=5, ge=1, le=5)
    remove_white_background: bool = False
    workflow_stage: Literal["auto", "crawl_and_review", "production"] = "auto"
    selected_candidates: list[dict[str, Any]] | None = None
    source_run_id: str | None = None

    @model_validator(mode="after")
    def validate_custom_size(self) -> "CreateJobRequest":
        if not self.niche.strip():
            raise ValueError("niche must not be blank")
        if self.product == "custom" and (self.width_px is None or self.height_px is None):
            raise ValueError("width_px and height_px are required when product is custom")
        if self.product != "custom" and (self.width_px is not None or self.height_px is not None):
            raise ValueError("width_px and height_px are only accepted when product is custom")
        return self


class Job:
    def __init__(self, request: CreateJobRequest) -> None:
        self.id = uuid.uuid4().hex
        self.request = request
        self.created_at = now()
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.status = "queued"
        self.logs: list[str] = [
            f"Khởi tạo Job {self.id}: niche='{request.niche}', product='{request.product}', stage='{request.workflow_stage}'"
        ]
        self.error: str | None = None
        self.result: PipelineResult | None = None
        self.review_package: Any | None = None
        self.run_dir: Path | None = None
        self.cancel_event = threading.Event()
        self.lock = threading.Lock()


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_config(request: CreateJobRequest) -> PipelineConfig:
    target = product_preset(request.product)
    if request.product == "custom":
        target = ProductTarget(
            name="custom",
            width_px=request.width_px or target.width_px,
            height_px=request.height_px or target.height_px,
            dpi=request.dpi,
            prefer_cmyk=target.prefer_cmyk,
            allow_custom_shape=True,
        )
    else:
        target = ProductTarget(
            name=target.name,
            width_px=target.width_px,
            height_px=target.height_px,
            dpi=request.dpi,
            prefer_cmyk=target.prefer_cmyk,
            allow_custom_shape=target.allow_custom_shape,
        )

    crawl_budget = min(250, max(30, request.desired_output_count * 10))
    return PipelineConfig(
        target=target,
        output_root=OUTPUT_ROOT,
        workflow_mode="trend_to_product",
        trend_niche=request.niche.strip(),
        trend_region=request.trend_region,
        trend_type=request.trend_type,
        trend_keyword_limit=env_int("TREND_KEYWORD_LIMIT", 50),
        trend_max_trends=env_int("TREND_MAX_TRENDS", 20),
        trend_min_semantic_fit=env_float("TREND_MIN_SEMANTIC_FIT", 35.0),
        trend_max_queries_per_trend=6,
        desired_output_count=request.desired_output_count,
        gemini_backend=env("GEMINI_BACKEND", "auto"),
        gemini_model=env("GEMINI_ANALYSIS_MODEL", "gemini-2.5-flash"),
        task5_token_path=task5_token_path_from_env(),
        task5_max_images_per_query=max(12, min(30, request.desired_output_count * 4)),
        task5_max_crawl_trends=5,
        task5_max_downloads=crawl_budget,
        task5_top_images=crawl_budget,
        task5_vision_mode="auto",
        task5_refresh_vision_cache=True,
        crop_mode="cover",
        dedupe_threshold=6,
        remove_white_background=request.remove_white_background,
        # This API is intentionally limited to the two requested deliverables:
        # AI lifestyle images and CMYK files for the production vendor.
        export_cmyk=True,
        design_mode=request.design_mode.replace("-", "_"),
        artwork_image_size=request.artwork_image_size,
        enhancement_mode="task2_local",
        task4_mockup_engine=request.mockup_engine,
        task4_ai_limit=request.desired_output_count,
        task4_variants_per_product=request.ai_background_variants,
        # Local placeholder mockups are not API deliverables.
        mockup_count=0,
    )


def run_job(job: Job) -> None:
    with job.lock:
        job.status = "running"
        job.started_at = now()

    def progress(message: str) -> None:
        with job.lock:
            job.logs.append(message)
            del job.logs[:-500]

    try:
        cfg = build_config(job.request)
        if job.request.workflow_stage == "crawl_and_review":
            review_pkg = run_crawl_and_review_stage(cfg, progress=progress, cancel_event=job.cancel_event)
            with job.lock:
                job.status = "ready_for_review"
                job.review_package = review_pkg
                job.run_dir = review_pkg.run_dir
        elif job.request.workflow_stage == "production":
            selected = job.request.selected_candidates or []
            src_dir = (OUTPUT_ROOT / job.request.source_run_id).resolve() if job.request.source_run_id else None
            prod_result = run_production_from_candidates(
                selected,
                cfg,
                run_dir=src_dir,
                progress=progress,
                cancel_event=job.cancel_event,
            )
            with job.lock:
                job.status = "completed"
                job.result = prod_result
                job.run_dir = prod_result.run_dir
        else:
            result = run_pipeline(cfg, progress=progress, cancel_event=job.cancel_event)
            with job.lock:
                job.status = "completed"
                job.result = result
                job.run_dir = result.run_dir
    except PipelineCancelled as exc:
        with job.lock:
            job.status = "cancelled"
            job.error = str(exc)
    except Exception as exc:
        with job.lock:
            job.status = "failed"
            job.error = str(exc)
    finally:
        with job.lock:
            job.finished_at = now()


def get_job_or_404(job_id: str) -> Job:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def asset(path: Path, run_dir: Path, kind: str) -> dict[str, Any]:
    relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
    return {
        "kind": kind,
        "name": path.name,
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "download_url": f"/v1/jobs/{{job_id}}/files/{relative}",
    }


def path_in_run(value: object, run_dir: Path) -> Path | None:
    """Return an existing file only when its resolved path stays inside this run."""
    if not value:
        return None
    candidate = Path(str(value)).resolve()
    root = run_dir.resolve()
    if candidate.is_file() and root in candidate.parents:
        return candidate
    return None


def marketing_image_paths(result: PipelineResult) -> list[Path]:
    """Find only accepted AI lifestyle/background outputs, never local mockups."""
    manifest_path = result.run_dir / "stage_manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(manifest, dict):
        return []

    paths: list[Path] = []
    for record_key, path_key in (
        ("template_mockup_records", "mockup_path"),
        ("ai_background_final_records", "lifestyle_path"),
    ):
        records = manifest.get(record_key, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict) or record.get("status") != "ok":
                continue
            path = path_in_run(record.get(path_key), result.run_dir)
            if path is not None and path not in paths:
                paths.append(path)
    return paths


def cmyk_print_paths(result: PipelineResult) -> list[Path]:
    return [
        path
        for path in result.final_images
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"} and "_cmyk" in path.stem.lower()
    ]


def deliverable_paths(result: PipelineResult) -> list[Path]:
    return [*marketing_image_paths(result), *cmyk_print_paths(result)]


def job_payload(job: Job) -> dict[str, Any]:
    with job.lock:
        payload: dict[str, Any] = {
            "job_id": job.id,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "logs": list(job.logs),
            "status_url": f"/v1/jobs/{job.id}",
            "cancel_url": f"/v1/jobs/{job.id}",
        }
        result = job.result
        review_pkg = job.review_package
        current_run_dir = job.run_dir

    if review_pkg is not None and current_run_dir is not None:
        cand_list = []
        for c in review_pkg.candidates:
            c_dict = c.to_dict() if hasattr(c, "to_dict") else dict(c)
            lp = c_dict.get("local_path")
            if lp:
                try:
                    rel = Path(lp).resolve().relative_to(current_run_dir.resolve()).as_posix()
                    c_dict["download_url"] = f"/v1/jobs/{job.id}/files/{rel}"
                except Exception:
                    pass
            cand_list.append(c_dict)
        payload["run_id"] = current_run_dir.name
        payload["candidates"] = cand_list
        payload["total_candidates"] = len(cand_list)
        payload["direct_printable_count"] = sum(1 for c in cand_list if c.get("is_direct_printable"))

    if result is None:
        return payload

    run_dir = result.run_dir
    marketing_images = [asset(path, run_dir, "marketing_image") for path in marketing_image_paths(result)]
    cmyk_print_files = [asset(path, run_dir, "print_cmyk") for path in cmyk_print_paths(result)]
    for item in [*marketing_images, *cmyk_print_files]:
        item["download_url"] = item["download_url"].format(job_id=job.id)

    rug_shape_records = []
    detected_shape = "rectangle"
    manifest_path = run_dir / "stage_manifest.json"
    if manifest_path.is_file():
        try:
            m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            rug_shape_records = m_data.get("rug_shape_records") or []
            if rug_shape_records and isinstance(rug_shape_records[0], dict):
                detected_shape = rug_shape_records[0].get("shape") or "rectangle"
        except Exception:
            pass

    payload["output"] = {
        "run_id": run_dir.name,
        "marketing_images": marketing_images,
        "print_cmyk_images": cmyk_print_files,
        "rug_shape": detected_shape,
        "rug_shape_records": rug_shape_records,
        "summary": {
            "marketing_images": len(marketing_images),
            "print_cmyk_images": len(cmyk_print_files),
        },
    }
    return payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/jobs")
def create_job(request: CreateJobRequest, wait: bool = Query(default=False)) -> JSONResponse:
    job = Job(request)
    with jobs_lock:
        jobs[job.id] = job
    worker = threading.Thread(target=run_job, args=(job,), name=f"trend-product-{job.id[:8]}", daemon=True)
    worker.start()
    if wait:
        worker.join()
        return JSONResponse(status_code=status.HTTP_200_OK, content=job_payload(job))
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=job_payload(job))


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return job_payload(get_job_or_404(job_id))


@app.delete("/v1/jobs/{job_id}", status_code=status.HTTP_202_ACCEPTED)
def cancel_job(job_id: str) -> dict[str, Any]:
    job = get_job_or_404(job_id)
    with job.lock:
        if job.status in {"completed", "failed", "cancelled"}:
            raise HTTPException(status_code=409, detail=f"Job is already {job.status}")
        job.cancel_event.set()
        if job.status == "queued":
            job.status = "cancelling"
    return job_payload(job)


@app.get("/v1/jobs/{job_id}/files/{relative_path:path}")
def download_file(job_id: str, relative_path: str) -> FileResponse:
    job = get_job_or_404(job_id)
    with job.lock:
        result = job.result
        run_dir = (result.run_dir if result else job.run_dir)
    if run_dir is None:
        raise HTTPException(status_code=409, detail="Files are available only after a run directory is initialized")
    run_dir = run_dir.resolve()
    candidate = (run_dir / relative_path).resolve()
    if candidate != run_dir and run_dir not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    return FileResponse(candidate, media_type=media_type, filename=candidate.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=os.getenv("TREND_PRODUCT_API_HOST", "127.0.0.1"), port=int(os.getenv("TREND_PRODUCT_API_PORT", "8000")), reload=False)

