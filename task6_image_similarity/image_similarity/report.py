from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .models import QueryFeatures, SimilarityResult
from .utils import dataclass_to_dict, relpath_for_html, write_csv, write_json


def result_rows(results: list[SimilarityResult]) -> list[dict[str, Any]]:
    rows = []
    for item in results:
        metadata = item.metadata or {}
        rows.append(
            {
                "rank": item.rank,
                "score": item.score,
                "semantic_score": item.semantic_score,
                "color_score": item.color_score,
                "edge_score": item.edge_score,
                "hash_score": item.hash_score,
                "aspect_score": item.aspect_score,
                "local_score": metadata.get("local_score", ""),
                "gemini_score": metadata.get("gemini_score", ""),
                "gemini_match_level": metadata.get("gemini_match_level", ""),
                "gemini_reason": metadata.get("gemini_reason", ""),
                "filename": item.filename,
                "path": item.path,
                "width": item.width,
                "height": item.height,
                "trend": metadata.get("trend", ""),
                "query": metadata.get("query", ""),
                "pin_url": metadata.get("pin_url", ""),
                "image_url": metadata.get("image_url", ""),
                "detected_product": metadata.get("detected_product", ""),
                "target_product_type": metadata.get("target_product_type", ""),
            }
        )
    return rows


def export_results(output_dir: Path, *, query: QueryFeatures, results: list[SimilarityResult], index_payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "similarity_results.json",
        {
            "query": dataclass_to_dict(query),
            "index": {
                "gallery": index_payload.get("gallery"),
                "image_count": index_payload.get("image_count"),
                "embedding_model": index_payload.get("embedding_model"),
                "generated_at": index_payload.get("generated_at"),
            },
            "results": results,
        },
    )
    write_csv(
        output_dir / "similarity_results.csv",
        result_rows(results),
        [
            "rank",
            "score",
            "semantic_score",
            "color_score",
            "edge_score",
            "hash_score",
            "aspect_score",
            "local_score",
            "gemini_score",
            "gemini_match_level",
            "gemini_reason",
            "filename",
            "path",
            "width",
            "height",
            "trend",
            "query",
            "pin_url",
            "image_url",
            "detected_product",
            "target_product_type",
        ],
    )
    render_report(output_dir / "similarity_report.html", query=query, results=results, index_payload=index_payload)


def gemini_chips(metadata: dict[str, Any]) -> str:
    score = metadata.get("gemini_score")
    if score == "" or score is None:
        return ""
    local_score = metadata.get("local_score")
    level = html.escape(str(metadata.get("gemini_match_level") or ""))
    try:
        gemini_text = f"{float(score):.1f}"
    except Exception:
        gemini_text = html.escape(str(score))
    try:
        local_text = f"{float(local_score):.1f}"
    except Exception:
        local_text = html.escape(str(local_score or ""))
    local_chip = f"<span>local {local_text}</span>" if local_text else ""
    return f"<span>gemini {gemini_text}</span><span>{level}</span>{local_chip}"


def render_report(path: Path, *, query: QueryFeatures, results: list[SimilarityResult], index_payload: dict) -> None:
    base = path.parent
    query_src = html.escape(relpath_for_html(Path(query.path), base))
    mode = index_payload.get("embedding_model") or "classic visual fallback"
    cards = []
    for item in results:
        metadata = item.metadata or {}
        src = html.escape(relpath_for_html(Path(item.path), base))
        trend = html.escape(str(metadata.get("trend") or ""))
        query_text = html.escape(str(metadata.get("query") or ""))
        product = html.escape(str(metadata.get("detected_product") or metadata.get("target_product_type") or ""))
        pin_url = str(metadata.get("pin_url") or "")
        semantic = "n/a" if item.semantic_score is None else f"{item.semantic_score:.1f}"
        gemini_reason = html.escape(str(metadata.get("gemini_reason") or ""))
        cards.append(
            f"""
<article class="card">
  <img src="{src}" alt="{html.escape(item.filename)}" loading="lazy">
  <div class="pad">
    <div class="score">#{item.rank} · {item.score:.1f}</div>
    <div class="name">{html.escape(item.filename)}</div>
    <div class="meta">{trend}</div>
    <div class="meta">{query_text}</div>
    <div class="chips">
      <span>semantic {semantic}</span>
      <span>color {item.color_score:.1f}</span>
      <span>edge {item.edge_score:.1f}</span>
      <span>hash {item.hash_score:.1f}</span>
      {gemini_chips(metadata)}
    </div>
    <p>{product}</p>
    {f'<p>{gemini_reason}</p>' if gemini_reason else ''}
    {f'<a href="{html.escape(pin_url)}" target="_blank" rel="noreferrer">Open pin</a>' if pin_url else ''}
  </div>
</article>
"""
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image Similarity Report</title>
  <style>
    :root {{
      color-scheme: light;
      --paper: #f6f7f9;
      --surface: #ffffff;
      --ink: #17202a;
      --muted: #667386;
      --rule: #dfe5ee;
      --accent: #197c86;
      font-family: Arial, Helvetica, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); }}
    header {{ padding: 24px 32px; background: var(--surface); border-bottom: 1px solid var(--rule); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.1; letter-spacing: 0; }}
    .muted, .meta {{ color: var(--muted); }}
    main {{ padding: 24px 32px 40px; }}
    .query {{
      display: grid;
      grid-template-columns: minmax(220px, 360px) minmax(0, 1fr);
      gap: 20px;
      align-items: start;
      margin-bottom: 24px;
    }}
    .query img, .card img {{
      width: 100%;
      display: block;
      object-fit: cover;
      background: #e9edf3;
      border: 1px solid var(--rule);
    }}
    .query img {{ max-height: 420px; object-fit: contain; background: var(--surface); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 14px; }}
    .metric small {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 20px; overflow-wrap: anywhere; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }}
    .card {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; }}
    .card img {{ aspect-ratio: 4 / 3; border-width: 0 0 1px; }}
    .pad {{ padding: 12px; }}
    .score {{ color: var(--accent); font-weight: 700; }}
    .name {{ margin-top: 4px; font-weight: 700; overflow-wrap: anywhere; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }}
    .chips span {{ padding: 3px 7px; background: #eef2f7; border-radius: 6px; font-size: 12px; }}
    p {{ margin: 8px 0; color: var(--muted); line-height: 1.4; }}
    a {{ color: var(--accent); }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .query {{ grid-template-columns: 1fr; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Image Similarity Report</h1>
    <div class="muted">Mode: {html.escape(str(mode))} · Gallery images: {html.escape(str(index_payload.get("image_count", 0)))}</div>
  </header>
  <main>
    <section class="query">
      <img src="{query_src}" alt="Query image">
      <div class="summary">
        <div class="metric"><small>Input</small><strong>{html.escape(Path(query.path).name)}</strong></div>
        <div class="metric"><small>Dimensions</small><strong>{query.width} x {query.height}</strong></div>
        <div class="metric"><small>Top results</small><strong>{len(results)}</strong></div>
        <div class="metric"><small>Best score</small><strong>{results[0].score if results else 0:.1f}</strong></div>
      </div>
    </section>
    <section class="grid">{''.join(cards)}</section>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
