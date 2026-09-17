from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

from .dedupe import DedupeDecision
from .comparison import ComparisonRow, build_comparison_rows


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_report(
    path: Path,
    config: object,
    decisions: list[DedupeDecision],
    final_images: list[Path],
    mockups: list[Path],
    stage_manifest: dict[str, object] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for decision in decisions:
        duplicate_of = html.escape(str(decision.duplicate_of or ""))
        rows.append(
            "<tr>"
            f"<td>{html.escape(decision.candidate.path.name)}</td>"
            f"<td>{html.escape(decision.candidate.keyword)}</td>"
            f"<td>{'kept' if decision.kept else 'rejected'}</td>"
            f"<td>{html.escape(decision.reason)}</td>"
            f"<td>{'' if decision.distance is None else decision.distance}</td>"
            f"<td>{duplicate_of}</td>"
            "</tr>"
        )

    final_cards = "\n".join(image_card(image_path, path.parent) for image_path in final_images)
    mockup_cards = "\n".join(image_card(image_path, path.parent) for image_path in mockups)
    comparison_rows = build_comparison_rows(path.parent, stage_manifest)
    comparison_html = comparison_table(comparison_rows, path.parent)
    payload = json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str)
    stage_payload = json.dumps(stage_manifest or {}, ensure_ascii=False, indent=2, default=str)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trend Product Tool Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f2ed; color: #1d1b18; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1, h2 {{ margin: 0 0 16px; }}
    section {{ margin: 28px 0; }}
    pre {{ overflow: auto; background: #fff; padding: 14px; border: 1px solid #ddd5ca; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ border-bottom: 1px solid #e2dbd0; padding: 9px; text-align: left; font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }}
    .card {{ background: #fff; border: 1px solid #ddd5ca; border-radius: 8px; padding: 10px; }}
    .card img {{ width: 100%; height: 260px; object-fit: contain; background: #fafafa; }}
    .name {{ font-size: 13px; margin-top: 8px; word-break: break-all; }}
    .compare {{ display: grid; gap: 16px; }}
    .compare-row {{ background: #fff; border: 1px solid #ddd5ca; border-radius: 8px; padding: 12px; }}
    .compare-head {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; font-size: 14px; }}
    .compare-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .compare-backgrounds {{ margin-top: 12px; display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }}
    .compare-cell {{ border: 1px solid #ece5db; background: #fafafa; padding: 8px; min-height: 210px; }}
    .compare-cell img {{ width: 100%; height: 220px; object-fit: contain; }}
    .label {{ font-weight: 700; margin-bottom: 6px; font-size: 13px; }}
    .missing {{ color: #8b2d20; font-size: 13px; padding-top: 80px; text-align: center; }}
  </style>
</head>
<body>
<main>
  <h1>Trend Product Tool Report</h1>
  <section>
    <h2>Config</h2>
    <pre>{html.escape(payload)}</pre>
  </section>
  <section>
    <h2>Final Print Files</h2>
    <div class="grid">{final_cards}</div>
  </section>
  <section>
    <h2>Comparison</h2>
    {comparison_html}
  </section>
  <section>
    <h2>Mockups</h2>
    <div class="grid">{mockup_cards}</div>
  </section>
  <section>
    <h2>Production Stages</h2>
    <pre>{html.escape(stage_payload)}</pre>
  </section>
  <section>
    <h2>Dedupe Decisions</h2>
    <table>
      <thead><tr><th>File</th><th>Keyword</th><th>Status</th><th>Reason</th><th>Distance</th><th>Duplicate Of</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return path


def image_card(path: Path | object, base_dir: Path) -> str:
    path = first_asset_path(path)
    if path is None:
        return ""
    src = relative_src(path, base_dir)
    return (
        '<div class="card">'
        f'<img src="{html.escape(src)}" alt="">'
        f'<div class="name">{html.escape(path.name)}</div>'
        "</div>"
    )


def comparison_table(rows: list[ComparisonRow], base_dir: Path) -> str:
    if not rows:
        return "<p>No comparison rows available.</p>"
    rendered = []
    for row in rows:
        status = html.escape(row.status or "unknown")
        reason = html.escape(row.reason)
        label = html.escape(row.product_label)
        rendered.append(
            '<div class="compare-row">'
            f'<div class="compare-head"><strong>#{row.index}</strong><span>{status}</span>'
            f'<span>{label}</span><span>{reason}</span></div>'
            '<div class="compare-grid">'
            f'{comparison_image_cell("Pinterest source", row.source_path, base_dir)}'
            f'{comparison_image_cell("Product cutout", row.cutout_white_path or row.cutout_path, base_dir)}'
            f'{comparison_image_cell("Final print", row.final_print_path, base_dir)}'
            "</div>"
            f'<div class="label" style="margin-top:12px">AI backgrounds ({len(row.ai_background_paths)})</div>'
            f'<div class="compare-backgrounds">{comparison_background_cells(row, base_dir)}</div>'
            "</div>"
        )
    return '<div class="compare">' + "\n".join(rendered) + "</div>"


def comparison_image_cell(label: str, path: Path | object | None, base_dir: Path) -> str:
    path = first_asset_path(path)
    if path is None or not path.exists():
        body = '<div class="missing">missing</div>'
        name = ""
    else:
        src = relative_src(path, base_dir)
        body = f'<img src="{html.escape(src)}" alt="">'
        name = f'<div class="name">{html.escape(path.name)}</div>'
    return f'<div class="compare-cell"><div class="label">{html.escape(label)}</div>{body}{name}</div>'


def comparison_background_cells(row: ComparisonRow, base_dir: Path) -> str:
    paths = row.ai_background_paths or (() if row.ai_background_path is None else (row.ai_background_path,))
    if not paths:
        return comparison_image_cell("AI background", None, base_dir)
    return "".join(
        comparison_image_cell(f"View {index}", image_path, base_dir)
        for index, image_path in enumerate(paths, start=1)
    )


def relative_src(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        return path.resolve().as_uri()


def first_asset_path(value: Path | object | None) -> Path | None:
    """Normalize legacy single paths and newer multi-view asset collections."""
    if isinstance(value, Path):
        return value
    if isinstance(value, (list, tuple, set)):
        for item in value:
            path = first_asset_path(item)
            if path is not None:
                return path
        return None
    if isinstance(value, str) and value.strip():
        return Path(value)
    return None

