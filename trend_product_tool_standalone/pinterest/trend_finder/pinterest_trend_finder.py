from __future__ import annotations

import argparse
import html
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pinterest.shared.cache import JsonCache
    from pinterest.shared.models import TrendPackage
    from pinterest.shared.utils import (
        clamp,
        configure_logging,
        env,
        html_page,
        normalize_text,
        stable_id,
        utc_now_iso,
        write_csv,
        write_json,
    )
    from pinterest.trend_finder.models import TrendCandidate
    from pinterest.trend_finder.pinterest_client import PinterestApiError, PinterestClient
    from pinterest.trend_finder.semantic_analyzer import GeminiSemanticAnalyzer
else:
    from ..shared.cache import JsonCache
    from ..shared.models import TrendPackage
    from ..shared.utils import (
        clamp,
        configure_logging,
        env,
        html_page,
        normalize_text,
        stable_id,
        utc_now_iso,
        write_csv,
        write_json,
    )
    from .models import TrendCandidate
    from .pinterest_client import PinterestApiError, PinterestClient
    from .semantic_analyzer import GeminiSemanticAnalyzer


LOG = logging.getLogger("pinterest.trend_finder")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[0]


def score_from_rank(rank: int, total: int) -> float:
    if total <= 1:
        return 70.0
    return round(100.0 - ((rank - 1) / max(1, total - 1)) * 45.0, 2)


def score_keyword(item: dict[str, Any], rank: int, total: int) -> float:
    base = score_from_rank(rank, total)
    growth_values = []
    for key in ("pct_growth_wow", "pct_growth_mom", "pct_growth_yoy"):
        try:
            growth_values.append(float(item.get(key)))
        except (TypeError, ValueError):
            pass
    growth_bonus = 0.0
    if growth_values:
        growth_bonus = min(18.0, max(growth_values) / 10000.0 * 18.0)
    return clamp(base + growth_bonus, default=base)


def first_text(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def payload_items(payload: Any, keys: tuple[str, ...]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = payload_items(value, keys)
            if nested:
                return nested
    return []


def collect_pinterest_trends(
    *,
    client: PinterestClient,
    region: str,
    trend_type: str,
    interest: str,
    limit: int,
) -> tuple[list[TrendCandidate], dict[str, Any], dict[str, str]]:
    raw: dict[str, Any] = {}
    errors: dict[str, str] = {}

    endpoints: list[tuple[str, str, dict[str, Any]]] = [
        (
            "trending_keywords",
            f"/trends/keywords/{region}/top/{trend_type}",
            {"limit": limit},
        ),
        (
            "featured_topics",
            "/trends/topics/featured",
            {"region": region},
        ),
        (
            "shopping_trends",
            "/trends/product_categories/trending",
            {"region": region},
        ),
        (
            "editorial_trends",
            "/trends/editorial_articles",
            {"region": region},
        ),
    ]
    if interest:
        endpoints[0][2]["interests"] = [interest]

    for name, path, params in endpoints:
        try:
            payload = client.get(path, params=params)
            raw[name] = payload
            LOG.info("Pinterest %s: ok", name)
        except PinterestApiError as exc:
            raw[name] = {"error": str(exc), "payload": exc.payload}
            errors[name] = str(exc)
            LOG.warning("Pinterest %s unavailable: %s", name, exc)
            if "WinError 10013" in str(exc):
                LOG.error("Stopping Pinterest collection early because Windows is blocking outbound HTTPS.")
                break

    candidates: list[TrendCandidate] = []
    seen: set[str] = set()

    keyword_items = payload_items(raw.get("trending_keywords"), ("trends", "items", "keywords"))
    if keyword_items:
        for index, item in enumerate(keyword_items[:limit], start=1):
            if not isinstance(item, dict):
                continue
            name = first_text(item, ("keyword", "name", "title"))
            if not name:
                continue
            key = normalize_text(name)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                TrendCandidate(
                    candidate_id=stable_id("keyword", name),
                    name=name,
                    source="trending_keywords",
                    rank=index,
                    strength=score_keyword(item, index, len(keyword_items)),
                    metrics={
                        "pct_growth_wow": item.get("pct_growth_wow"),
                        "pct_growth_mom": item.get("pct_growth_mom"),
                        "pct_growth_yoy": item.get("pct_growth_yoy"),
                    },
                    raw=item,
                )
            )

    featured = raw.get("featured_topics")
    groups = payload_items(featured, ("items", "featured_topics", "trends"))
    if isinstance(groups, list):
        rank = 0
        for group in groups:
            trends = group.get("trends") if isinstance(group, dict) else None
            if not isinstance(trends, list):
                trends = [group]
            for item in trends:
                if not isinstance(item, dict):
                    continue
                name = first_text(item, ("title", "name", "keyword"))
                if not name:
                    continue
                key = normalize_text(name)
                if key in seen:
                    continue
                seen.add(key)
                rank += 1
                candidates.append(
                    TrendCandidate(
                        candidate_id=stable_id("featured", name),
                        name=name,
                        source="featured_topics",
                        rank=rank,
                        strength=score_from_rank(rank, max(rank, 50)),
                        metrics={},
                        raw=item,
                    )
                )

    for source_name in ("shopping_trends", "editorial_trends"):
        payload = raw.get(source_name)
        items = payload_items(payload, ("items", "trends", "categories", "articles"))
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items[:limit], start=1):
            if not isinstance(item, dict):
                continue
            name = first_text(item, ("title", "name", "keyword", "category"))
            if not name:
                continue
            key = normalize_text(name)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                TrendCandidate(
                    candidate_id=stable_id(source_name, name),
                    name=name,
                    source=source_name,
                    rank=index,
                    strength=score_from_rank(index, len(items)),
                    metrics={k: item.get(k) for k in ("pct_growth_wow", "pct_growth_mom", "pct_growth_yoy") if k in item},
                    raw=item,
                )
            )

    candidates.sort(key=lambda item: item.strength, reverse=True)
    return candidates, raw, errors


def export_package_csv(path: Path, package: TrendPackage) -> None:
    rows = []
    for item in package.trends:
        rows.append(
            {
                "trend_id": item.trend_id,
                "trend": item.trend,
                "trend_strength": item.trend_strength,
                "relationship": item.relationship,
                "semantic_fit": item.semantic_fit,
                "queries": " | ".join(query.query for query in item.queries),
                "sources": " | ".join(item.sources),
                "reason": item.reason,
            }
        )
    write_csv(
        path,
        rows,
        [
            "trend_id",
            "trend",
            "trend_strength",
            "relationship",
            "semantic_fit",
            "queries",
            "sources",
            "reason",
        ],
    )


def render_report(path: Path, package: TrendPackage) -> None:
    rows = []
    for item in package.trends:
        query_html = "".join(
            f'<span class="pill">{html.escape(query.query)}</span>'
            for query in item.queries
        )
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(item.trend)}</strong><br><span class='muted'>{html.escape(item.trend_id)}</span></td>"
            f"<td class='score'>{item.trend_strength:.1f}</td>"
            f"<td>{html.escape(item.relationship)}</td>"
            f"<td class='score'>{item.semantic_fit:.1f}</td>"
            f"<td>{query_html}</td>"
            f"<td>{html.escape(item.reason)}</td>"
            "</tr>"
        )

    body = f"""
<header>
  <h1>Trend Package</h1>
  <div class="muted">Niche: {html.escape(package.niche)} · Region: {html.escape(package.region)} · Generated: {html.escape(package.generated_at)}</div>
</header>
<main>
  <table>
    <thead>
      <tr><th>Trend</th><th>Strength</th><th>Relationship</th><th>Fit</th><th>Queries</th><th>Reason</th></tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</main>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_page("Trend Package", body), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find Pinterest hot trends for a niche and export a trend package.")
    parser.add_argument("--niche", required=True)
    parser.add_argument("--region", default=env("PINTEREST_REGION", "US"))
    parser.add_argument("--trend-type", default=env("PINTEREST_TREND_TYPE", "growing"), choices=["growing", "monthly", "yearly", "seasonal"])
    parser.add_argument("--interest", default=env("PINTEREST_INTEREST", ""))
    parser.add_argument("--keyword-limit", type=int, default=50)
    parser.add_argument("--output", default="trend_output")
    parser.add_argument("--token-path", default="")
    parser.add_argument("--timeout", type=int, default=int(env("PINTEREST_TIMEOUT", "30") or 30))
    parser.add_argument("--gemini-model", default=env("GEMINI_ANALYSIS_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--gemini-backend", choices=["auto", "enterprise", "api-key"], default="auto")
    parser.add_argument("--gemini-batch-size", type=int, default=20)
    parser.add_argument("--min-semantic-fit", type=float, default=35.0)
    parser.add_argument("--max-trends", type=int, default=40)
    parser.add_argument("--no-semantic-cache", action="store_true")
    parser.add_argument("--refresh-semantic-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = PinterestClient(
            token_path=Path(args.token_path) if args.token_path else None,
            timeout=args.timeout,
        )
        candidates, raw, errors = collect_pinterest_trends(
            client=client,
            region=args.region,
            trend_type=args.trend_type,
            interest=args.interest,
            limit=max(1, min(50, args.keyword_limit)),
        )
        write_json(output_dir / "trends_raw.json", raw)
    except Exception as exc:
        LOG.error("Pinterest trend collection failed: %s", exc)
        return 1

    LOG.info("Trend candidates collected: %d", len(candidates))
    if not candidates:
        write_json(
            output_dir / "trend_collection_error.json",
            {
                "niche": args.niche,
                "region": args.region,
                "trend_type": args.trend_type,
                "errors": errors,
                "message": (
                    "No trend candidates were collected. Fix the Pinterest API connection or token "
                    "permissions before running semantic analysis."
                ),
            },
        )
        LOG.error(
            "No trend candidates collected. Wrote diagnostics to %s",
            output_dir / "trend_collection_error.json",
        )
        return 2

    analyzer = GeminiSemanticAnalyzer(
        niche=args.niche,
        model=args.gemini_model,
        backend=args.gemini_backend,
        batch_size=args.gemini_batch_size,
        cache=JsonCache(output_dir / ".semantic_cache.json", enabled=not args.no_semantic_cache),
        refresh_cache=args.refresh_semantic_cache,
    )
    accepted, rejected = analyzer.analyze(candidates[: max(1, args.max_trends * 3)])
    accepted = [item for item in accepted if item.semantic_fit >= args.min_semantic_fit]
    accepted = accepted[: max(1, args.max_trends)]
    for index, item in enumerate(accepted, start=1):
        item.trend_id = f"trend_{index:03d}"

    package = TrendPackage(
        schema_version="1.0",
        generated_at=utc_now_iso(),
        niche=args.niche,
        region=args.region,
        source={
            "trend_provider": "pinterest_trends_api",
            "semantic_model": args.gemini_model,
            "trend_type": args.trend_type,
            "interest": args.interest,
            "errors": errors,
        },
        trends=accepted,
        rejected_trends=rejected,
        raw_summary={
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
        },
    )

    write_json(output_dir / "trend_package.json", package)
    export_package_csv(output_dir / "trend_package.csv", package)
    render_report(output_dir / "trend_report.html", package)
    LOG.info("DONE: %s", output_dir / "trend_package.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
