# Trend Product Tool

Workflow tool for turning trend research into print-ready rug/blanket/custom-shape assets and product mockups.

The default flow is non-destructive: every run creates `output/run_YYYYMMDD_HHMMSS` with config, intermediate data, final print files, mockups, and an HTML report.

## What It Does

1. Set product output target: rug `4000x6400 px` at `300 DPI`, blanket `10000x11000 px` at `300 DPI`, or custom dimensions.
2. Use Pinterest Trends API to discover hot trend keywords and create `trend_package.json`. If trend discovery fails, the run stops.
3. Crawl and rank Pinterest images using the original queries returned in `trend_package.json`.
4. Convert each hot trend image into a printable artwork candidate (`product_design`, `pattern_repeat`, or `direct`).
5. Render that artwork as a rug/blanket product asset with an alpha mask.
6. Generate a product-aware background from the rendered product when AI background replacement is enabled.
7. Enhance/upscale with the local print production stage, optional white-background removal, and crop/fit to the print target.
8. Export `RGB PNG`, optional `CMYK JPG`, rendered product assets, local mockups, optional AI background outputs, `stage_manifest.json`, and `report.html`.

## Folder Layout

```text
trend_product_tool/
  app.py
  run_pipeline.py
  requirements.txt
  trend_tool/
    config.py
    crawler.py
    dedupe.py
    image_ops.py
    pipeline.py
    product.py
    report.py
    design.py
    enhancement.py
    task3_adapter.py
    task4_adapter.py
    task5_adapter.py
    task6_adapter.py
```

## Quick Start

Install dependencies:

```powershell
pip install -r trend_product_tool/requirements.txt
```

Run the UI:

```powershell
streamlit run trend_product_tool/app.py
```


## Environment Values

The tool automatically loads `trend_product_tool/.env` first, then the repo-level `.env` if present. Existing process environment values are kept.

Useful values include:

```text
PINTEREST_NICHE=coastal grandmother decor
PINTEREST_REGION=US
PINTEREST_TREND_TYPE=growing
PINTEREST_INTEREST=
GEMINI_ANALYSIS_MODEL=gemini-2.5-flash
GEMINI_BACKEND=auto
TREND_PRODUCT_OUTPUT=trend_product_tool/output
PINTEREST_BROWSER_PROFILE_DIR=optional\path\to\profile
PINTEREST_OAUTH_TOKEN_PATH=optional\path\to\.pinterest_oauth_tokens.json
PINTEREST_ACCESS_TOKEN=optional-existing-token
```

Auto discovery uses the Pinterest Trends API to find trend keywords/queries. Image discovery uses Pinterest Browser only. Log in once with the UI button `Open Pinterest login` before crawling images.

## UI Flow

Enter a trend niche, choose a product target, then click `Find and build`. The niche is sent to Pinterest Trends API for trend discovery and ranking; Pinterest Browser crawls the original hot-trend keyword returned by the API. The product target is used only for artwork, print dimensions, rendering, and mockups.

```text
Niche: coastal grandmother decor
Product: blanket
```

Then click `Find and build`. The tool first tries Pinterest Trends API + semantic analysis to create `trend_package.json`, then crawls Pinterest in the saved browser profile, ranks images, and creates final product files.

If the Pinterest Trends API token/permission is unavailable, the run stops. Bing image search and generic keyword fallback are not used.

The `Niche` field is the Pinterest Trends API request.

`Preview designs` controls how many RGB PNG design previews are rendered in Streamlit after a run; all print files are still written to disk.

## CLI Examples

Auto trend discovery and Pinterest image crawl:

```powershell
python trend_product_tool/run_pipeline.py --workflow-mode trend-to-product --niche "coastal grandmother decor" --product rug
```

Create AI lifestyle backgrounds from the rendered rug/blanket products:

```powershell
python trend_product_tool/run_pipeline.py --workflow-mode trend-to-product --niche "coastal grandmother decor" --product blanket --task4-mockup-engine task4-ai --task4-ai-limit 3
```

Use similarity matching as a stronger duplicate gate:

```powershell
python trend_product_tool/run_pipeline.py --niche "coastal grandmother decor" --product rug --task6-gallery task6_image_similarity\gallery --task6-mode auto --task6-threshold 92
```

Create print-oriented product designs instead of resizing source photos directly:

```powershell
python trend_product_tool/run_pipeline.py --niche "coastal grandmother decor" --product rug --design-mode product-design
```

Use repeat-style blanket/rug patterns:

```powershell
python trend_product_tool/run_pipeline.py --niche "coastal grandmother decor" --product blanket --design-mode pattern-repeat
```

Generate AI artwork reference outputs for the first final design:

```powershell
python trend_product_tool/run_pipeline.py --niche "coastal grandmother decor" --product rug --task3-reference-dir task3_image_replace\dataset\references --task3-output-limit 1
```

Replace the background for crawled product images:

```powershell
python trend_product_tool/run_pipeline.py --niche "coastal grandmother decor" --product rug --task4-mockup-engine task4-ai --task4-ai-limit 1 --task4-final-integration off
```

## Product Workflow

- Trend discovery: Pinterest Trends API discovers the keywords. If the API token/permission is unavailable, the run stops so generic product queries are not mislabeled as trends.
- Pinterest crawl: the image source is Pinterest Browser only. It uses a saved Chromium login profile and writes fresh results into the current run folder. Default trend-to-product runs pass `crawl_purpose=inspiration` so the vision gate looks for printable trend visuals, not only existing rug/blanket product photos.
- Similarity and dedupe: perceptual dedupe runs by default. Optional CLIP/classic similarity matching can reject images that are too close to an existing gallery.
- Printability gates: candidates must be reusable artwork references, then generated artwork must pass the final print rubric before export. Decisions and reasons are recorded in `stage_manifest.json`.
- Product design: `product_design`, `pattern_repeat`, or `direct` convert trend imagery into print-oriented design candidates. Trend-to-product then renders the design as a rug/blanket product asset before background generation.
- Print production: local enhance/upscale, optional white-background removal, target crop/fit, RGB PNG export, and optional CMYK JPG export.
- Mockup/background: local mockups are generated from final PNGs. Optional AI background replacement runs on rendered product assets in trend-to-product mode, keeps that generated rug/blanket product as the hero, and creates a product-aware background brief from the niche/product.

The AI-heavy artwork/background stages can trigger paid Google Cloud calls. They stay disabled unless their output limit is greater than zero.


## Pinterest Credentials

Auto trend discovery can use a Pinterest API token. Browser image crawl uses a saved Chromium profile. In the UI, click `Open Pinterest login`, finish login, then run `Find and build`.

```text
task5_hottrend/.pinterest_browser_profile/
task5_hottrend/.pinterest_oauth_tokens.json
```

You can override the browser profile folder with:

```powershell
PINTEREST_BROWSER_PROFILE_DIR=path\to\profile
```

You can override the API token path with `PINTEREST_OAUTH_TOKEN_PATH`, `PINTEREST_TOKEN_PATH`, or `TASK5_TOKEN_PATH`. OAuth token files are used only for trend/keyword discovery; the image crawl path uses the browser login profile.

## Notes

- A niche is required and is sent unchanged to Pinterest Trends API. If the API is unavailable, the run stops.
- Pinterest Browser always crawls fresh images into the current `run_*/task5_crawl` folder.
- `Refresh vision analysis` is on by default so the product vision gate is recalculated for the current fresh crawl.
- Auto mode requires a saved Pinterest browser login and Gemini settings when semantic or vision filtering is enabled.
- Work files stay in RGB/sRGB because most image tooling is safer there.
- Final CMYK JPG export is attempted with Pillow. Always soft-proof with the print vendor when color matching is critical.
- Custom shape products should use transparent PNG or a white-background reference. The tool can remove near-white background for supplier review.









