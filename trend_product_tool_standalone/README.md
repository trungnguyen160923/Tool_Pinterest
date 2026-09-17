# Trend Product Tool — Standalone

Standalone API that discovers Pinterest trends, creates print-ready rug or
blanket artwork, exports CMYK production files, and creates AI lifestyle images.
It has no runtime dependency on the sibling `task2_*` through `task6_*` folders.

## Install and configure

```powershell
cd trend_product_tool_standalone
pip install -r requirements.txt
playwright install chromium
Copy-Item .env.example .env
```

Set Pinterest and Gemini credentials in `.env`. If browser login is needed, run:

```powershell
python pinterest/pinterest_browser_login.py
```

## Run UI and API together

Use one command for normal operation. It starts Streamlit at port `8502` and
the API at port `8000`; closing it stops both processes.

```powershell
python run.py
```

For another pair of ports:

```powershell
python run.py --ui-port 8502 --api-port 8000 --host 0.0.0.0
```

## Run the API only

```powershell
python api.py
```

For a trusted local network, set `TREND_PRODUCT_API_HOST=0.0.0.0` in `.env` and
call `http://MACHINE_LAN_IP:8000/v1/jobs`. OpenAPI documentation is available at
`/docs`.

## Run the UI only

The standalone project also retains the Streamlit interface:

```powershell
streamlit run app.py --server.port 8501
```

Open `http://127.0.0.1:8501`. The **Open Pinterest login** button uses the
bundled `pinterest/` code in this project, not a sibling task folder.

## Create a product

```json
POST /v1/jobs
{
  "niche": "rug",
  "product": "rug",
  "desired_output_count": 1,
  "ai_background_variants": 4
}
```

`POST /v1/jobs` always waits: the caller remains loading until the final
deliverables are ready, then receives them in the response. A completed job
returns only:

- `output.marketing_images`: approved AI lifestyle/background images.
- `output.print_cmyk_images`: CMYK JPG print masters.

Use the `download_url` on each returned item to download that image. `DELETE
/v1/jobs/{job_id}` requests cancellation.
