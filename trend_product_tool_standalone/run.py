"""Start the Streamlit UI and HTTP API together for the standalone tool."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Trend Product Tool UI and API together.")
    parser.add_argument("--ui-port", type=int, default=int(os.getenv("TREND_PRODUCT_UI_PORT", "8502")))
    parser.add_argument("--api-port", type=int, default=int(os.getenv("TREND_PRODUCT_API_PORT", "8000")))
    parser.add_argument("--host", default=os.getenv("TREND_PRODUCT_HOST", "0.0.0.0"), help="Bind host for trusted LAN use.")
    return parser.parse_args()


def stop_process(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True)
    else:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> None:
    args = parse_args()
    environment = os.environ.copy()
    environment["TREND_PRODUCT_API_HOST"] = args.host
    environment["TREND_PRODUCT_API_PORT"] = str(args.api_port)

    api = subprocess.Popen([sys.executable, str(ROOT / "api.py")], cwd=ROOT, env=environment)
    try:
        # Give Uvicorn a moment to claim its port before Streamlit becomes the
        # foreground process. The API itself remains available independently.
        time.sleep(1)
        if api.poll() is not None:
            raise RuntimeError(f"API process stopped early with exit code {api.returncode}.")
        print(f"API: http://{args.host}:{args.api_port}/docs", flush=True)
        print(f"UI:  http://{args.host}:{args.ui_port}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "app.py"),
                "--server.port",
                str(args.ui_port),
                "--server.address",
                args.host,
            ],
            cwd=ROOT,
            env=environment,
            check=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        stop_process(api)


if __name__ == "__main__":
    main()
