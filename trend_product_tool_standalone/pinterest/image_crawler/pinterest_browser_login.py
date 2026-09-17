from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pinterest.image_crawler.discovery import browser_profile_dir
    from pinterest.shared.utils import env
else:
    from .discovery import browser_profile_dir
    from ..shared.utils import env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a persistent Pinterest browser profile for manual login.")
    parser.add_argument("--profile-dir", default=env("PINTEREST_BROWSER_PROFILE_DIR", ""))
    parser.add_argument("--timeout", type=int, default=int(env("PINTEREST_BROWSER_LOGIN_TIMEOUT", "600") or 600))
    parser.add_argument("--url", default="https://www.pinterest.com/login/")
    return parser.parse_args()


def has_login_cookie(context) -> bool:
    try:
        cookies = context.cookies("https://www.pinterest.com")
    except Exception:
        return False
    names = {cookie.get("name") for cookie in cookies}
    return bool({"_pinterest_sess", "_auth"} & names)


def page_looks_logged_in(page) -> bool:
    try:
        return bool(
            page.evaluate(
                """
                () => {
                  const path = location.pathname.toLowerCase();
                  if (path.includes('/login') || path.includes('/signup')) return false;
                  const text = Array.from(document.querySelectorAll('button, a'))
                    .map((el) => el.textContent || '')
                    .join('\\n');
                  return !/\\b(log in|sign up)\\b/i.test(text);
                }
                """
            )
        )
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    profile_dir = browser_profile_dir(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print(f"ERROR: Playwright is not installed or unavailable: {exc}", flush=True)
        print("Install with: pip install playwright && playwright install chromium", flush=True)
        return 1

    print(f"Pinterest browser profile: {profile_dir}", flush=True)
    print("A Chromium window will open. Log in to Pinterest there.", flush=True)
    print("The window will close automatically after login is detected, or when timeout expires.", flush=True)

    deadline = time.time() + max(30, args.timeout)
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            viewport={"width": 1366, "height": 900},
            locale=env("PINTEREST_LOCALE", "en-US"),
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=45_000)
        except Exception as exc:
            print(f"WARNING: Could not open Pinterest login page: {exc}", flush=True)

        while time.time() < deadline:
            try:
                if has_login_cookie(context) and page_looks_logged_in(page):
                    print("Pinterest login detected. Saved browser profile.", flush=True)
                    context.close()
                    return 0
            except Exception:
                break
            time.sleep(2)

        print("Login wait timed out. If you finished login, the profile may still be saved.", flush=True)
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
