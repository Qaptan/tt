"""Save TikTok authenticated browser session state for automated uploads.

Usage:
  python scripts/save_tiktok_session.py --out storage/accounts/fitness_account/session_state.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save TikTok session state")
    parser.add_argument("--out", required=True, help="Output storage_state JSON path")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.tiktok.com/login")
        print("Log in manually, then press ENTER to save session state...")
        input()
        context.storage_state(path=str(out_path))
        browser.close()

    print(f"Saved session state to: {out_path}")


if __name__ == "__main__":
    main()
