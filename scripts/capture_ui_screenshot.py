"""Capture Streamlit UI screenshots for the README/model card.

The Streamlit app must already be running at http://localhost:8501.
Outputs three full-page PNGs to results/, one per tab.

Usage:
    streamlit run app.py --server.headless true --server.port 8501 &
    python scripts/capture_ui_screenshot.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).parent.parent


def screenshot_app(url: str, output_dir: Path, tabs: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1100})
        page = context.new_page()

        page.goto(url, wait_until="networkidle", timeout=120_000)
        # Streamlit needs a beat to hydrate after networkidle
        page.wait_for_selector("h1", timeout=30_000)
        time.sleep(4)

        landing = output_dir / "ui_screenshot.png"
        page.screenshot(path=str(landing), full_page=True)
        paths.append(landing)
        print(f"  saved {landing.name}")

        for i, label in enumerate(tabs):
            try:
                # Click the i-th tab (Streamlit renders tabs as button[role=tab])
                page.locator("button[role='tab']").nth(i).click()
                time.sleep(3)
                out = output_dir / f"ui_tab{i+1}_{label}.png"
                page.screenshot(path=str(out), full_page=True)
                paths.append(out)
                print(f"  saved {out.name}")
            except Exception as e:
                print(f"  skipped tab {i+1} ({label}): {e}")

        browser.close()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    tabs = ["browse", "text_search", "experiments"]
    paths = screenshot_app(args.url, args.output, tabs)
    print(f"\n{len(paths)} screenshot(s) written.")


if __name__ == "__main__":
    main()
