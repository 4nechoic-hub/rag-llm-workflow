#!/usr/bin/env python3
"""Best-effort Playwright capture script for the README screenshots.

It can optionally launch Streamlit, drive the UI, and overwrite the two preview
assets used by README.md:

- assets/readme/ui-chatbot-preview.png
- assets/readme/ui-explorer-preview.png

Requirements:
  pip install playwright pillow
  python -m playwright install chromium

Examples:
  python scripts/capture_readme_screenshots.py --launch-streamlit
  python scripts/capture_readme_screenshots.py --base-url http://localhost:8501
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional


DEFAULT_CHAT_QUESTION = "What measurement techniques were used across these documents?"
DEFAULT_EXPLORER_QUERY = "Compare the experimental approaches used across the papers."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture README screenshots from the Streamlit app.")
    parser.add_argument("--base-url", default="http://localhost:8501", help="Streamlit URL.")
    parser.add_argument("--launch-streamlit", action="store_true", help="Launch app/streamlit_app.py before capturing.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root containing app/streamlit_app.py.")
    parser.add_argument("--assets-dir", type=Path, default=Path("assets/readme"), help="Directory for output assets.")
    parser.add_argument("--chat-question", default=DEFAULT_CHAT_QUESTION, help="Question used in chatbot mode.")
    parser.add_argument("--explorer-query", default=DEFAULT_EXPLORER_QUERY, help="Query used in explorer mode.")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout in seconds for page/app readiness.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--skip-chatbot", action="store_true", help="Skip chatbot capture.")
    parser.add_argument("--skip-explorer", action="store_true", help="Skip explorer capture.")
    return parser


def wait_for_http(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: Optional[Exception] = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {url!r}. Last error: {last_error}")


def launch_streamlit(repo_root: Path) -> subprocess.Popen[str]:
    app_path = repo_root / "app" / "streamlit_app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Could not find {app_path}")

    env = os.environ.copy()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def click_first_text(page, text: str) -> bool:
    locator = page.get_by_text(text, exact=True)
    if locator.count() > 0:
        locator.first.click()
        return True
    locator = page.get_by_text(text)
    if locator.count() > 0:
        locator.first.click()
        return True
    return False


def maybe_click(page, text: str) -> None:
    try:
        click_first_text(page, text)
    except Exception:  # noqa: BLE001
        return


def fill_by_placeholder(page, placeholder: str, value: str) -> bool:
    locator = page.get_by_placeholder(placeholder)
    if locator.count() == 0:
        return False
    locator.first.click()
    locator.first.fill(value)
    return True


def wait_for_text(page, text: str, timeout_ms: int = 15000) -> None:
    page.get_by_text(text).first.wait_for(timeout=timeout_ms)


def reveal_sources_if_present(page) -> None:
    for label in ["sources retrieved", "Sources ("]:
        locator = page.get_by_text(label)
        if locator.count() > 0:
            try:
                locator.first.click()
            except Exception:  # noqa: BLE001
                pass
            return


def capture_chatbot(page, output_path: Path, question: str) -> None:
    maybe_click(page, "Document Chatbot")
    wait_for_text(page, "Document Chatbot", timeout_ms=30000)

    if not fill_by_placeholder(page, "Ask about your documents...", question):
        raise RuntimeError("Could not find the chatbot input field.")

    maybe_click(page, "Send")
    page.keyboard.press("Enter")
    page.wait_for_timeout(7000)
    reveal_sources_if_present(page)
    page.screenshot(path=str(output_path), full_page=True)


def capture_explorer(page, output_path: Path, query: str) -> None:
    maybe_click(page, "Pipeline Explorer")
    wait_for_text(page, "Pipeline Explorer", timeout_ms=30000)
    maybe_click(page, "Compare All")
    maybe_click(page, "Document Comparison")

    if not fill_by_placeholder(page, "e.g. What experimental setup was used in this study?", query):
        textarea = page.locator("textarea")
        if textarea.count() == 0:
            raise RuntimeError("Could not find the explorer query input.")
        textarea.first.click()
        textarea.first.fill(query)

    maybe_click(page, "Run Query")
    page.wait_for_timeout(9000)
    page.screenshot(path=str(output_path), full_page=True)


def main() -> int:
    args = build_parser().parse_args()
    proc: Optional[subprocess.Popen[str]] = None

    try:
        if args.launch_streamlit:
            proc = launch_streamlit(args.repo_root)

        wait_for_http(args.base_url, timeout=args.timeout)

        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Playwright is required. Install it with `pip install playwright pillow` and "
                "then run `python -m playwright install chromium`."
            ) from exc

        assets_dir = args.assets_dir
        assets_dir.mkdir(parents=True, exist_ok=True)
        chatbot_out = assets_dir / "ui-chatbot-preview.png"
        explorer_out = assets_dir / "ui-explorer-preview.png"

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=args.headless)
            context = browser.new_context(viewport={"width": 1600, "height": 1100}, device_scale_factor=1)

            if not args.skip_chatbot:
                page = context.new_page()
                page.goto(args.base_url, wait_until="networkidle")
                capture_chatbot(page, chatbot_out, args.chat_question)
                page.close()

            if not args.skip_explorer:
                page = context.new_page()
                page.goto(args.base_url, wait_until="networkidle")
                capture_explorer(page, explorer_out, args.explorer_query)
                page.close()

            browser.close()

        print(f"Saved screenshot assets to {assets_dir}")
        return 0
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
