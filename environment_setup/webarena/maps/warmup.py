#!/usr/bin/env python3
"""Warm up the OpenStreetMap UI with a real browser."""

from __future__ import annotations

import argparse
import sys
import time

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


MAP_READY_SELECTOR = ".leaflet-container, #map"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up the WebArena maps UI.")
    parser.add_argument("url", help="Maps URL to open, for example http://host:443")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=90_000,
        help="Per-attempt Playwright timeout in milliseconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of browser attempts before failing.",
    )
    parser.add_argument(
        "--retry-delay-s",
        type=float,
        default=5.0,
        help="Seconds to wait between failed attempts.",
    )
    return parser.parse_args()


def warmup(url: str, timeout_ms: int) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox"],
        )
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.set_default_timeout(timeout_ms)
            page.set_default_navigation_timeout(timeout_ms)
            page.goto(url, wait_until="load", timeout=timeout_ms)
        finally:
            browser.close()


def main() -> int:
    args = parse_args()
    last_error: Exception | None = None

    for attempt in range(1, args.retries + 1):
        try:
            print(f"Map browser warmup attempt {attempt}/{args.retries}: {args.url}", flush=True)
            warmup(args.url, args.timeout_ms)
            print(f"Map browser warmup succeeded: {args.url}", flush=True)
            return 0
        except (PlaywrightError, PlaywrightTimeoutError) as exc:
            last_error = exc
            print(f"Map browser warmup failed on attempt {attempt}: {exc}", file=sys.stderr, flush=True)
            if attempt < args.retries:
                time.sleep(args.retry_delay_s)

    print(f"Map browser warmup failed after {args.retries} attempts: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
