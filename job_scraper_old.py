#!/usr/bin/env python3
"""
job_scraper.py

Playwright-based job scraping skeleton.

GOAL:
  - For a given JOB_xxxxx folder, read client_meta.json
  - Use keywords/locations to search job sites
  - Save cleaned job descriptions as job_description_01.txt, etc.

IMPORTANT:
  - This script is intentionally generic to avoid violating site-specific TOS.
  - You provide the actual job board URLs via a simple config file.
"""

from pathlib import Path
from typing import List
from playwright.sync_api import sync_playwright
import json

BASE = Path("/home/mykl/webui/filesystem/FiverrMachine")
PROCESSING = BASE / "PROCESSING"


def load_client_meta(job_dir: Path) -> dict:
    """Load client_meta.json if present, otherwise return empty dict."""
    meta_path = job_dir / "client_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"raw": meta_path.read_text(encoding="utf-8")}


def load_job_source_urls(job_dir: Path) -> List[str]:
    """
    Load job source URLs from job_sources.txt, one URL per line.

    This keeps scraping logic generic; you decide which sites to target.
    """
    urls_path = job_dir / "job_sources.txt"
    if not urls_path.exists():
        return []
    urls = [line.strip() for line in urls_path.read_text(encoding="utf-8").splitlines()]
    return [u for u in urls if u]


def scrape_job_pages(job_dir: Path) -> None:
    """
    Use Playwright to fetch HTML from configured job URLs and
    store cleaned text into job_description_XX.txt files.

    This is a skeleton: it grabs page text; you can later refine selectors
    for specific sites.
    """
    urls = load_job_source_urls(job_dir)
    if not urls:
        print(f"[SCRAPER] No job_sources.txt found for {job_dir}, writing a dummy job description.")
        dummy = """Dummy Job Title
Company: Example Corp
Location: Remote

Responsibilities:
- Do interesting AI automation work.
- Support clients using Python and APIs.

Requirements:
- Strong problem-solving.
- Experience with automation and LLM tools.
"""
        (job_dir / "job_description_01.txt").write_text(dummy, encoding="utf-8")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for idx, url in enumerate(urls, start=1):
            try:
                print(f"[SCRAPER] Fetching {url} ...")
                page.goto(url, wait_until="networkidle")
                text_content = page.inner_text("body")
                out_path = job_dir / f"job_description_{idx:02}.txt"
                out_path.write_text(text_content, encoding="utf-8")
                print(f"[SCRAPER] Wrote {out_path}")
            except Exception as e:
                print(f"[SCRAPER] Error scraping {url}: {e}")

        browser.close()


def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 job_scraper.py /path/to/JOB_xxxxx")
        raise SystemExit(1)

    job_dir = Path(sys.argv[1]).resolve()
    if not job_dir.is_dir():
        print(f"[SCRAPER] Not a directory: {job_dir}")
        raise SystemExit(1)

    scrape_job_pages(job_dir)


if __name__ == "__main__":
    main()
