import json
from pathlib import Path

import types
import sys


class DummyOpenAI:
    def __init__(self, *args, **kwargs):
        pass


sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=DummyOpenAI))
sync_api = types.ModuleType("playwright.sync_api")
sync_api.sync_playwright = lambda: None
sync_api.TimeoutError = Exception
sys.modules.setdefault("playwright", types.ModuleType("playwright"))
sys.modules.setdefault("playwright.sync_api", sync_api)

import job_scraper
from packet_builder_qwen import write_selection_summary
from report_builder import build_markdown_report, parse_job_sources
from config import DEFAULT_CONFIG


def _dummy_playwright():
    class DummyBrowser:
        def new_page(self):
            return self

        def close(self):
            pass

    class DummyChromium:
        def launch(self, headless=True):
            return DummyBrowser()

    class DummyContext:
        chromium = DummyChromium()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    return DummyContext()


def test_scraper_collects_multiple_sites(monkeypatch, tmp_path):
    monkeypatch.setattr(job_scraper, "sync_playwright", lambda: _dummy_playwright())
    monkeypatch.setattr(job_scraper, "fetch_full_descriptions", lambda browser, jobs: None)
    monkeypatch.setattr(job_scraper, "load_client_meta", lambda path: {})
    monkeypatch.setattr(job_scraper, "build_search_query", lambda meta: ("analyst", "remote", ["sql"]))

    fake_job = {
        "title": "Data Analyst",
        "company": "Acme",
        "location": "Remote",
        "snippet": "Work with data.",
        "url": "https://example.com",
        "score": 5.0,
        "site": "stub",
        "full_text": "Long description",
    }

    for engine in [
        "scrape_indeed",
        "scrape_linkedin",
        "scrape_ziprecruiter",
        "scrape_glassdoor",
        "scrape_usajobs",
        "scrape_monster",
    ]:
        monkeypatch.setattr(
            job_scraper,
            engine,
            lambda *args, site=engine, **kwargs: [dict(fake_job, site=site)],
        )

    job_scraper.scrape_job_boards(tmp_path)

    manifest_path = tmp_path / "jobs_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    sites = {entry["site"] for entry in manifest}
    assert {"scrape_indeed", "scrape_linkedin", "scrape_ziprecruiter"}.issubset(sites)


def test_selection_summary_includes_required_fields(tmp_path):
    manifest = [
        {
            "job_id": "job01",
            "index": 1,
            "title": "Data Analyst",
            "company": "Acme",
            "location": "Remote",
            "site": "indeed",
            "url": "https://example.com",
            "salary": "$120,000",
            "raw_scraper_score": 5.0,
        }
    ]
    (tmp_path / "jobs_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "score_job01.json").write_text(json.dumps({"match_score": 88}), encoding="utf-8")
    (tmp_path / "job_description_01.txt").write_text("Salary: $120,000", encoding="utf-8")

    write_selection_summary(tmp_path, DEFAULT_CONFIG)

    summary = json.loads((tmp_path / "selection_summary.json").read_text())
    assert summary["focus_shortlist"]
    job_entry = summary["jobs"][0]
    for key in [
        "job_id",
        "title",
        "company",
        "location",
        "source_site",
        "match_bucket",
        "raw_score",
        "normalized_score",
    ]:
        assert key in job_entry


def test_report_builder_contains_key_sections(tmp_path):
    manifest = [
        {
            "job_id": "job01",
            "index": 1,
            "title": "Data Analyst",
            "company": "Acme",
            "location": "Remote",
            "site": "indeed",
            "url": "https://example.com",
            "salary": "$120,000",
            "raw_scraper_score": 5.0,
        }
    ]
    desc = tmp_path / "job_description_01.txt"
    desc.write_text("Salary: $120,000", encoding="utf-8")

    jobs_info = parse_job_sources(tmp_path, manifest)
    selection = {
        "focus_shortlist": ["job01"],
        "jobs": [
            {
                "job_id": "job01",
                "label": "01",
                "title": "Data Analyst",
                "company": "Acme",
                "location": "Remote",
                "source_site": "indeed",
                "normalized_score": 90,
                "match_bucket": "High",
            }
        ],
    }

    report_md = build_markdown_report(
        tmp_path,
        {},
        jobs_info,
        selection,
        manifest,
        DEFAULT_CONFIG,
    )

    assert "Job Search Report for Client" in report_md
    assert "Focus Roles" in report_md
    assert "Job List (Best Matches First)" in report_md
