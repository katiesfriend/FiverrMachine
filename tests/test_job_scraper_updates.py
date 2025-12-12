import json
import sys
import types
from pathlib import Path


class _DummyOpenAI:
    def __init__(self, *args, **kwargs):
        pass


# Provide a stub OpenAI module so job_scraper imports without external deps during tests
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=_DummyOpenAI))

import job_scraper


class DummyPage:
    def goto(self, *args, **kwargs):
        return None

    def wait_for_timeout(self, *args, **kwargs):
        return None

    def query_selector_all(self, *args, **kwargs):
        return []

    def close(self):
        return None

    def text_content(self, *args, **kwargs):
        return ""


class DummyContext:
    def new_page(self):
        return DummyPage()

    def close(self):
        return None


class DummyBrowser:
    def __init__(self):
        self._page = DummyPage()

    def new_page(self):
        return DummyPage()

    def new_context(self):
        return DummyContext()

    def close(self):
        return None


class DummyChromium:
    def launch(self, headless=True):
        return DummyBrowser()


class DummyPlaywright:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @property
    def chromium(self):
        return DummyChromium()


def test_dedupe_by_url():
    jobs = [
        {"url": "http://example.com/role", "title": "A", "company": "ACME", "location": "Remote"},
        {"url": "http://example.com/role/", "title": "B", "company": "Beta", "location": "Remote"},
    ]
    deduped = job_scraper.dedupe_jobs(jobs)
    assert len(deduped) == 1


def test_board_failure_does_not_abort(monkeypatch, tmp_path):
    monkeypatch.setattr(job_scraper, "sync_playwright", lambda: DummyPlaywright())

    def fake_fetch(browser, jobs):
        for job in jobs:
            job["full_text"] = job.get("snippet", "")

    monkeypatch.setattr(job_scraper, "fetch_full_descriptions", fake_fetch)

    # Fail LinkedIn scraper underneath the wrapper
    def failing_scraper(*args, **kwargs):
        raise RuntimeError("blocked")

    monkeypatch.setattr(job_scraper, "scrape_linkedin", failing_scraper)

    def success_search(*args, **kwargs):
        return [
            {
                "title": "Engineer",
                "company": "ACME",
                "location": "Remote",
                "site": "indeed",
                "url": "http://example.com/job1",
                "score": 10.0,
                "snippet": "desc",
            }
        ], {"state": "ok", "reason": ""}

    monkeypatch.setattr(job_scraper, "search_indeed", success_search)

    # Silence other boards
    monkeypatch.setattr(job_scraper, "search_ziprecruiter", success_search)
    monkeypatch.setattr(job_scraper, "search_usajobs", success_search)
    monkeypatch.setattr(job_scraper, "scrape_glassdoor", lambda *a, **k: [])
    monkeypatch.setattr(job_scraper, "scrape_monster", lambda *a, **k: [])

    # Minimal client request
    client_meta = {"target_roles": ["Engineer"], "location": "Remote", "skills": ["python"]}
    (tmp_path / "client_request.json").write_text(json.dumps(client_meta), encoding="utf-8")

    job_scraper.scrape_job_boards(tmp_path)

    manifest_path = tmp_path / "jobs_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest, "Manifest should have at least one job entry"
    assert manifest[0].get("board") == manifest[0].get("site")


def test_manifest_city_state_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(job_scraper, "sync_playwright", lambda: DummyPlaywright())
    monkeypatch.setattr(job_scraper, "fetch_full_descriptions", lambda browser, jobs: None)

    def single_board(*args, **kwargs):
        return [
            {
                "title": "Analyst",
                "company": "DataCo",
                "location": "Indianapolis, IN",
                "site": "indeed",
                "url": "http://example.com/job2",
                "score": 5.0,
                "snippet": "desc",
            }
        ], {"state": "ok", "reason": ""}

    monkeypatch.setattr(job_scraper, "search_indeed", single_board)
    monkeypatch.setattr(job_scraper, "search_linkedin", single_board)
    monkeypatch.setattr(job_scraper, "search_ziprecruiter", single_board)
    monkeypatch.setattr(job_scraper, "search_usajobs", single_board)
    monkeypatch.setattr(job_scraper, "scrape_glassdoor", lambda *a, **k: [])
    monkeypatch.setattr(job_scraper, "scrape_monster", lambda *a, **k: [])

    client_meta = {"target_roles": ["Analyst"], "location": "Indianapolis, IN", "skills": ["sql"]}
    (tmp_path / "client_request.json").write_text(json.dumps(client_meta), encoding="utf-8")

    job_scraper.scrape_job_boards(tmp_path)

    manifest = json.loads((tmp_path / "jobs_manifest.json").read_text())
    assert manifest[0]["city"] == "Indianapolis"
    assert manifest[0]["state"].lower() in {"in", "indiana"}
    assert manifest[0]["board"] == "indeed"

