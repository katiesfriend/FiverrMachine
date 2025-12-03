#!/usr/bin/env python3
"""
job_scraper.py

Multi-board job scraper for FiverrMachine.

Given a JOB directory (with client_request.json), this script:

1. Reads client metadata (skills, target_title, location)
2. Searches multiple job boards, first page only:
   - Indeed
   - LinkedIn (public search)
   - ZipRecruiter
   - Glassdoor
   - USAJobs
   - Monster
3. Collects job cards (title, company, location, snippet, URL)
4. Scores each job against client skills
5. Picks the top N jobs across all boards (default 20)
6. Visits each job URL to pull the full job description
7. Writes:
   - job_description_XX.txt for each selected job
   - job_sources.txt listing scores and URLs

This script is designed to be called by run_fiverr_pipeline.py:

    python3 job_scraper.py /abs/path/to/JOB_xxx
"""

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
from urllib.parse import quote_plus

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


TOP_N_JOBS_DEFAULT = 20  # how many jobs to turn into job_description_XX.txt
MAX_FETCH_MULTIPLIER = 2  # how many jobs to fetch full descriptions for, relative to TOP_N
PRIMARY_FIT_THRESHOLD = 2.5  # heuristic score cutoff for "high fit"


# -----------------------------
# Utility functions
# -----------------------------

def log(msg: str) -> None:
    print(f"[SCRAPER] {msg}", flush=True)


def load_client_meta(job_dir: Path) -> Dict[str, Any]:
    """Load client_request.json from job_dir."""
    meta_path = job_dir / "client_request.json"
    if not meta_path.exists():
        log(f"WARNING: {meta_path} not found; using empty metadata.")
        return {}

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        log(f"WARNING: Failed to parse {meta_path}: {e}")
        return {}


def normalize_skills(meta: Dict[str, Any]) -> List[str]:
    """
    Extract skills from metadata as a list of lowercase phrases.

    Supports both:
      - "key_skills": ["Agile project management", "Scrum", ...]
      - "skills": ["..."]

    Falls back to an empty list if neither is present.
    """
    skills = meta.get("skills") or meta.get("key_skills") or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    elif isinstance(skills, list):
        skills = [str(s).strip() for s in skills if str(s).strip()]
    else:
        skills = []

    return [s.lower() for s in skills]


def extract_title_and_location(meta: Dict[str, Any]) -> Tuple[str, str]:
    """
    Determine search title and location from metadata.

    Title priority:
      1. target_roles[0]
      2. preferred_titles[0]
      3. target_title / headline / job_title

    Location priority:
      - If remote_preference == "remote_only" -> "Remote"
      - If remote_preference == "remote_plus_local" -> zip if present, else "Remote"
      - Else: zip if present, else `location`, else "Remote"
    """
    target_roles = meta.get("target_roles") or []
    preferred_titles = meta.get("preferred_titles") or []

    # Normalize lists
    if isinstance(target_roles, str):
        target_roles = [s.strip() for s in target_roles.split(",") if s.strip()]
    if isinstance(preferred_titles, str):
        preferred_titles = [s.strip() for s in preferred_titles.split(",") if s.strip()]

    title = ""
    if isinstance(target_roles, list) and target_roles:
        title = target_roles[0]
    elif isinstance(preferred_titles, list) and preferred_titles:
        title = preferred_titles[0]
    else:
        title = (
            meta.get("target_title")
            or meta.get("headline")
            or meta.get("job_title")
            or ""
        )

    title = str(title).strip()

    remote_pref = str(meta.get("remote_preference") or "").lower()
    zip_code = str(meta.get("location_zip") or "").strip()
    location_fallback = str(meta.get("location") or "").strip()

    if remote_pref == "remote_only":
        location = "Remote"
    elif remote_pref == "remote_plus_local":
        # Prefer local zip, but remote is acceptable
        location = zip_code or "Remote"
    else:
        # Local-only or unspecified
        if zip_code:
            location = zip_code
        elif location_fallback:
            location = location_fallback
        else:
            location = "Remote"

    return title, location


def build_search_query(meta: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """
    Build (query, location, skills_list) for searching job boards.

    Uses:
      - Derived title from target_roles / preferred_titles
      - key_skills / skills
      - search_aggressiveness: "narrow", "balanced", "wide"
    """
    title, location = extract_title_and_location(meta)
    skills = normalize_skills(meta)
    search_aggr = str(meta.get("search_aggressiveness") or "balanced").lower()

    # Base query: job title or, if missing, top skills
    if not title:
        if skills:
            base_query = " ".join(skills[:3])
        else:
            base_query = "project manager"
    else:
        base_query = title

    # Adjust query based on aggressiveness
    if search_aggr == "narrow":
        # Just the title
        query = base_query

    elif search_aggr == "wide":
        # Title + first 3 skills
        extra = " ".join(skills[:3])
        query = f"{base_query} {extra}".strip()

    else:  # "balanced" and unknown values
        # Add 1–2 high-signal keywords often useful in tech roles
        high_signal = []
        for kw in ["agile", "scrum", "cloud", "python"]:
            for s in skills:
                if kw in s and kw not in high_signal:
                    high_signal.append(kw)
                    break
        if high_signal:
            query = f"{base_query} {' '.join(high_signal[:2])}".strip()
        else:
            query = base_query

    return query, location, skills


def compute_score(text: str, skills: List[str], title_hint: str = "") -> float:
    """
    Simple heuristic score: count skill phrase matches + mild title match bonus.
    """
    t = text.lower()
    score = 0.0

    for skill in skills:
        if not skill:
            continue
        # Phrase-level match
        if skill in t:
            score += 2.0

        # Token-level rough match
        for token in skill.split():
            token = token.strip()
            if token and token in t:
                score += 0.5

    if title_hint:
        th = title_hint.lower()
        if th in t:
            score += 3.0

    return score


def clean_text(s: str) -> str:
    """Normalize whitespace a bit."""
    if not s:
        return ""
    return " ".join(s.split())


# -----------------------------
# Site-specific search functions (result list only)
# -----------------------------

def scrape_indeed(browser, query: str, location: str, skills: List[str], title_hint: str) -> List[Dict[str, Any]]:
    """
    Scrape Indeed first page results.
    """
    log("Searching Indeed...")
    jobs: List[Dict[str, Any]] = []
    page = browser.new_page()
    try:
        q = quote_plus(query)
        loc = quote_plus(location)
        url = f"https://www.indeed.com/jobs?q={q}&l={loc}"
        page.goto(url, timeout=60000)
        page.wait_for_timeout(4000)

        cards = page.query_selector_all("a.tapItem")
        for card in cards:
            try:
                title_el = card.query_selector("h2.jobTitle") or card.query_selector("h2")
                title = clean_text(title_el.inner_text()) if title_el else "Untitled"

                company_el = card.query_selector(".companyName")
                company = clean_text(company_el.inner_text()) if company_el else ""

                loc_el = card.query_selector(".companyLocation")
                loc_text = clean_text(loc_el.inner_text()) if loc_el else ""

                snip_el = card.query_selector(".job-snippet")
                snippet = clean_text(snip_el.inner_text()) if snip_el else ""

                href = card.get_attribute("href") or ""
                if href.startswith("/"):
                    href = f"https://www.indeed.com{href}"

                text_for_score = f"{title}\n{company}\n{loc_text}\n{snippet}"
                score = compute_score(text_for_score, skills, title_hint=title_hint)

                if score <= 0:
                    continue

                jobs.append(
                    {
                        "site": "indeed",
                        "title": title,
                        "company": company,
                        "location": loc_text,
                        "snippet": snippet,
                        "url": href,
                        "score": score,
                        "full_text": None,
                    }
                )
            except Exception:
                continue

    except Exception as e:
        log(f"ERROR scraping Indeed: {e}")
    finally:
        page.close()

    log(f"Indeed: collected {len(jobs)} scored jobs.")
    return jobs


def scrape_ziprecruiter(browser, query: str, location: str, skills: List[str], title_hint: str) -> List[Dict[str, Any]]:
    """
    Scrape ZipRecruiter first page.
    """
    log("Searching ZipRecruiter...")
    jobs: List[Dict[str, Any]] = []
    page = browser.new_page()
    try:
        q = quote_plus(query)
        loc = quote_plus(location)
        url = f"https://www.ziprecruiter.com/candidate/search?search={q}&location={loc}"
        page.goto(url, timeout=60000)
        page.wait_for_timeout(4000)

        cards = page.query_selector_all("article")
        if not cards:
            cards = page.query_selector_all(".job_content, .job_result")

        for card in cards:
            try:
                title_el = card.query_selector("a[name='job_title']") or card.query_selector("a.job_link") or card.query_selector("h2")
                title = clean_text(title_el.inner_text()) if title_el else "Untitled"

                company_el = card.query_selector(".job_org, .t_org_link, .company")
                company = clean_text(company_el.inner_text()) if company_el else ""

                loc_el = card.query_selector(".job_loc, .job_location")
                loc_text = clean_text(loc_el.inner_text()) if loc_el else ""

                snip_el = card.query_selector(".job_snippet, .job_snippet_text, p")
                snippet = clean_text(snip_el.inner_text()) if snip_el else ""

                href = ""
                if title_el:
                    href = title_el.get_attribute("href") or ""
                if href.startswith("/"):
                    href = f"https://www.ziprecruiter.com{href}"

                text_for_score = f"{title}\n{company}\n{loc_text}\n{snippet}"
                score = compute_score(text_for_score, skills, title_hint=title_hint)
                if score <= 0:
                    continue

                jobs.append(
                    {
                        "site": "ziprecruiter",
                        "title": title,
                        "company": company,
                        "location": loc_text,
                        "snippet": snippet,
                        "url": href,
                        "score": score,
                        "full_text": None,
                    }
                )
            except Exception:
                continue

    except Exception as e:
        log(f"ERROR scraping ZipRecruiter: {e}")
    finally:
        page.close()

    log(f"ZipRecruiter: collected {len(jobs)} scored jobs.")
    return jobs


def scrape_glassdoor(browser, query: str, location: str, skills: List[str], title_hint: str) -> List[Dict[str, Any]]:
    """
    Scrape Glassdoor first page.
    """
    log("Searching Glassdoor...")
    jobs: List[Dict[str, Any]] = []
    page = browser.new_page()
    try:
        q = quote_plus(query)
        loc = quote_plus(location)
        url = f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={q}&locT=C&locId=&locKeyword={loc}"
        page.goto(url, timeout=60000)
        page.wait_for_timeout(5000)

        cards = page.query_selector_all("article") or page.query_selector_all("li[data-test='job-listing']")
        for card in cards:
            try:
                title_el = card.query_selector("a[data-test='job-link'], a") or card.query_selector("span")
                title = clean_text(title_el.inner_text()) if title_el else "Untitled"

                company_el = card.query_selector("div[data-test='job-layout'] span") or card.query_selector("div[data-test='employerName']")
                company = clean_text(company_el.inner_text()) if company_el else ""

                loc_el = card.query_selector("div[data-test='job-location']")
                loc_text = clean_text(loc_el.inner_text()) if loc_el else ""

                snip_el = card.query_selector("div[data-test='job-snippet']") or card.query_selector("p")
                snippet = clean_text(snip_el.inner_text()) if snip_el else ""

                href = ""
                if title_el:
                    href = title_el.get_attribute("href") or ""
                if href.startswith("/"):
                    href = f"https://www.glassdoor.com{href}"

                text_for_score = f"{title}\n{company}\n{loc_text}\n{snippet}"
                score = compute_score(text_for_score, skills, title_hint=title_hint)
                if score <= 0:
                    continue

                jobs.append(
                    {
                        "site": "glassdoor",
                        "title": title,
                        "company": company,
                        "location": loc_text,
                        "snippet": snippet,
                        "url": href,
                        "score": score,
                        "full_text": None,
                    }
                )
            except Exception:
                continue

    except Exception as e:
        log(f"ERROR scraping Glassdoor: {e}")
    finally:
        page.close()

    log(f"Glassdoor: collected {len(jobs)} scored jobs.")
    return jobs


def scrape_linkedin(browser, query: str, location: str, skills: List[str], title_hint: str) -> List[Dict[str, Any]]:
    """
    Scrape LinkedIn public jobs search (no auth).
    This is best-effort and may be limited by LinkedIn's UX or anti-bot behavior.
    """
    log("Searching LinkedIn (public)...")
    jobs: List[Dict[str, Any]] = []
    page = browser.new_page()
    try:
        q = quote_plus(query)
        loc = quote_plus(location)
        url = f"https://www.linkedin.com/jobs/search?keywords={q}&location={loc}"
        page.goto(url, timeout=60000)
        page.wait_for_timeout(6000)

        cards = page.query_selector_all("li.jobs-search-results__list-item") or page.query_selector_all("div.base-card")
        for card in cards:
            try:
                title_el = card.query_selector("h3") or card.query_selector("a") or card.query_selector("span")
                title = clean_text(title_el.inner_text()) if title_el else "Untitled"

                company_el = card.query_selector("h4") or card.query_selector("a[data-tracking-control-name*='company-name']")
                company = clean_text(company_el.inner_text()) if company_el else ""

                loc_el = card.query_selector(".job-search-card__location")
                loc_text = clean_text(loc_el.inner_text()) if loc_el else ""

                snip_el = card.query_selector("p")  # LinkedIn often hides full desc behind click; snippet is minimal
                snippet = clean_text(snip_el.inner_text()) if snip_el else ""

                href = ""
                link_el = card.query_selector("a[href*='/jobs/view/']")
                if link_el:
                    href = link_el.get_attribute("href") or ""
                if href and href.startswith("/"):
                    href = f"https://www.linkedin.com{href}"

                text_for_score = f"{title}\n{company}\n{loc_text}\n{snippet}"
                score = compute_score(text_for_score, skills, title_hint=title_hint)
                if score <= 0:
                    continue

                jobs.append(
                    {
                        "site": "linkedin",
                        "title": title,
                        "company": company,
                        "location": loc_text,
                        "snippet": snippet,
                        "url": href,
                        "score": score,
                        "full_text": None,
                    }
                )
            except Exception:
                continue

    except Exception as e:
        log(f"ERROR scraping LinkedIn: {e}")
    finally:
        page.close()

    log(f"LinkedIn: collected {len(jobs)} scored jobs.")
    return jobs


def scrape_usajobs(browser, query: str, location: str, skills: List[str], title_hint: str) -> List[Dict[str, Any]]:
    """
    Scrape USAJobs first page.
    """
    log("Searching USAJobs...")
    jobs: List[Dict[str, Any]] = []
    page = browser.new_page()
    try:
        q = quote_plus(query)
        loc = quote_plus(location)
        url = f"https://www.usajobs.gov/Search/Results?Keyword={q}&Location={loc}"
        page.goto(url, timeout=60000)
        page.wait_for_timeout(5000)

        cards = page.query_selector_all("usajobs-search-result-card") or page.query_selector_all("li")
        for card in cards:
            try:
                title_el = card.query_selector("a usajobs-link, a") or card.query_selector("h2")
                title = clean_text(title_el.inner_text()) if title_el else "Untitled"

                company_el = card.query_selector("[data-testid='hiring-organization-name']")
                company = clean_text(company_el.inner_text()) if company_el else "US Federal Government"

                loc_el = card.query_selector("[data-testid='location']") or card.query_selector("usajobs-search-location")
                loc_text = clean_text(loc_el.inner_text()) if loc_el else ""

                snip_el = card.query_selector("p") or card.query_selector("[data-testid='summary']")
                snippet = clean_text(snip_el.inner_text()) if snip_el else ""

                href = ""
                link_el = card.query_selector("a")
                if link_el:
                    href = link_el.get_attribute("href") or ""
                if href.startswith("/"):
                    href = f"https://www.usajobs.gov{href}"

                text_for_score = f"{title}\n{company}\n{loc_text}\n{snippet}"
                score = compute_score(text_for_score, skills, title_hint=title_hint)
                if score <= 0:
                    continue

                jobs.append(
                    {
                        "site": "usajobs",
                        "title": title,
                        "company": company,
                        "location": loc_text,
                        "snippet": snippet,
                        "url": href,
                        "score": score,
                        "full_text": None,
                    }
                )
            except Exception:
                continue

    except Exception as e:
        log(f"ERROR scraping USAJobs: {e}")
    finally:
        page.close()

    log(f"USAJobs: collected {len(jobs)} scored jobs.")
    return jobs


def scrape_monster(browser, query: str, location: str, skills: List[str], title_hint: str) -> List[Dict[str, Any]]:
    """
    Scrape Monster first page (yes, it's still a thing).
    """
    log("Searching Monster...")
    jobs: List[Dict[str, Any]] = []
    page = browser.new_page()
    try:
        q = quote_plus(query)
        loc = quote_plus(location)
        url = f"https://www.monster.com/jobs/search/?q={q}&where={loc}"
        page.goto(url, timeout=60000)
        page.wait_for_timeout(5000)

        cards = page.query_selector_all("section.card-content") or page.query_selector_all("article")
        for card in cards:
            try:
                title_el = card.query_selector("h2 a") or card.query_selector("h2")
                title = clean_text(title_el.inner_text()) if title_el else "Untitled"

                company_el = card.query_selector(".company, .company-name")
                company = clean_text(company_el.inner_text()) if company_el else ""

                loc_el = card.query_selector(".location")
                loc_text = clean_text(loc_el.inner_text()) if loc_el else ""

                snip_el = card.query_selector("div.summary, p")
                snippet = clean_text(snip_el.inner_text()) if snip_el else ""

                href = ""
                if title_el:
                    href = title_el.get_attribute("href") or ""
                if href.startswith("/"):
                    href = f"https://www.monster.com{href}"

                text_for_score = f"{title}\n{company}\n{loc_text}\n{snippet}"
                score = compute_score(text_for_score, skills, title_hint=title_hint)
                if score <= 0:
                    continue

                jobs.append(
                    {
                        "site": "monster",
                        "title": title,
                        "company": company,
                        "location": loc_text,
                        "snippet": snippet,
                        "url": href,
                        "score": score,
                        "full_text": None,
                    }
                )
            except Exception:
                continue

    except Exception as e:
        log(f"ERROR scraping Monster: {e}")
    finally:
        page.close()

    log(f"Monster: collected {len(jobs)} scored jobs.")
    return jobs


# -----------------------------
# Full-run orchestration
# -----------------------------

def fetch_full_descriptions(browser, jobs: List[Dict[str, Any]]) -> None:
    """
    For each selected job, visit its URL and pull full body text.
    """
    if not jobs:
        return

    context = browser.new_context()
    page = context.new_page()

    for idx, job in enumerate(jobs, start=1):
        url = job.get("url") or ""
        if not url:
            job["full_text"] = job.get("snippet", "")
            continue

        try:
            log(f"Fetching full description for job {idx}: {url}")
            page.goto(url, timeout=70000)
            page.wait_for_timeout(4000)
            body_text = page.text_content("body") or ""
            job["full_text"] = clean_text(body_text) or job.get("snippet", "")
        except PlaywrightTimeoutError:
            log(f"Timeout fetching {url}")
            job["full_text"] = job.get("snippet", "")
        except Exception as e:
            log(f"Error fetching {url}: {e}")
            job["full_text"] = job.get("snippet", "")

    page.close()
    context.close()


def write_outputs(job_dir: Path, jobs: List[Dict[str, Any]], top_n: int) -> None:
    """
    Write job_description_XX.txt and job_sources.txt.
    top_n controls how many job_description_XX.txt files we emit.
    Uses a primary/secondary split so high-fit jobs are preferred.
    """
    # Clean old descriptions
    for old in job_dir.glob("job_description_*.txt"):
        try:
            old.unlink()
        except Exception:
            # Best-effort cleanup only
            pass

    # Sort by score desc
    jobs_sorted = sorted(jobs, key=lambda j: j.get("score", 0.0), reverse=True)

    # Partition into primary (high-fit) and secondary (stretch / backup)
    primary = [j for j in jobs_sorted if j.get("score", 0.0) >= PRIMARY_FIT_THRESHOLD]
    secondary = [j for j in jobs_sorted if j.get("score", 0.0) < PRIMARY_FIT_THRESHOLD]

    if len(primary) >= top_n:
        # Plenty of high-fit jobs: only use those
        selected = primary[:top_n]
    else:
        # Use all high-fit jobs, then fill the rest with the best of the secondary
        needed = top_n - len(primary)
        selected = primary + secondary[:needed]

    # In tiny markets, we may still have fewer than top_n total jobs.
    selected = selected[:len(selected)]

    # Write descriptions
    for i, job in enumerate(selected, start=1):
        fname = job_dir / f"job_description_{i:02d}.txt"
        content_lines = [
            f"Title: {job.get('title', '')}",
            f"Company: {job.get('company', '')}",
            f"Location: {job.get('location', '')}",
            f"Site: {job.get('site', '')}",
            f"Source URL: {job.get('url', '')}",
            "",
            "Job Summary:",
            job.get("snippet", ""),
            "",
            "Full Description:",
            job.get("full_text", job.get("snippet", "")),
            "",
        ]
        try:
            with fname.open("w", encoding="utf-8") as f:
                f.write("\n".join(content_lines))
        except Exception as e:
            log(f"ERROR writing {fname}: {e}")

    # Write job_sources.txt
    src_path = job_dir / "job_sources.txt"
    try:
        with src_path.open("w", encoding="utf-8") as f:
            high_fit_count = len(
                [j for j in selected if j.get("score", 0.0) >= PRIMARY_FIT_THRESHOLD]
            )
            f.write(
                f"# Selected {len(selected)} jobs "
                f"(high-fit >= {PRIMARY_FIT_THRESHOLD}: {high_fit_count})\n\n"
            )
            for i, job in enumerate(selected, start=1):
                f.write(
                    f"{i:02d}. [{job.get('site','')}] "
                    f"score={job.get('score', 0.0):.2f} "
                    f"url={job.get('url','')}\n"
                )
                f.write(
                    f"    title={job.get('title','')}\n"
                    f"    company={job.get('company','')}\n"
                    f"    location={job.get('location','')}\n\n"
                )
    except Exception as e:
        log(f"ERROR writing {src_path}: {e}")

    log(f"Wrote {len(selected)} job_description_XX.txt files and job_sources.txt")

def scrape_job_boards(job_dir: Path) -> None:
    meta = load_client_meta(job_dir)
    query, location, skills = build_search_query(meta)

    # Determine how many jobs we want to *deliver* and how many we are willing to *scrape*
    job_volume_target = int(meta.get("job_volume_target", TOP_N_JOBS_DEFAULT) or TOP_N_JOBS_DEFAULT)
    if job_volume_target <= 0:
        job_volume_target = TOP_N_JOBS_DEFAULT

    max_jobs_to_scrape = int(meta.get("max_jobs_to_scrape", job_volume_target * 10) or (job_volume_target * 10))
    if max_jobs_to_scrape < job_volume_target:
        max_jobs_to_scrape = job_volume_target

    top_n = job_volume_target

    log(f"Job dir: {job_dir}")
    log(f"Search query: '{query}'  | location: '{location}'")
    log(f"Skills: {skills}")
    log(f"Target job volume: {top_n}, max jobs to scrape: {max_jobs_to_scrape}")

    all_jobs: List[Dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        try:
            # Each function returns a list of scored jobs
            all_jobs.extend(scrape_indeed(browser, query, location, skills, title_hint=query))
            all_jobs.extend(scrape_linkedin(browser, query, location, skills, title_hint=query))
            all_jobs.extend(scrape_ziprecruiter(browser, query, location, skills, title_hint=query))
            all_jobs.extend(scrape_glassdoor(browser, query, location, skills, title_hint=query))
            all_jobs.extend(scrape_usajobs(browser, query, location, skills, title_hint=query))
            all_jobs.extend(scrape_monster(browser, query, location, skills, title_hint=query))

            log(f"Total collected scored jobs across all boards: {len(all_jobs)}")

            if not all_jobs:
                log("No scored jobs found. Writing a single fallback job_description_01.txt")
                fallback = job_dir / "job_description_01.txt"
                with fallback.open("w", encoding="utf-8") as f:
                    f.write(
                        "No matching jobs were found from the boards.\n"
                        "This is a fallback placeholder. The packet builder will still run.\n"
                    )
                return

            # Sort jobs and decide how many to fetch full descriptions for
            all_jobs_sorted = sorted(all_jobs, key=lambda j: j["score"], reverse=True)

            max_fetch = max(top_n * MAX_FETCH_MULTIPLIER, top_n)
            max_fetch = min(len(all_jobs_sorted), max_fetch, max_jobs_to_scrape)

            to_fetch = all_jobs_sorted[:max_fetch]

            fetch_full_descriptions(browser, to_fetch)

            # Now write outputs for the top_n jobs based on updated list
            write_outputs(job_dir, to_fetch, top_n)

        finally:
            browser.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: job_scraper.py /abs/path/to/JOB_xxxx", file=sys.stderr)
        sys.exit(1)

    job_dir = Path(sys.argv[1]).resolve()
    if not job_dir.exists():
        print(f"Job directory does not exist: {job_dir}", file=sys.stderr)
        sys.exit(1)

    log(f"Processing job folder: {job_dir}")
    try:
        scrape_job_boards(job_dir)
        log("Job scraping completed.")
    except Exception as e:
        log(f"UNHANDLED ERROR: {e}")
        traceback.print_exc()
        # Let run_fiverr_pipeline see this as a failure if needed
        sys.exit(1)


if __name__ == "__main__":
    main()
