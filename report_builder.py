#!/usr/bin/env python3
"""
report_builder.py

Builds a human-readable summary report for a FiverrMachine job folder.

Inputs (inside JOB dir):
  - client_request.json       (optional but preferred)
  - job_sources.txt           (from job_scraper.py)
  - selection_summary.json    (from deliverable_packager / matcher)

Output:
  - final_report.md           (Markdown summary for the client / VSOP)
"""

import json
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import DEFAULT_CONFIG, PipelineConfig


def log(msg: str) -> None:
    print(f"[REPORT] {msg}", flush=True)


def _clamp_score(val: float) -> float:
    try:
        return max(0.0, min(100.0, float(val)))
    except Exception:
        return 0.0


# -----------------------------
# Load client metadata
# -----------------------------

def load_client_meta(job_dir: Path) -> Dict[str, Any]:
    meta_path = job_dir / "client_request.json"
    if not meta_path.exists():
        log("client_request.json not found; proceeding with minimal metadata.")
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"Failed to parse client_request.json: {e}")
        return {}


def load_jobs_manifest(job_dir: Path) -> List[Dict[str, Any]]:
    path = job_dir / "jobs_manifest.json"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as exc:
        log(f"Failed to parse jobs_manifest.json: {exc}")
    return []


# -----------------------------
# Parse job_sources.txt
# -----------------------------

def parse_job_sources(job_dir: Path, manifest: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Parse job_sources.txt into:
      {
        "high_fit_threshold": float or None,
        "high_fit_count": int or None,
        "jobs": [
          {
            "index": int,
            "site": str,
            "score": float,
            "url": str,
            "title": str,
            "company": str,
            "location": str,
          },
          ...
        ]
      }
    """
    if manifest:
        jobs = [
            {
                "index": int(item.get("index", idx + 1)),
                "site": item.get("site", ""),
                "score": float(item.get("raw_scraper_score", 0.0) or 0.0),
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "company": item.get("company", ""),
                "location": item.get("location", ""),
                "salary": item.get("salary", ""),
            }
            for idx, item in enumerate(manifest)
        ]
        return {"high_fit_threshold": None, "high_fit_count": None, "jobs": jobs}

    src_path = job_dir / "job_sources.txt"
    if not src_path.exists():
        log("job_sources.txt not found; no jobs to report.")
        return {"high_fit_threshold": None, "high_fit_count": None, "jobs": []}

    lines = src_path.read_text(encoding="utf-8").splitlines()

    high_fit_threshold: Optional[float] = None
    high_fit_count: Optional[int] = None
    jobs: List[Dict[str, Any]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Header example:
        # # Selected 20 jobs (high-fit >= 2.5: 6)
        if line.startswith("# Selected") and "high-fit" in line:
            try:
                # crude parse for "high-fit >= X: Y"
                part = line.split("high-fit", 1)[1]
                part = part.replace("(", "").replace(")", "")
                # e.g. " >= 2.5: 6"
                part = part.replace(">=", "").replace(":", " ").strip()
                pieces = part.split()
                if len(pieces) >= 2:
                    high_fit_threshold = float(pieces[0])
                    high_fit_count = int(pieces[1])
            except Exception:
                pass
            i += 1
            continue

        # Job entry example:
        # 01. [linkedin] score=3.00 url=...
        if "." in line and line[0:2].isdigit() and "score=" in line:
            try:
                # Parse job index
                idx_str = line.split(".", 1)[0]
                index = int(idx_str)

                # Extract site in brackets
                site_part = line.split("]", 1)[0]
                site = site_part.split("[", 1)[1] if "[" in site_part else ""

                # Extract score
                score_str = "0.0"
                if "score=" in line:
                    score_part = line.split("score=", 1)[1]
                    score_str = score_part.split()[0]
                score = float(score_str)

                # Extract url
                url = ""
                if "url=" in line:
                    url = line.split("url=", 1)[1].strip()

                # Next 3 lines should be title / company / location
                title = ""
                company = ""
                location = ""

                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("title="):
                        title = next_line.split("=", 1)[1].strip()

                if i + 2 < len(lines):
                    next_line = lines[i + 2].strip()
                    if next_line.startswith("company="):
                        company = next_line.split("=", 1)[1].strip()

                if i + 3 < len(lines):
                    next_line = lines[i + 3].strip()
                    if next_line.startswith("location="):
                        location = next_line.split("=", 1)[1].strip()

                jobs.append(
                    {
                        "index": index,
                        "site": site,
                        "score": score,
                        "url": url,
                        "title": title,
                        "company": company,
                        "location": location,
                    }
                )
            except Exception:
                # If parsing fails, skip this block
                pass

            # Skip ahead past the 3 detail lines
            i += 4
            continue

        i += 1

    log(f"Parsed {len(jobs)} jobs from job_sources.txt")
    return {
        "high_fit_threshold": high_fit_threshold,
        "high_fit_count": high_fit_count,
        "jobs": jobs,
    }


# -----------------------------
# Simple bucketing for match scores
# -----------------------------

def bucket_match_score(
    score: float, high_fit_threshold: Optional[float], config: PipelineConfig = DEFAULT_CONFIG
) -> str:
    """
    Bucket the job_scraper score into human labels.

    Note: this is not the same as the LLM-based match score,
    but it's still a useful "job match" signal.
    """
    if high_fit_threshold is None:
        # Fallback heuristic based on pipeline config
        if score >= config.min_score_for_high_fit:
            return "High"
        elif score >= config.min_score_for_high_fit - 10:
            return "Medium"
        elif score > 0:
            return "Low"
        else:
            return "Very Low"

    # If we have a high-fit threshold, use that and derive others
    if score >= high_fit_threshold:
        return "High"
    elif score >= high_fit_threshold * 0.7:
        return "Medium"
    elif score > 0:
        return "Low"
    else:
        return "Very Low"


# -----------------------------
# Load selection_summary.json
# -----------------------------

def load_selection_summary(job_dir: Path) -> Dict[str, Any]:
    """
    Load LLM match scores and selected labels from selection_summary.json, if present.
    """
    path = job_dir / "selection_summary.json"
    if not path.exists():
        log("selection_summary.json not found; proceeding without LLM match scores.")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log(f"Failed to parse selection_summary.json: {exc}")
        return {}


# -----------------------------
# Salary band inference
# -----------------------------

def infer_salary_band(job_dir: Path, selected_labels: List[str]) -> Dict[str, Any]:
    """
    Very simple heuristic salary extraction from the selected job_description_XX.txt files.

    We only look for dollar amounts like "$120,000" or "$95000" and then keep
    values that look like annual salaries (between 20k and 400k).
    """
    salary_values: List[int] = []
    posting_count: int = 0

    for label in selected_labels:
        try:
            idx = int(label)
        except (TypeError, ValueError):
            continue

        desc_path = job_dir / f"job_description_{idx:02d}.txt"
        if not desc_path.exists():
            continue

        try:
            text = desc_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        found_in_this_posting = False

        # Only treat numbers with a dollar sign as salary to avoid "401(k)" noise, etc.
        for m in re.finditer(r"\$\s*([0-9]{2,3}(?:,[0-9]{3})?)", text):
            raw = m.group(1)
            try:
                num = int(raw.replace(",", ""))
            except ValueError:
                continue

            # Rough annual-salary bounds
            if 20_000 <= num <= 400_000:
                salary_values.append(num)
                found_in_this_posting = True

        if found_in_this_posting:
            posting_count += 1

    if not salary_values:
        return {}

    salary_values.sort()
    min_s = salary_values[0]
    max_s = salary_values[-1]
    avg_s = int(round(sum(salary_values) / len(salary_values)))

    mid = len(salary_values) // 2
    if len(salary_values) % 2 == 1:
        median_s = salary_values[mid]
    else:
        median_s = int(round((salary_values[mid - 1] + salary_values[mid]) / 2))

    return {
        "min": min_s,
        "max": max_s,
        "avg": avg_s,
        "median": median_s,
        "sample_count": len(salary_values),
        "posting_count": posting_count,
    }


# -----------------------------
# Build the markdown report
# -----------------------------

def build_markdown_report(
    job_dir: Path,
    client: Dict[str, Any],
    jobs_info: Dict[str, Any],
    selection: Optional[Dict[str, Any]] = None,
    manifest: Optional[List[Dict[str, Any]]] = None,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> str:
    selection = selection or {}
    manifest = manifest or []

    manifest_by_index = {int(m.get("index", i + 1)): m for i, m in enumerate(manifest)}
    manifest_by_id = {m.get("job_id") or f"job{m.get('index', i + 1):02d}": m for i, m in enumerate(manifest)}

    selection_jobs = selection.get("jobs") or []
    jobs: List[Dict[str, Any]] = []

    if selection_jobs:
        for sj in selection_jobs:
            label = sj.get("label") or sj.get("job_id") or ""
            try:
                idx = int(str(label).lstrip("job"))
            except (TypeError, ValueError):
                idx = None
            job_id = sj.get("job_id") or (f"job{idx:02d}" if idx else "")
            meta = manifest_by_id.get(job_id) or (manifest_by_index.get(idx) if idx else {})

            norm_score = _clamp_score(sj.get("normalized_score", sj.get("raw_score", 0.0)))
            bucket = sj.get("match_bucket") or bucket_match_score(norm_score, jobs_info.get("high_fit_threshold"), config)

            jobs.append(
                {
                    "index": idx or len(jobs) + 1,
                    "job_id": job_id or f"job{len(jobs) + 1:02d}",
                    "title": sj.get("title") or meta.get("title", ""),
                    "company": sj.get("company") or meta.get("company", ""),
                    "location": sj.get("location") or meta.get("location", ""),
                    "site": sj.get("source_site") or meta.get("site", ""),
                    "url": sj.get("url") or meta.get("url", ""),
                    "match_bucket": bucket,
                    "normalized_score": norm_score,
                    "salary": sj.get("salary_range") or meta.get("salary", ""),
                }
            )
    else:
        for j in jobs_info.get("jobs", []):
            idx = j.get("index", len(jobs) + 1)
            bucket = bucket_match_score(j.get("score", 0.0), jobs_info.get("high_fit_threshold"), config)
            jobs.append(
                {
                    "index": idx,
                    "job_id": f"job{int(idx):02d}",
                    "title": j.get("title", ""),
                    "company": j.get("company", ""),
                    "location": j.get("location", ""),
                    "site": j.get("site", ""),
                    "url": j.get("url", ""),
                    "match_bucket": bucket,
                    "normalized_score": _clamp_score(j.get("score", 0.0)),
                    "salary": j.get("salary", ""),
                }
            )

    jobs = sorted(jobs, key=lambda j: j.get("normalized_score", 0.0), reverse=True)

    focus_ids = selection.get("focus_shortlist") or []
    focus_rows = [j for j in jobs if j.get("job_id") in focus_ids]
    if not focus_rows:
        focus_rows = jobs[: config.focus_shortlist_size]

    selected_labels = [str(j.get("index")) for j in focus_rows]
    salary_band = infer_salary_band(job_dir, selected_labels)

    lines: List[str] = []
    lines.append("# Job Search Report for Client")
    lines.append("")

    client_name = client.get("name") or client.get("client_name") or client.get("customer_name")
    if client_name:
        lines.append(f"### Prepared for: {client_name}")
        lines.append("")
    lines.append(
        "A focused, AI-assisted job search packet built from your resume and intake data. "
        "Use this report to see where your best-fit opportunities are and what to do next."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    site_counts: Dict[str, int] = {}
    bucket_counts: Dict[str, int] = {"High": 0, "Medium": 0, "Low": 0, "Very Low": 0}
    score_values: List[float] = []

    for j in jobs:
        site = j.get("site") or "?"
        site_counts[site] = site_counts.get(site, 0) + 1
        bucket = j.get("match_bucket") or bucket_match_score(
            j.get("normalized_score", 0.0), jobs_info.get("high_fit_threshold"), config
        )
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if isinstance(j.get("normalized_score"), (int, float)):
            score_values.append(float(j["normalized_score"]))

    lines.append("## Summary at a Glance")
    lines.append("")
    if score_values:
        avg_score = sum(score_values) / len(score_values)
        lines.append(f"Average match score across all roles: **{avg_score:.1f}** (0–100 scale)")
        lines.append("")
    lines.append("**By match bucket:**")
    for b in ["High", "Medium", "Low", "Very Low"]:
        count = bucket_counts.get(b, 0)
        lines.append(f"- {b}: {count} job(s)")
    lines.append("")
    lines.append("**By source site:**")
    for site, count in sorted(site_counts.items(), key=lambda x: x[0]):
        lines.append(f"- {site}: {count} job(s)")
    lines.append("")
    lines.append("---")
    lines.append("")

    if focus_rows:
        lines.append("## Focus Roles (AI-selected shortlist)")
        lines.append("")
        lines.append(
            "These are the roles FiverrMachine pulled into your final packet, ranked by the AI match score. "
            "If you only have time to apply to a handful of roles, **start here**."
        )
        lines.append("")
        lines.append("| Job | Match score | Title | Company | Location | Where to apply | Salary signal |")
        lines.append("|-----|-------------|-------|---------|----------|----------------|---------------|")

        match_values: List[float] = []

        for row in focus_rows:
            label_str = str(row.get("index") or row.get("job_id") or "")
            ms = row.get("normalized_score")
            title = row.get("title") or "(no title)"
            company = row.get("company") or "(no company)"
            location = row.get("location") or "(unspecified)"
            url = row.get("url") or ""
            salary = row.get("salary") or "—"

            title_md = title.replace("|", "\|")
            company_md = company.replace("|", "\|")
            location_md = location.replace("|", "\|")

            if isinstance(ms, (int, float)):
                match_values.append(float(ms))
                ms_str = f"{ms:.1f}"
            else:
                ms_str = "—"

            apply_md = f"[Open posting]({url})" if url else "Search title + company"

            lines.append(
                f"| {label_str} | {ms_str} | {title_md} | {company_md} | {location_md} | {apply_md} | {salary} |"
            )

        lines.append("")

        if salary_band:
            min_s = salary_band["min"]
            max_s = salary_band["max"]
            median_s = salary_band["median"]
            posting_count = salary_band.get("posting_count", 0)

            lines.append("### Market Compensation Signal (Approximate)")
            lines.append("")
            lines.append(
                f"Based on the focus roles that actually list explicit annual salary ranges "
                f"(**{posting_count} of {len(focus_rows)}**), similar roles around your target area "
                f"cluster roughly between **${min_s:,.0f}** and **${max_s:,.0f}** per year, with a central "
                f"band near **${median_s:,.0f}**."
            )
            lines.append("")

        if match_values:
            avg_match = sum(match_values) / len(match_values)
            lines.append(
                f"Average AI match score for the focus list: **{avg_match:.1f}** (0–100)."
            )

        lines.append("")
        lines.append(
            "> This is **not** a formal salary survey or negotiation advice. Treat it as a directional "
            "signal so you don't accidentally underprice yourself when an offer comes in far below this band."
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## What This Report Is (and Isn’t)")
    lines.append("")
    lines.append("**This report IS:**")
    lines.append("- A scraped and filtered list of roles that match your target titles and skills.")
    lines.append("- A way to **focus your energy** on the jobs most aligned with your background.")
    lines.append("- A starting point for tailored resumes and cover letters, not a final verdict on your worth.")
    lines.append("")
    lines.append("**This report is NOT:**")
    lines.append("- A guarantee of interviews or offers.")
    lines.append("- A magic ATS bypass or insider connection.")
    lines.append("- A judgement on your value — it’s just how the text in your profile and skills line up with the text in job ads.")
    lines.append("")
    lines.append("> Translation: this report gives you **leverage**, not lottery tickets.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Job List (Best Matches First)")
    lines.append("")
    lines.append(
        "Jobs are ordered by how closely the job description text matches your target roles and key skills. "
        "Use this to decide where to apply next once you've exhausted the focus shortlist above."
    )
    lines.append("")
    lines.append("| # | Title | Company | Location | Site | Match bucket | Match score |")
    lines.append("|---|-------|---------|----------|------|--------------|-------------|")

    for j in jobs:
        idx = j["index"]
        title = j.get("title") or "(no title)"
        company = j.get("company") or "(no company)"
        location = j.get("location") or "(unspecified)"
        site = j.get("site") or "?"
        score = j.get("normalized_score", 0.0)
        bucket = j.get("match_bucket") or bucket_match_score(score, jobs_info.get("high_fit_threshold"), config)

        title_md = title.replace("|", "\|")
        company_md = company.replace("|", "\|")
        location_md = location.replace("|", "\|")

        lines.append(
            f"| {int(idx):02d} | {title_md} | {company_md} | {location_md} | {site} | {bucket} | {score:.1f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## How to Use This Report")
    lines.append("")
    lines.append("Think of this as your **attack plan**, not a wall of links.")
    lines.append("")
    lines.append("1. **Start with the focus shortlist.**")
    lines.append("   - These roles combine **textual match** with your resume and **salary signals** from the market.")
    lines.append("   - If you only have limited time each week, start here.")
    lines.append("")
    lines.append("2. **Then move into High bucket roles from the full list.**")
    lines.append("   - These are the next closest matches and can broaden your options without going off-track.")
    lines.append("")
    lines.append("3. **Use Medium / Low buckets to explore adjacent paths.**")
    lines.append("   - Apply if you want more volume or are intentionally exploring different directions.")
    lines.append("   - You’re not expected to chase everything on this list.")
    lines.append("")
    lines.append("4. **Pair this report with tailored resumes and cover letters.**")
    lines.append("   - The power move is: *strong match list* + *tailored application materials*.")
    lines.append("   - That combination can meaningfully raise your odds of getting interviews over time.")
    lines.append("")
    lines.append(
        "> The scores here are based purely on text matching between your target roles/skills "
        "and the job descriptions. They’re meant to **guide your focus**, not decide your fate."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Skill Demand Snapshot (across your job list)")
    lines.append("")
    lines.append(
        "The chart below shows how often the job descriptions mention key skill clusters "
        "(automation, data, leadership, cloud, etc.) across the roles we pulled for you."
    )
    lines.append("")
    lines.append("![Skill demand across job list](career_insights_skills_pie.png)")
    lines.append("")
    lines.append(
        "_Tip: if one slice is much smaller than the others (for example, **Automation & "
        "Scripting**), that’s a clear signal about where to invest learning time to unlock more "
        "opportunities over the next few months._"
    )
    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: report_builder.py /path/to/JOB_xxxx", file=sys.stderr)
        sys.exit(1)

    job_dir = Path(sys.argv[1]).resolve()
    if not job_dir.exists():
        print(f"Job directory does not exist: {job_dir}", file=sys.stderr)
        sys.exit(1)

    log(f"Building report for job folder: {job_dir}")

    client_meta = load_client_meta(job_dir)
    manifest = load_jobs_manifest(job_dir)
    jobs_info = parse_job_sources(job_dir, manifest if manifest else None)

    if not jobs_info["jobs"]:
        log("No jobs found in job_sources.txt; aborting report creation.")
        sys.exit(0)

    selection = load_selection_summary(job_dir)

    report_md = build_markdown_report(
        job_dir, client_meta, jobs_info, selection, manifest, DEFAULT_CONFIG
    )
    out_path = job_dir / "final_report.md"
    out_path.write_text(report_md, encoding="utf-8")

    log(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
