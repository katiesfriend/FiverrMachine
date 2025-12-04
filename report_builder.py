#!/usr/bin/env python3
"""
report_builder.py

Builds a human-readable summary report for a FiverrMachine job folder.

Inputs (inside JOB dir):
  - client_request.json   (optional but preferred)
  - job_sources.txt       (from job_scraper.py)

Output:
  - final_report.md       (Markdown summary for the client / VSOP)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REPORT_FILENAME = "Job_Search_Report.md"


def log(msg: str) -> None:
    print(f"[REPORT] {msg}", flush=True)


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


# -----------------------------
# Parse job_sources.txt
# -----------------------------

def parse_job_sources(job_dir: Path) -> Dict[str, Any]:
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

def bucket_match_score(score: float, high_fit_threshold: Optional[float]) -> str:
    """
    Bucket the job_scraper score into human labels.

    Note: this is not the same as the LLM-based ATS,
    but it's still a useful "job match" signal.
    """
    if high_fit_threshold is None:
        # Fallback heuristic
        if score >= 3.0:
            return "High"
        elif score >= 1.5:
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
# Build the markdown report
# -----------------------------

def build_markdown_report(
    job_dir: Path,
    client: Dict[str, Any],
    jobs_info: Dict[str, Any],
) -> str:
    jobs = sorted(jobs_info["jobs"], key=lambda j: j.get("score", 0.0), reverse=True)
    high_fit_threshold = jobs_info.get("high_fit_threshold")
    high_fit_count = jobs_info.get("high_fit_count")

    client_name = client.get("client_name", "Client")
    target_roles = client.get("target_roles") or []
    preferred_titles = client.get("preferred_titles") or []
    location_zip = client.get("location_zip", "")
    remote_pref = client.get("remote_preference", "")
    job_volume_target = client.get("job_volume_target", len(jobs))

    title_str = ""
    if target_roles:
        title_str = ", ".join(target_roles)
    elif preferred_titles:
        title_str = ", ".join(preferred_titles)

    lines: List[str] = []

    # -----------------------------
    # Header
    # -----------------------------
    lines.append(f"# Job Search Report for {client_name}")
    lines.append("")
    lines.append(
        "> This report doesn’t promise job offers — it gives you a realistic, data-backed "
        "shortlist of roles where your skills are more likely to land interviews."
    )
    lines.append("")

    if title_str:
        lines.append(f"**Target roles:** {title_str}")
    lines.append(f"**Location focus:** zip `{location_zip}` | remote preference: `{remote_pref}`")
    lines.append(f"**Requested job volume:** {job_volume_target}")
    lines.append(f"**Jobs matched by scraper:** {len(jobs)}")
    if high_fit_threshold is not None and high_fit_count is not None:
        lines.append(f"**High-fit threshold:** score ≥ {high_fit_threshold}  → {high_fit_count} jobs")
    lines.append("")
    lines.append("---")
    lines.append("")

    # -----------------------------
    # Summary at a glance
    # -----------------------------
    site_counts: Dict[str, int] = {}
    bucket_counts: Dict[str, int] = {"High": 0, "Medium": 0, "Low": 0, "Very Low": 0}

    for j in jobs:
        site_counts[j["site"]] = site_counts.get(j["site"], 0) + 1
        b = bucket_match_score(j["score"], high_fit_threshold)
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    lines.append("## Summary at a Glance")
    lines.append("")
    lines.append("These numbers show you **where your best bets are** so you can stop guessing and start focusing.")
    lines.append("")
    lines.append("**By match bucket (scraper score):**")
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

    # -----------------------------
    # What this report IS / IS NOT
    # -----------------------------
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

    # -----------------------------
    # Detailed table
    # -----------------------------
    lines.append("## Job List (Best Matches First)")
    lines.append("")
    lines.append(
        "Jobs are ordered by how closely the job description text matches your target roles and key skills. "
        "Use this to decide where to apply first."
    )
    lines.append("")
    lines.append("| # | Title | Company | Location | Site | Match bucket | Raw score |")
    lines.append("|---|-------|---------|----------|------|--------------|-----------|")

    for j in jobs:
        idx = j["index"]
        title = j["title"] or "(no title)"
        company = j["company"] or "(no company)"
        location = j["location"] or "(unspecified)"
        site = j["site"] or "?"
        score = j["score"]
        bucket = bucket_match_score(score, high_fit_threshold)

        # Escape pipes in markdown
        title_md = title.replace("|", "\\|")
        company_md = company.replace("|", "\\|")
        location_md = location.replace("|", "\\|")

        lines.append(
            f"| {idx:02d} | {title_md} | {company_md} | {location_md} | {site} | {bucket} | {score:.2f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # -----------------------------
    # Coaching / narrative
    # -----------------------------
    lines.append("## How to Use This Report")
    lines.append("")
    lines.append("Think of this as your **attack plan**, not a wall of links.")
    lines.append("")
    lines.append("1. **Start with the High bucket.**")
    lines.append("   - These roles are the closest text match to your skills and target titles.")
    lines.append("   - If you only have limited time each week, start here.")
    lines.append("")
    lines.append("2. **Then move into Medium matches (if present).**")
    lines.append("   - These are adjacent or slightly stretched roles that can still be great opportunities.")
    lines.append("   - Good for expanding your options without going totally off-track.")
    lines.append("")
    lines.append("3. **Use Low / Very Low as optional practice or stretch targets.**")
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

    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------

def build_report(job_dir: Path) -> Path:
    """
    Build the Job Search Report for the given job directory and
    return the Path to the report file.
    """
    job_dir = job_dir.resolve()
    if not job_dir.exists() or not job_dir.is_dir():
        raise FileNotFoundError(f"Job directory does not exist: {job_dir}")

    log(f"Building report for job folder: {job_dir}")

    client_meta = load_client_meta(job_dir)
    jobs_info = parse_job_sources(job_dir)

    if not jobs_info["jobs"]:
        log("No jobs found in job_sources.txt; creating an empty report.")

    report_md = build_markdown_report(job_dir, client_meta, jobs_info)
    out_path = job_dir / REPORT_FILENAME
    out_path.write_text(report_md, encoding="utf-8")

    log(f"Report written to {out_path}")
    return out_path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: report_builder.py /path/to/JOB_xxxx", file=sys.stderr)
        sys.exit(1)

    job_dir = Path(sys.argv[1]).resolve()

    try:
        report_path = build_report(job_dir)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(report_path)


if __name__ == "__main__":
    main()
