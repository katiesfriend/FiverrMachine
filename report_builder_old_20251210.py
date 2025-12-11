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

    Note: this is not the same as the LLM-based match score,
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
) -> str:
    jobs = sorted(jobs_info["jobs"], key=lambda j: j.get("score", 0.0), reverse=True)
    high_fit_threshold = jobs_info.get("high_fit_threshold")
    high_fit_count = jobs_info.get("high_fit_count")

    # LLM selection summary (focus list)
    selection = selection or {}
    selected_labels: List[str] = selection.get("selected_labels") or []
    selected_labels_set = set(selected_labels)
    jobs_by_index = {j["index"]: j for j in jobs}

    llm_scores: Dict[int, float] = {}
    selection_jobs = selection.get("jobs") or []
    for sj in selection_jobs:
        label_str = sj.get("label")
        ms = sj.get("match_score")
        try:
            idx = int(label_str)
        except (TypeError, ValueError):
            continue
        if isinstance(ms, (int, float)):
            llm_scores[idx] = float(ms)

    # Build ordered focus rows (only the actually selected labels)
    focus_rows: List[Dict[str, Any]] = []
    for sj in selection_jobs:
        label_str = sj.get("label")
        if label_str not in selected_labels_set:
            continue

        try:
            idx = int(label_str)
        except (TypeError, ValueError):
            continue

        job_info = jobs_by_index.get(idx)
        if not job_info:
            continue

        focus_rows.append(
            {
                "label": label_str,
                "match_score": llm_scores.get(idx),
                "job": job_info,
            }
        )

    salary_band = infer_salary_band(job_dir, selected_labels) if selected_labels else {}

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
    if location_zip and remote_pref:
        lines.append(f"**Location focus:** zip `{location_zip}` | remote preference: `{remote_pref}`")
    elif location_zip:
        lines.append(f"**Location focus:** zip `{location_zip}`")
    elif remote_pref:
        lines.append(f"**Location focus:** remote preference: `{remote_pref}`")
    else:
        lines.append("**Location focus:** (not specified; search used remote-friendly defaults)")
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
    # Focus shortlist (LLM-selected)
    # -----------------------------
    if focus_rows:
        lines.append("## Focus Roles (AI-selected shortlist)")
        lines.append("")
        lines.append(
            "These are the roles FiverrMachine pulled into your final packet, ranked by the AI match score. "
            "If you only have time to apply to a handful of roles, **start here**."
        )
        lines.append("")
        lines.append("| Job | Match score | Title | Company | Location | Where to apply |")
        lines.append("|-----|-------------|-------|---------|----------|----------------|")

        match_values: List[float] = []

        for row in focus_rows:
            label_str = row["label"]
            ms = row.get("match_score")
            job = row["job"]

            title = job["title"] or "(no title)"
            company = job["company"] or "(no company)"
            location = job["location"] or "(unspecified)"
            url = job.get("url") or ""

            # Escape pipes for Markdown
            title_md = title.replace("|", "\\|")
            company_md = company.replace("|", "\\|")
            location_md = location.replace("|", "\\|")

            if isinstance(ms, (int, float)):
                match_values.append(float(ms))
                ms_str = f"{ms:.1f}"
            else:
                ms_str = "—"

            if url:
                apply_md = f"[Open posting]({url})"
            else:
                apply_md = "Search title + company"

            lines.append(
                f"| {int(label_str):02d} | {ms_str} | {title_md} | {company_md} | {location_md} | {apply_md} |"
            )

        lines.append("")

        # -----------------------------
        # Market compensation signal
        # -----------------------------
        if salary_band:
            min_s = salary_band["min"]
            max_s = salary_band["max"]
            median_s = salary_band["median"]
            sample_count = salary_band.get("sample_count", 0)
            posting_count = salary_band.get("posting_count", 0)

            min_s_str = f"${min_s:,.0f}"
            max_s_str = f"${max_s:,.0f}"
            median_s_str = f"${median_s:,.0f}"

            lines.append("### Market Compensation Signal (Approximate)")
            lines.append("")
            lines.append(
                f"Based on the focus roles that actually list explicit annual salary ranges "
                f"(**{posting_count} of {len(focus_rows)}**), similar roles around your target area "
                f"cluster roughly between **{min_s_str}** and **{max_s_str}** per year, with a central "
                f"band near **{median_s_str}**."
            )
            lines.append("")
            lines.append(
                "For a profile like yours targeting these roles, that band is a realistic expectation "
                "for market-aligned compensation. Offers far below this range should usually be treated "
                "as starting points for negotiation, not your default target."
            )

        if match_values:
            avg_match = sum(match_values) / len(match_values)
            lines.append("")
            lines.append(
                f"Your average AI match score across these focus roles is about **{avg_match:.1f}** "
                "on a 0–100 scale. In plain language: these postings describe the kind of work your "
                "current resume is already pointing at, so this band is a reasonable starting point "
                "for your current market value in this lane."
            )

        lines.append("")
        lines.append(
            "> This is **not** a formal salary survey or negotiation advice. Treat it as a directional "
            "signal so you don't accidentally underprice yourself when an offer comes in far below this band."
        )
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
    # Detailed table (all jobs)
    # -----------------------------
    lines.append("## Job List (Best Matches First)")
    lines.append("")
    lines.append(
        "Jobs are ordered by how closely the job description text matches your target roles and key skills. "
        "Use this to decide where to apply next once you've exhausted the focus shortlist above."
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

        # Escape pipes for Markdown
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
    jobs_info = parse_job_sources(job_dir)

    if not jobs_info["jobs"]:
        log("No jobs found in job_sources.txt; aborting report creation.")
        sys.exit(0)

    selection = load_selection_summary(job_dir)

    report_md = build_markdown_report(job_dir, client_meta, jobs_info, selection)
    out_path = job_dir / "final_report.md"
    out_path.write_text(report_md, encoding="utf-8")

    log(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
