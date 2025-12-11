#!/usr/bin/env python3
"""
career_insights.py

Reads selection_summary.json + job descriptions for a JOB_xxx folder
and generates:

- A JSON summary (printed to stdout)
- A pie chart of local skill demand
- A LaTeX PDF report (career_insights_report.pdf) that includes:
  * Overview + skill breakdown
  * Selected roles ranked by heuristic match score
  * Certification signals
  * Approximate salary range estimate based on local postings
"""

import json
import sys
import re
import subprocess
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tex_escape(text: str) -> str:
    """Escape LaTeX special characters in a minimal way."""
    if not text:
        return ""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def fmt_money(n: int) -> str:
    return "${:,}".format(int(n))


def load_selection_summary(job_root: Path) -> dict:
    summary_path = job_root / "selection_summary.json"
    if not summary_path.exists():
        # Very defensive fallback
        return {
            "mode": "unknown",
            "jobs": [],
            "selected_labels": [],
        }
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_job_text(job_root: Path, label: str) -> str:
    path = job_root / f"job_description_{label}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_resume_text(job_root: Path) -> str:
    path = job_root / "base_resume.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def compute_skill_demand(job_texts: dict, selected_labels: list) -> dict:
    """Very simple keyword buckets over selected postings."""
    skill_groups = {
        "Agile & Scrum": {"agile", "scrum"},
        "Cloud Platforms": {"aws", "azure", "gcp", "google cloud", "cloud"},
        "Automation & Scripting": {"python", "bash", "powershell", "automation", "script"},
        "Data & Analytics": {"data", "analytics", "sql", "tableau", "power bi"},
        "Stakeholder & Leadership": {"stakeholder", "leadership", "executive", "communicate", "presentation"},
        "Vendor & Delivery": {"vendor", "supplier", "delivery", "release", "deployment"},
    }
    counts = {k: 0 for k in skill_groups.keys()}

    for label in selected_labels:
        text = job_texts.get(label, "").lower()
        if not text:
            continue
        for group, keywords in skill_groups.items():
            if any(kw in text for kw in keywords):
                counts[group] += 1

    return counts


def compute_cert_stats(job_texts: dict, selected_labels: list, resume_text: str):
    """Count how often key certifications appear in postings and resume."""
    resume_lower = resume_text.lower()

    cert_defs = [
        ("PMP (Project Management Professional)",
         [r"\bpmp\b", "project management professional"]),
        ("Certified Scrum Master (CSM)",
         [r"certified scrum master", r"\bcsm\b"]),
        ("SAFe / Scaled Agile cert",
         [r"\bsafe\b", "scaled agile"]),
        ("AWS certification",
         ["aws certified", "aws certification"]),
        ("Azure certification",
         ["azure certified", "azure certification"]),
        ("ITIL certification",
         ["itil certification", r"\bitil\b"]),
    ]

    cert_stats = []
    for cert_label, patterns in cert_defs:
        postings_count = 0
        for label in selected_labels:
            text = job_texts.get(label, "").lower()
            if not text:
                continue
            if any(
                (pat.startswith(r"\b") and re.search(pat, text))
                or (not pat.startswith(r"\b") and pat in text)
                for pat in patterns
            ):
                postings_count += 1

        in_resume = any(
            (pat.startswith(r"\b") and re.search(pat, resume_lower))
            or (not pat.startswith(r"\b") and pat in resume_lower)
            for pat in patterns
        )

        cert_stats.append(
            {
                "cert": cert_label,
                "in_selected_postings": postings_count,
                "in_resume": bool(in_resume),
            }
        )

    return cert_stats


def extract_salary_numbers(text: str):
    """
    Pull approximate annual salary figures out of a posting text.

    Supports:
    - $120,000 patterns
    - $120k patterns
    - $60/hour, $60 hr, etc. (converted to annual ~2080 hours)
    """
    nums = []

    # $120,000 style
    for m in re.finditer(r"\$([0-9]{2,3}(?:,[0-9]{3})+)", text):
        val = int(m.group(1).replace(",", ""))
        nums.append(val)

    # 120k or $120k
    for m in re.finditer(r"\$?([0-9]{2,3})k", text, flags=re.IGNORECASE):
        val = int(m.group(1)) * 1000
        nums.append(val)

    # $60/hour, $60 hr, 60 per hour, etc.
    for m in re.finditer(
        r"\$?([0-9]{2,3})\s*(?:/hour|/hr|per hour| hr\b| hours\b)",
        text,
        flags=re.IGNORECASE,
    ):
        hourly = int(m.group(1))
        val = hourly * 2080  # 40h/week * 52
        nums.append(val)

    return nums


def build_selected_job_rows(selection_summary: dict, job_texts: dict, selected_labels: list):
    """Return list of dicts {label, score, snapshot} sorted by score desc."""
    jobs = selection_summary.get("jobs", [])
    score_map = {j["label"]: j.get("match_score") for j in jobs}

    rows = []
    for label in selected_labels:
        text = job_texts.get(label, "")
        # first non-empty line as a quick snapshot
        snapshot = ""
        for line in text.splitlines():
            line = line.strip()
            if line:
                snapshot = line
                break
        if len(snapshot) > 120:
            snapshot = snapshot[:117] + "..."
        rows.append(
            {
                "label": label,
                "score": score_map.get(label),
                "snapshot": snapshot,
            }
        )

    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))
    return rows


def make_pie_chart(job_root: Path, skill_demand: dict) -> str | None:
    nonzero = {k: v for k, v in skill_demand.items() if v > 0}
    if not nonzero:
        return None

    labels = list(nonzero.keys())
    sizes = list(nonzero.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.axis("equal")
    fig.suptitle("Share of selected postings mentioning each skill group")

    path = job_root / "career_insights_skills_pie.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def build_latex_report(
    job_root: Path,
    selection_mode: str,
    total_jobs: int,
    selected_jobs_count: int,
    skill_demand: dict,
    selected_rows: list,
    cert_stats: list,
    salary_estimate: dict | None,
    pie_chart_path: str | None,
) -> str | None:
    tex_path = job_root / "career_insights_report.tex"

    if pie_chart_path:
        chart_block = (
            "\\begin{center}\n"
            "Share of selected postings mentioning each skill group\\\\[4pt]\n"
            f"\\includegraphics[width=0.6\\textwidth]{{{Path(pie_chart_path).name}}}\n"
            "\\end{center}\n"
        )
    else:
        chart_block = (
            "No skill-demand chart is available for this run; the selected postings "
            "did not trigger any of the current keyword groups.\n"
        )

    # Selected roles table
    if selected_rows:
        table_lines = [
            "\\begin{tabular}{llp{9cm}}\n",
            "\\hline\n",
            "Job & Match score & Snapshot \\\\\n",
            "\\hline\n",
        ]
        for row in selected_rows:
            score = "--" if row["score"] is None else f"{row['score']:.1f}"
            table_lines.append(
                f"{tex_escape(row['label'])} & {tex_escape(score)} & "
                f"{tex_escape(row['snapshot'])} \\\\\n"
            )
        table_lines.append("\\hline\n\\end{tabular}\n")
        selected_table = "".join(table_lines)
    else:
        selected_table = "No selected postings were available for this run.\n"

    # Salary section text
    if salary_estimate is not None:
        salary_section = (
            "\\section*{Market Compensation Signal (Approximate)}\n\n"
            f"Based on the selected postings that include pay information "
            f"({salary_estimate['samples']} of {selected_jobs_count}), the approximate "
            f"annual salary range for similar roles in your area is "
            f"{fmt_money(salary_estimate['min'])} to {fmt_money(salary_estimate['max'])}, "
            f"with a central tendency around {fmt_money(salary_estimate['median'])}. "
            "Treat this as a directional signal only; individual offers depend on company, "
            "benefits, and negotiation.\n"
        )
    else:
        salary_section = (
            "\\section*{Market Compensation Signal (Approximate)}\n\n"
            "The selected postings in this sample do not include explicit salary ranges. "
            "Use public compensation tools (salary surveys, level guides, and recruiter "
            "conversations) alongside this report to estimate your personal target range.\n"
        )

    # Certification bullets
    cert_lines = ["\\begin{itemize}\n"]
    for c in cert_stats:
        in_sel = c["in_selected_postings"]
        if in_sel == 0:
            freq = "does not appear in these selected postings"
        elif in_sel == 1:
            freq = "appears in 1 of the selected postings"
        else:
            freq = f"appears in {in_sel} of the selected postings"

        if c["in_resume"]:
            resume_note = "and IS already visible in your resume."
        else:
            resume_note = "and is NOT currently highlighted in your resume."

        line = f"{c['cert']} — {freq} {resume_note}"
        cert_lines.append("\\item " + tex_escape(line) + "\n")
    cert_lines.append("\\end{itemize}\n")
    cert_block = "".join(cert_lines)

    doc = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{helvet}}
\\renewcommand\\familydefault{{\\sfdefault}}

\\begin{{document}}

\\begin{{center}}
\\Large Career Insights Report\\\\[4pt]
\\normalsize Client
\\end{{center}}

\\section*{{Overview}}

This report summarizes the roles FiverrMachine selected for you, the skill patterns in your local
market, and practical ways to strengthen your long-term positioning.

\\begin{{itemize}}
\\item Total postings scraped in this run: {total_jobs}
\\item Postings included in the final packet: {selected_jobs_count}
\\item Selection mode: {tex_escape(selection_mode)}
\\end{{itemize}}

\\section*{{Local Skill Demand (Selected Jobs)}}

The chart and counts below show how often each skill group appears across the selected postings.

{chart_block}

Agile \\& Scrum: {skill_demand['Agile & Scrum']} of {selected_jobs_count} selected postings mention this.\\\\
Cloud Platforms: {skill_demand['Cloud Platforms']} of {selected_jobs_count} selected postings mention this.\\\\
Automation \\& Scripting: {skill_demand['Automation & Scripting']} of {selected_jobs_count} selected postings mention this.\\\\
Data \\& Analytics: {skill_demand['Data & Analytics']} of {selected_jobs_count} selected postings mention this.\\\\
Stakeholder \\& Leadership: {skill_demand['Stakeholder & Leadership']} of {selected_jobs_count} selected postings mention this.\\\\
Vendor \\& Delivery: {skill_demand['Vendor & Delivery']} of {selected_jobs_count} selected postings mention this.\\\\

\\section*{{Selected Roles and Match Scores}}

The table below lists the roles included in your final packet, ranked by their heuristic match score
computed by FiverrMachine.

{selected_table}

{salary_section}

\\section*{{Certification Signals}}

Recruiters often scan for specific certifications as a shorthand for capability. In this local sample,
the following certifications show up repeatedly:

{cert_block}

Based on the selected postings, the certifications above represent high-leverage opportunities to
pursue or foreground in your profile.

\\section*{{How to Use This Report}}

\\begin{{enumerate}}
\\item Use the skill-demand breakdown to prioritize what you emphasize at the top of your resume
      and LinkedIn profile.
\\item Where you see strong demand but weaker representation in your resume, add concrete bullet
      points and examples.
\\item For recommended certifications, decide which ones align with your desired direction and make
      a simple plan (timeline, budget, and study resources) to obtain or foreground them.
\\item As you gain new skills or credentials, re-run FiverrMachine to see how your positioning in
      the market improves.
\\end{{enumerate}}

\\end{{document}}
"""

    tex_path.write_text(doc, encoding="utf-8")

    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=job_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        pdf_path = job_root / "career_insights_report.pdf"
        if pdf_path.exists():
            return str(pdf_path)
    except Exception:
        # If LaTeX fails, we still return JSON without a PDF path.
        pass

    return None


def main():
    if len(sys.argv) < 2:
        print(
            json.dumps(
                {"ok": False, "error": "Usage: career_insights.py JOB_ROOT"},
                indent=2,
            )
        )
        sys.exit(1)

    job_root = Path(sys.argv[1]).resolve()
    if not job_root.exists():
        print(
            json.dumps(
                {"ok": False, "error": f"Job root does not exist: {job_root}"},
                indent=2,
            )
        )
        sys.exit(1)

    summary = load_selection_summary(job_root)
    selection_mode = summary.get("mode", "unknown")
    jobs_list = summary.get("jobs", [])
    total_jobs = len(jobs_list) or 20  # loose fallback

    selected_labels = summary.get("selected_labels") or [
        j["label"] for j in jobs_list
    ]
    selected_labels = list(selected_labels)
    selected_jobs_count = len(selected_labels)

    # Load texts
    job_texts = {label: load_job_text(job_root, label) for label in selected_labels}
    resume_text = load_resume_text(job_root)

    # Skill demand + cert stats
    skill_demand = compute_skill_demand(job_texts, selected_labels)
    cert_stats = compute_cert_stats(job_texts, selected_labels, resume_text)

    # Selected jobs table + salary samples
    selected_rows = build_selected_job_rows(summary, job_texts, selected_labels)
    all_salary_samples = []
    salary_postings_with_data = 0
    for label in selected_labels:
        text = job_texts.get(label, "")
        samples = extract_salary_numbers(text)
        if samples:
            salary_postings_with_data += 1
            all_salary_samples.extend(samples)

    salary_estimate = None
    if all_salary_samples:
        all_salary_samples.sort()
        salary_estimate = {
            "min": int(all_salary_samples[0]),
            "max": int(all_salary_samples[-1]),
            "median": int(median(all_salary_samples)),
            "samples": int(salary_postings_with_data),
        }

    # Pie chart + PDF report
    pie_chart_path = make_pie_chart(job_root, skill_demand)
    report_pdf_path = build_latex_report(
        job_root=job_root,
        selection_mode=selection_mode,
        total_jobs=total_jobs,
        selected_jobs_count=selected_jobs_count,
        skill_demand=skill_demand,
        selected_rows=selected_rows,
        cert_stats=cert_stats,
        salary_estimate=salary_estimate,
        pie_chart_path=pie_chart_path,
    )

    # Final JSON summary (used by pipeline logs)
    result = {
        "ok": True,
        "job_root": str(job_root),
        "selection_mode": selection_mode,
        "total_jobs": total_jobs,
        "selected_jobs_count": selected_jobs_count,
        "skill_demand": skill_demand,
        "cert_stats": cert_stats,
        "pie_chart": pie_chart_path,
        "report_pdf": report_pdf_path,
        "salary_estimate": salary_estimate,
        "note": "Career insights report with skill demand, selected roles, certifications, and salary signal.",
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
