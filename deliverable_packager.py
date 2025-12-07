#!/usr/bin/env python3
"""
deliverable_packager.py

Builds the final Fiverr ZIP for a single JOB_xxx folder.

Selection logic:

1. Try to load match scores from packet_jobXX.json (if such files exist).
2. If no JSON is found, compute heuristic scores from:
   - job_description_XX.txt
   - resume_jobXX.txt (or base_resume.txt as fallback)
   - cover_letter_jobXX.txt (if present)

3. Use tunable thresholds:
   - FIVERR_MIN_MATCH_SCORE (float, default 0.0)
   - FIVERR_TOP_K          (int,   default 8; 0 = no limit)
   - FIVERR_ALWAYS_INCLUDE_TOP (0/1, default 1)

4. Include in the ZIP:
   - resume_jobXX.pdf
   - cover_letter_jobXX.pdf
   - job_description_XX.txt
   - packet_jobXX.json (if present)
   - selection_summary.json
"""

import json
import os
import re
import sys
import zipfile
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

BASE = Path(__file__).resolve().parent
DELIVERABLES = BASE / "DELIVERABLES"

# Defaults are soft; you can override with environment variables.
MIN_MATCH_SCORE = float(os.environ.get("FIVERR_MIN_MATCH_SCORE", "0.0"))
TOP_K = int(os.environ.get("FIVERR_TOP_K", "8"))
ALWAYS_INCLUDE_TOP = os.environ.get("FIVERR_ALWAYS_INCLUDE_TOP", "1") == "1"

STOPWORDS = {
    "the", "and", "or", "a", "an", "to", "of", "in", "for", "on", "at", "by",
    "with", "from", "as", "is", "are", "be", "this", "that", "it", "its",
    "their", "they", "you", "your", "i", "we", "our"
}


@dataclass
class JobScore:
    label: str        # "01", "02", ...
    match_score: float


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z']+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def heuristic_score(job_text: str, profile_text: str) -> float:
    jd_terms = set(tokenize(job_text))
    profile_terms = set(tokenize(profile_text))

    if not jd_terms:
        return 0.0

    overlap = len(jd_terms & profile_terms)
    if overlap == 0:
        return 0.0

    raw_ratio = overlap / len(jd_terms)
    # Scale to 0–100, clamp, 1 decimal
    score = max(0.0, min(100.0, raw_ratio * 100.0))
    return round(score, 1)


def load_job_scores_from_json(job_root: Path) -> List[JobScore]:
    """Load scores from packet_jobXX.json, if present."""
    scores: List[JobScore] = []

    for path in sorted(job_root.glob("packet_job*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[PACKAGER]   [WARN] Could not parse {path.name}: {exc}")
            continue

        label = data.get("job_label")
        if not label:
            m = re.search(r"packet_job(\d+)\.json$", path.name)
            if m:
                label = m.group(1)

        if not label:
            print(f"[PACKAGER]   [WARN] Could not determine job label from {path.name}, skipping.")
            continue

        raw_score = data.get("match_score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            print(f"[PACKAGER]   [WARN] Non-numeric match_score in {path.name}: {raw_score!r}. Using 0.0.")
            score = 0.0

        scores.append(JobScore(label=label, match_score=score))

    return scores


def compute_heuristic_job_scores(job_root: Path) -> List[JobScore]:
    """
    Build scores purely from text artifacts if we don't have packet_jobXX.json.

    Uses:
      - job_description_XX.txt
      - resume_jobXX.txt (or base_resume.txt fallback)
      - cover_letter_jobXX.txt (if present)
    """
    jd_files = sorted(job_root.glob("job_description_*.txt"))
    if not jd_files:
        print("[PACKAGER]   [WARN] No job_description_*.txt found; cannot compute heuristic scores.")
        return []

    base_resume_text = ""
    base_resume_path = job_root / "base_resume.txt"
    if base_resume_path.exists():
        try:
            base_resume_text = base_resume_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"[PACKAGER]   [WARN] Could not read base_resume.txt: {exc}")

    scores: List[JobScore] = []

    for jd_path in jd_files:
        m = re.search(r"job_description_(\d+)\.txt$", jd_path.name)
        if not m:
            continue
        label = m.group(1)

        try:
            jd_text = jd_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"[PACKAGER]   [WARN] Could not read {jd_path.name}: {exc}")
            jd_text = ""

        # Try job-specific resume text, fall back to base_resume.
        resume_txt_path = job_root / f"resume_job{label}.txt"
        resume_text = ""
        if resume_txt_path.exists():
            try:
                resume_text = resume_txt_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                print(f"[PACKAGER]   [WARN] Could not read {resume_txt_path.name}: {exc}")
        else:
            resume_text = base_resume_text

        cover_txt_path = job_root / f"cover_letter_job{label}.txt"
        cover_text = ""
        if cover_txt_path.exists():
            try:
                cover_text = cover_txt_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                print(f"[PACKAGER]   [WARN] Could not read {cover_txt_path.name}: {exc}")

        profile_text = resume_text + "\n" + cover_text
        score = heuristic_score(jd_text, profile_text)
        scores.append(JobScore(label=label, match_score=score))

    # Normalize so the best job is at most 100 (already mostly true, but just in case).
    if scores:
        max_score = max(s.match_score for s in scores) or 1.0
        normalized: List[JobScore] = []
        for s in scores:
            norm = round(100.0 * (s.match_score / max_score), 1)
            normalized.append(JobScore(label=s.label, match_score=norm))
        scores = normalized

    print("[PACKAGER]   Heuristic scores (0–100, max normalized to 100):")
    for s in sorted(scores, key=lambda x: x.match_score, reverse=True):
        print(f"[PACKAGER]     job{s.label}: {s.match_score:.1f}")

    return scores


def fallback_labels_from_pdfs(job_root: Path) -> List[str]:
    """If we truly have no way to score, include all resume_jobXX.pdf."""
    labels = []
    for path in sorted(job_root.glob("resume_job*.pdf")):
        m = re.search(r"resume_job(\d+)\.pdf$", path.name)
        if m:
            labels.append(m.group(1))
    return labels


def select_labels(job_root: Path, scores: List[JobScore]) -> Tuple[List[str], dict]:
    """
    Decide which job labels to include.

    Returns:
        selected_labels, summary_dict
    """
    if not scores:
        print("[PACKAGER] No packet_job*.json found; computing heuristic scores from text.")
        scores = compute_heuristic_job_scores(job_root)

    if not scores:
        print("[PACKAGER]   [FALLBACK] No scores available at all; including ALL PDFs.")
        labels = fallback_labels_from_pdfs(job_root)
        labels = sorted(labels)
        summary = {
            "mode": "fallback_all",
            "reason": "no_scores_available",
            "selected_labels": labels,
            "jobs": [],
        }
        return labels, summary

    # Sort by score descending
    scores_sorted = sorted(scores, key=lambda s: s.match_score, reverse=True)

    print(
        f"[PACKAGER] Selection params: "
        f"MIN_MATCH_SCORE={MIN_MATCH_SCORE}, TOP_K={TOP_K}, "
        f"ALWAYS_INCLUDE_TOP={ALWAYS_INCLUDE_TOP}"
    )

    # Filter by minimum score
    candidates = [s for s in scores_sorted if s.match_score >= MIN_MATCH_SCORE]

    # Ensure at least one packet if requested
    if not candidates and ALWAYS_INCLUDE_TOP and scores_sorted:
        print("[PACKAGER]   No jobs met minimum score; including top-scoring job anyway.")
        candidates = [scores_sorted[0]]

    # Apply TOP_K limit
    if TOP_K > 0:
        candidates = candidates[:TOP_K]

    selected_labels = [s.label for s in candidates]

    # Logging for transparency
    selected_set = set(selected_labels)
    for s in scores_sorted:
        flag = "SELECTED" if s.label in selected_set else "skip"
        print(f"[PACKAGER]   job{s.label}: score={s.match_score:.1f} -> {flag}")

    summary = {
        "mode": "score_filter",
        "min_match_score": MIN_MATCH_SCORE,
        "top_k": TOP_K,
        "always_include_top": ALWAYS_INCLUDE_TOP,
        "jobs": [asdict(s) for s in scores_sorted],
        "selected_labels": selected_labels,
    }
    return selected_labels, summary

def build_zip(job_root: Path, selected_labels: List[str], summary_path: Path) -> Path:
    """Create the final ZIP containing only the selected jobs plus global reports."""
    job_folder_name = job_root.name
    DELIVERABLES.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = DELIVERABLES / f"fiverr_deliverables_{job_folder_name}_{ts}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Per-job artifacts
        for label in sorted(selected_labels):
            # PDFs
            for filename in (
                f"resume_job{label}.pdf",
                f"cover_letter_job{label}.pdf",
            ):
                path = job_root / filename
                if path.exists():
                    zf.write(path, arcname=path.name)
                else:
                    print(f"[PACKAGER]   [WARN] Missing file for selected job {label}: {filename}")

            # Optional extras per job
            for filename in (
                f"job_description_{label}.txt",
                f"packet_job{label}.json",
            ):
                path = job_root / filename
                if path.exists():
                    zf.write(path, arcname=path.name)

        # Global artifacts for the whole job
        if summary_path.exists():
            zf.write(summary_path, arcname=summary_path.name)

        final_report = job_root / "final_report.md"
        if final_report.exists():
            zf.write(final_report, arcname=final_report.name)

    return zip_path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: deliverable_packager.py JOB_ROOT", file=sys.stderr)
        sys.exit(1)

    job_root = Path(sys.argv[1]).resolve()
    if not job_root.exists():
        print(f"[PACKAGER] ERROR: job root does not exist: {job_root}", file=sys.stderr)
        sys.exit(1)

    print(f"[PACKAGER] Building deliverables ZIP for {job_root}", flush=True)

    # 1) Try JSON scores (future Qwen integration)
    scores = load_job_scores_from_json(job_root)

    # 2) Selection logic (may compute heuristic scores instead)
    selected_labels, summary = select_labels(job_root, scores)

    # 3) Persist summary for later debugging / client report
    summary_path = job_root / "selection_summary.json"
    try:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception as exc:
        print(f"[PACKAGER]   [WARN] Failed to write selection_summary.json: {exc}")

    # 4) Build ZIP
    zip_path = build_zip(job_root, selected_labels, summary_path)

    print(f"[PACKAGER] Created zip at {zip_path}")
    # Optional: generate and append Career Insights report
    try:
        print("[PACKAGER] Generating career insights report...")
        insights_proc = subprocess.run(
            [sys.executable, str(BASE / "career_insights.py"), str(job_root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        try:
            insights_json = json.loads(insights_proc.stdout)
        except json.JSONDecodeError:
            print("[PACKAGER] Could not parse career_insights output as JSON.")
            insights_json = {}

        report_pdf = insights_json.get("report_pdf")
        pie_chart = insights_json.get("pie_chart")

        # Append report + chart into the ZIP if they exist
        if report_pdf and os.path.isfile(report_pdf):
            with zipfile.ZipFile(zip_path, "a") as zf:
                zf.write(
                    report_pdf,
                    arcname="career_insights/career_insights_report.pdf",
                )
                if pie_chart and os.path.isfile(pie_chart):
                    zf.write(
                        pie_chart,
                        arcname="career_insights/career_insights_skills_pie.png",
                    )
            print("[PACKAGER] Career insights report added to ZIP.")
        else:
            print("[PACKAGER] Career insights report not found; skipping ZIP attach.")
    except Exception as e:
        print(f"[PACKAGER] Career insights generation failed: {e}")

    print(f"[PACKAGER] Zip ready for upload: {zip_path}")


if __name__ == "__main__":
    main()
