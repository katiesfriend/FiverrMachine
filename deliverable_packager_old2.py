#!/usr/bin/env python3
"""
deliverable_packager.py

Builds the final Fiverr ZIP for a single JOB_xxx folder.

New behavior:
- Reads packet_jobXX.json for match_score.
- Selects only the best jobs according to:
    FIVERR_MIN_MATCH_SCORE (float, default 55.0)
    FIVERR_TOP_K          (int,   default 8; 0 = no limit)
    FIVERR_ALWAYS_INCLUDE_TOP (0/1, default 1)

- If no per-job JSON is found, falls back to "include all".
"""

import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


BASE = Path(__file__).resolve().parent
DELIVERABLES = BASE / "DELIVERABLES"

MIN_MATCH_SCORE = float(os.environ.get("FIVERR_MIN_MATCH_SCORE", "55.0"))
TOP_K = int(os.environ.get("FIVERR_TOP_K", "8"))
ALWAYS_INCLUDE_TOP = os.environ.get("FIVERR_ALWAYS_INCLUDE_TOP", "1") == "1"


@dataclass
class JobScore:
    label: str        # "01", "02", ...
    match_score: float


def load_job_scores(job_root: Path) -> List[JobScore]:
    """Load match scores from packet_jobXX.json files, if present."""
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


def fallback_labels_from_pdfs(job_root: Path) -> List[str]:
    """If we have no JSON, infer job labels from resume_jobXX.pdf."""
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
        print("[PACKAGER] No packet_job*.json found; including ALL resumes/cover letters.")
        labels = fallback_labels_from_pdfs(job_root)
        labels = sorted(labels)
        summary = {
            "mode": "fallback_all",
            "reason": "no_packet_json_found",
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
    """Create the final ZIP containing only the selected jobs."""
    job_folder_name = job_root.name
    DELIVERABLES.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = DELIVERABLES / f"fiverr_deliverables_{job_folder_name}_{ts}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
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

        # Global selection summary
        if summary_path.exists():
            zf.write(summary_path, arcname=summary_path.name)

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

    scores = load_job_scores(job_root)
    selected_labels, summary = select_labels(job_root, scores)

    # Persist summary for later debugging / client report
    summary_path = job_root / "selection_summary.json"
    try:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception as exc:
        print(f"[PACKAGER]   [WARN] Failed to write selection_summary.json: {exc}")

    zip_path = build_zip(job_root, selected_labels, summary_path)

    print(f"[PACKAGER] Created zip at {zip_path}")
    print(f"[PACKAGER] Zip ready for upload: {zip_path}")


if __name__ == "__main__":
    main()
