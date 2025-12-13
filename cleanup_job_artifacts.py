#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional, Set, Dict, Any


LABEL_RE = re.compile(r"(\d+)")


def _norm_label(raw: object) -> Optional[str]:
    if raw is None:
        return None
    m = LABEL_RE.search(str(raw))
    if not m:
        return None
    try:
        return f"{int(m.group(1)):02d}"
    except ValueError:
        return None


def get_valid_labels(job_root: Path) -> Set[str]:
    """
    Prefer jobs_manifest.json when present (authoritative for this run).
    Fall back to job_description_XX.txt filenames.
    """
    labels: Set[str] = set()

    manifest = job_root / "jobs_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for j in data.get("jobs", []):
                lab = _norm_label(j.get("label") or j.get("job_id") or j.get("id"))
                if lab:
                    labels.add(lab)
        except Exception:
            pass

    if not labels:
        for p in job_root.glob("job_description_*.txt"):
            lab = _norm_label(p.name)
            if lab:
                labels.add(lab)

    return labels


def cleanup(job_root: Path) -> Dict[str, Any]:
    valid = get_valid_labels(job_root)

    # If we can't determine valid labels, do nothing (safer than deleting).
    if not valid:
        return {
            "ok": True,
            "job_root": str(job_root),
            "valid_labels": [],
            "removed": [],
            "note": "No valid labels detected; cleanup skipped.",
        }

    patterns = [
        "job_description_*.txt",
        "resume_job*.txt",
        "resume_job*.pdf",
        "resume_job*.tex",
        "cover_letter_job*.txt",
        "cover_letter_job*.pdf",
        "cover_letter_job*.tex",
        "packet_job*.json",
        "score_job*.json",
    ]

    removed = []
    for pat in patterns:
        for p in job_root.glob(pat):
            lab = _norm_label(p.name)
            # Only touch files that clearly have a label.
            if lab and lab not in valid:
                try:
                    p.unlink()
                    removed.append(p.name)
                except Exception:
                    pass

    return {
        "ok": True,
        "job_root": str(job_root),
        "valid_labels": sorted(valid),
        "removed": sorted(removed),
        "removed_count": len(removed),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: cleanup_job_artifacts.py JOB_ROOT", file=sys.stderr)
        sys.exit(1)

    job_root = Path(sys.argv[1]).resolve()
    if not job_root.exists():
        print(json.dumps({"ok": False, "error": f"job root does not exist: {job_root}"}))
        sys.exit(2)

    result = cleanup(job_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
