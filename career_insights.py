#!/usr/bin/env python3
"""
CLI shim for the Career Insights module.

Usage:
    career_insights.py JOB_FOLDER_NAME
    career_insights.py /full/path/to/PROCESSING/JOB_xxxxx
"""

import sys
from pathlib import Path


def resolve_job_dir(arg: str) -> Path:
    base = Path(__file__).resolve().parent
    p = Path(arg)

    # Absolute path? Use as-is.
    if p.is_absolute():
        return p

    # Allow plain JOB_xxx
    if str(p).startswith("JOB_"):
        return base / "PROCESSING" / str(p)

    # Allow relative PROCESSING/JOB_xxx style
    return base / str(p)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: career_insights.py JOB_FOLDER_NAME_OR_PATH",
            file=sys.stderr,
        )
        return 1

    job_dir = resolve_job_dir(sys.argv[1]).resolve()
    print(f"[CAREER_INSIGHTS] (stub) would generate insights for: {job_dir}")

    # TODO: wire this to modules.career_insights + engines.ai_model
    # from modules.career_insights.career_insights import build_career_report
    # build_career_report(job_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
