#!/usr/bin/env python3
"""
cover_letter_latex.py

Rebuilds cover letter PDFs for a given job folder using LaTeX,
based on the existing cover_letter_jobNN.txt files.
"""

import sys
from pathlib import Path

from latex_cover_letter_builder import build_cover_letter_pdf


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: cover_letter_latex.py JOB_DIR", file=sys.stderr)
        sys.exit(1)

    job_root = Path(sys.argv[1]).resolve()
    print(f"[LATEX-CL] Rebuilding cover letter PDFs in: {job_root}", flush=True)

    # We follow the same 01..20 naming you already use.
    for idx in range(1, 21):
        txt_path = job_root / f"cover_letter_job{idx:02d}.txt"
        if not txt_path.is_file():
            continue

        pdf_path = job_root / f"cover_letter_job{idx:02d}.pdf"
        try:
            text = txt_path.read_text(encoding="utf-8")
            build_cover_letter_pdf(text, pdf_path)
            print(f"[LATEX-CL]   -> {pdf_path.name}", flush=True)
        except Exception as exc:
            print(
                f"[LATEX-CL]   [WARN] Failed to build cover letter PDF for job "
                f"{idx:02d}: {exc}",
                flush=True,
            )

    print("[LATEX-CL] Done.", flush=True)


if __name__ == "__main__":
    main()
