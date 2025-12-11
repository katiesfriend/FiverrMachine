#!/usr/bin/env python3
"""
latex_report_builder.py

Convert the markdown final_report.md produced by report_builder.py
into a polished PDF letter using the KOMA-Script scrlttr2 class.

This script is intended to be called from run_fiverr_pipeline.py after
the report_builder step has created final_report.md.
"""

from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path
from typing import Optional, Dict, Any

from report_builder import load_client_meta

REPORT_MD_NAME = "final_report.md"
REPORT_TEX_NAME = "final_report.tex"
REPORT_PDF_NAME = "final_report.pdf"

def escape_latex(text: str) -> str:
    """
    Escape LaTeX special characters in normal text.
    (Local copy so this module does not depend on latex_resume_builder.)
    """
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
    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out

def log(msg: str) -> None:
    print(f"[REPORT-LATEX] {msg}")


def markdown_to_latex_body(md_text: str) -> str:
    """
    Very small markdown-to-LaTeX converter that understands:

    - # / ## / ### headings
    - unordered lists with "-" or "*"
    - ordered lists like "1. text"
    - paragraphs separated by blank lines

    Everything else is emitted as escaped plain text.
    """
    lines = md_text.splitlines()

    out: list[str] = []
    in_itemize = False
    in_enumerate = False

    def close_lists() -> None:
        nonlocal in_itemize, in_enumerate
        if in_itemize:
            out.append(r"\end{itemize}")
            in_itemize = False
        if in_enumerate:
            out.append(r"\end{enumerate}")
            in_enumerate = False

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Blank line -> paragraph break
        if stripped == "":
            close_lists()
            out.append("")  # blank line in LaTeX = paragraph break
            continue

        # Headings (render as bold text instead of LaTeX \section in scrlttr2)
        if stripped.startswith("### "):
            close_lists()
            title = escape_latex(stripped[4:].strip())
            out.append(r"\textbf{" + title + r"}" + "\n" + r"\par\smallskip")
            continue
        if stripped.startswith("## "):
            close_lists()
            title = escape_latex(stripped[3:].strip())
            out.append(r"\bigskip" + "\n" + r"\textbf{" + title + r"}" + "\n" + r"\par\medskip")
            continue
        if stripped.startswith("# "):
            close_lists()
            title = escape_latex(stripped[2:].strip())
            out.append(r"\bigskip" + "\n" + r"\textbf{" + title + r"}" + "\n" + r"\par\medskip")
            continue

        # Unordered list
        if stripped.startswith("- ") or stripped.startswith("* "):
            if in_enumerate:
                out.append(r"\end{enumerate}")
                in_enumerate = False
            if not in_itemize:
                out.append(r"\begin{itemize}")
                in_itemize = True
            content = stripped[2:].strip()
            out.append(r"\item " + escape_latex(content))
            continue

        # Ordered list: "1. something"
        if re.match(r"^\d+\.\s+", stripped):
            if in_itemize:
                out.append(r"\end{itemize}")
                in_itemize = False
            if not in_enumerate:
                out.append(r"\begin{enumerate}")
                in_enumerate = True
            # strip leading "1. "
            content = re.sub(r"^\d+\.\s+", "", stripped).strip()
            out.append(r"\item " + escape_latex(content))
            continue

        # Horizontal rule style --- or *** or ___
        if set(stripped) in ({"-"}, {"*"}, {"_"}):
            close_lists()
            out.append(r"\medskip\hrule\medskip")
            continue

        # Fallback: regular paragraph line
        close_lists()
        out.append(escape_latex(stripped))

    # Close any dangling lists
    close_lists()
    return "\n".join(out)


def make_letter_latex(job_dir: Path) -> str:
    """
    Build the complete scrlttr2 LaTeX document as a string.
    """
    md_path = job_dir / REPORT_MD_NAME
    if not md_path.exists():
        raise FileNotFoundError(f"{md_path} not found; run report_builder first.")

    md_text = md_path.read_text(encoding="utf-8")
    body_block = markdown_to_latex_body(md_text)

    meta: Dict[str, Any] = load_client_meta(job_dir)
    client_name = (
        meta.get("client_name")
        or meta.get("name")
        or meta.get("full_name")
        or "Client"
    )

    # Prefer explicit job title from meta if present; otherwise first inferred role
    target_roles = (
        meta.get("target_roles")
        or meta.get("preferred_titles")
        or meta.get("job_title")
    )
    primary_role: Optional[str] = None
    if isinstance(target_roles, list):
        primary_role = target_roles[0] if target_roles else None
    elif isinstance(target_roles, str) and target_roles.strip():
        primary_role = target_roles.strip()

    if primary_role:
        subject_line = f"Job Search Report – {primary_role}"
    else:
        subject_line = "Job Search Report"

    subject_tex = escape_latex(subject_line)
    client_name_tex = escape_latex(str(client_name))

    intro_parts = [
        "Thank you for working with TLZ Career Services. "
        "This report summarizes the job search we performed on your behalf."
    ]
    if primary_role:
        intro_parts.append(
            f"We focused on roles aligned with your experience as a {primary_role} "
            f"and closely related positions."
        )
    intro = " ".join(intro_parts)

    intro_tex = escape_latex(intro)

    latex_source = rf"""
\documentclass[11pt]{{scrlttr2}}

\usepackage[margin=1in]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}

\defaultfontfeatures{{Ligatures=TeX}}
\setmainfont{{TeX Gyre Heros}}

\setkomavar{{fromname}}{{TLZ Career Services}}
\setkomavar{{fromaddress}}{{Remote \\\ United States}}
\setkomavar{{subject}}{{{subject_tex}}}

\begin{{document}}

\begin{{letter}}{{{client_name_tex}}}
\opening{{Dear {client_name_tex},}}

{intro_tex}

\medskip

% --- Auto-generated summary from Job_Search_Report.md ---
{body_block}

\medskip

Thank you again for trusting us with this important transition. If you'd like,
we can refine this search further or retarget it toward additional roles.

\closing{{Sincerely,}}

\end{{letter}}
\end{{document}}
""".lstrip()

    return latex_source


def build_report_letter(job_dir: Path) -> Path:
    """
    Entry point for pipeline: create Job_Search_Report.tex and Job_Search_Report.pdf
    inside the given job_dir.
    """
    job_dir = job_dir.resolve()
    log(f"Building LaTeX report for {job_dir}")

    latex_source = make_letter_latex(job_dir)

    tex_path = job_dir / REPORT_TEX_NAME
    tex_path.write_text(latex_source, encoding="utf-8")
    log(f"Wrote {tex_path.name}")

    cmd = [
        "xelatex",
        "-interaction=nonstopmode",
        tex_path.name,
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(job_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    log(proc.stdout)
    if proc.returncode != 0:
        log(proc.stderr)
        raise RuntimeError(f"xelatex failed with code {proc.returncode}")

    pdf_path = job_dir / REPORT_PDF_NAME
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected {pdf_path} to be created by xelatex.")

    log(f"Created {pdf_path.name}")
    return pdf_path


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: latex_report_builder.py JOB_DIR", file=sys.stderr)
        return 1

    job_dir = Path(argv[0])
    try:
        build_report_letter(job_dir)
    except Exception as e:
        log(f"ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
