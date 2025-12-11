from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path
from typing import List

REPORT_MD_NAME = "final_report.md"
REPORT_TEX_NAME = "final_report.tex"
REPORT_PDF_NAME = "final_report.pdf"

LATEX_SPECIAL_CHARS = {
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

def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    for ch, repl in LATEX_SPECIAL_CHARS.items():
        text = text.replace(ch, repl)
    return text

def sanitize_cell(text: str) -> str:
    """
    Clean up a table cell:
    - Turn [Open posting](URL) into just 'Open posting'
    - Escape LaTeX special characters
    """
    text = text.strip()
    m = re.match(r"\[([^\]]+)\]\([^)]+\)", text)
    if m:
        text = m.group(1)
    return escape_latex(text)

def parse_table(lines: List[str], idx: int):
    """
    Parse a GitHub-style markdown table starting at lines[idx].
    lines[idx]   -> header row
    lines[idx+1] -> separator row
    returns (headers, rows, new_index)
    """
    header_line = lines[idx].strip().strip("|")
    headers = [h.strip() for h in header_line.split("|")]
    col_count = len(headers)

    rows: List[List[str]] = []
    i = idx + 2
    while i < len(lines):
        line = lines[i]
        # stop if no pipe or blank line
        if "|" not in line or not line.strip():
            break
        # alignment row again? stop
        if re.match(r"^\s*\|?\s*-", line):
            break

        row = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(row) < col_count:
            row += [""] * (col_count - len(row))
        elif len(row) > col_count:
            row = row[:col_count]
        rows.append(row)
        i += 1

    return headers, rows, i

def markdown_to_latex_body(md: str) -> str:
    """
    Very small markdown subset -> LaTeX:
    - #, ##, ### headings
    - bullet lists (-, *, •)
    - numbered lists (1. ...)
    - blockquotes (> ...)
    - markdown tables
    - plain paragraphs
    """
    lines = md.splitlines()
    out: List[str] = []

    in_itemize = False
    in_enumerate = False
    in_quote = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Markdown table?
        if (
            "|" in line
            and i + 1 < len(lines)
            and "|" in lines[i + 1]
            and re.match(r"^\s*\|?\s*-", lines[i + 1])
        ):
            if in_itemize:
                out.append(r"\end{itemize}")
                in_itemize = False
            if in_enumerate:
                out.append(r"\end{enumerate}")
                in_enumerate = False
            if in_quote:
                out.append(r"\end{quote}")
                in_quote = False

            headers, rows, new_i = parse_table(lines, i)
            col_spec = " | ".join(["l"] * len(headers))

            out.append(r"\begin{tabular}{%s}" % col_spec)
            out.append(r"\hline")
            out.append(" & ".join(sanitize_cell(h) for h in headers) + r" \\")
            out.append(r"\hline")
            for row in rows:
                out.append(" & ".join(sanitize_cell(c) for c in row) + r" \\")
            out.append(r"\hline")
            out.append(r"\end{tabular}")
            out.append("")
            i = new_i
            continue

        # Blank line: close open environments and add paragraph break
        if stripped == "":
            if in_itemize:
                out.append(r"\end{itemize}")
                in_itemize = False
            if in_enumerate:
                out.append(r"\end{enumerate}")
                in_enumerate = False
            if in_quote:
                out.append(r"\end{quote}")
                in_quote = False
            out.append("")
            i += 1
            continue

        # Headings: #, ##, ###
        m = re.match(r"^(#{1,3})\s+(.*)", stripped)
        if m:
            level = len(m.group(1))
            text = escape_latex(m.group(2))
            if in_itemize:
                out.append(r"\end{itemize}")
                in_itemize = False
            if in_enumerate:
                out.append(r"\end{enumerate}")
                in_enumerate = False
            if in_quote:
                out.append(r"\end{quote}")
                in_quote = False

            if level == 1:
                out.append(r"\section*{%s}" % text)
            elif level == 2:
                out.append(r"\subsection*{%s}" % text)
            else:
                out.append(r"\subsubsection*{%s}" % text)
            i += 1
            continue

        # Blockquote: > ...
        if stripped.startswith("> "):
            if not in_quote:
                in_quote = True
                out.append(r"\begin{quote}")
            content = escape_latex(stripped[2:].strip())
            out.append(content)
            i += 1
            continue

        # Bullet list
        if stripped.startswith(("- ", "* ", "• ")):
            if in_enumerate:
                out.append(r"\end{enumerate}")
                in_enumerate = False
            if not in_itemize:
                in_itemize = True
                out.append(r"\begin{itemize}")
            if stripped[0] in "-*":
                bullet_text = stripped[2:].strip()
            else:
                bullet_text = stripped[1:].strip()
            out.append(r"\item %s" % escape_latex(bullet_text))
            i += 1
            continue

        # Numbered list: 1. ...
        m = re.match(r"^\d+\.\s+(.*)", stripped)
        if m:
            if in_itemize:
                out.append(r"\end{itemize}")
                in_itemize = False
            if not in_enumerate:
                in_enumerate = True
                out.append(r"\begin{enumerate}")
            out.append(r"\item %s" % escape_latex(m.group(1)))
            i += 1
            continue

        # Normal paragraph line
        text = escape_latex(stripped)
        out.append(text)
        i += 1

    # Close any open environments at EOF
    if in_itemize:
        out.append(r"\end{itemize}")
    if in_enumerate:
        out.append(r"\end{enumerate}")
    if in_quote:
        out.append(r"\end{quote}")

    return "\n".join(out)

LETTER_TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{fontspec}

\setmainfont{TeX Gyre Heros}

\begin{document}
%s
\end{document}
""".lstrip()

def build_latex_report(job_dir: Path) -> Path:
    print(f"[REPORT-LATEX] Building LaTeX report for {job_dir}")
    md_path = job_dir / REPORT_MD_NAME
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown report not found at {md_path}")

    md_text = md_path.read_text(encoding="utf-8")
    body = markdown_to_latex_body(md_text)
    tex_content = LETTER_TEMPLATE % body

    tex_path = job_dir / REPORT_TEX_NAME
    tex_path.write_text(tex_content, encoding="utf-8")
    print("[REPORT-LATEX] Wrote final_report.tex")

    # Run XeLaTeX to produce the PDF
    try:
        completed = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", REPORT_TEX_NAME],
            cwd=str(job_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    except subprocess.CalledProcessError as e:
        sys.stdout.write(e.stdout or "")
        sys.stderr.write(e.stderr or "")
        raise

    print("[REPORT-LATEX] Created final_report.pdf")
    return job_dir / REPORT_PDF_NAME

def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: latex_report_builder.py JOB_FOLDER", file=sys.stderr)
        raise SystemExit(1)

    job_dir = Path(argv[0]).resolve()
    build_latex_report(job_dir)

if __name__ == "__main__":
    main()
