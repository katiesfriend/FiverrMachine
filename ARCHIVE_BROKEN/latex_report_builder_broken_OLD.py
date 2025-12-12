from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

from report_builder import load_client_meta

REPORT_MD_NAME = "final_report.md"
REPORT_TEX_NAME = "final_report.tex"
REPORT_PDF_NAME = "final_report.pdf"


# --------- Small helpers --------- #

def escape_latex(text: str) -> str:
    """Escape characters that LaTeX treats as special."""
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


def render_inline(text: str) -> str:
    """
    Very small markdown -> LaTeX inline renderer:
    - **bold**
    - *italic*
    - [label](url)  ->  \href{url}{label}
    """
    bold_parts: List[str] = []
    italic_parts: List[str] = []
    link_parts: List[tuple[str, str]] = []

    # First: markdown links [label](url) -> placeholders
    def _link_repl(m: re.Match) -> str:
        idx = len(link_parts)
        label = m.group(1)
        url = m.group(2)
        link_parts.append((label, url))
        return f"@@LINK{idx}@@"

    text0 = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link_repl, text)

    # Then bold **...**
    def _bold_repl(m: re.Match) -> str:
        idx = len(bold_parts)
        bold_parts.append(m.group(1))
        return f"@@BOLD{idx}@@"

    text1 = re.sub(r"\*\*(.+?)\*\*", _bold_repl, text0)

    # Then italics *...*
    def _italic_repl(m: re.Match) -> str:
        idx = len(italic_parts)
        italic_parts.append(m.group(1))
        return f"@@ITAL{idx}@@"

    text2 = re.sub(r"\*(?!\s)(.+?)\*(?!\*)", _italic_repl, text1)

    # Escape remaining LaTeX specials
    escaped = escape_latex(text2)

    # Restore bold
    for i, inner in enumerate(bold_parts):
        escaped_inner = escape_latex(inner)
        escaped = escaped.replace(
            f"@@BOLD{i}@@", r"\textbf{" + escaped_inner + r"}"
        )

    # Restore italics
    for i, inner in enumerate(italic_parts):
        escaped_inner = escape_latex(inner)
        escaped = escaped.replace(
            f"@@ITAL{i}@@", r"\emph{" + escaped_inner + r"}"
        )

    # Restore links as clickable but short labels
    for i, (label, url) in enumerate(link_parts):
        label_tex = escape_latex(label)
        url_tex = escape_latex(url)
        link_tex = r"\href{" + url_tex + "}{" + label_tex + "}"
        escaped = escaped.replace(f"@@LINK{i}@@", link_tex)

    return escaped

def log(msg: str) -> None:
    print(f"[REPORT-LATEX] {msg}")


# --------- Table conversion (pipe markdown -> LaTeX table) --------- #

def _split_pipe_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_alignment_row(cells: list[str]) -> bool:
    # Row like: |---|:---:|---|
    for c in cells:
        stripped = c.strip()
        if not stripped:
            continue
        if not set(stripped) <= set("-:"):
            return False
    return True


def convert_pipe_table(start_idx: int, lines: list[str], out: list[str]) -> int:
    """
    Convert a block of GitHub-style pipe table lines into a LaTeX tabularx table
    with alternating TLZ-blue stripes. Returns the index of the first line
    *after* the table block.
    """
    table_lines: list[str] = []
    i = start_idx
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        table_lines.append(lines[i].rstrip("\n"))
        i += 1

    # If it's too short, just print literally
    if len(table_lines) < 2:
        for raw in table_lines:
            out.append(render_inline(raw))
        return i

    header_cells = _split_pipe_row(table_lines[0])
    data_rows: list[list[str]] = []

    for line in table_lines[1:]:
        cells = _split_pipe_row(line)
        if _is_alignment_row(cells):
            continue
        data_rows.append(cells)

    num_cols = len(header_cells)
    if num_cols == 0:
        return i

    # First column centered, the rest flexible-width columns
    col_spec = "c" if num_cols == 1 else "c " + " ".join("X" for _ in range(num_cols - 1))

    out.append(r"\medskip")
    out.append(r"\rowcolors{2}{TLZBlueLight!15}{white}")
    out.append(r"\begin{tabularx}{\textwidth}{@{}" + col_spec + r"@{}}")
    out.append(r"\toprule")
    out.append(" & ".join(render_inline(h) for h in header_cells) + r" \\")
    out.append(r"\midrule")

    for row in data_rows:
        if len(row) < num_cols:
            row = row + [""] * (num_cols - len(row))
        elif len(row) > num_cols:
            row = row[: num_cols - 1] + [" ".join(row[num_cols - 1:])]
        out.append(" & ".join(render_inline(cell) for cell in row) + r" \\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabularx}")
    out.append(r"\rowcolors{2}{}{}")
    out.append(r"\medskip")

    return i


# --------- Extract numbers for visuals (pie + comp bar) --------- #

def extract_bucket_counts(md_text: str):
    """
    Find:
      High: 15 job(s)
      Medium: 5 job(s)
      Low: 0 job(s)
      Very Low: 0 job(s)
    """
    pattern = (
        r"High:\s*(\d+)\s*job\(s\).*?"
        r"Medium:\s*(\d+)\s*job\(s\).*?"
        r"Low:\s*(\d+)\s*job\(s\).*?"
        r"Very Low:\s*(\d+)\s*job\(s\)"
    )
    m = re.search(pattern, md_text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())  # (High, Medium, Low, VeryLow)


def extract_comp_range(md_text: str):
    """
    Find salary range phrase like:
      between **$40,000** and **$205,000** ... near **$61,500**
    """
    pattern = (
        r"between\s+\*\*\$(\d[\d,]*)\*\*\s+and\s+\*\*\$(\d[\d,]*)\*\*"
        r".*?near\s+\*\*\$(\d[\d,]*)\*\*"
    )
    m = re.search(pattern, md_text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    vals = [float(v.replace(",", "")) for v in m.groups()]
    return tuple(vals)  # (low, high, center)

def build_pie_code(counts) -> str:
    """
    Build a pgf-pie chart for the match buckets.
    We keep the labels in a legend so they don't collide.
    """
    h, m, l, v = counts
    return rf"""
\medskip
\begin{{center}}
\begin{{tikzpicture}}
\pie[text=legend,radius=2.8,sum=auto,color={{TLZBlue,TLZBlueLight,gray!55,gray!25}}]{{{h}/High, {m}/Medium, {l}/Low, {v}/Very Low}}
\end{{tikzpicture}}
\end{{center}}
\medskip
"""


def build_comp_bar(low: float, high: float, center: float) -> str:
    """
    Placeholder for a future horizontal compensation band graphic.
    For now we rely on the textual explanation only.
    """
    rng = max(high - low, 1.0)
    ratio = (center - low) / rng
    ratio = max(0.0, min(1.0, ratio))
    pos = 8.0 * ratio

    # We keep the implementation available but do not inject it anywhere yet.
    return rf"""
\medskip
\begin{{center}}
\begin{{tikzpicture}}[x=1cm,y=1cm]
  \draw[gray!40, line width=0.4pt] (0,0) -- (8,0);
  \draw[fill=blue!20, draw=blue!60] (0,-0.18) rectangle (8,0.18);
  \draw[fill=black] ({pos:.2f},-0.28) -- ({pos:.2f}+0.12,0) -- ({pos:.2f},0.28) -- cycle;
\end{{tikzpicture}}
\end{{center}}
\medskip
"""


def inject_visuals(md_text: str, body: str) -> str:
    """
    Inject visuals (currently just the pie chart) into the LaTeX body.
    """
    # Pie chart after "By match bucket..."
    counts = extract_bucket_counts(md_text)
    if counts:
        pie = build_pie_code(counts)
        marker = r"\textbf{By match bucket (scraper score):}"
        if marker in body:
            body = body.replace(marker, marker + "\n" + pie, 1)

    # No horizontal comp bar for now – keep the textual explanation only
    return body

# --------- Block-level markdown -> LaTeX --------- #

def markdown_to_latex_body(md_text: str) -> str:
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

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if stripped == "":
            close_lists()
            out.append("")
            i += 1
            continue

        # GitHub-style tables
        if stripped.startswith("|"):
            close_lists()
            i = convert_pipe_table(i, lines, out)
            continue

        # Headings ###
        if stripped.startswith("### "):
            close_lists()
            title = render_inline(stripped[4:].strip())
            out.append(r"\textbf{" + title + r"}")
            out.append(r"\par\smallskip")
            i += 1
            continue
        if stripped.startswith("## "):
            close_lists()
            title = render_inline(stripped[3:].strip())
            out.append(r"\bigskip")
            out.append(r"\textbf{" + title + r"}")
            out.append(r"\par\medskip")
            i += 1
            continue
        if stripped.startswith("# "):
            close_lists()
            title = render_inline(stripped[2:].strip())
            out.append(r"\bigskip")
            out.append(r"\textbf{" + title + r"}")
            out.append(r"\par\medskip")
            i += 1
            continue

        # Bullets
        if stripped.startswith("- ") or stripped.startswith("* "):
            if in_enumerate:
                out.append(r"\end{enumerate}")
                in_enumerate = False
            if not in_itemize:
                out.append(r"\begin{itemize}")
                in_itemize = True
            content = stripped[2:].strip()
            out.append(r"\item " + render_inline(content))
            i += 1
            continue

        # Numbered list "1. Foo"
        if re.match(r"^\d+\.\s+", stripped):
            if in_itemize:
                out.append(r"\end{itemize}")
                in_itemize = False
            if not in_enumerate:
                out.append(r"\begin{enumerate}")
                in_enumerate = True
            content = re.sub(r"^\d+\.\s+", "", stripped).strip()
            out.append(r"\item " + render_inline(content))
            i += 1
            continue

        # Horizontal rule
        if set(stripped) in ({"-"}, {"*"}, {"_"}):
            close_lists()
            out.append(r"\medskip\hrule\medskip")
            i += 1
            continue

        # Plain paragraph
        close_lists()
        out.append(render_inline(stripped))
        i += 1

    close_lists()
    return "\n".join(out)


# --------- Build the letter document --------- #

def make_letter_latex(job_dir: Path) -> str:
    md_path = job_dir / REPORT_MD_NAME
    if not md_path.exists():
        raise FileNotFoundError(f"{md_path} not found; run report_builder first.")

    md_text = md_path.read_text(encoding="utf-8")
    # Remove Unicode em-dashes from the markdown; they cause ugly output
    md_text = md_text.replace("—", " - ")

    body_block = markdown_to_latex_body(md_text)
    body_block = inject_visuals(md_text, body_block)

    meta: Dict[str, Any] = load_client_meta(job_dir)

    # Derive client full name from metadata or folder name (e.g. JOB_Michael_Wood)
    client_full_name = (
        meta.get("client_name")
        or meta.get("name")
        or meta.get("full_name")
    )

    if not client_full_name:
        token = job_dir.name  # e.g. JOB_James or JOB_Michael_Wood
        if token.upper().startswith("JOB_"):
            token = token[4:]
        client_full_name = token.replace("_", " ") if token else "Client"

    client_full_name = str(client_full_name).strip() or "Client"
    client_first_name = client_full_name.split()[0]

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
        subject_line = f"Job Search Report - {primary_role}"
    else:
        subject_line = "Job Search Report"

    subject_tex = escape_latex(subject_line)
    client_name_tex = escape_latex(client_full_name)
    client_first_tex = escape_latex(client_first_name)

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
\documentclass[11pt,foldmarks=false]{{scrlttr2}}

\usepackage[margin=1in]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{graphicx}}
\usepackage[table]{{xcolor}}
\usepackage{{colortbl}}
\definecolor{{TLZBlue}}{{HTML}}{{284B63}}      % main brand blue
\definecolor{{TLZBlueLight}}{{HTML}}{{8FB1FF}} % lighter blue for pie / stripes
\usepackage{{tabularx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{tikz}}
\usepackage{{pgf-pie}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}

\defaultfontfeatures{{Ligatures=TeX}}
\setmainfont{{TeX Gyre Heros}}

\setkomavar{{fromname}}{{TLZ Career Services}}
\setkomavar{{fromaddress}}{{}}          % no address lines
\KOMAoptions{{backaddress=false}}       % remove underlined backaddress line
\setkomavar{{signature}}{{TLZ Career Services}}
\setkomavar{{subject}}{{{subject_tex}}}

\begin{{document}}

\begin{{letter}}{}
\opening{{Dear {client_first_tex},}}

{intro_tex}

\medskip

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
