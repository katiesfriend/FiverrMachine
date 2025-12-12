from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from report_builder import load_client_meta, REPORT_MD_NAME


# -----------------------------------------------------------------------------
# Basic LaTeX escaping
# -----------------------------------------------------------------------------

def escape_latex(text: str) -> str:
    """
    Escape LaTeX special characters in normal text.
    (Local copy so this module does not depend on latex_resume_builder.)
    """
    replacements = {
        "\": r"\textbackslash{}",
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
    out_chars: List[str] = []
    for ch in text:
        out_chars.append(replacements.get(ch, ch))
    return "".join(out_chars)


# -----------------------------------------------------------------------------
# Inline markdown handling: links, emphasis, and simple formatting
# -----------------------------------------------------------------------------

LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")


def render_inline(text: str) -> str:
    """
    Convert simple inline markdown to LaTeX-safe content:

    - [text](url)  -> \href{url}{text}
    - *italic* or _italic_ -> \emph{text}
    - **bold** or __bold__ -> \textbf{text}

    Everything else is escaped.
    """

    def replace_link(match: re.Match) -> str:
        label = escape_latex(match.group(1).strip())
        url = match.group(2).strip()
        # No escaping for URL to keep it clickable
        return r"\href{" + url + "}{" + label + "}"

    # First handle links
    with_links = LINK_PATTERN.sub(replace_link, text)

    # Handle strong emphasis (**text** or __text__)
    def replace_bold(t: str) -> str:
        return re.sub(
            r"(\*\*|__)(.+?)\1",
            lambda m: r"\textbf{" + escape_latex(m.group(2)) + "}",
            t,
        )

    # Handle italics (*text* or _text_)
    def replace_italic(t: str) -> str:
        return re.sub(
            r"(\*|_)([^*_]+?)\1",
            lambda m: r"\emph{" + escape_latex(m.group(2)) + "}",
            t,
        )

    # First bold, then italic, on the link-processed text
    tmp = replace_bold(with_links)
    tmp = replace_italic(tmp)

    # Whatever is left, escape as plain text (but links are already protected)
    # To avoid double-escaping LaTeX macros we've introduced (like \href),
    # we do a simple pass: split on backslashes and only escape the first chunk.
    parts = tmp.split("\\")
    if not parts:
        return ""

    escaped_first = escape_latex(parts[0])
    rebuilt = [escaped_first]
    # For subsequent parts, we assume they start a LaTeX command we created.
    for tail in parts[1:]:
        rebuilt.append("\\" + tail)

    return "".join(rebuilt)


# -----------------------------------------------------------------------------
# Pipe table support
# -----------------------------------------------------------------------------

def _split_pipe_row(row: str) -> List[str]:
    """
    Split a pipe table row into cells, ignoring leading/trailing pipes.
    """
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    # Split on | not escaped (we don't generate escaped ones anyway)
    parts = [c.strip() for c in row.split("|")]
    return parts


def is_pipe_table_header(line1: str, line2: str) -> bool:
    """
    Detect markdown pipe table header: e.g.

    | Col A | Col B |
    | ----- | ----- |

    We only validate that line2 contains dashes and pipes in all cells.
    """
    if "|" not in line1 or "|" not in line2:
        return False
    if not line2.strip().startswith("|") or not line2.strip().endswith("|"):
        return False

    cells = _split_pipe_row(line2)
    if not cells:
        return False
    for c in cells:
        c = c.strip()
        if not c or not all(ch in "-: " for ch in c):
            return False
    return True


def convert_pipe_table(lines: List[str], start_idx: int) -> tuple[str, int]:
    """
    Convert a markdown pipe table starting at lines[start_idx] into a LaTeX tabular.

    Returns (latex_string, new_index) where new_index is the index of the first
    line after the table.
    """
    header_line = lines[start_idx]
    separator_line = lines[start_idx + 1]
    header_cells = _split_pipe_row(header_line)
    ncols = len(header_cells)

    # Determine alignment from separator line (very simple heuristic)
    sep_cells = _split_pipe_row(separator_line)
    aligns = []
    for c in sep_cells:
        c = c.strip()
        left = c.startswith(":")
        right = c.endswith(":")
        if left and right:
            aligns.append("c")
        elif right:
            aligns.append("r")
        else:
            aligns.append("l")
    if len(aligns) < ncols:
        aligns.extend(["l"] * (ncols - len(aligns)))
    aligns = aligns[:ncols]

    body_lines: List[str] = []
    idx = start_idx + 2
    while idx < len(lines):
        l = lines[idx]
        if not l.strip().startswith("|"):
            break
        if set(l.strip()) <= {"|", "-"}:
            # Another separator row – treat as table end
            break
        body_lines.append(l)
        idx += 1

    # Build LaTeX tabular
    out: List[str] = []
    out.append("\\begin{tabular}{" + " | ".join(aligns) + "}")
    out.append("\\hline")

    # Header row
    escaped_headers = [render_inline(c) for c in header_cells]
    out.append(" & ".join(escaped_headers) + r" \\")
    out.append("\\hline")

    # Body rows
    for row in body_lines:
        cells = _split_pipe_row(row)
        if not cells:
            continue
        if len(cells) < ncols:
            cells.extend([""] * (ncols - len(cells)))
        cells = cells[:ncols]
        tex_cells = [render_inline(c) for c in cells]
        out.append(" & ".join(tex_cells) + r" \\")
    out.append("\\hline")
    out.append("\\end{tabular}")

    return "\n".join(out), idx


# -----------------------------------------------------------------------------
# Markdown-to-LaTeX body
# -----------------------------------------------------------------------------

def markdown_to_latex_body(md_text: str) -> str:
    """
    Small markdown-to-LaTeX converter that understands:

    - # / ## / ### headings
    - unordered lists with "-" or "*"
    - ordered lists like "1. text"
    - pipe tables
    - simple inline markdown formatting (links, bold, italics)

    Everything else is emitted as escaped plain text.
    """
    lines = md_text.splitlines()

    out: list[str] = []
    in_itemize = False
    in_enumerate = False

    def close_lists() -> None:
        nonlocal in_itemize, in_enumerate
        if in_itemize:
            out.append(r"\\end{itemize}")
            in_itemize = False
        if in_enumerate:
            out.append(r"\\end{enumerate}")
            in_enumerate = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n")

        # Blank line -> paragraph break
        if not stripped.strip():
            close_lists()
            out.append("")
            i += 1
            continue

        # Pipe table detection
        if (
            i + 1 < len(lines)
            and is_pipe_table_header(stripped, lines[i + 1])
        ):
            close_lists()
            table_tex, new_idx = convert_pipe_table(lines, i)
            out.append(table_tex)
            i = new_idx
            continue

        # Headings (render as bold text instead of LaTeX \\section in scrlttr2)
        if stripped.startswith("### "):
            close_lists()
            title = render_inline(stripped[4:].strip())
            out.append(r"\\textbf{" + title + r"}" + "\n" + r"\\par\\smallskip")
            i += 1
            continue
        if stripped.startswith("## "):
            close_lists()
            title = render_inline(stripped[3:].strip())
            out.append(
                r"\\bigskip"
                + "\n"
                + r"\\textbf{" + title + r"}"
                + "\n"
                + r"\\par\\medskip"
            )
            i += 1
            continue
        if stripped.startswith("# "):
            close_lists()
            title = render_inline(stripped[2:].strip())
            out.append(
                r"\\bigskip"
                + "\n"
                + r"\\textbf{" + title + r"}"
                + "\n"
                + r"\\par\\medskip"
            )
            i += 1
            continue

        # Ordered list "1. text"
        m_num = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m_num:
            if in_itemize:
                out.append(r"\\end{itemize}")
                in_itemize = False
            if not in_enumerate:
                out.append(r"\\begin{enumerate}")
                in_enumerate = True
            content = m_num.group(2).strip()
            out.append(r"\\item " + render_inline(content))
            i += 1
            continue

        # Unordered list
        if stripped.startswith("- ") or stripped.startswith("* "):
            if in_enumerate:
                out.append(r"\\end{enumerate}")
                in_enumerate = False
            if not in_itemize:
                out.append(r"\\begin{itemize}")
                in_itemize = True
            content = stripped[2:].strip()
            out.append(r"\\item " + render_inline(content))
            i += 1
            continue

        # Default paragraph
        close_lists()
        out.append(render_inline(stripped))
        i += 1

    # Close any dangling lists
    close_lists()
    return "\n".join(out)


# -----------------------------------------------------------------------------
# Letter wrapper
# -----------------------------------------------------------------------------

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
    elif isinstance(target_roles, str):
        primary_role = target_roles

    # Use a friendlier default subject if we don't have a role
    subject_role = primary_role or "career opportunities"

    client_first_tex = escape_latex(client_name.split()[0])
    subject_tex = escape_latex(subject_role)

    # Compose the full LaTeX document
    latex_source = rf"""
\\documentclass[11pt,foldmarks=false]{{scrlttr2}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{fontspec}}
\\usepackage{{graphicx}}
\\usepackage[table]{{xcolor}}
\\usepackage{{hyperref}}

\\setmainfont{{TeX Gyre Heros}}

\\KOMAoptions{{
    fromalign=left,
    fromrule=aftername,
    backaddress=false,
    parskip=half
}}

\\setkomavar{{fromname}}{{TLZ Career Services}}
\\setkomavar{{fromaddress}}{{}}          % no address lines
\\KOMAoptions{{backaddress=false}}       % remove return address line

\\setkomavar{{subject}}{{Selected {subject_tex} opportunities}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=blue
}}

\\begin{{document}}

\\begin{{letter}}{{}}
\\opening{{Dear {client_first_tex},}}

{body_block}

\\closing{{Sincerely,}}

\\end{{letter}}
\\end{{document}}
""".lstrip()
    return latex_source


# -----------------------------------------------------------------------------
# CLI wrapper
# -----------------------------------------------------------------------------

def run_xelatex(tex_path: Path) -> None:
    """
    Run xelatex on the given .tex file and raise if it fails.
    """
    cmd = ["xelatex", "-interaction=nonstopmode", tex_path.name]
    proc = subprocess.run(
        cmd,
        cwd=str(tex_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print("XeLaTeX failed with return code", proc.returncode, file=sys.stderr)
        print("STDOUT:", proc.stdout, file=sys.stderr)
        print("STDERR:", proc.stderr, file=sys.stderr)
        raise RuntimeError("xelatex failed; see above for details.")


def build_latex_report(job_dir: Path) -> Path:
    """
    Main entry point: generate final_report.tex and final_report.pdf in job_dir.
    """
    job_dir = job_dir.resolve()
    tex_path = job_dir / "final_report.tex"

    latex_source = make_letter_latex(job_dir)
    tex_path.write_text(latex_source, encoding="utf-8")

    # Run xelatex twice for stable refs (even though we don't use many)
    run_xelatex(tex_path)
    run_xelatex(tex_path)

    pdf_path = job_dir / "final_report.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError("XeLaTeX did not produce final_report.pdf")
    return pdf_path


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("Usage: python latex_report_builder.py JOB_DIR", file=sys.stderr)
        raise SystemExit(1)
    job_dir = Path(argv[1])
    pdf_path = build_latex_report(job_dir)
    print(f"[REPORT-LATEX] Created {pdf_path}")


if __name__ == "__main__":
    main(sys.argv)
