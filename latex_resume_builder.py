#!/usr/bin/env python3
"""
latex_resume_builder.py

Takes a plaintext tailored resume (resume_jobXX.txt), combines it with
base_resume.txt (for stable sections like Education and Certifications),
builds a structured representation, writes a JSON snapshot, and then
renders a LaTeX PDF.

If pdflatex is missing or compilation fails with no PDF output, the
caller should catch the RuntimeError and fall back to a simpler path.
"""

import json
import os
import subprocess
from typing import Dict, List, Tuple, Optional


def _normalize_unicode(text: str) -> str:
    """
    Normalize common Unicode punctuation into LaTeX-friendly ASCII.
    This prevents classic "Unicode character U+XXXX not set up" errors.
    """
    replacements = {
        "\u2022": "*",    # bullet
        "\u2013": "-",    # en dash
        "\u2014": "-",    # long dash
        "\u201c": "\"",
        "\u201d": "\"",
        "\u201e": "\"",
        "\u2018": "'",
        "\u2019": "'",
        "\u2026": "...",
        "\u00a0": " ",    # non-breaking space
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _escape_latex(text: str) -> str:
    """
    Escape basic LaTeX special characters in a conservative way.
    Also normalizes Unicode punctuation first.
    """
    text = _normalize_unicode(text)

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
    out: List[str] = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def _split_header_body(lines: List[str], header_lines: int = 3) -> Tuple[List[str], List[str]]:
    """
    Heuristic: first N non-empty lines are treated as the "header block"
    (name, title, contact line). Everything else is body.
    """
    non_empty_indices = [i for i, ln in enumerate(lines) if ln.strip()]
    if not non_empty_indices:
        return [], lines

    cutoff_count = min(header_lines, len(non_empty_indices))
    cutoff_index = non_empty_indices[cutoff_count - 1]

    header = lines[: cutoff_index + 1]
    body = lines[cutoff_index + 1 :]
    return header, body


# Map of section headers in the raw text to internal keys
_HEADER_MAP: Dict[str, str] = {
    "PROFESSIONAL SUMMARY": "summary",
    "SUMMARY": "summary",
    "CORE COMPETENCIES": "core_competencies",
    "CORE COMPETENCY": "core_competencies",
    "PROFESSIONAL EXPERIENCE": "experience",
    "EXPERIENCE": "experience",
    "EDUCATION": "education",
    "CERTIFICATIONS": "certifications",
    "CERTIFICATION": "certifications",
    "NOTABLE ACHIEVEMENTS": "achievements",
    "ACHIEVEMENTS": "achievements",
}


def _parse_sections(text: str) -> Dict[str, str]:
    """
    Very simple section parser:

    - Everything before the first known header is "preamble" (name, contact).
    - Known headers start a new section.
    - Content is all lines until the next header.
    """
    lines = text.splitlines()
    sections: Dict[str, List[str]] = {}
    current_key = "preamble"
    buf: List[str] = []

    def flush() -> None:
        nonlocal buf, current_key
        if buf:
            existing = sections.get(current_key, [])
            existing.extend(buf)
            sections[current_key] = existing
            buf = []

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if upper in _HEADER_MAP:
            # New section header
            flush()
            current_key = _HEADER_MAP[upper]
            continue
        buf.append(line)

    flush()
    # Join lines into blocks
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _build_structured_plaintext(
    tailored_text: str,
    base_text: Optional[str],
) -> Tuple[str, Dict[str, str]]:
    """
    Combine tailored resume text with base resume text into a structured set
    of sections and return a reconstructed plaintext resume plus the section
    dict (for JSON debugging).

    Rules:
    - Header/preamble: prefer tailored, else base.
    - Summary, core competencies, experience, achievements:
        prefer tailored, fall back to base if missing.
    - Education and certifications:
        always keep if present in tailored; otherwise backfill from base.
    - No empty sections are emitted.
    """
    tail_secs = _parse_sections(tailored_text)
    base_secs = _parse_sections(base_text) if base_text else {}

    def pick(key: str, stable: bool = False) -> str:
        """
        If stable is True, use tailored if non-empty, otherwise base.
        Otherwise, prefer tailored but fall back to base if missing.
        """
        t = tail_secs.get(key, "").strip()
        b = base_secs.get(key, "").strip()
        if stable:
            return t if t else b
        return t or b

    header_block = tail_secs.get("preamble", base_secs.get("preamble", "")).strip()

    summary_block = pick("summary")
    core_block = pick("core_competencies")
    experience_block = pick("experience")
    education_block = pick("education", stable=True)
    certifications_block = pick("certifications", stable=True)
    achievements_block = pick("achievements")

    structured: Dict[str, str] = {
        "header": header_block,
        "summary": summary_block,
        "core_competencies": core_block,
        "experience": experience_block,
        "education": education_block,
        "certifications": certifications_block,
        "achievements": achievements_block,
    }

    parts: List[str] = []

    if header_block:
        parts.append(header_block)

    def add_section(label: str, body: str) -> None:
        if body and body.strip():
            parts.append(label)
            parts.append(body.strip())

    add_section("PROFESSIONAL SUMMARY", summary_block)
    add_section("CORE COMPETENCIES", core_block)
    add_section("PROFESSIONAL EXPERIENCE", experience_block)
    add_section("EDUCATION", education_block)
    add_section("CERTIFICATIONS", certifications_block)
    add_section("NOTABLE ACHIEVEMENTS", achievements_block)

    rebuilt_text = "\n\n".join(parts) + "\n"
    return rebuilt_text, structured


def _build_resume_tex(content: str, doc_title: Optional[str] = None) -> str:
    """
    Build a LaTeX source string for a clean, modern single-column resume.

    We treat the first 2-3 non-empty lines as a header and render them larger.
    The remaining lines are rendered as body text, preserving blank lines as
    vertical space and turning list-like lines into simple bullets.
    """
    raw_lines = content.splitlines()
    header_lines, body_lines = _split_header_body(raw_lines, header_lines=3)

    header_lines_esc = [_escape_latex(ln.strip()) for ln in header_lines if ln.strip()]
    body_lines_esc = [_escape_latex(ln.rstrip()) for ln in body_lines]

    header_block = ""
    if header_lines_esc:
        name = header_lines_esc[0]
        title_line = header_lines_esc[1] if len(header_lines_esc) > 1 else ""
        contact_line = header_lines_esc[2] if len(header_lines_esc) > 2 else ""

        header_block = r"""
\begin{center}
    {\fontsize{20}{22}\selectfont \textbf{""" + name + r"""}}\\[4pt]
"""

        if title_line:
            header_block += r"    {\large " + title_line + r"}\\[2pt]" + "\n"
        if contact_line:
            header_block += r"    {\small " + contact_line + r"}\\[4pt]" + "\n"

        header_block += r"\rule{0.9\textwidth}{0.4pt}" + "\n" + r"\end{center}" + "\n\n"

    # Body: blank lines become vertical space, regular lines get light spacing
    body_block_lines: List[str] = []
    for ln in body_lines_esc:
        if ln.strip() == "":
            body_block_lines.append(r"\par\vspace{0.6\baselineskip}")
        else:
            body_block_lines.append(ln + r"\\[0.1\baselineskip]")

    body_block = "\n".join(body_block_lines)

    title_comment = f"% {doc_title}\n" if doc_title else ""
    tex = r"""% Auto-generated by latex_resume_builder.py
""" + title_comment + r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{textcomp}
\usepackage{enumitem}
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    urlcolor=black,
    linkcolor=black
}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.2\baselineskip}

\begin{document}
""" + header_block + body_block + r"""

\end{document}
"""

    return tex


def render_plaintext_resume_to_pdf(
    txt_path: str,
    pdf_path: str,
    doc_title: Optional[str] = "Tailored Resume",
) -> None:
    """
    Read a plaintext tailored resume and compile a LaTeX PDF at pdf_path.

    Also writes a JSON snapshot of the structured sections next to the PDF.

    Raises RuntimeError if pdflatex is unavailable or if compilation fails
    and no PDF is produced.
    """
    if not os.path.isfile(txt_path):
        raise RuntimeError(f"Resume text file does not exist: {txt_path}")

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        tailored_text = f.read()

    job_dir = os.path.dirname(os.path.abspath(pdf_path))
    base_resume_path = os.path.join(job_dir, "base_resume.txt")
    base_text: Optional[str] = None
    if os.path.isfile(base_resume_path):
        with open(base_resume_path, "r", encoding="utf-8", errors="ignore") as f:
            base_text = f.read()

    # Combine tailored and base into structured sections
    rebuilt_text, structured = _build_structured_plaintext(tailored_text, base_text)

    # Write JSON snapshot for debugging and introspection
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = os.path.join(job_dir, base_name + ".json")
    try:
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(structured, jf, indent=2)
    except OSError:
        # Non-fatal if JSON cannot be written
        pass

    # Build LaTeX source from the rebuilt text
    tex_source = _build_resume_tex(rebuilt_text, doc_title=doc_title)

    tex_name = base_name + ".tex"
    tex_path = os.path.join(job_dir, tex_name)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_source)

    compiled_pdf = os.path.join(job_dir, base_name + ".pdf")

    # Run pdflatex; we treat a non-zero return as fatal only if no PDF appears
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_name],
            cwd=job_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError("pdflatex is not installed or not on PATH") from e

    if result.returncode != 0 and not os.path.isfile(compiled_pdf):
        msg = f"LaTeX compilation failed and no PDF was produced: returncode={result.returncode}"
        raise RuntimeError(msg)

    if not os.path.isfile(compiled_pdf):
        raise RuntimeError(f"Expected LaTeX output PDF not found: {compiled_pdf}")

    if os.path.abspath(compiled_pdf) != os.path.abspath(pdf_path):
        os.replace(compiled_pdf, pdf_path)

    # Clean up aux/log files (best-effort)
    for ext in (".aux", ".log", ".out"):
        p = os.path.join(job_dir, base_name + ext)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
