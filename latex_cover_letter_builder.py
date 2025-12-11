#!/usr/bin/env python3
"""
latex_cover_letter_builder.py

Builds a simple, clean LaTeX PDF cover letter from plain text.
"""

from pathlib import Path
import subprocess


def _escape_latex(text: str) -> str:
    """
    Escape characters that LaTeX treats as special.
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
    for bad, repl in replacements.items():
        text = text.replace(bad, repl)
    return text


def _format_body(letter_text: str) -> str:
    """
    Turn the raw letter text into LaTeX paragraphs.
    Blank lines become paragraph breaks.
    """
    text = letter_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    latex_paras = []

    for p in paragraphs:
        latex_paras.append(_escape_latex(p))

    # Blank line between paragraphs -> new paragraph in LaTeX
    return "\n\n".join(latex_paras)


def build_cover_letter_pdf(letter_text: str, output_pdf: Path) -> None:
    """
    Render a single cover letter into a LaTeX PDF at output_pdf.
    """
    output_pdf = Path(output_pdf)
    workdir = output_pdf.parent

    body = _format_body(letter_text)

    tex_source = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{parskip}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}

\begin{document}
%s
\end{document}
""" % body

    tex_path = workdir / f"{output_pdf.stem}.tex"
    tex_path.write_text(tex_source, encoding="utf-8")

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        tex_path.name,
    ]

    completed = subprocess.run(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"pdflatex failed for {output_pdf} with code {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    # pdflatex leaves .aux, .log etc. around; you can clean them later if desired.
