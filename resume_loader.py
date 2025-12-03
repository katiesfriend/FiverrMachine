#!/usr/bin/env python3
"""
resume_loader.py

Centralized logic for finding and converting a client's resume
(from whatever format they uploaded) into plain text.

Supported formats:
- .txt
- .docx
- .pdf
- .rtf
- .doc
- .odt

Output:
- Writes base_resume.txt into the job folder
- Returns the resume text as a Python string
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


ALLOWED_EXTS = [".txt", ".docx", ".pdf", ".rtf", ".doc", ".odt"]


class ResumeError(RuntimeError):
    pass


def _debug(msg: str) -> None:
    # You can upgrade this later to real logging if you want
    print(f"[RESUME_LOADER] {msg}")


def find_resume_file(job_dir: Path) -> Optional[Path]:
    """
    Look for a file in job_dir that appears to be the client's resume.

    Priority:
    1) Files starting with 'base_resume'
    2) Files containing 'resume' anywhere in the name

    Only considers files with ALLOWED_EXTS extensions.
    """
    candidates = []

    for entry in job_dir.iterdir():
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue

        name_lower = entry.name.lower()
        if name_lower.startswith("base_resume"):
            candidates.append((0, entry))  # highest priority
        elif "resume" in name_lower:
            candidates.append((1, entry))

    if not candidates:
        return None

    # Sort by priority, then by name to keep deterministic
    candidates.sort(key=lambda x: (x[0], x[1].name))
    chosen = candidates[0][1]
    _debug(f"Selected resume file: {chosen}")
    return chosen


def _run_cmd(cmd: list[str], *, expect_stdout: bool = True) -> str:
    """
    Run a system command and return stdout.

    Raises ResumeError on failure.
    """
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise ResumeError(f"Required tool not found for command: {' '.join(cmd)}")
    except subprocess.CalledProcessError as e:
        raise ResumeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"Return code: {e.returncode}\n"
            f"stdout: {e.stdout}\n"
            f"stderr: {e.stderr}"
        )

    if expect_stdout:
        return result.stdout
    return ""


def extract_text_from_file(path: Path) -> str:
    """
    Convert the given resume file to plain text based on extension.
    """
    ext = path.suffix.lower()
    _debug(f"Extracting text from resume: {path} (ext={ext})")

    if ext == ".txt":
        # Plain text: just read
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback if encoding is weird
            return path.read_text(encoding="latin-1")

    # Use a temporary directory when tools expect an output file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        out_file = tmpdir_path / "resume.txt"

        if ext == ".docx":
            # docx2txt infile outfile
            cmd = ["docx2txt", str(path), str(out_file)]
            _run_cmd(cmd, expect_stdout=False)
            return out_file.read_text(encoding="utf-8", errors="ignore")

        elif ext == ".pdf":
            # pdftotext infile outfile
            cmd = ["pdftotext", str(path), str(out_file)]
            _run_cmd(cmd, expect_stdout=False)
            return out_file.read_text(encoding="utf-8", errors="ignore")

        elif ext == ".rtf":
            # unrtf --text infile → stdout (plain text with some headers)
            cmd = ["unrtf", "--text", str(path)]
            raw = _run_cmd(cmd, expect_stdout=True)
            # unrtf usually outputs some headers; we can keep it simple for now
            return raw

        elif ext == ".doc":
            # antiword infile → stdout
            cmd = ["antiword", str(path)]
            raw = _run_cmd(cmd, expect_stdout=True)
            return raw

        elif ext == ".odt":
            # odt2txt infile → stdout
            cmd = ["odt2txt", str(path)]
            raw = _run_cmd(cmd, expect_stdout=True)
            return raw

        else:
            # Should never reach here because we filter extensions earlier
            raise ResumeError(f"Unsupported resume extension: {ext}")


def load_base_resume(job_dir: str | Path) -> str:
    """
    Main entrypoint for FiverrMachine.

    - If base_resume.txt already exists, read and return it.
    - Otherwise:
        - Find a resume file in the job_dir (any supported format)
        - Convert to plain text
        - Save as base_resume.txt in job_dir
        - Return the text
    """
    job_dir_path = Path(job_dir).resolve()
    base_resume_path = job_dir_path / "base_resume.txt"

    # 1) If base_resume.txt already exists, respect it and just load it
    if base_resume_path.exists():
        _debug(f"Using existing base_resume.txt at {base_resume_path}")
        try:
            return base_resume_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return base_resume_path.read_text(encoding="latin-1")

    # 2) Find a resume file
    resume_file = find_resume_file(job_dir_path)
    if resume_file is None:
        raise FileNotFoundError(
            f"No resume file found in {job_dir_path}. "
            f"Expected something like 'resume.docx', 'base_resume.pdf', etc."
        )

    # 3) Convert to text
    text = extract_text_from_file(resume_file)

    # Basic cleanup
    cleaned = text.replace('\r\n', '\n').replace('\r', '\n').strip()

    # 4) Save as base_resume.txt for downstream steps
    base_resume_path.write_text(cleaned, encoding="utf-8")
    _debug(f"Wrote normalized base_resume.txt to {base_resume_path}")

    return cleaned
