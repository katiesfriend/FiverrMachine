#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


def _strip_bold(line: str) -> str:
    # Remove common markdown emphasis markers that leak into LaTeX output
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
    line = re.sub(r"__(.+?)__", r"\1", line)
    line = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\1", line)
    line = re.sub(r"(^|[^\w])_(?!\s)([^_]+?)(?<!\s)_([^\w]|$)", r"\1\2\3", line)
    return line


def _ws_len(s: str) -> int:
    return len(s) - len(s.lstrip(" \t"))


def sanitize_markdown(text: str) -> str:
    """
    Fixes two issues that show up in final_report.pdf:
      1) literal **bold** markers leaking into LaTeX-rendered output
      2) ordered lists that render as 1/1/1 when markdown's "all 1." trick is used,
         especially when there are nested bullets between items.

    Strategy:
      - Strip bold markers everywhere.
      - Renumber top-level ordered list items explicitly while keeping the list "active"
        across indented sub-lines (nested bullets, wrapped lines, blank lines).
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []

    list_active = False
    base_indent_len = 0
    counter = 1

    ordered_re = re.compile(r"^([ \t]*)(\d+)\.\s+(.*)$")

    for raw in lines:
        line = _strip_bold(raw)

        m = ordered_re.match(line)
        if m:
            indent = m.group(1)
            ilen = _ws_len(indent)
            body = m.group(3)

            # Start or continue a list at this indent level
            if not list_active:
                list_active = True
                base_indent_len = ilen
                counter = 1
            elif ilen != base_indent_len:
                # New list at a different indent: restart numbering for that level
                base_indent_len = ilen
                counter = 1

            newline = "\n" if line.endswith("\n") else ""
            out.append(f"{indent}{counter}. {body}{newline}")
            counter += 1
            continue

        if list_active:
            # Keep list active across blank lines or indented subcontent (nested bullets)
            if line.strip() == "":
                out.append(line)
                continue

            ilen = _ws_len(line)
            if ilen > base_indent_len:
                out.append(line)
                continue

            # A non-indented, non-ordered line ends the list
            list_active = False

        out.append(line)

    return "".join(out)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: sanitize_report_md.py /path/to/final_report.md", file=sys.stderr)
        return 2

    md_path = Path(sys.argv[1]).resolve()
    if not md_path.exists():
        print(f"[SANITIZE] ERROR: file not found: {md_path}", file=sys.stderr)
        return 2

    original = md_path.read_text(encoding="utf-8", errors="replace")
    cleaned = sanitize_markdown(original)

    if cleaned != original:
        md_path.write_text(cleaned, encoding="utf-8")
        print("[SANITIZE] Updated final_report.md")
    else:
        print("[SANITIZE] No changes needed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
