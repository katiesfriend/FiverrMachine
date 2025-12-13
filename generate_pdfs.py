#!/usr/bin/env python3
import json
import os
import re
import sys
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# Try to import the LaTeX resume renderer (optional)
try:
    from latex_resume_builder import render_plaintext_resume_to_pdf
except Exception:
    render_plaintext_resume_to_pdf = None


BASE_DIR = "/home/mykl/webui/filesystem/FiverrMachine"
PROCESSING_DIR = os.path.join(BASE_DIR, "PROCESSING")


def draw_multiline_text(c, text, left_margin=1 * inch, top_margin=10 * inch, line_height=14):
    """
    Very simple text renderer: wraps on '\\n', moves down each line.
    No fancy layout yet; this keeps things robust.
    """
    lines = text.splitlines()
    x = left_margin
    y = top_margin

    for line in lines:
        # If we run off the page, start a new one
        if y < 1 * inch:
            c.showPage()
            y = top_margin

        c.drawString(x, y, line)
        y -= line_height


def render_text_file_to_pdf(txt_path, pdf_path, title=None):
    """
    Basic ReportLab-based PDF generator used for:
      - Fallback if LaTeX is missing / fails
      - Cover letters (for now)
    """
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    c = canvas.Canvas(pdf_path, pagesize=LETTER)
    width, height = LETTER

    # Basic header
    if title:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1 * inch, 10.5 * inch, title)
        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, 10.2 * inch, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    else:
        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, 10.5 * inch, f"Generated: {datetime.now().isoformat(timespec='seconds')}")

    c.setFont("Helvetica", 11)
    # Start content a bit below header
    draw_multiline_text(c, content, left_margin=1 * inch, top_margin=9.7 * inch, line_height=14)

    c.showPage()
    c.save()


def _label_from_name(name: str):
    match = re.search(r"(\d+)", name)
    if not match:
        return None
    try:
        return f"{int(match.group(1)):02d}"
    except ValueError:
        return None


def _selected_labels_from_summary(job_folder_path: str):
    summary_path = os.path.join(job_folder_path, "selection_summary.json")
    if not os.path.isfile(summary_path):
        return []

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    labels = []
    for raw in data.get("selected_labels", []):
        norm = _label_from_name(str(raw))
        if norm:
            labels.append(norm)

    return sorted(set(labels))


def _labels_from_job_descriptions(job_folder_path: str):
    labels = []
    for name in os.listdir(job_folder_path):
        if name.startswith("job_description_") and name.endswith(".txt"):
            norm = _label_from_name(name)
            if norm:
                labels.append(norm)
    return sorted(set(labels))


def generate_pdfs_for_job_folder(job_folder_path):
    """
    Finds resume_jobXX.txt and cover_letter_jobXX.txt in a job folder
    and generates matching PDFs.

    Resumes:
        - Prefer LaTeX (modern layout)
        - Fallback to basic ReportLab if LaTeX fails or is unavailable

    Cover letters:
        - Use basic ReportLab renderer for now
    """
    if not os.path.isdir(job_folder_path):
        raise RuntimeError(f"Job folder does not exist: {job_folder_path}")

    files = os.listdir(job_folder_path)
    resumes = sorted(f for f in files if f.startswith("resume_job") and f.endswith(".txt"))
    covers = sorted(f for f in files if f.startswith("cover_letter_job") and f.endswith(".txt"))

    allowed_labels = _selected_labels_from_summary(job_folder_path)
    if not allowed_labels:
        allowed_labels = _labels_from_job_descriptions(job_folder_path)
    if not allowed_labels:
        detected = []
        for name in resumes + covers:
            norm = _label_from_name(name)
            if norm:
                detected.append(norm)
        allowed_labels = sorted(set(detected))

    if allowed_labels:
        allowed_set = set(allowed_labels)
        resumes = [f for f in resumes if _label_from_name(f) in allowed_set]
        covers = [f for f in covers if _label_from_name(f) in allowed_set]

    if not resumes and not covers:
        print(f"[PDF] No resume/cover_letter text files found in {job_folder_path}")
        return

    print(f"[PDF] Generating PDFs in: {job_folder_path}")

    # ----- Resumes (LaTeX preferred) -----
    for txt_name in resumes:
        txt_path = os.path.join(job_folder_path, txt_name)
        pdf_name = txt_name.replace(".txt", ".pdf")
        pdf_path = os.path.join(job_folder_path, pdf_name)

        # If we have the LaTeX builder, try that first
        if render_plaintext_resume_to_pdf is not None:
            try:
                render_plaintext_resume_to_pdf(txt_path, pdf_path, doc_title="Tailored Resume")
                print(f"[PDF] Resume (LaTeX) -> {pdf_name}")
                continue  # Done with this resume
            except RuntimeError as e:
                # Log failure, fall back to basic renderer
                print(f"[PDF] LaTeX resume generation failed for {pdf_name}: {e}. Falling back to basic renderer.")

        # Fallback: simple ReportLab renderer
        print(f"[PDF] Resume (basic) -> {pdf_name}")
        render_text_file_to_pdf(txt_path, pdf_path, title="Tailored Resume")

    # ----- Cover letters (basic for now) -----
    for txt_name in covers:
        txt_path = os.path.join(job_folder_path, txt_name)
        pdf_name = txt_name.replace(".txt", ".pdf")
        pdf_path = os.path.join(job_folder_path, pdf_name)
        print(f"[PDF] Cover Letter -> {pdf_name}")
        render_text_file_to_pdf(txt_path, pdf_path, title="Tailored Cover Letter")

    print("[PDF] Done.")


def main():
    if len(sys.argv) != 2:
        print("Usage: ./generate_pdfs.py JOB_xxxxxxx")
        sys.exit(1)

    job_id = sys.argv[1]

    # For now, assume we're working in PROCESSING; you can adjust if you prefer DELIVERABLES
    job_folder_path = os.path.join(PROCESSING_DIR, job_id)

    generate_pdfs_for_job_folder(job_folder_path)


if __name__ == "__main__":
    main()
