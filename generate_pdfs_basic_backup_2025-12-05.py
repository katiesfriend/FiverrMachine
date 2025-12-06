#!/usr/bin/env python3
import os
import sys
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


BASE_DIR = "/home/mykl/webui/filesystem/FiverrMachine"
PROCESSING_DIR = os.path.join(BASE_DIR, "PROCESSING")


def draw_multiline_text(c, text, left_margin=1 * inch, top_margin=10 * inch, line_height=14):
    """
    Very simple text renderer: wraps on '\n', moves down each line.
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


def generate_pdfs_for_job_folder(job_folder_path):
    """
    Finds resume_jobXX.txt and cover_letter_jobXX.txt in a job folder
    and generates matching PDFs.
    """
    if not os.path.isdir(job_folder_path):
        raise RuntimeError(f"Job folder does not exist: {job_folder_path}")

    files = os.listdir(job_folder_path)
    resumes = sorted(f for f in files if f.startswith("resume_job") and f.endswith(".txt"))
    covers = sorted(f for f in files if f.startswith("cover_letter_job") and f.endswith(".txt"))

    if not resumes and not covers:
        print(f"[PDF] No resume/cover_letter text files found in {job_folder_path}")
        return

    print(f"[PDF] Generating PDFs in: {job_folder_path}")

    for txt_name in resumes:
        txt_path = os.path.join(job_folder_path, txt_name)
        pdf_name = txt_name.replace(".txt", ".pdf")
        pdf_path = os.path.join(job_folder_path, pdf_name)
        print(f"[PDF] Resume -> {pdf_name}")
        render_text_file_to_pdf(txt_path, pdf_path, title="Tailored Resume")

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
