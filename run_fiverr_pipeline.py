#!/usr/bin/env python3
import subprocess
import sys
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
VENV_PY = BASE / "venv" / "bin" / "python3"

INTAKE = BASE / "INTAKE"
PROCESSING = BASE / "PROCESSING"
DELIVERABLES = BASE / "DELIVERABLES"

def run_step(label, argv):
    try:
        completed = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return {
            "step": label,
            "ok": True,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.CalledProcessError as e:
        return {
            "step": label,
            "ok": False,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "returncode": e.returncode,
        }

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Usage: run_fiverr_pipeline.py JOB_FOLDER_NAME"}))
        sys.exit(1)

    job_folder_name = sys.argv[1]

    # We assume you've dropped a job folder into INTAKE/JOB_xxx
    intake_job_dir = INTAKE / job_folder_name
    processing_job_dir = PROCESSING / job_folder_name

    if not intake_job_dir.exists() and not processing_job_dir.exists():
        print(json.dumps({
            "ok": False,
            "error": f"Job folder {job_folder_name} not found in INTAKE or PROCESSING"
        }))
        sys.exit(1)

    # If it's still in INTAKE, move it to PROCESSING (mimics what watcher does)
    if intake_job_dir.exists() and not processing_job_dir.exists():
        processing_job_dir.parent.mkdir(parents=True, exist_ok=True)
        intake_job_dir.rename(processing_job_dir)

    steps = []

    # 1) Scrape job description (Playwright, etc.)
    steps.append(run_step(
        "job_scraper",
        [str(VENV_PY), str(BASE / "job_scraper.py"), str(processing_job_dir)]
    ))

    # 2) Build resume + cover letters with your Qwen pipeline
    steps.append(run_step(
        "packet_builder_qwen",
        [str(VENV_PY), str(BASE / "packet_builder_qwen.py"), str(processing_job_dir)]
    ))

    # 3) Build the Job Search Report
    steps.append(run_step(
        "report_builder",
        [str(VENV_PY), str(BASE / "report_builder.py"), str(processing_job_dir)]
    ))

    # 4) Generate PDFs (resume + cover letters) inside the job folder
    steps.append(run_step(
        "generate_pdfs",
        [str(VENV_PY), str(BASE / "generate_pdfs.py"), job_folder_name]
    ))

    # 5) Package deliverables (zip, scoring json, PDFs, etc.)
    steps.append(run_step(
        "deliverable_packager",
        [str(VENV_PY), str(BASE / "deliverable_packager.py"), str(processing_job_dir)]
    ))
    # 5) Career Insights Letter
    steps.append(run_step(
        "career_insights",
        [str(VENV_PY), str(BASE / "career_insights.py"), str(processing_job_dir)]
    ))

    # (Optional, later) 4) Upload to Google Drive, notify phone
    # steps.append(run_step(
    #     "google_drive_uploader",
    #     [str(VENV_PY), str(BASE / "google_drive_uploader.py"), str(processing_job_dir)]
    # ))

    ok = all(s["ok"] for s in steps)

    print(json.dumps({
        "ok": ok,
        "job_folder": job_folder_name,
        "processing_dir": str(processing_job_dir),
        "deliverables_dir": str(DELIVERABLES),
        "report_file": str(processing_job_dir / "Job_Search_Report.md"),
        "steps": steps,
    }))

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
