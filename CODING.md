# FiverrMachine quickstart

## Running a full client job
1. Place a resume (PDF or DOCX) and optional `client_request.json` into `PROCESSING/JOB_ID/`.
2. Run the pipeline from repo root:
   ```bash
   python3 run_fiverr_pipeline.py JOB_ID
   ```
3. PDFs and artifacts will be written back to `PROCESSING/JOB_ID/` (tailored resumes, cover letters, `final_report.pdf`, and `selection_summary.json`).

## Key configuration knobs
Configuration lives in `config.py` (override with environment variables):
- `FIVERR_MAX_JOBS_TOTAL` – cap total jobs collected per run (default 50).
- `FIVERR_MAX_JOBS_PER_SITE` – per-engine cap to keep sources balanced (default 15).
- `FIVERR_MIN_SCORE_HIGH_FIT` – threshold for High match bucket (default 70.0).
- `FIVERR_FOCUS_SHORTLIST_SIZE` – number of roles in the AI shortlist for resumes/cover letters (default 8).

## Outputs and where to find them
- `job_description_XX.txt` / `jobs_manifest.json` – structured scrape outputs with site/location/salary metadata.
- `resume_jobXX.*` and `cover_letter_jobXX.*` – tailored application packets for the shortlist.
- `selection_summary.json` – normalized scores, match buckets, and the chosen shortlist.
- `final_report.md` / `final_report.pdf` – polished client-facing report; PDFs live alongside the job folder.
