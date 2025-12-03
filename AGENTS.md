# FiverrMachine – Agent Instructions

## Purpose

FiverrMachine takes an intake job (client resume + job description), generates tailored resumes and cover letters, scores them, and outputs deliverables for the client.

## Code layout (Dec 2025)

- `watcher.py` – Watches INTAKE/INCOMING folders, assigns files, creates job folders under PROCESSING.
- `run_fiverr_pipeline.py` – Orchestrates the full job pipeline (scraper, builder, scorer, etc.).
- `SCRAPER/` – Job scraping logic (reads job description, maybe web scraping).
- `BUILDER/` – Resume and cover-letter generation / transformation.
- `SCORER/` – Scoring of packets (0–100) and routing based on score.
- `DELIVERABLES/` – Final client-facing output.

*(Update this list as we refine things.)*

## How to run the pipeline locally

From repo root:

```bash
python3 run_fiverr_pipeline.py JOB_TEST123
