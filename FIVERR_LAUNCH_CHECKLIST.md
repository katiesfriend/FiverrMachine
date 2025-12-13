Stage A – Intake & context
Inputs:
    • Client DOCX/PDF resume → normalized to base_resume.txt
    • Optional client_request.json with:
        ◦ target title(s)
        ◦ location (city/state/ZIP) and radius
        ◦ remote OK? (bool)
        ◦ desired job volume (e.g. 20)
Outputs:
    • PROCESSING/JOB_X/ containing:
        ◦ base_resume.txt
        ◦ resume_ai_meta.json with:
            ▪ inferred skills (clusters)
            ▪ inferred target roles
            ▪ inferred location (from resume if not provided)
            ▪ seniority level estimate

Stage B – Job search (multi-board, local-biased)
Using resume_ai_meta + client request:
    • Query multiple boards (at least):
        ◦ LinkedIn Jobs
        ◦ Indeed
        ◦ ZipRecruiter
        ◦ USAJobs (for .gov roles)
        ◦ (optional later: Glassdoor / Monster if easy)
    • Respect:
        ◦ location + radius
        ◦ remote flag
        ◦ max total jobs (e.g. 60 across boards)
Behavior:
    • Each board has a dedicated function: search_linkedin(...), search_indeed(...), etc.
    • All results normalized into a single JobPosting dataclass:
        ◦ id (per board)
        ◦ title
        ◦ company
        ◦ city, state
        ◦ URL
        ◦ salary info if present
        ◦ board name
        ◦ raw description text
    • Logs clearly say, per board:
        ◦ search query
        ◦ location
        ◦ count returned
        ◦ reason if 0 (no results vs error vs blocked)
Outputs:
    • job_description_01.txt … job_description_N.txt
    • job_sources.txt summarizing boards used + counts

Stage C – Scoring, selection, insights
For each job:
    • Compute:
        ◦ scraper match score (0–? scale you already use)
        ◦ AI match score (0–100) using local or API model
    • Assign bucket: High / Medium / Low / Very Low.
Select:
    • Focus shortlist (e.g. top 8 by AI score).
    • Full ranked list of all jobs.
Generate insights:
    • Salary band:
        ◦ From postings that include explicit salary → min, max, median/central.
    • Skill demand:
        ◦ Count how often job descriptions mention key skill groups (Automation/Scripting, Data/Analytics, Cloud, Stakeholder, Vendor/Customer, etc.).
        ◦ Save an image career_insights_skills_pie.png.
Outputs:
    • score_jobXX.json for each role
    • selection_summary.json (focus jobs, salary band, stats)
    • career_insights_skills_pie.png

Stage D – Tailored resumes & cover letters
For each selected job (you’re already doing this, we just define the goal):
    • Build:
        ◦ resume_jobXX.pdf + .tex + .txt
        ◦ cover_letter_jobXX.pdf + .tex + .txt
    • Each resume/CL:
        ◦ Maintains consistent layout and styling.
        ◦ Is clearly customized to that job’s title, company, and key requirements.
        ◦ Avoids obviously spammy repetition.

Stage E – Report & packaging
    1. Report markdown (final_report.md) with sections:
        ◦ Job Search Report for Client (short intro)
        ◦ Summary at a Glance
            ▪ by match bucket
            ▪ by source site
        ◦ Focus Roles (shortlist table)
        ◦ Market Compensation Signal
        ◦ What This Report Is / Isn’t (short)
        ◦ Job List (Best Matches First) – compact table
        ◦ Skill Demand Snapshot
            ▪ embedded pie chart
            ▪ 2–3 bullets explaining what to learn next
       Tables:
        ◦ Shortlist:
            ▪ Job | Match | Title | Company | City, ST
        ◦ Full list:
            ▪ # | Title | Company | City, ST | Site | Match bucket
       URLs stay as [Open posting](…) text, not huge displayed URLs.
    2. LaTeX → PDF via latex_report_builder.py:
        ◦ No table runs off the page.
        ◦ List numbering correct (no “1. 1. 1.” glitches).
        ◦ Chart image inserted with caption.
    3. Deliverable packager creates a ZIP in DELIVERABLES/ named like:
        ◦ JOB_James_full_packet.zip
       Containing:
        ◦ The PDF report
        ◦ All tailored resumes & cover letters
        ◦ A README.txt (how to use this packet)

2. Definition of Done (for v1.0 / Fiverr launch)
You’re allowed to list this on Fiverr when ALL of this is true:
Pipeline reliability
    1. run_fiverr_pipeline.py JOB_Example completes without errors on at least:
        ◦ one IT Desktop Support resume (James)
        ◦ one totally different role (e.g. logistics / PM)
and produces a ZIP each time.
    2. pytest passes, including new tests we’ll add for:
        ◦ multi-board scraping
        ◦ report structure / markdown
        ◦ LaTeX conversion with image + tables.

Job search quality
    3. For a typical support-tech resume with local Indy-ish address:
        ◦ At least 20 jobs total, from ≥ 3 different boards.
        ◦ At least 7 High bucket jobs (or we log clearly why not).
        ◦ Logs show each board’s query, location, counts.
    4. selection_summary.json correctly reports:
        ◦ number of jobs per bucket
        ◦ number per source site
        ◦ salary band (min / max / “center”).

Report & documents
    5. final_report.pdf:
        ◦ 4–5 pages max.
        ◦ No table bleeding off page.
        ◦ Pie chart visible and readable.
        ◦ “How to Use This Report” is at most:
            ▪ 4 numbered steps
            ▪ each with 1–2 bullet subpoints.
    6. At least one sample job’s:
        ◦ resume_jobXX.pdf
        ◦ cover_letter_jobXX.pdf
       …pass your “would I be proud to send this as my resume?” gut check.

UX & Fiverr readiness
    7. Simple usage instructions documented in a small README.md:
        ◦ how to set up venv
        ◦ how to run a job
        ◦ where to find outputs
        ◦ where to tweak config (location, radius, job volume).
    8. You have:
        ◦ One before/after screenshot set for a sample client (for the Fiverr gallery).
        ◦ At least one real pipeline run zipped and stored as a “portfolio example”.

Safety / backups / GitHub
    9. Code is safely stored:
        ◦ Local backup ZIP
        ◦ GitHub repo with a main (stable) and dev branch.
    10. You can clone fresh on another folder and run the pipeline successfully following your own README.

