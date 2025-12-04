# modules/career_insights/career_insights.py
"""
Career Insights Module
Generates a personalized career roadmap letter for the client.

Inputs:
    resume_json (dict)
    job_matches (list of dict)
    demographics (dict) – optional
    zipcode (str) – optional

Output:
    dict with the following keys:
        'insights_letter' (str)

This module is intentionally pure/isolated so Codex and GPT can fill in
the content without side-effects.
"""

import json

def generate_career_insights_letter(
    resume_json: dict,
    job_matches: list,
    demographics: dict = None,
    zipcode: str = None
) -> dict:
    """
    Generate a personalized career insights letter.

    The AI model (OpenAI / local Codex equivalent) will fill the content.
    This function simply prepares the structured prompt and returns
    the result in a clean JSON structure.
    """

    # ---- Extract useful info for the prompt ----
    name = resume_json.get("name", "the client")
    current_title = resume_json.get("current_title", "")
    skills = resume_json.get("skills", [])
    experience = resume_json.get("experience", [])
    education = resume_json.get("education", [])

    top_jobs = job_matches[:5] if job_matches else []

    # Build a data block for the AI prompt
    data_block = {
        "client_name": name,
        "current_title": current_title,
        "skills": skills,
        "experience": experience,
        "education": education,
        "top_job_matches": top_jobs,
        "zipcode": zipcode,
        "demographics": demographics,
    }

    # ---- Prompt Template (Codex/GPT reads this) ----
    prompt = f"""
    You are a senior career strategist writing a personalized
    career roadmap for the client.

    Here is the client data (JSON):
    {json.dumps(data_block, indent=2)}

    Write a detailed career insights letter with these qualities:

    - Warm, confident, and motivating.
    - Explain how their current skills map to local job demand.
    - Highlight strengths you see in their resume.
    - For each of the top job matches:
        • How well they fit
        • What to emphasize in interviews
        • What skill gaps to close
    - Provide 2–4 specific certifications that would raise salary potential.
    - Mention realistic salary increases tied to those certifications.
    - If demographic data shows weak demand in their ZIP:
        • Recommend nearby ZIP codes with stronger markets.
    - End with a clear, encouraging next-steps section.

    Do NOT reference the prompt or instructions. Speak directly to the client.
    """

    # ---- Call to AI Model (placeholder, you wire this) ----
    from engines.ai_model import run_ai
    result = run_ai(prompt)

    return {
        "insights_letter": result,
        "prompt_used": prompt
    }
