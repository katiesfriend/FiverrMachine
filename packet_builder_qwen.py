#!/usr/bin/env python3
"""
packet_builder_qwen.py

Builds tailored resume + cover letter packets for each job description in a job folder.

Inputs (inside JOB_DIR):
- base_resume.txt          : base resume text (already extracted from your .docx)
- job_description_*.txt    : one or more job description files
- client_request.json      : small JSON with client preferences (optional)

Outputs (inside JOB_DIR):
- resume_jobXX.txt
- cover_letter_jobXX.txt
- score_jobXX.json         : includes match_score, model info, notes, polishing flags

Heavy lifting is done by Ollama with model qwen2.5:14b.
Optional polishing is done by OpenAI if OPENAI_API_KEY is set.
"""

import json
import os
import sys
import glob
import textwrap
from typing import Dict, Any, Tuple, Optional, List
from resume_loader import load_base_resume as _load_base_resume
from datetime import datetime
from config import DEFAULT_CONFIG, PipelineConfig

import requests
import re
from collections import Counter
from pathlib import Path

def build_intake_hint_for_prompt(client_request: dict) -> str:
    """
    Turn the client_request JSON for a Fiverr job into a plain text block
    that can be appended to the job description to give the model more
    context. This is defensive and works even if the JSON schema changes.
    """
    if not isinstance(client_request, dict):
        return ""

    meta = client_request.get("meta") or {}
    source = meta.get("source")

    lines: list[str] = []
    lines.append("FIVERR CLIENT INTAKE DATA")
    lines.append("This job came from a Fiverr order. The following fields")
    lines.append("summarize what the client filled out in the intake form.")
    lines.append("")

    if source:
        lines.append(f"source: {source}")
        lines.append("")

    # High level fields we expect to have in the intake
    interesting_keys = [
        "job_search_goal",
        "target_title",
        "target_industry",
        "role_focus",
        "salary_target",
        "location_preferences",
        "remote_preferences",
        "commute_limit",
        "must_have_skills",
        "nice_to_have_skills",
        "tools_and_tech",
        "certifications",
        "experiences_to_highlight",
        "achievements_to_highlight",
        "company_preferences",
        "job_search_constraints",
        "tone_preferences",
        "notes_for_writer",
    ]

    for key in interesting_keys:
        value = client_request.get(key)
        if value:
            label = key.replace("_", " ").title()
            lines.append(f"{label}:")
            lines.append(str(value))
            lines.append("")

    # Shallow dump of any extra user level fields
    skip_keys = set(interesting_keys + ["meta"])
    extra_keys = sorted(k for k in client_request.keys() if k not in skip_keys)

    if extra_keys:
        lines.append("Additional intake fields:")
        for key in extra_keys:
            value = client_request.get(key)
            if value:
                lines.append(f"{key}: {value}")

    return "\n".join(lines).strip()


# -----------------------------
# Configuration
# -----------------------------

# Ollama config
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("FIVERR_QWEN_MODEL", "qwen2.5:14b")

# OpenAI models:
# - Draft: GPT-4.1-mini (cheap, good structure)
# - Polish: GPT-5.1 (expensive, high quality)
OPENAI_DRAFT_MODEL = os.environ.get("FIVERR_OPENAI_DRAFT_MODEL", "gpt-4.1-mini")
OPENAI_MODEL = os.environ.get("FIVERR_OPENAI_MODEL", "gpt-5.1")

# Safety: we keep temperature modest so outputs are stable
QWEN_TEMPERATURE = float(os.environ.get("FIVERR_QWEN_TEMPERATURE", "0.4"))

# Match Score thresholds (configurable via env, with ATS fallbacks for backwards compatibility)
TARGET_MATCH_SCORE = float(
    os.environ.get(
        "FIVERR_TARGET_MATCH_SCORE",
        os.environ.get("FIVERR_ATS_TARGET", "88.0"),
    )
)
LOW_FIT_FLOOR = float(
    os.environ.get(
        "FIVERR_LOW_FIT_FLOOR",
        os.environ.get("FIVERR_ATS_LOW_FIT", "40.0"),
    )
)

def _strip_markdown_fences(text: str) -> str:
    """
    If the model wraps JSON in ``` or ```json fences, strip them
    so json.loads() can work on the inner content.
    """
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if not lines:
        return text

    first_line = lines[0]
    if not first_line.startswith("```"):
        return text

    closing_index = None
    for i in range(len(lines) - 1, 0, -1):
        if lines[i].startswith("```"):
            closing_index = i
            break

    if closing_index is None or closing_index <= 0:
        return text

    inner = "\n".join(lines[1:closing_index])
    return inner.strip()

# -----------------------------
# Helpers: file loading
# -----------------------------
def scrub_unverified_certifications(base_resume: str, resume_text: str) -> str:
    """
    Remove certification lines from the final resume if they are not
    literally present in the base resume text.

    Assumes a section labeled 'Certifications' followed by bullet lines.
    Any bullet whose text is not found (case-insensitive) in base_resume
    will be dropped.
    """
    base_lower = base_resume.lower()
    lines = resume_text.splitlines()
    out_lines: List[str] = []
    in_certs = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # Detect start of Certifications section
        if lower.startswith("certifications"):
            in_certs = True
            out_lines.append(line)
            continue

        if in_certs:
            # If we hit a blank line, or an obvious new section heading,
            # end the Certifications section.
            if stripped == "" or stripped.lower().endswith("summary") or stripped.lower().endswith("experience"):
                out_lines.append(line)
                in_certs = False
                continue

            # Handle bullet vs plain text lines
            content = stripped
            if stripped.startswith(("-", "•", "*")):
                content = stripped.lstrip("-•*").strip()

            # If the bullet text doesn't appear in the base resume, drop it.
            if content and content.lower() not in base_lower:
                continue

            # Otherwise keep it
            out_lines.append(line)
            continue

        # Outside Certifications section, just pass lines through
        out_lines.append(line)

    return "\n".join(out_lines)
def extract_education_block(text: str) -> Optional[str]:
    """
    Pull out the Education section from the base resume text.

    We look for a line starting with 'Education' and then collect
    all following lines up until the next obvious section heading
    (e.g., 'Certifications', 'Notable Achievements', etc.).
    """
    lines = text.splitlines()
    out: List[str] = []
    in_edu = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if not in_edu:
            if lower.startswith("education"):
                in_edu = True
                out.append(line)
            continue

        # Once we're inside Education, stop when we hit the next section.
        if (
            lower.startswith("certifications")
            or lower.startswith("notable achievements")
            or lower.endswith("summary")
            or lower.endswith("experience")
        ):
            break

        out.append(line)

    return "\n".join(out) if out else None


def force_education_from_base(base_resume: str, resume_text: str) -> str:
    """
    Replace the Education section in the generated resume with the exact
    Education block from the base resume. If no Education section exists
    in the generated resume, we append the base Education block at the end.
    """
    edu_block = extract_education_block(base_resume)
    if not edu_block:
        return resume_text  # nothing to enforce

    lines = resume_text.splitlines()
    out: List[str] = []
    in_edu = False
    replaced = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if not in_edu:
            if lower.startswith("education"):
                # Start of the existing Education section: replace it once.
                in_edu = True
                if not replaced:
                    out.append(edu_block)
                    replaced = True
                # Skip the original 'Education' line and its content;
                # they'll be replaced by edu_block.
                continue
            else:
                out.append(line)
        else:
            # We're in the old Education section. Skip its lines until the next section.
            if (
                lower.startswith("certifications")
                or lower.startswith("notable achievements")
                or lower.endswith("summary")
                or lower.endswith("experience")
            ):
                in_edu = False
                out.append(line)  # keep the next section heading
            # otherwise, swallow lines inside old Education section

    # If we never found an Education heading in the generated resume,
    # append the base Education block at the end.
    if not replaced:
        out.append("")
        out.append(edu_block)

    return "\n".join(out)

def load_base_resume(job_dir: str) -> str:
    """
    Compatibility wrapper that delegates to resume_loader.load_base_resume.
    This lets us support many resume formats without changing the rest
    of the pipeline.
    """
    return _load_base_resume(job_dir)


def load_client_meta(job_dir: str) -> Dict[str, Any]:
    path = os.path.join(job_dir, "client_request.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        # if malformed, just ignore; pipeline should still run
        return {}


def load_job_descriptions(job_dir: str):
    pattern = os.path.join(job_dir, "job_description_*.txt")
    files = sorted(glob.glob(pattern))
    return files


# -----------------------------
# Core: call Qwen via Ollama
# -----------------------------

def call_qwen_draft(base_resume: str, job_desc: str, client_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Draft tailored resume + cover letter using OpenAI GPT-4.1-mini
    instead of Qwen/Ollama.

    Returns a dict with:
      - resume
      - cover_letter
      - match_score
      - notes
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot generate resume draft.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package is not installed in this environment.") from e

    client = OpenAI(api_key=api_key)

    system_prompt = textwrap.dedent(
        """
        You are an expert resume and cover letter writer who specializes
        in honest, high-conversion applications.

        Your job:

        - Use ONLY facts present in the base resume and client metadata.
        - NEVER invent or upgrade:
            * employers
            * job titles
            * dates
            * degrees
            * certifications
            * tools / technologies
        - Use the job description ONLY to:
            * choose which existing facts to highlight
            * mirror relevant terminology and focus
        - Do NOT copy or restate the job posting as the resume.
        - Do NOT include salary, posting dates, or sections like
          "Responsibilities" / "Qualifications" copied from the job ad.

        OUTPUT FORMAT (STRICT):

        Respond with a SINGLE valid JSON object with EXACTLY these keys:

        - "resume"        (string, full candidate resume)
        - "cover_letter"  (string, full cover letter)
        - "match_score"   (number 0–100)
        - "notes"         (string, brief explanation)

        No markdown, no code fences, no extra commentary outside the JSON.
        """
    ).strip()

    user_prompt = textwrap.dedent(
        f"""
        CLIENT METADATA (JSON):
        {json.dumps(client_meta, indent=2)}

        BASE RESUME (CANDIDATE FACTS):
        {base_resume}

        JOB DESCRIPTION (TARGET ROLE):
        {job_desc}

        Please:
        1) Tailor the RESUME to this job using ONLY candidate-supported facts.
        2) Write a COVER LETTER that sells the candidate for this role.
        3) Estimate a numeric MATCH_SCORE between 0 and 100.
        4) Explain key tailoring decisions in NOTES.
        """
    ).strip()

    completion = client.chat.completions.create(
        model=OPENAI_DRAFT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=QWEN_TEMPERATURE,
    )

    content = completion.choices[0].message.content or ""
    content = _strip_markdown_fences(content.strip())

    # Try to parse JSON
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        decoded = {}

    if not isinstance(decoded, dict):
        decoded = {}

    resume = str(decoded.get("resume", "")).strip()
    cover_letter = str(decoded.get("cover_letter", "")).strip()
    raw_score = decoded.get("match_score", decoded.get("ats_score", 80))
    try:
        raw_score = float(raw_score)
    except (TypeError, ValueError):
        raw_score = 80.0
    raw_score = max(0.0, min(100.0, raw_score))
    notes = str(decoded.get("notes", "")).strip()

    # Safety fallback: never dump a job description blob as "the resume".
    if not resume:
        resume = base_resume
        if notes:
            notes += "\n"
        notes += "Draft step: model output missing/invalid 'resume'; fell back to base resume text."

    if not cover_letter:
        cover_letter = (
            "Cover letter could not be generated automatically; "
            "please draft manually based on the resume and job description."
        )

    return {
        "resume": resume,
        "cover_letter": cover_letter,
        "match_score": raw_score,
        "ats_score": raw_score,
        "notes": notes,
    }

def call_qwen_revision(
    base_resume: str,
    job_desc: str,
    client_meta: Dict[str, Any],
    current_resume: str,
    current_cover: str,
    current_match_score: float,
) -> Dict[str, Any]:
    """
    Ask GPT-4.1-mini to revise an existing resume + cover letter to better
    match the job description, without inventing any new facts.

    Returns the same JSON structure as call_qwen_draft().
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot generate revision.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package is not installed in this environment.") from e

    client = OpenAI(api_key=api_key)

    system_prompt = textwrap.dedent(
        """
        You are revising an existing resume and cover letter to improve
        their match to a specific job description and increase the Match Score.

        RULES:

        - You may NOT invent or upgrade any facts about the candidate.
        - You must stay within the employers, titles, dates, degrees,
          certifications, and tools present in the BASE RESUME / metadata.
        - Use the job description to improve keyword alignment and phrasing,
          not to fabricate new experience.
        - Do NOT copy long sections of the job ad into the resume verbatim.

        OUTPUT FORMAT (STRICT):

        Respond with a SINGLE valid JSON object with EXACTLY these keys:
        - "resume"        (string, full revised resume text)
        - "cover_letter"  (string, full revised cover letter text)
        - "match_score"   (number 0–100)
        - "notes"         (string, brief explanation of changes)
        """
    ).strip()

    user_prompt = textwrap.dedent(
        f"""
        CLIENT METADATA (JSON):
        {json.dumps(client_meta, indent=2)}

        BASE RESUME (CANDIDATE FACTS):
        {base_resume}

        JOB DESCRIPTION (TARGET ROLE):
        {job_desc}

        CURRENT TAILORED RESUME:
        {current_resume}

        CURRENT COVER LETTER:
        {current_cover}

        CURRENT MATCH SCORE (HEURISTIC):
        {current_match_score}

        Please:
        1) Revise the RESUME and COVER LETTER to better fit this job,
           WITHOUT inventing any new facts.
        2) Improve keyword alignment and clarity.
        3) Estimate an updated MATCH_SCORE between 0 and 100.
        4) Explain your key changes in NOTES.
        """
    ).strip()

    completion = client.chat.completions.create(
        model=OPENAI_DRAFT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=QWEN_TEMPERATURE,
    )

    content = completion.choices[0].message.content or ""
    content = _strip_markdown_fences(content.strip())

    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        decoded = {}

    if not isinstance(decoded, dict):
        decoded = {}

    resume = str(decoded.get("resume", current_resume)).strip()
    cover_letter = str(decoded.get("cover_letter", current_cover)).strip()
    raw_score = decoded.get("match_score", decoded.get("ats_score", current_match_score))
    try:
        raw_score = float(raw_score)
    except (TypeError, ValueError):
        raw_score = current_match_score
    raw_score = max(0.0, min(100.0, raw_score))
    notes = str(decoded.get("notes", "")).strip()

    return {
        "resume": resume or current_resume,
        "cover_letter": cover_letter or current_cover,
        "match_score": raw_score,
        "ats_score": raw_score,
        "notes": notes or "Revision step: model returned no notes.",
    }


# -----------------------------
# Optional: OpenAI polishing
# -----------------------------

def maybe_polish_with_openai(text: str, kind: str) -> Tuple[str, bool]:
    """
    Optionally polish text with OpenAI if OPENAI_API_KEY is set.
    - kind: "resume" or "cover_letter"
    - Enforces: no EM dashes, only normal hyphen '-' if needed.

    Returns (polished_text, used_openai_flag).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return text, False

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return text, False

    client = OpenAI(api_key=api_key)

    system_prompt = textwrap.dedent(
        f"""
        You are a professional editor for a {kind.replace('_', ' ')}.

        TASK:
        - Preserve all factual content.
        - Improve clarity, flow, and professionalism.
        - Keep it concise and strongly targeted to the job description context.
        - Do NOT invent facts or add fake achievements.
        - Do NOT use EM DASH characters (—). Use a normal hyphen '-' instead if needed.
        - Return ONLY the final edited text, with no explanations.
        """
    ).strip()

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )
        polished = completion.choices[0].message.content or ""
        polished = polished.strip().replace("—", "-")
        if polished:
            return polished, True
        return text, False
    except Exception:
        # If OpenAI fails for any reason, just return original.
        return text, False
# -----------------------------
# Match Score (skills + title + location)
# -----------------------------

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had",
    "are", "was", "were", "will", "would", "could", "should", "can", "may",
    "might", "into", "onto", "over", "under", "above", "below", "about",
    "your", "their", "they", "them", "you", "our", "ours", "but", "not",
    "any", "all", "each", "other", "than", "then", "also", "such", "more",
    "most", "some", "many", "much", "very", "via", "per", "etc"
}


def _tokenize(text: str):
    """
    Lowercase, strip non-letters, split into words, drop short/stopwords.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9+]+", " ", text)
    tokens = []
    for tok in text.split():
        if len(tok) < 3:
            continue
        if tok in _STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def _extract_bullet_lines(text: str):
    """
    Extract lines that look like bullet points in the resume:
    starting with '-', '*', or '•'.
    """
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("-", "*", "•")):
            lines.append(stripped)
    return lines


def _dedupe_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        key = str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(str(item).strip())
    return ordered


def _extract_client_skills(client_meta: Dict[str, Any], resume_text: str):
    skills = []
    for key in ("skills", "key_skills", "core_skills", "technical_skills", "keywords"):
        val = client_meta.get(key)
        if isinstance(val, list):
            skills.extend([str(v) for v in val])
        elif isinstance(val, str):
            skills.extend([s.strip() for s in re.split(r"[,;\n]", val) if s.strip()])

    skills = _dedupe_preserve_order(skills)
    if skills:
        return skills

    # Fallback: infer top tokens from the resume (favor bullet lines)
    token_source = []
    bullet_lines = _extract_bullet_lines(resume_text)
    for line in bullet_lines:
        token_source.extend(_tokenize(line))
    if not token_source:
        token_source = _tokenize(resume_text)

    counts = Counter(token_source)
    inferred = [w for w, _ in counts.most_common(25)]
    return inferred


def _extract_target_titles(client_meta: Dict[str, Any]):
    titles = []
    for key in ("target_roles", "preferred_titles", "titles", "role"):
        val = client_meta.get(key)
        if isinstance(val, list):
            titles.extend([str(v) for v in val])
        elif isinstance(val, str):
            titles.extend([s.strip() for s in re.split(r"[,;/\n]", val) if s.strip()])
    return _dedupe_preserve_order(titles)


def _extract_job_title(jd_text: str) -> str:
    for line in jd_text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"(?i)^job\s*title\s*[:\-]\s*", "", cleaned)
        return cleaned[:120]
    return ""


def _compute_title_similarity(job_title: str, target_titles) -> float:
    if not job_title or not target_titles:
        return 0.0

    job_lower = job_title.lower()
    job_tokens = set(_tokenize(job_title))
    best = 0.0
    for target in target_titles:
        target_lower = target.lower()
        if not target_lower:
            continue
        if target_lower in job_lower:
            best = max(best, 1.0)
            continue
        target_tokens = set(_tokenize(target))
        if not target_tokens:
            continue
        overlap = job_tokens.intersection(target_tokens)
        ratio = len(overlap) / len(target_tokens)
        best = max(best, ratio)
    return best


def _skill_in_job_description(skill: str, jd_tokens, jd_text_lower: str) -> bool:
    skill_tokens = set(_tokenize(skill))
    if not skill_tokens:
        return False
    if all(tok in jd_tokens for tok in skill_tokens):
        return True
    return skill.lower() in jd_text_lower


def _extract_locations(client_meta: Dict[str, Any]):
    locations = []
    for key in (
        "location",
        "location_city",
        "location_state",
        "location_country",
        "location_zip",
        "preferred_location",
        "preferred_locations",
    ):
        val = client_meta.get(key)
        if isinstance(val, list):
            locations.extend([str(v) for v in val])
        elif isinstance(val, str):
            locations.extend([s.strip() for s in re.split(r"[,;/\n]", val) if s.strip()])
    return _dedupe_preserve_order(locations)


def _is_remote_friendly(client_meta: Dict[str, Any]) -> bool:
    pref = str(client_meta.get("remote_preference", "")).lower()
    return any(word in pref for word in ["remote", "hybrid", "either", "flexible"])


def compute_match_score(
    resume_text: str,
    jd_text: str,
    client_meta: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute a deterministic Match Score (0-100) blending:

    - Skill coverage (largest weight): percentage of client skills present in the JD.
    - Title similarity (medium weight): overlap between target titles and the JD title line.
    - Location/remote alignment (small weight): simple boosts for location or remote fit.
    """

    if not resume_text or not jd_text:
        return 0.0

    client_meta = client_meta or {}
    jd_tokens = set(_tokenize(jd_text))
    jd_text_lower = jd_text.lower()

    # Skill coverage
    skills = _extract_client_skills(client_meta, resume_text)
    skill_hits = sum(
        1 for skill in skills if _skill_in_job_description(skill, jd_tokens, jd_text_lower)
    )
    skill_ratio = skill_hits / len(skills) if skills else 0.0
    skill_score = skill_ratio * 70.0

    # Title similarity
    job_title = _extract_job_title(jd_text)
    target_titles = _extract_target_titles(client_meta)
    title_similarity = _compute_title_similarity(job_title, target_titles)
    title_score = title_similarity * 20.0

    # Location / remote sanity
    remote_in_jd = "remote" in jd_text_lower or "work from home" in jd_text_lower
    remote_pref = _is_remote_friendly(client_meta)
    location_terms = _extract_locations(client_meta)
    location_hit = any(term.lower() in jd_text_lower for term in location_terms)

    location_score = 0.0
    if remote_in_jd and remote_pref:
        location_score = 10.0
    elif location_hit:
        location_score = 8.0
    elif remote_in_jd:
        location_score = 5.0
    elif remote_pref:
        location_score = 3.0

    score = skill_score + title_score + location_score
    score = max(0.0, min(100.0, round(score, 1)))
    return score

def _clamp_score(val: float) -> float:
    try:
        return max(0.0, min(100.0, float(val)))
    except Exception:
        return 0.0


def _load_manifest(job_dir: Path) -> List[Dict[str, Any]]:
    path = Path(job_dir) / "jobs_manifest.json"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def _parse_job_description_header(desc_path: Path) -> Dict[str, str]:
    """Read the top of job_description_XX.txt to recover metadata."""
    info: Dict[str, str] = {}
    try:
        lines = desc_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return info

    for line in lines[:10]:
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        info[label.strip().lower()] = value.strip()
    return info


def _fallback_manifest(job_dir: Path) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    for desc_path in sorted(Path(job_dir).glob("job_description_*.txt")):
        name = desc_path.stem
        try:
            idx = int(name.split("_")[-1])
        except (TypeError, ValueError):
            continue
        header = _parse_job_description_header(desc_path)
        manifest.append(
            {
                "job_id": f"job{idx:02d}",
                "index": idx,
                "title": header.get("title", ""),
                "company": header.get("company", ""),
                "location": header.get("location", ""),
                "site": header.get("site", ""),
                "url": header.get("source url", ""),
                "salary": header.get("salary", ""),
                "raw_scraper_score": 0.0,
                "snippet": "",
            }
        )
    return manifest


def _bucket_match_score(score: float, config: PipelineConfig) -> str:
    if score >= config.min_score_for_high_fit:
        return "High"
    if score >= config.min_score_for_high_fit - 10:
        return "Medium"
    if score >= config.min_score_for_high_fit - 25:
        return "Low"
    return "Very Low"


def write_selection_summary(job_dir: Path, config: PipelineConfig = DEFAULT_CONFIG) -> None:
    manifest = _load_manifest(job_dir) or _fallback_manifest(job_dir)
    if not manifest:
        print("[BUILDER]   -> No manifest found; skipping selection_summary.json", flush=True)
        return

    summary_jobs: List[Dict[str, Any]] = []

    for entry in manifest:
        job_id = entry.get("job_id") or f"job{entry.get('index', 0):02d}"
        label_num = int(str(entry.get("index") or "0") or 0)
        score_path = job_dir / f"score_job{label_num:02d}.json"

        raw_score = 0.0
        if score_path.exists():
            try:
                with score_path.open("r", encoding="utf-8") as f:
                    score_payload = json.load(f)
                raw_score = float(score_payload.get("match_score") or 0.0)
            except Exception:
                raw_score = 0.0

        normalized = _clamp_score(raw_score)
        bucket = _bucket_match_score(normalized, config)

        summary_jobs.append(
            {
                "job_id": job_id,
                "label": f"{label_num:02d}",
                "title": entry.get("title", ""),
                "company": entry.get("company", ""),
                "location": entry.get("location", ""),
                "source_site": entry.get("site", ""),
                "url": entry.get("url", ""),
                "salary_range": entry.get("salary", ""),
                "raw_score": raw_score,
                "normalized_score": normalized,
                "match_bucket": bucket,
            }
        )

    bucket_priority = {"High": 3, "Medium": 2, "Low": 1, "Very Low": 0}
    summary_jobs = sorted(
        summary_jobs,
        key=lambda j: (
            bucket_priority.get(j.get("match_bucket", "Very Low"), 0),
            1 if j.get("salary_range") else 0,
            j.get("normalized_score", 0.0),
        ),
        reverse=True,
    )

    focus_shortlist = summary_jobs[: config.focus_shortlist_size]

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "focus_shortlist": [j.get("job_id") for j in focus_shortlist],
        "jobs": summary_jobs,
    }

    out_path = job_dir / "selection_summary.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(
            f"[BUILDER]   -> selection_summary.json written ({len(summary_jobs)} jobs, "
            f"shortlist size {len(focus_shortlist)})",
            flush=True,
        )
    except Exception as exc:
        print(f"[BUILDER]   -> Failed to write selection_summary.json: {exc}", flush=True)


def process_job(job_dir: str) -> None:
    # High-level job banner
    print("\n====================================================", flush=True)
    print(f"[BUILDER] Processing job folder: {job_dir}", flush=True)
    print("====================================================", flush=True)

    # Stage 1: load inputs
    print("[BUILDER]   [1/3] Loading base_resume...", flush=True)
    base_resume = load_base_resume(job_dir)

    # Step 2: load optional client_request.json for Fiverr jobs only
    print("[BUILDER]   [2/3] Loading client_request.json (if present)...")

    client_request = None
    client_request_path = Path(os.path.join(job_dir, "client_request.json"))

    if client_request_path.is_file():
        try:
            with client_request_path.open("r", encoding="utf-8") as f:
                raw_request = json.load(f)
        except Exception as exc:
            print(f"[BUILDER]   [WARN] Could not parse client_request.json: {exc}")
            raw_request = None

        if isinstance(raw_request, dict):
            meta = raw_request.get("meta") or {}
            source = meta.get("source")
            if source == "fiverr":
                client_request = raw_request
                print("[BUILDER]   [FIVERR] Using intake from client_request.json (source is fiverr).")
            else:
                print("[BUILDER]   [FIVERR] client_request.json present but meta.source is not 'fiverr'. Ignoring intake for this job.")
        else:
            print("[BUILDER]   [WARN] client_request.json did not contain a JSON object. Ignoring intake.")
    else:
        print("[BUILDER]   [FIVERR] No client_request.json found. Proceeding without intake data.")

    # Build a lightweight meta object for downstream model calls.
    # For Fiverr jobs this includes the intake; for all others it is None.
    if client_request is not None:
        client_meta = {
            "source": "fiverr",
            "intake": client_request.get("intake") or {},
            "raw": client_request,
        }
    else:
        client_meta = None

    print("[BUILDER]   [3/3] Loading job_description_*.txt files...", flush=True)
    jd_files = load_job_descriptions(job_dir)
    print(f"[BUILDER]   -> Found {len(jd_files)} job description files.", flush=True)

    if not jd_files:
        print("[BUILDER] No job_description_*.txt files found, nothing to do.", flush=True)
        return

    # Self-revision parameters
    max_revisions = int(os.environ.get("FIVERR_MAX_REVISIONS", "2"))
    target_match_score = TARGET_MATCH_SCORE
    low_fit_floor = LOW_FIT_FLOOR

    # Stage 2: per-job generation
    total = len(jd_files)
    # Stage 2: per-job generation
    total = len(jd_files)
    for idx, jd_path in enumerate(jd_files, start=1):
        job_label = f"{idx:02d}"
        print(
            f"\n[BUILDER] ===== Job {job_label}/{total}: {os.path.basename(jd_path)} =====",
            flush=True,
        )

        with open(jd_path, "r", encoding="utf-8", errors="ignore") as f:
            jd_text = f.read().strip()

        # -------------------------------------------------
        # If this job came from Fiverr and we have intake
        # data, append a compact intake summary to the
        # job description that goes to the model.
        # -------------------------------------------------
        jd_for_model = jd_text

        if client_meta and isinstance(client_meta, dict):
            # Prefer a nested "intake" object if present.
            raw_intake = client_meta.get("intake")
            if isinstance(raw_intake, dict):
                intake = raw_intake
            else:
                intake = client_meta

            summary_lines = []

            def add_line(label: str, key: str) -> None:
                value = intake.get(key)
                if value:
                    # Handle lists vs strings
                    if isinstance(value, list):
                        value_str = ", ".join(str(v) for v in value if str(v).strip())
                    else:
                        value_str = str(value).strip()
                    if value_str:
                        summary_lines.append(f"{label}: {value_str}")

            # These keys match the intake JSON we are designing.
            add_line("Target role", "target_role")
            add_line("Target seniority", "target_seniority")
            add_line("Target industries", "target_industries")
            add_line("Preferred locations", "preferred_locations")
            add_line("Remote preference", "remote_preference")
            add_line("Minimum salary (USD)", "salary_min_usd")
            add_line("Work schedule constraints", "schedule_constraints")
            add_line("Must have technologies", "must_have_tech")
            add_line("Nice to have technologies", "nice_to_have_tech")
            add_line("Deal breakers", "deal_breakers")
            add_line("Notes to writer", "notes_to_writer")

            if summary_lines:
                intake_block = (
                    "=== CLIENT INTAKE SUMMARY (FIVERR) ===\n"
                    + "\n".join(summary_lines)
                )
                jd_for_model = jd_text + "\n\n" + intake_block

        print(f"[BUILDER]   -> Calling Qwen draft for job {job_label}...", flush=True)
        draft = call_qwen_draft(base_resume, jd_for_model, client_meta)

        resume_text = draft["resume"]
        cover_text = draft["cover_letter"]

        # Keep the model's own guess separately
        model_raw_match = draft.get("match_score", draft.get("ats_score", 0.0))
        notes = draft["notes"] or ""
        revisions_used = 0
        match_score = compute_match_score(resume_text, jd_text, client_meta)

        # -----------------------------
        # Self-revision loop (pre-polish)
        # -----------------------------
        while revisions_used < max_revisions:

            # If Match Score is extremely low, treat this as a low-fit job and skip revisions.
            if match_score < low_fit_floor:
                print(
                    f"[BUILDER]   -> MatchScore={match_score:.1f} is below low-fit floor {low_fit_floor:.1f}. "
                    f"Skipping revisions for this job.",
                    flush=True,
                )
                if notes:
                    notes += "\n"
                notes += (
                    f"Low-fit job: MatchScore={match_score:.1f} below low-fit floor "
                    f"{low_fit_floor:.1f}; no revisions attempted."
                )
                break

            # If we've already hit the target, no need to revise.
            if match_score >= target_match_score:
                break

            print(
                f"[BUILDER]   -> MatchScore={match_score:.1f} < target {target_match_score:.1f}, "
                f"requesting revision {revisions_used + 1}...",
                flush=True,
            )

            revision = call_qwen_revision(
                base_resume=base_resume,
                job_desc=jd_text,
                client_meta=client_meta,
                current_resume=resume_text,
                current_cover=cover_text,
                current_match_score=match_score,
            )

            resume_text = revision["resume"]
            cover_text = revision["cover_letter"]
            model_raw_match = revision.get("match_score", revision.get("ats_score", model_raw_match))
            rev_note = revision.get("notes", "").strip()
            if rev_note:
                if notes:
                    notes += "\n"
                notes += f"Revision {revisions_used + 1}: {rev_note}"

            revisions_used += 1
            match_score = compute_match_score(resume_text, jd_text, client_meta)

        # Compute final Match Score *before* polish (for reference)
        match_score = compute_match_score(resume_text, jd_text, client_meta)

        # Optional polishing
        print(f"[BUILDER]   -> Optional OpenAI polish for job {job_label}...", flush=True)
        resume_text, resume_polished = maybe_polish_with_openai(resume_text, "resume")
        cover_text, cover_polished = maybe_polish_with_openai(cover_text, "cover_letter")

        # Hard-guard: remove any certifications not present in the base resume
        resume_text = scrub_unverified_certifications(base_resume, resume_text)

        # Recompute Match Score after polish (final stored score)
        match_score = compute_match_score(resume_text, jd_text, client_meta)

        # Write outputs
        print(f"[BUILDER]   -> Writing outputs for job {job_label}...", flush=True)
        resume_out = os.path.join(job_dir, f"resume_job{job_label}.txt")
        cover_out = os.path.join(job_dir, f"cover_letter_job{job_label}.txt")
        score_out = os.path.join(job_dir, f"score_job{job_label}.json")

        with open(resume_out, "w", encoding="utf-8") as f:
            f.write(resume_text)

        with open(cover_out, "w", encoding="utf-8") as f:
            f.write(cover_text)

        score_payload = {
            "match_score": match_score,
            "ats_score": match_score,  # Deprecated alias for backward compatibility
            "model_raw_match_score": model_raw_match,
            "target_match_score": target_match_score,
            "low_fit_floor": low_fit_floor,
            "revisions_used": revisions_used,
            "notes": notes,
            "model": OLLAMA_MODEL,
            "ollama_url": OLLAMA_URL,
            "openai_polish": {
                "used": resume_polished or cover_polished,
                "model": OPENAI_MODEL if (resume_polished or cover_polished) else None,
            },
        }

        with open(score_out, "w", encoding="utf-8") as f:
            json.dump(score_payload, f, indent=2)

        print(
            f"[BUILDER]   -> Done job {job_label}: "
            f"MatchScore={match_score}, revisions_used={revisions_used}, "
            f"OpenAI polish={'yes' if (resume_polished or cover_polished) else 'no'}",
            flush=True,
        )

    write_selection_summary(Path(job_dir), config=DEFAULT_CONFIG)


def main():
    if len(sys.argv) != 2:
        print("Usage: packet_builder_qwen.py JOB_DIR", file=sys.stderr)
        sys.exit(1)

    job_dir = sys.argv[1]
    process_job(job_dir)


if __name__ == "__main__":
    main()
