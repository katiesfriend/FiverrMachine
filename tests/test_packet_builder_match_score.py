import sys
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Provide a lightweight requests stub so we can import without optional dependencies
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace(post=None)

import packet_builder_qwen as builder


def test_compute_match_score_high_fit():
    resume = (
        "Senior Software Engineer skilled in Python, Django, REST APIs, and AWS. "
        "Open to remote roles."
    )
    jd = (
        "Senior Software Engineer - Remote\n"
        "We need expertise with Python, Django, REST APIs, and AWS cloud services."
    )
    meta = {
        "skills": ["Python", "Django", "REST APIs", "AWS"],
        "target_roles": ["Senior Software Engineer"],
        "remote_preference": "Remote",
    }

    score = builder.compute_match_score(resume, jd, meta)
    assert score > builder.TARGET_MATCH_SCORE


def test_compute_match_score_medium_fit():
    resume = "Backend engineer experienced with Python, SQL, Docker, and Kubernetes."
    jd = (
        "Backend Engineer (Data Pipelines)\n"
        "Responsibilities include building services in Python, SQL, and Docker."
    )
    meta = {
        "skills": ["Python", "SQL", "Docker", "Kubernetes"],
        "target_roles": ["Analytics Manager"],
        "remote_preference": "Hybrid",
    }

    score = builder.compute_match_score(resume, jd, meta)
    assert builder.LOW_FIT_FLOOR <= score < builder.TARGET_MATCH_SCORE


def test_compute_match_score_low_fit():
    resume = "Graphic designer proficient in Adobe Illustrator and Photoshop."
    jd = (
        "Senior Backend Engineer needed onsite.\n"
        "Must have deep experience with C++ services and distributed systems."
    )
    meta = {
        "skills": ["Adobe Illustrator", "Photoshop"],
        "target_roles": ["Graphic Designer"],
        "remote_preference": "Onsite",
    }

    score = builder.compute_match_score(resume, jd, meta)
    assert score < builder.LOW_FIT_FLOOR
