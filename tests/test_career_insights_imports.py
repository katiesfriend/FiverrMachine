import sys
from pathlib import Path

# Make repo root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def test_career_insights_imports():
    """
    Smoke test: career insights modules should import cleanly.
    This protects us from syntax/import errors sneaking into the repo.
    """
    import importlib

    importlib.import_module("modules.career_insights.career_insights")
    importlib.import_module("engines.ai_model")
