import importlib
import pathlib
import sys


# Ensure the repo root is on sys.path so top-level modules import cleanly
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_core_modules_import():
    """
    Smoke test: make sure core modules import without errors.
    This protects us from refactors that break basic imports.
    """
    modules = [
        "watcher",
        "run_fiverr_pipeline",
    ]

    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None
