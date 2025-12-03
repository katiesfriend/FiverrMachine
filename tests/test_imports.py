def test_core_modules_import():
    """
    Smoke test: make sure core modules import without errors.
    This protects us from refactors that break basic imports.
    """
    import importlib

    modules = [
        "watcher",
        "run_fiverr_pipeline",
        # Add others as we go, e.g. "SCRAPER.job_scraper"
    ]

    for name in modules:
        importlib.import_module(name)
