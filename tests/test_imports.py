def test_core_modules_import():
    """
    Smoke test: make sure core modules import without errors.
    This protects us from refactors that break basic imports.
    """
    import importlib
    import sys
    import types

    # Provide lightweight stubs so we can import without optional dependencies
    if "watchdog" not in sys.modules:
        observer_stub = type("Observer", (), {})
        event_handler_stub = type("FileSystemEventHandler", (), {})
        watchdog_stub = types.SimpleNamespace()
        watchdog_stub.observers = types.SimpleNamespace(Observer=observer_stub)
        watchdog_stub.events = types.SimpleNamespace(FileSystemEventHandler=event_handler_stub)
        sys.modules["watchdog"] = watchdog_stub
        sys.modules["watchdog.observers"] = watchdog_stub.observers
        sys.modules["watchdog.events"] = watchdog_stub.events

    modules = [
        "watcher",
        "run_fiverr_pipeline",
        # Add others as we go, e.g. "SCRAPER.job_scraper"
    ]

    for name in modules:
        importlib.import_module(name)
