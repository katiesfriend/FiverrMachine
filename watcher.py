#!/usr/bin/env python3
"""
watcher.py

File watcher for FiverrMachine.

Responsibilities:
  - Watch the INTAKE directory for new client request bundles
  - When a new bundle appears, create a JOB_xxxxx folder in PROCESSING
  - Move the bundle into the JOB folder
  - (Optionally) trigger job_scraper and packet_builder_qwen for that JOB

This is designed to run continuously (e.g., via systemd or screen/tmux).
"""

import time
import uuid
import shutil
import subprocess
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

BASE = Path("/home/mykl/webui/filesystem/FiverrMachine")
INTAKE = BASE / "INTAKE"
PROCESSING = BASE / "PROCESSING"
LOGS = BASE / "LOGS"

# Avoid side effects in CI or non-mykl environments:
# only create these directories when NOT running under GitHub Actions.
if os.getenv("GITHUB_ACTIONS") != "true":
    INTAKE.mkdir(parents=True, exist_ok=True)
    PROCESSING.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    """Append a message to watcher.log and print to stdout."""
    line = f"[WATCHER] {msg}\n"
    print(line, end="")
    (LOGS / "watcher.log").open("a", encoding="utf-8").write(line)


class IntakeHandler(FileSystemEventHandler):
    """
    Handles filesystem events in INTAKE.

    Current behavior:
      - When a new file appears, wrap it into a new JOB_xxxxx directory.
      - Future upgrade: handle client-specific subfolders or JSON.
    """

    def on_created(self, event):
        path = Path(event.src_path)

        # Ignore temporary or directory events
        if path.is_dir():
            return
        if path.name.startswith("."):
            return

        # Create a new JOB folder
        job_id = f"JOB_{uuid.uuid4().hex[:8]}"
        job_dir = PROCESSING / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        dest = job_dir / path.name
        shutil.move(str(path), str(dest))

        log(f"New file {path.name} -> created {job_dir}")

        # OPTIONAL: kick off downstream processing here.
        # For now we just log. Later we can chain:
        #   python job_scraper.py JOB_xxxxx
        #   python packet_builder_qwen.py JOB_xxxxx
        #
        # Example (commented for safety):
        # subprocess.Popen(
        #     ["python3", str(BASE / "job_scraper.py"), str(job_dir)],
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        # )


def main():
    observer = Observer()
    handler = IntakeHandler()
    observer.schedule(handler, str(INTAKE), recursive=False)
    observer.start()
    log(f"Watching {INTAKE} for new client files...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("Stopping watcher via KeyboardInterrupt...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
