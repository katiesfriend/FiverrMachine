#!/usr/bin/env python3
"""
deliverable_packager.py

If called with an argument:
    deliverable_packager.py /path/to/job_folder

    -> Create a zip archive containing everything under that folder.
       (Used by run_fiverr_pipeline.py, which passes the PROCESSING/JOB_xxxxx path.)

If called with NO arguments:
    -> Fallback to legacy behavior: zip everything under FINAL_DELIVERY.
"""

import sys
import zipfile
from datetime import datetime
from pathlib import Path

BASE = Path("/home/mykl/webui/filesystem/FiverrMachine")
FINAL = BASE / "FINAL_DELIVERY"
DELIVERABLES = BASE / "DELIVERABLES"


def package_tree(root: Path) -> Path:
    """
    Create a zip archive containing everything under the given root path.
    Returns the path to the created zip.
    """
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use the root folder name in the zip file for easier identification
    root_name = root.name or "all"
    zip_path = DELIVERABLES / f"fiverr_deliverables_{root_name}_{timestamp}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in root.rglob("*"):
            if item.is_file():
                arcname = item.relative_to(root)
                zf.write(item, arcname)

    print(f"[PACKAGER] Created zip at {zip_path}")
    return zip_path


def main():
    # Decide what root directory to zip
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = FINAL

    if not root.exists():
        print(f"[PACKAGER] ERROR: root path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    zip_path = package_tree(root)
    print(f"[PACKAGER] Zip ready for upload: {zip_path}")


if __name__ == "__main__":
    main()
