import zipfile
from pathlib import Path

import deliverable_packager


def test_packager_includes_report(monkeypatch, tmp_path: Path) -> None:
    job_dir = tmp_path / "JOB_abc"
    job_dir.mkdir()

    report_path = job_dir / deliverable_packager.REPORT_FILENAME
    report_path.write_text("Report body", encoding="utf-8")

    (job_dir / "resume.txt").write_text("resume", encoding="utf-8")

    output_dir = tmp_path / "deliverables"
    monkeypatch.setattr(deliverable_packager, "DELIVERABLES", output_dir)

    zip_path = deliverable_packager.package_tree(job_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    assert "01_Job_Search_Report.md" in names
    assert "resume.txt" in names
