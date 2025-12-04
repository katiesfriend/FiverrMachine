from pathlib import Path
import json

from report_builder import REPORT_FILENAME, build_report


def test_build_report_creates_file_with_jobs(tmp_path: Path) -> None:
    job_dir = tmp_path / "JOB_test"
    job_dir.mkdir()

    client_request = {
        "client_name": "Test Client",
        "target_roles": ["Engineer"],
        "location_zip": "12345",
    }
    (job_dir / "client_request.json").write_text(json.dumps(client_request), encoding="utf-8")

    job_sources = "\n".join(
        [
            "# Selected 2 jobs (high-fit >= 2.5: 1)",
            "01. [linkedin] score=3.00 url=https://example.com/job1",
            "title=Software Engineer",
            "company=Acme Corp",
            "location=Remote",
            "02. [indeed] score=1.00 url=https://example.com/job2",
            "title=Junior Developer",
            "company=Beta LLC",
            "location=NYC",
        ]
    )
    (job_dir / "job_sources.txt").write_text(job_sources, encoding="utf-8")

    report_path = build_report(job_dir)

    assert report_path.exists()
    assert report_path.name == REPORT_FILENAME

    contents = report_path.read_text(encoding="utf-8")
    assert contents.strip()
    assert "Software Engineer" in contents
