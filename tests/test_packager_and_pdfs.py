import json
import sys
import types
import zipfile
from pathlib import Path

if "reportlab" not in sys.modules:
    dummy_canvas = types.SimpleNamespace(
        Canvas=lambda *args, **kwargs: types.SimpleNamespace(
            setFont=lambda *a, **k: None,
            drawString=lambda *a, **k: None,
            showPage=lambda *a, **k: None,
            save=lambda *a, **k: None,
        )
    )
    sys.modules["reportlab"] = types.SimpleNamespace()
    sys.modules["reportlab.lib"] = types.SimpleNamespace()
    sys.modules["reportlab.lib.pagesizes"] = types.SimpleNamespace(LETTER=(612, 792))
    sys.modules["reportlab.lib.units"] = types.SimpleNamespace(inch=72)
    sys.modules["reportlab.pdfgen"] = types.SimpleNamespace(canvas=dummy_canvas)
    sys.modules["reportlab.pdfgen.canvas"] = dummy_canvas

import deliverable_packager
import generate_pdfs


def _run_packager(monkeypatch, job_root: Path, stdout_payload: dict):
    deliverables_dir = job_root.parent / "DELIVERABLES"
    monkeypatch.setattr(deliverable_packager, "DELIVERABLES", deliverables_dir)

    def fake_run(*args, **kwargs):
        return deliverable_packager.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(stdout_payload),
            stderr="",
        )

    monkeypatch.setattr(deliverable_packager.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["deliverable_packager.py", str(job_root)])

    deliverable_packager.main()
    zips = list(deliverables_dir.glob("*.zip"))
    assert zips, "zip not created"
    return zips[0]


def test_packager_uses_selection_summary_and_adds_career_insights(monkeypatch, tmp_path):
    job_root = tmp_path / "JOB_TEST"
    job_root.mkdir()

    # Seed selection summary and artifacts
    (job_root / "selection_summary.json").write_text(json.dumps({"selected_labels": ["01"]}), encoding="utf-8")
    (job_root / "resume_job01.pdf").write_bytes(b"resume")
    (job_root / "cover_letter_job01.pdf").write_bytes(b"cover")
    pie_chart = job_root / "career_insights_skills_pie.png"
    pie_chart.write_bytes(b"pie")

    zip_path = _run_packager(monkeypatch, job_root, {"pie_chart": str(pie_chart)})

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "career_insights/career_insights_skills_pie.png" in names
        assert "resume_job01.pdf" in names
        # Ensure we carried selected labels through instead of recomputing
        assert "selection_summary.json" in names


def test_packager_prefers_selection_summary_over_heuristic(monkeypatch, tmp_path):
    job_root = tmp_path / "JOB_TEST2"
    job_root.mkdir()

    (job_root / "selection_summary.json").write_text(json.dumps({"selected_labels": ["02"]}), encoding="utf-8")
    (job_root / "resume_job01.pdf").write_bytes(b"resume1")
    (job_root / "cover_letter_job01.pdf").write_bytes(b"cover1")
    (job_root / "resume_job02.pdf").write_bytes(b"resume2")
    (job_root / "cover_letter_job02.pdf").write_bytes(b"cover2")

    # If heuristic scoring is consulted, this will raise and fail the test
    monkeypatch.setattr(
        deliverable_packager,
        "compute_heuristic_job_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("heuristic should not run")),
    )

    zip_path = _run_packager(monkeypatch, job_root, {})

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "resume_job02.pdf" in names
        assert "resume_job01.pdf" not in names


def test_generate_pdfs_respects_selection_labels(monkeypatch, tmp_path):
    job_root = tmp_path / "JOB_TEST3"
    job_root.mkdir()

    (job_root / "selection_summary.json").write_text(json.dumps({"selected_labels": ["01"]}), encoding="utf-8")

    monkeypatch.setattr(generate_pdfs, "render_plaintext_resume_to_pdf", None)

    def fake_render(txt_path, pdf_path, title=None):
        Path(pdf_path).write_text("pdf", encoding="utf-8")

    monkeypatch.setattr(generate_pdfs, "render_text_file_to_pdf", fake_render)

    for idx in ("01", "02", "03"):
        (job_root / f"resume_job{idx}.txt").write_text(f"Resume {idx}", encoding="utf-8")
        (job_root / f"cover_letter_job{idx}.txt").write_text(f"Cover {idx}", encoding="utf-8")

    generate_pdfs.generate_pdfs_for_job_folder(str(job_root))

    assert (job_root / "resume_job01.pdf").exists()
    assert (job_root / "cover_letter_job01.pdf").exists()
    assert not (job_root / "resume_job02.pdf").exists()
    assert not (job_root / "cover_letter_job02.pdf").exists()
    assert not (job_root / "resume_job03.pdf").exists()
    assert not (job_root / "cover_letter_job03.pdf").exists()
