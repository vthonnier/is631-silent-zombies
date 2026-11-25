"""
build_submission_zip.py -- Builds the KLMS Assignment 3 deliverable archive
(20266480_ThonnierVirgile_assignment3.zip) containing the PDF report, full
research paper PDF, and clean reproducible artifact directory.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "submission"
ZIP_NAME = "20266480_ThonnierVirgile_assignment3.zip"
REPORT_PDF = ROOT / "report" / "20266480_ThonnierVirgile_assignment3.pdf"
PAPER_PDF = ROOT / "paper" / "paper.pdf"

INCLUDE_TOP_DIRS = ["latent_mas_attack", "tests", "scripts", "notebooks", "paper"]
INCLUDE_TOP_FILES = ["README.md", "LICENSE", "pyproject.toml", "requirements.txt"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".git", ".venv", "submission", "remote_backup"}


def build_zip(copy_to_term_project: bool = True) -> Path:
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SUB_DIR / ZIP_NAME

    if not REPORT_PDF.exists():
        raise FileNotFoundError(f"Missing Assignment 3 report PDF: {REPORT_PDF}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(REPORT_PDF, arcname="20266480_ThonnierVirgile_assignment3.pdf")
        if PAPER_PDF.exists():
            zf.write(PAPER_PDF, arcname="Silent_Zombies_Full_Paper.pdf")

        for fname in INCLUDE_TOP_FILES:
            p = ROOT / fname
            if p.exists():
                zf.write(p, arcname=f"artifact/{fname}")

        for dname in INCLUDE_TOP_DIRS:
            dpath = ROOT / dname
            if not dpath.exists():
                continue
            for fp in sorted(dpath.rglob("*")):
                if fp.is_dir():
                    continue
                if any(part in EXCLUDE_PARTS for part in fp.parts):
                    continue
                if fp.suffix in {".pyc", ".pyo", ".pt", ".pth", ".aux", ".log", ".bbl", ".blg"}:
                    continue
                rel = fp.relative_to(ROOT).as_posix()
                zf.write(fp, arcname=f"artifact/{rel}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Zip integrity check failed on {bad}")
        count = len(zf.namelist())

    print(f"Built {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB, {count} files)")

    if copy_to_term_project:
        tp_dir = Path(r"C:\Users\virgi\Documents\Cours\3A_KAIST\IS631_Kernel_System_Security\Term_Project")
        if tp_dir.exists():
            shutil.copy2(REPORT_PDF, tp_dir / "20266480_ThonnierVirgile_assignment3.pdf")
            shutil.copy2(zip_path, tp_dir / ZIP_NAME)
            print(f"Copied PDF + ZIP to {tp_dir}")

    return zip_path


if __name__ == "__main__":
    build_zip()
