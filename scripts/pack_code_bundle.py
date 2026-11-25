"""
pack_code_bundle.py -- Packs minimal source files into code_bundle.tar.gz (~50 KB)
and generates a lightweight ~3 KB fallback Colab notebook.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"__pycache__", ".pytest_cache", ".git", ".venv", "remote_backup"}


def pack_bundle() -> Path:
    bundle = ROOT / "submission" / "code_bundle.tar.gz"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    include_paths = [
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
        ROOT / "scripts" / "run_colab_suite.py",
        ROOT / "paper" / "results" / "m3_pgd.json",
    ]
    for d in ("latent_mas_attack", "tests"):
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not any(part in EXCLUDE for part in p.parts):
                include_paths.append(p)

    with tarfile.open(bundle, "w:gz") as tf:
        for p in include_paths:
            if p.exists():
                tf.add(p, arcname=p.relative_to(ROOT).as_posix())

    print(f"Packed {bundle} ({bundle.stat().st_size / 1024:.1f} KB, {len(include_paths)} files)")
    return bundle


def write_thin_notebook() -> Path:
    def md(s: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in s.splitlines()]}

    def code(s: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in s.splitlines()]}

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "cells": [
            md("# IS631 Assignment 3: Silent Zombies (Drive-Backed Runner)\n\n"
               "1. In **Runtime -> Change runtime type**, select **A100 GPU** (or **L4 GPU**).\n"
               "2. Press **Run all (`Ctrl+F9`)** and approve Google Drive mount once.\n"
               "3. Every experiment auto-saves to `MyDrive/IS631_Silent_Zombies/results/` and auto-resumes if disconnected."),
            code(
                "from google.colab import drive\n"
                "drive.mount('/content/drive')\n\n"
                "!pip install -q \"transformers==4.53.3\" \"bitsandbytes>=0.45.0\" \"accelerate>=0.34.0\" "
                "\"agentdojo==0.1.35\" \"datasets>=2.19.0\" \"pytest>=8.0\"\n\n"
                "import os, shutil, tarfile\n"
                "DRIVE_DIR = \"/content/drive/MyDrive/IS631_Silent_Zombies\"\n"
                "BUNDLE = f\"{DRIVE_DIR}/code_bundle.tar.gz\"\n"
                "assert os.path.exists(BUNDLE), f\"Missing {BUNDLE}\"\n"
                "if os.path.exists(\"/content/workspace\"):\n"
                "    shutil.rmtree(\"/content/workspace\")\n"
                "os.makedirs(\"/content/workspace\", exist_ok=True)\n"
                "with tarfile.open(BUNDLE, \"r:gz\") as tf:\n"
                "    tf.extractall(\"/content/workspace\")\n"
                "print(\"Unpacked workspace from\", BUNDLE)"
            ),
            code(
                "import os\n"
                "os.environ[\"LMAS_OUT_DIR\"] = \"/content/drive/MyDrive/IS631_Silent_Zombies/results\"\n"
                "os.makedirs(os.environ[\"LMAS_OUT_DIR\"], exist_ok=True)\n"
                "!python -u /content/workspace/scripts/run_colab_suite.py"
            ),
        ],
    }
    out_nb = ROOT / "notebooks" / "IS631_Assignment3_Colab.ipynb"
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    out_nb.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Wrote thin notebook {out_nb} ({out_nb.stat().st_size} bytes)")
    return out_nb


if __name__ == "__main__":
    pack_bundle()
    write_thin_notebook()
