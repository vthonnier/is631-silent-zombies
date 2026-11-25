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
            md("# IS631 Assignment 3: Silent Zombies (Lightweight Runner)\n\n"
               "Pulls `code_bundle.tar.gz` (or clones via HF/Drive) and runs `scripts/run_colab_suite.py`."),
            code(
                "!pip install -q 'transformers==4.53.3' 'bitsandbytes>=0.45.0' 'accelerate>=0.34.0' "
                "'agentdojo==0.1.35' 'datasets>=2.19.0' 'pytest>=8.0' 'huggingface_hub>=0.26.0'\n"
                "import os; os.makedirs('/content/workspace', exist_ok=True)\n"
                "# Upload code_bundle.tar.gz into /content/workspace or pull from Drive/HF, then:\n"
                "!tar -xzf /content/workspace/code_bundle.tar.gz -C /content/workspace\n"
                "!python /content/workspace/scripts/run_colab_suite.py"
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
