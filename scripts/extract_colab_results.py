"""
extract_colab_results.py -- Extracts the base64-encoded JSON result bundle from
an executed Colab .ipynb file into paper/results/.
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
import tarfile
from pathlib import Path


def extract(ipynb_path: Path, out_dir: Path) -> list[str]:
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    stdout_chunks: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and "text" in out:
                txt = out["text"]
                stdout_chunks.append("".join(txt) if isinstance(txt, list) else txt)
            elif out.get("output_type") == "execute_result":
                data = out.get("data", {}).get("text/plain", "")
                stdout_chunks.append("".join(data) if isinstance(data, list) else data)

    full_text = "\n".join(stdout_chunks)
    m = re.search(
        r"===BUNDLE_B64_BEGIN===\s*(.*?)\s*===BUNDLE_B64_END===",
        full_text,
        re.DOTALL,
    )
    if not m:
        raise RuntimeError(
            "Could not find ===BUNDLE_B64_BEGIN===...===BUNDLE_B64_END=== in notebook outputs."
        )
    b64_clean = re.sub(r"\s+", "", m.group(1))
    raw_tar = base64.b64decode(b64_clean)
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            data = tf.extractfile(member).read()
            dest = out_dir / Path(member.name).name
            dest.write_bytes(data)
            extracted.append(dest.name)
    return extracted


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_colab_results.py <executed.ipynb> [out_dir]")
        sys.exit(1)
    nb_path = Path(sys.argv[1])
    target_dir = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path(__file__).resolve().parents[1] / "paper" / "results"
    )
    names = extract(nb_path, target_dir)
    print(f"Extracted {len(names)} JSON files into {target_dir}: {names}")
