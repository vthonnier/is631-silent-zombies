"""
build_colab_notebook.py -- Generates a self-contained Google Colab notebook
(notebooks/IS631_Assignment3_Colab.ipynb) from the actual repository files.

Why self-contained?
  The project GitHub repository is private. Embedding the exact verified source
  files directly into the Colab notebook allows one-click GPU execution on
  Google Colab without needing GitHub PATs or SSH keys inside the Colab runtime.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FILES_TO_EMBED = [
    "pyproject.toml",
    "latent_mas_attack/lmas/__init__.py",
    "latent_mas_attack/lmas/config.py",
    "latent_mas_attack/lmas/models.py",
    "latent_mas_attack/lmas/handoff.py",
    "latent_mas_attack/lmas/agent.py",
    "latent_mas_attack/lmas/pipeline.py",
    "latent_mas_attack/lmas/metrics.py",
    "latent_mas_attack/lmas/runtime_env.py",
    "latent_mas_attack/lmas/bus/__init__.py",
    "latent_mas_attack/lmas/bus/shm_bus.py",
    "latent_mas_attack/lmas/bus/edge_monitor.py",
    "latent_mas_attack/lmas/detectors/__init__.py",
    "latent_mas_attack/lmas/detectors/acmas.py",
    "latent_mas_attack/lmas/detectors/text_monitor.py",
    "latent_mas_attack/run_m3_pgd.py",
    "latent_mas_attack/run_m4d_purelatent.py",
    "latent_mas_attack/run_m6_benchmark.py",
    "latent_mas_attack/run_m7_text_baseline.py",
    "latent_mas_attack/run_m8_shm_edge_defense.py",
    "tests/test_shm_bus.py",
    "tests/test_edge_monitor.py",
    "tests/test_parsing_and_metrics.py",
    "paper/results/m3_pgd.json",
]


def _md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


def build_notebook() -> Path:
    embedded_b64: dict[str, str] = {}
    for rel in FILES_TO_EMBED:
        p = ROOT / rel
        if p.exists():
            embedded_b64[rel] = base64.b64encode(p.read_bytes()).decode("ascii")

    cell1 = """# Cell 1: Pin dependencies BEFORE importing transformers/torch & capture OS/kernel telemetry
import subprocess, sys, os, json, platform
from pathlib import Path

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.53.3", "bitsandbytes>=0.45.0", "accelerate>=0.34.0",
    "agentdojo==0.1.35", "pytest>=8.0", "huggingface_hub>=0.26.0"
])

import torch
import transformers
import bitsandbytes
import agentdojo

OUT_DIR = Path("/content/workspace/paper/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["LMAS_OUT_DIR"] = str(OUT_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
gpu_vram_gb = (
    round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    if torch.cuda.is_available()
    else 0.0
)

os_release = ""
if os.path.exists("/etc/os-release"):
    os_release = Path("/etc/os-release").read_text(encoding="utf-8")

uname = platform.uname()
env_info = {
    "ok": True,
    "system": uname.system,
    "node": uname.node,
    "kernel_release": uname.release,
    "kernel_version": uname.version,
    "machine": uname.machine,
    "os_release": os_release,
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "transformers_version": transformers.__version__,
    "bitsandbytes_version": bitsandbytes.__version__,
    "agentdojo_version": getattr(agentdojo, "__version__", "0.1.35"),
    "gpu_name": gpu_name,
    "gpu_vram_gb": gpu_vram_gb,
    "dev_shm_exists": os.path.exists("/dev/shm"),
}
(OUT_DIR / "env_info.json").write_text(json.dumps(env_info, indent=2), encoding="utf-8")
print(json.dumps(env_info, indent=2))
"""

    payload_json = json.dumps(embedded_b64)
    cell2 = f"""# Cell 2: Materialize verified repository modules, drivers, and unit tests
import base64, sys
from pathlib import Path

WORKSPACE = Path("/content/workspace")
EMBEDDED = {payload_json}

for rel_path, b64_str in EMBEDDED.items():
    dest = WORKSPACE / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(b64_str))

sys.path.insert(0, str(WORKSPACE / "latent_mas_attack"))
print(f"Materialized {{len(EMBEDDED)}} files into {{WORKSPACE}}")
"""

    cell3 = """# Cell 3: Run Pytest Suite + M8 POSIX /dev/shm IPC & Tamper-Rejection Benchmark
import subprocess, sys, json
from pathlib import Path

res = subprocess.run(
    [sys.executable, "-m", "pytest", "-v"],
    cwd="/content/workspace",
    capture_output=True,
    text=True,
)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise RuntimeError("Unit tests failed!")

from run_m8_shm_edge_defense import benchmark_ipc
ipc_res = benchmark_ipc()
print("M8 IPC Summary:", json.dumps(ipc_res, indent=2))
"""

    cell4 = """# Cell 4: Load Qwen/Qwen2.5-3B-Instruct (4-bit NF4, bf16) once & run M3 + M4d + M8 Dual-Evasion
import gc, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lmas.runtime_env import resolve_model_path

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
path = resolve_model_path(MODEL_ID)
tok = AutoTokenizer.from_pretrained(path)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    path, quantization_config=bnb, device_map={"": 0}, attn_implementation="sdpa"
).eval()
print(f"Loaded {MODEL_ID} on {model.device} ({torch.cuda.get_device_name(0)})")

from run_m3_pgd import run_m3_with_model
from run_m4d_purelatent import run_m4d_with_model
from run_m8_shm_edge_defense import run_m8_with_model

print("=== Running M3 (completing deep6_n16 + verification) ===")
m3_out = run_m3_with_model(model, tok, only_missing=True)
print("M3 ok:", m3_out.get("ok"), "configs:", list(m3_out.get("configs", {}).keys()))
torch.cuda.empty_cache(); gc.collect()

print("=== Running M4d (Pure-Latent Depths 2..6, n=3) ===")
m4d_out = run_m4d_with_model(model, tok)
print("M4d ok:", m4d_out.get("ok"))
torch.cuda.empty_cache(); gc.collect()

print("=== Running M8 (POSIX /dev/shm IPC + Dual-Evasion Edge Monitor) ===")
m8_out = run_m8_with_model(model, tok)
print("M8 ok:", m8_out.get("ok"))
torch.cuda.empty_cache(); gc.collect()
"""

    cell5 = """# Cell 5: Run M6 (12-Pair AgentDojo Latent Attack + Active Edge Defense) & M7 (12-Pair Text Baseline)
import gc, json, torch
from run_m6_benchmark import run_m6_with_model
from run_m7_text_baseline import run_m7_with_model

print("=== Running M6 (12-pair AgentDojo Latent Attack + Edge Reference Monitor) ===")
m6_out = run_m6_with_model(model, tok)
print("M6 Summary:", json.dumps(m6_out.get("summary", {}), indent=2))
torch.cuda.empty_cache(); gc.collect()

print("=== Running M7 (12-pair Text-Channel Baseline) ===")
m7_out = run_m7_with_model(model, tok)
print("M7 Summary:", json.dumps(m7_out.get("summary", {}), indent=2))
torch.cuda.empty_cache(); gc.collect()
"""

    cell6 = """# Cell 6: Bundle all JSON artifacts into base64 tar.gz & print summary
import base64, io, json, tarfile
from pathlib import Path

res_dir = Path("/content/workspace/paper/results")
files = sorted(res_dir.glob("*.json"))
print("Result files:", [f.name for f in files])

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    for f in files:
        tf.add(f, arcname=f.name)

b64_bundle = base64.b64encode(buf.getvalue()).decode("ascii")
print("===BUNDLE_B64_BEGIN===")
for i in range(0, len(b64_bundle), 120):
    print(b64_bundle[i:i+120])
print("===BUNDLE_B64_END===")

# Optional: push to private HF dataset if HF_TOKEN is stored in Colab Secrets
try:
    from google.colab import userdata
    hf_tok = userdata.get("HF_TOKEN")
    if hf_tok:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_tok)
        api.upload_folder(
            folder_path=str(res_dir),
            path_in_repo="results",
            repo_id="vthonnier/is631-silent-zombies-artifacts",
            repo_type="dataset",
            commit_message="Upload Colab A100/L4 benchmark results",
        )
        print("Uploaded results to Hugging Face dataset.")
except Exception as e:
    print(f"HF optional upload skipped: {e}")

print("===ALL_EXPERIMENTS_COMPLETE===")
"""

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "accelerator": "GPU",
        },
        "cells": [
            _md_cell(
                "# IS631 Term Project (Assignment 3): Silent Zombies\n\n"
                "**Inference-Time Latent-Handoff Attacks & Edge-Level Defenses in Tool-Using Multi-Agent Systems**\n\n"
                "This notebook runs the complete GPU & POSIX `/dev/shm` experimental suite on `Qwen/Qwen2.5-3B-Instruct` (4-bit NF4, `bfloat16` compute) and AgentDojo `workspace`."
            ),
            _code_cell(cell1),
            _code_cell(cell2),
            _code_cell(cell3),
            _code_cell(cell4),
            _code_cell(cell5),
            _code_cell(cell6),
        ],
    }

    out_nb = ROOT / "notebooks" / "IS631_Assignment3_Colab.ipynb"
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    out_nb.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Wrote Colab notebook: {out_nb} ({out_nb.stat().st_size} bytes)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
