"""
run_colab_suite.py -- Unified headless GPU runner for Colab Pro (A100/L4/T4).

Loads Qwen2.5-3B-Instruct (4-bit NF4) ONCE for M3..M12 query phase, then
loads Qwen2.5-1.5B-Instruct (4-bit NF4) for BB-3 surrogate training, and finally
loads unquantized bfloat16 Qwen2.5-3B-Instruct ONCE for M10 Cross-Precision
transfer AND M12 BB-3 cross-model alignment evaluation.
Packs all result JSONs into results_bundle.tar.gz.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "latent_mas_attack"))


def capture_env_info(out_dir: Path) -> dict:
    import agentdojo
    import bitsandbytes
    import torch
    import transformers

    uname = platform.uname()
    os_release = Path("/etc/os-release").read_text(encoding="utf-8") if os.path.exists("/etc/os-release") else ""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if torch.cuda.is_available()
        else 0.0
    )
    info = {
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
    (out_dir / "env_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("ENV_INFO:", json.dumps(info, indent=2), flush=True)
    return info


def main() -> None:
    out_dir = Path(os.environ.get("LMAS_OUT_DIR", str(ROOT / "paper" / "results")))
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LMAS_OUT_DIR"] = str(out_dir)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 1. Unit tests on target OS / Linux /dev/shm
    res = subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=str(ROOT), text=True)
    if res.returncode != 0:
        raise RuntimeError("Unit tests failed on remote runtime!")

    capture_env_info(out_dir)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from lmas.runtime_env import resolve_model_path

    # 2. Load Qwen2.5-3B-Instruct (4-bit NF4, bf16) ONCE
    victim_id = "Qwen/Qwen2.5-3B-Instruct"
    vpath = resolve_model_path(victim_id)
    tok = AutoTokenizer.from_pretrained(vpath)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_4bit = AutoModelForCausalLM.from_pretrained(
        vpath, quantization_config=bnb, device_map={"": 0}, attn_implementation="sdpa"
    ).eval()

    from run_m3_pgd import run_m3_with_model
    from run_m4d_purelatent import run_m4d_with_model
    from run_m6_benchmark import run_m6_with_model
    from run_m7_text_baseline import run_m7_with_model
    from run_m8_shm_edge_defense import run_m8_with_model
    from run_m9_whitebox_matrix_convpoint import run_m9_with_model
    from run_m10_transferability import eval_cross_precision_bf16, run_m10_with_model
    from run_m11_mas_benchmarks import run_m11_with_model
    from run_m12_blackbox import (
        run_m12_query_phase,
        run_m12_surrogate_phase,
        train_surrogate_and_collect_caches,
    )

    for label, fn in [
        ("M3", lambda: run_m3_with_model(model_4bit, tok, only_missing=True)),
        ("M4d", lambda: run_m4d_with_model(model_4bit, tok)),
        ("M8", lambda: run_m8_with_model(model_4bit, tok)),
        ("M6", lambda: run_m6_with_model(model_4bit, tok)),
        ("M7", lambda: run_m7_with_model(model_4bit, tok)),
        ("M9", lambda: run_m9_with_model(model_4bit, tok)),
        ("M10-NF4", lambda: run_m10_with_model(model_4bit, tok)),
        ("M11", lambda: run_m11_with_model(model_4bit, tok)),
        ("M12-Query", lambda: run_m12_query_phase(model_4bit, tok)),
    ]:
        print(f"\n==================== {label} ====================", flush=True)
        fn()
        torch.cuda.empty_cache()
        gc.collect()

    del model_4bit
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Load Surrogate Qwen2.5-1.5B-Instruct (4-bit NF4, bf16)
    print("\n==================== M12-Surrogate-1.5B ====================", flush=True)
    spath = resolve_model_path("Qwen/Qwen2.5-1.5B-Instruct")
    smodel = AutoModelForCausalLM.from_pretrained(
        spath, quantization_config=bnb, device_map={"": 0}, attn_implementation="sdpa"
    ).eval()
    sbundle = train_surrogate_and_collect_caches(smodel, tok)
    del smodel
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Load Unquantized bfloat16 Qwen2.5-3B-Instruct ONCE for M10-bf16 + M12-Surrogate-Transfer
    print("\n==================== M10-bf16 + M12-Bridge ====================", flush=True)
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        vpath, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    ).eval()
    eval_cross_precision_bf16(model_bf16, tok)
    run_m12_surrogate_phase(model_bf16, tok, sbundle)
    del model_bf16
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Bundle results_bundle.tar.gz
    bundle_path = ROOT / "results_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        for f in sorted(out_dir.glob("*.json")):
            tf.add(f, arcname=f.name)
    print(f"Packed {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KB)", flush=True)
    print("===ALL_EXPERIMENTS_COMPLETE===", flush=True)


if __name__ == "__main__":
    main()
