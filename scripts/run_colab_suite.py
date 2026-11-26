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
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Callable, Optional

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
    cap = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [0, 0]
    native_bf16 = bool(torch.cuda.is_available() and cap[0] >= 8)
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
        "gpu_compute_capability": cap,
        "native_bf16": native_bf16,
        "compute_dtype": "bfloat16" if native_bf16 else "float16",
        "dev_shm_exists": os.path.exists("/dev/shm"),
    }
    (out_dir / "env_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("ENV_INFO:", json.dumps(info, indent=2), flush=True)
    return info


def _json_done(path: Path, check: Optional[Callable[[dict], bool]] = None) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("ok") is not True:
        return False
    return True if check is None else bool(check(data))


def main() -> None:
    out_dir = Path(os.environ.get("LMAS_OUT_DIR", str(ROOT / "paper" / "results")))
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LMAS_OUT_DIR"] = str(out_dir)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    seed_m3 = ROOT / "paper" / "results" / "m3_pgd.json"
    dst_m3 = out_dir / "m3_pgd.json"
    if not dst_m3.exists() and seed_m3.exists():
        shutil.copy2(seed_m3, dst_m3)

    # 1. Unit tests on target OS / Linux /dev/shm
    res = subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=str(ROOT), text=True)
    if res.returncode != 0:
        raise RuntimeError("Unit tests failed on remote runtime!")

    capture_env_info(out_dir)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from lmas.runtime_env import resolve_model_path

    victim_id = "Qwen/Qwen2.5-3B-Instruct"
    vpath = resolve_model_path(victim_id)
    tok = AutoTokenizer.from_pretrained(vpath)
    native_bf16 = bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
    compute_dtype = torch.bfloat16 if native_bf16 else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

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

    stage_checks = [
        ("M3", _json_done(out_dir / "m3_pgd.json", lambda d: len(d.get("configs", {})) >= 4)),
        ("M4d", _json_done(out_dir / "m4d_purelatent.json")),
        ("M8", _json_done(out_dir / "m8_shm_edge_defense.json")),
        ("M6", _json_done(out_dir / "m6_benchmark.json", lambda d: d.get("n_pairs", 0) >= 24)),
        ("M7", _json_done(out_dir / "m7_text_baseline.json", lambda d: d.get("n_pairs", 0) >= 24)),
        ("M9", _json_done(out_dir / "m9_whitebox_convpoint.json")),
        ("M10-NF4", _json_done(out_dir / "m10_transferability.json")),
        ("M11", _json_done(out_dir / "m11_mas_benchmarks.json")),
        ("M12-Query", _json_done(out_dir / "m12_blackbox.json", lambda d: "BB1_subspace_nes" in d)),
    ]

    # 2. Load Qwen2.5-3B-Instruct (4-bit NF4, bf16) ONCE if any NF4 stage is incomplete
    if any(not done for _, done in stage_checks):
        model_4bit = AutoModelForCausalLM.from_pretrained(
            vpath, quantization_config=bnb, device_map={"": 0}, attn_implementation="sdpa"
        ).eval()

        stage_fns = {
            "M3": lambda: run_m3_with_model(model_4bit, tok, only_missing=True),
            "M4d": lambda: run_m4d_with_model(model_4bit, tok),
            "M8": lambda: run_m8_with_model(model_4bit, tok),
            "M6": lambda: run_m6_with_model(model_4bit, tok),
            "M7": lambda: run_m7_with_model(model_4bit, tok),
            "M9": lambda: run_m9_with_model(model_4bit, tok),
            "M10-NF4": lambda: run_m10_with_model(model_4bit, tok),
            "M11": lambda: run_m11_with_model(model_4bit, tok),
            "M12-Query": lambda: run_m12_query_phase(model_4bit, tok),
        }

        for label, done in stage_checks:
            if done:
                print(f"\n==================== {label} (SKIPPED - ALREADY DONE) ====================", flush=True)
                continue
            print(f"\n==================== {label} ====================", flush=True)
            stage_fns[label]()
            torch.cuda.empty_cache()
            gc.collect()

        del model_4bit
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n==================== NF4 PHASE (ALL STAGES ALREADY DONE) ====================", flush=True)

    need_m10_bf16 = not _json_done(
        out_dir / "m10_transferability.json", lambda d: len(d.get("cross_precision_bf16", [])) > 0
    )
    need_m12_bridge = not _json_done(
        out_dir / "m12_blackbox.json", lambda d: "BB3_surrogate_transfer" in d
    )

    # 3. Load Surrogate Qwen2.5-1.5B-Instruct (4-bit NF4, bf16) if M12-Bridge is needed
    sbundle = None
    if need_m12_bridge:
        print("\n==================== M12-Surrogate-1.5B ====================", flush=True)
        spath = resolve_model_path("Qwen/Qwen2.5-1.5B-Instruct")
        smodel = AutoModelForCausalLM.from_pretrained(
            spath, quantization_config=bnb, device_map={"": 0}, attn_implementation="sdpa"
        ).eval()
        sbundle = train_surrogate_and_collect_caches(smodel, tok)
        del smodel
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n==================== M12-Surrogate-1.5B (SKIPPED - ALREADY DONE) ====================", flush=True)

    # 4. Load Unquantized 16-bit Qwen2.5-3B-Instruct ONCE for M10-bf16 + M12-Surrogate-Transfer
    if need_m10_bf16 or need_m12_bridge:
        print(f"\n==================== M10-16bit ({compute_dtype}) + M12-Bridge ====================", flush=True)
        model_bf16 = AutoModelForCausalLM.from_pretrained(
            vpath, torch_dtype=compute_dtype, device_map={"": 0}, attn_implementation="sdpa"
        ).eval()
        if need_m10_bf16:
            eval_cross_precision_bf16(model_bf16, tok)
        if need_m12_bridge and sbundle is not None:
            run_m12_surrogate_phase(model_bf16, tok, sbundle)
        del model_bf16
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n==================== M10-bf16 + M12-Bridge (SKIPPED - ALREADY DONE) ====================", flush=True)

    # 5. Bundle results_bundle.tar.gz
    bundle_path = ROOT / "results_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        for f in sorted(out_dir.glob("*.json")):
            tf.add(f, arcname=f.name)
    if out_dir.parent.resolve() != ROOT.resolve():
        shutil.copy2(bundle_path, out_dir.parent / "results_bundle.tar.gz")
    print(f"Packed {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KB)", flush=True)
    print("===ALL_EXPERIMENTS_COMPLETE===", flush=True)


if __name__ == "__main__":
    main()
