"""
generate_figures.py -- Generates publication-quality PDF & PNG figures from
all experimental JSON files in paper/results/ into paper/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "paper" / "results"
FIG_DIR = ROOT / "paper" / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


def _load(name: str) -> dict | None:
    p = RES_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.pdf / .png")


def fig1_whitebox_matrix_convpoint() -> None:
    m9 = _load("m9_whitebox_convpoint.json")
    if not m9 or "summary" not in m9:
        return
    sa = m9["summary"].get("part_a_whitebox_matrix", {})
    sb = m9["summary"].get("part_b_conversation_points", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.3))

    # Left: White-box matrix
    regs_a = [
        ("clean", "Clean"),
        ("wb_gcg_text_on_text_comm", "WB-GCG\n(Text)"),
        ("wb_gcg_text_on_latent_comm", "WB-GCG\n(Latent)"),
        ("outofsight_1d_on_latent_comm", "1D Subspace\n(Latent)"),
        ("wb_fullspace_pgd_on_latent_comm", "Full PGD\n(Latent)"),
    ]
    labels_a = [lbl for k, lbl in regs_a if k in sa]
    asrs_a = [sa[k]["asr"] * 100 for k, _ in regs_a if k in sa]
    txt_catch = [sa[k]["text_or_ppl_catch_rate"] * 100 for k, _ in regs_a if k in sa]
    edge_catch = [sa[k]["edge_catch_rate"] * 100 for k, _ in regs_a if k in sa]

    x = np.arange(len(labels_a))
    w = 0.26
    ax1.bar(x - w, asrs_a, width=w, label="Attack ASR (%)", color="#1f497d")
    ax1.bar(x, txt_catch, width=w, label="Text/PPL Monitor (%)", color="#d95f02")
    ax1.bar(x + w, edge_catch, width=w, label="Edge Monitor (%)", color="#1b9e77")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_a)
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("(a) White-Box Text vs. Latent Expressivity")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left", frameon=True)

    # Right: Conversation Points
    regs_b = [
        ("0_conv_pure_latent", "0 Conv Pts\n(Pure Latent)"),
        ("1_conv_late_unadapted", "1 Conv Pt\n(Unadapted)"),
        ("1_conv_late_relay_adapted", "1 Conv Pt\n(Relay PGD)"),
        ("1_conv_early_wb_text", "1 Conv Pt\n(Early GCG)"),
        ("2_conv_all_text", "2 Conv Pts\n(All Text)"),
    ]
    labels_b = [lbl for k, lbl in regs_b if k in sb]
    asrs_b = [sb[k]["asr"] * 100 for k, _ in regs_b if k in sb]
    catch_b = [sb[k]["text_monitor_catch_rate"] * 100 for k, _ in regs_b if k in sb]
    xb = np.arange(len(labels_b))
    w2 = 0.34
    ax2.bar(xb - w2 / 2, asrs_b, width=w2, label="Executor ASR (%)", color="#1f497d")
    ax2.bar(xb + w2 / 2, catch_b, width=w2, label="Text Monitor Catch (%)", color="#d95f02")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels_b)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("(b) Causal Impact of Text Conversation Points (3-Agent)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    _save(fig, "fig1_whitebox_matrix_convpoint")


def fig2_m3_nll_convergence() -> None:
    m3 = _load("m3_pgd.json")
    if not m3 or "configs" not in m3:
        return
    steps_120 = [0, 15, 30, 45, 60, 75, 90, 105, 119]
    styles = {
        "late4_n8": ("Late Block {30..33}, rho=8", "#1f77b4", "o"),
        "spread4_n8": ("Spread {20,24,28,32}, rho=8", "#ff7f0e", "s"),
        "spread4_n16": ("Spread {20,24,28,32}, rho=16", "#2ca02c", "^"),
        "deep6_n16": ("Deep6 {12,16,20,24,28,32}, rho=16", "#d62728", "D"),
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for k, (label, col, mk) in styles.items():
        cfg = m3["configs"].get(k)
        if not cfg or "target_logp_hist" not in cfg:
            continue
        ys = cfg["target_logp_hist"]
        xs = steps_120[: len(ys)]
        ax.plot(xs, ys, marker=mk, linewidth=2, color=col, label=label)
    ax.set_xlabel("PGD Optimization Step")
    ax.set_ylabel("Target Tool-Call Log-Probability")
    ax.set_title("M3: Target Action Log-Probability Optimization Across Layer Placements")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    _save(fig, "fig2_m3_nll_convergence")


def fig3_transferability() -> None:
    m10 = _load("m10_transferability.json")
    if not m10 or "summary" not in m10:
        return
    s = m10["summary"]
    ct = s.get("cross_task", {})
    topo = s.get("cross_topology", {})
    sch = s.get("cross_schema", {})
    bf16 = s.get("cross_precision_bf16", {})

    cats = [
        "Cross-Task\n(Unseen Prompts)",
        "Star\n(3 Workers)",
        "3-Hop\nPure Latent",
        "DAG\n(Dual Merge)",
        "Schema &\nPrompt Shift",
        "Precision\n(NF4 -> bf16)",
    ]
    patch_vals = [
        ct.get("patch_asr", 0) * 100,
        topo.get("star_patch_asr", 0) * 100,
        topo.get("pure_latent_3agent_patch_asr", 0) * 100,
        topo.get("reconvergent_dag_dual_patch_asr", 0) * 100,
        sch.get("patch_asr", 0) * 100,
        bf16.get("patch_asr", 0) * 100,
    ]
    bcast_vals = [
        ct.get("broadcast_asr", 0) * 100,
        topo.get("star_broadcast_asr", 0) * 100,
        0.0,
        0.0,
        sch.get("broadcast_asr", 0) * 100,
        bf16.get("broadcast_asr", 0) * 100,
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.1))
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w / 2, patch_vals, width=w, label="Universal KV Prefix Patch (16 slots, 0 test steps)", color="#1f497d")
    ax.bar(x + w / 2, bcast_vals, width=w, label="Universal Broadcast Delta (1 slot, 0 test steps)", color="#6baed6")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Zero-Shot Transfer ASR (%)")
    ax.set_title("M10: Zero-Shot Latent Attack Transfer Across Tasks, Topologies, Schemas, and Precision")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    _save(fig, "fig3_transferability")


def fig4_mas_benchmarks() -> None:
    m6 = _load("m6_benchmark.json")
    m7 = _load("m7_text_baseline.json")
    m11 = _load("m11_mas_benchmarks.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.2))

    # Left: TAMAS 5-domain ASR & ERS
    if m11 and "tamas" in m11 and m11["tamas"].get("by_domain"):
        by_dom = m11["tamas"]["by_domain"]
        doms = list(by_dom.keys())
        x = np.arange(len(doms))
        w = 0.25
        dpi_asr = [by_dom[d]["tamas_text_dpi"]["asr"] * 100 for d in doms]
        lat_asr = [by_dom[d]["silent_zombie_latent"]["asr"] * 100 for d in doms]
        def_asr = [by_dom[d]["edge_defended"]["asr"] * 100 for d in doms]
        ax1.bar(x - w, dpi_asr, width=w, label="TAMAS Text DPI ASR", color="#d95f02")
        ax1.bar(x, lat_asr, width=w, label="Silent Zombie Latent ASR", color="#1f497d")
        ax1.bar(x + w, def_asr, width=w, label="Edge-Defended ASR", color="#1b9e77")
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.capitalize() for d in doms])
        ax1.set_ylim(0, 115)
        ax1.set_ylabel("Malicious Tool Execution ASR (%)")
        ax1.set_title("(a) Microsoft TAMAS 4-Agent MAS Across 5 Domains")
        ax1.grid(axis="y", alpha=0.25)
        ax1.legend(loc="upper right", frameon=True)

    # Right: AgentDojo Multi-Suite Comparison (Clean vs Text vs Latent vs Edge-Defended)
    if m6 and m7 and "by_suite" in m6:
        suites = [s for s in ("workspace", "banking", "slack") if s in m6["by_suite"]]
        x2 = np.arange(len(suites))
        w2 = 0.20
        txt_asr = [m7["by_suite"].get(s, {}).get("inter_agent_text", {}).get("asr", 0) * 100 for s in suites]
        lat_asr = [m6["by_suite"][s]["asr_compromise_attacked"] * 100 for s in suites]
        def_asr = [m6["by_suite"][s]["asr_defended_edge_gate"] * 100 for s in suites]
        edge_tpr = [m6["by_suite"][s]["edge_monitor_tpr"] * 100 for s in suites]
        ax2.bar(x2 - 1.5 * w2, txt_asr, width=w2, label="Text-Channel ASR", color="#d95f02")
        ax2.bar(x2 - 0.5 * w2, lat_asr, width=w2, label="Latent-PGD ASR", color="#1f497d")
        ax2.bar(x2 + 0.5 * w2, def_asr, width=w2, label="Edge-Gated ASR", color="#1b9e77")
        ax2.bar(x2 + 1.5 * w2, edge_tpr, width=w2, label="Edge Monitor TPR", color="#7570b3")
        ax2.set_xticks(x2)
        ax2.set_xticklabels([s.capitalize() for s in suites])
        ax2.set_ylim(0, 115)
        ax2.set_ylabel("Rate (%)")
        ax2.set_title("(b) AgentDojo Multi-Suite Evaluation (24 Pairs)")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    _save(fig, "fig4_mas_benchmarks")


def fig5_reach_and_stealth() -> None:
    m4b = _load("m4b_reach.json")
    m4d = _load("m4d_purelatent.json")
    m5 = _load("m5_stealth.json")
    m8 = _load("m8_shm_edge_defense.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.1))

    # Left: Reach vs Depth
    if m4b and "reach_horizon" in m4b:
        rh = m4b["reach_horizon"]["8.0"]["reach_by_depth"]
        ax1.plot(range(1, len(rh) + 1), [v * 100 for v in rh], "o--", color="#d95f02", linewidth=2, label="Text-Mediated Chain (rho=8)")
    if m4d and "depths" in m4d:
        ds = sorted(int(k) for k in m4d["depths"].keys())
        ys = [m4d["depths"][str(d)]["purelatent_reach"] * 100 for d in ds]
        ax1.plot(ds, ys, "s-", color="#1f497d", linewidth=2.2, label="Pure-Latent Differentiable Chain (rho=8)")
    ax1.set_xlabel("Hop Distance from Compromised Zombie")
    ax1.set_ylabel("Target Control / Marker Rate (%)")
    ax1.set_ylim(-5, 112)
    ax1.set_title("(a) Multi-Hop Reach: Decoding Bottleneck vs. Pure Latent")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower left", frameon=True)

    # Right: Stealth vs Dual-Evasion
    if m8 and "dual_evasion" in m8 and "regimes" in m8["dual_evasion"]:
        regs = m8["dual_evasion"]["regimes"]
        order = ["naive_pgd", "acmas_evasion_only", "dual_evasion_low", "dual_evasion_high", "dual_evasion_strict"]
        names = [k for k in order if k in regs]
        lbls = ["Naive\nPGD", "AcMAS\nEvasion", "Dual\n(le=50)", "Dual\n(le=500)", "Dual\n(le=2500)"][: len(names)]
        asr = [regs[k]["injection_asr"] * 100 for k in names]
        ac_f = [regs[k]["acmas_flag_rate"] * 100 for k in names]
        ed_f = [regs[k]["edge_flag_rate"] * 100 for k in names]
        x = np.arange(len(names))
        w = 0.26
        ax2.bar(x - w, asr, width=w, label="Attack Success (%)", color="#1f497d")
        ax2.bar(x, ac_f, width=w, label="AcMAS Flag Rate (%)", color="#e7298a")
        ax2.bar(x + w, ed_f, width=w, label="Edge Monitor Flag (%)", color="#1b9e77")
        ax2.set_xticks(x)
        ax2.set_xticklabels(lbls)
        ax2.set_ylim(0, 115)
        ax2.set_ylabel("Rate (%)")
        ax2.set_title("(b) Adaptive Stealth: Evading AcMAS vs. Edge Monitor")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(loc="upper right", frameon=True)
    elif m5 and "lambdas" in m5:
        ls = sorted(float(k) for k in m5["lambdas"].keys())
        inj = [m5["lambdas"][str(l)]["injection_rate"] * 100 for l in ls]
        flg = [m5["lambdas"][str(l)]["acmas_flag_rate"] * 100 for l in ls]
        ax2.plot(ls, inj, "o-", label="Injection Success (%)")
        ax2.plot(ls, flg, "s--", label="AcMAS Flag Rate (%)")
        ax2.legend()

    fig.tight_layout()
    _save(fig, "fig5_reach_and_stealth")


def fig6_shm_bus_latency() -> None:
    m8 = _load("m8_shm_edge_defense.json")
    if not m8 or "ipc_benchmark" not in m8:
        return
    benches = m8["ipc_benchmark"].get("benchmarks", [])
    if not benches:
        return
    lens = [str(b["seq_len"]) + f"\n({b['payload_mb']:.1f} MB)" for b in benches]
    shm_ms = [b["shm_ms_mean"] for b in benches]
    ser_ms = [b["serialized_ms_mean"] for b in benches]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(lens))
    w = 0.35
    ax.bar(x - w / 2, shm_ms, width=w, label="POSIX /dev/shm Zero-Copy + HMAC-SHA256", color="#1b9e77")
    ax.bar(x + w / 2, ser_ms, width=w, label="Serialized Pickle/torch.save IPC Baseline", color="#7570b3")
    for idx, b in enumerate(benches):
        ax.text(idx - w / 2, shm_ms[idx] + max(ser_ms) * 0.02, f"{b['speedup_x']}x faster", ha="center", fontsize=8, fontweight="bold", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(lens)
    ax.set_ylabel("Round-Trip IPC Latency (ms)")
    ax.set_title("M8: POSIX /dev/shm Tensor Bus vs. Serialized KV-Cache IPC (36 Layers, bf16)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    _save(fig, "fig6_shm_bus_latency")


def main() -> None:
    fig1_whitebox_matrix_convpoint()
    fig2_m3_nll_convergence()
    fig3_transferability()
    fig4_mas_benchmarks()
    fig5_reach_and_stealth()
    fig6_shm_bus_latency()


if __name__ == "__main__":
    main()
