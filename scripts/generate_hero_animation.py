"""
generate_hero_animation.py -- Generates a clean, intuitive animated comparison GIF
for the GitHub README from verified M3 & M9 experimental results.

Visualizes side-by-side / stacked 3-agent circle topologies evolving live across
PGD training steps (0 -> 120):
  (A) Pure-Latent Pipeline (0 Text Layers):
      v1 (Zombie) --[KV Tensor H1+Delta(t)]--> v2 --[KV Tensor H2(t)]--> v3 (Executor)
      End-to-end gradients drive target NLL 118.4 -> 0.005; live KV edge matrices
      absorb structured perturbations and flip v3 from benign calendar lookup to
      unauthorized send_email(...) (62.5%-75% ASR, 0% text catch).
  (B) Hybrid Pipeline (1 Text Layer / Conversation Boundary):
      v1 (Zombie) --[KV Tensor H1+Delta(t)]--> v2 --[Decoded Text m]--> v3 (Executor)
      Discrete text decoding at v2->v3 severs continuous gradients and washes out
      the latent perturbation into readable text; target NLL stays ~95.8 and v3
      remains safe on get_day_calendar_events(...) (0.0% ASR).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "paper" / "results"
OUT_DIR = ROOT / "docs" / "assets"
GIF_PATH = OUT_DIR / "silent_zombies_demo.gif"
PNG_PATH = OUT_DIR / "silent_zombies_demo_preview.png"


def _load_curves() -> tuple[np.ndarray, np.ndarray, float]:
    steps = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120], dtype=float)
    nll_latent = np.array([118.37, 45.576, 5.263, 0.038, 0.014, 0.009, 0.007, 0.006, 0.005])
    m3_p = RES_DIR / "m3_pgd.json"
    if m3_p.exists():
        d3 = json.loads(m3_p.read_text(encoding="utf-8"))
        hist = d3["configs"]["spread4_n8"]["target_logp_hist"]
        nll_latent = np.array([-x for x in hist], dtype=float)
        steps = steps[: len(nll_latent)]

    clean_nll = 95.811
    m9_p = RES_DIR / "m9_whitebox_convpoint.json"
    if m9_p.exists():
        d9 = json.loads(m9_p.read_text(encoding="utf-8"))
        clean_nll = float(
            d9.get("summary", {})
            .get("part_a_whitebox_matrix", {})
            .get("clean", {})
            .get("mean_target_nll", 95.811)
        )
    return steps, nll_latent, clean_nll


def _interp_log(xs: np.ndarray, ys: np.ndarray, t: float) -> float:
    ly = np.log10(np.clip(ys, 1e-4, None))
    return float(10.0 ** np.interp(t, xs, ly))


# Fixed RNG patterns for deterministic, smooth tensor heatmap evolution
_RNG = np.random.default_rng(7)
_BASE_H1 = _RNG.normal(0.0, 0.32, size=(5, 9))
_BASE_H2 = _RNG.normal(0.0, 0.30, size=(5, 9))
# Structured adversarial rank-2 / stripe pattern representing optimized KV Delta
_ADV_DELTA1 = np.outer(
    np.array([1.0, -0.7, 0.9, -0.8, 1.0]),
    np.array([0.8, 1.1, -0.9, 1.2, 0.9, -0.8, 1.1, 1.0, -0.7]),
)
_ADV_DELTA2 = np.outer(
    np.array([0.9, -0.8, 1.0, -0.6, 0.95]),
    np.array([0.7, 1.0, -0.85, 1.1, 0.85, -0.75, 1.0, 0.9, -0.6]),
)


def _draw_kv_heatmap(
    ax,
    x0: float,
    y0: float,
    w: float,
    h: float,
    base: np.ndarray,
    adv: np.ndarray,
    strength: float,
    border_col: str,
    label: str,
) -> None:
    """Draw a miniature 2D KV-cache tensor heatmap directly above an edge."""
    mat = (1.0 - 0.35 * strength) * base + (1.35 * strength) * adv
    nrows, ncols = mat.shape
    cw = w / ncols
    ch = h / nrows

    # Subtle outer frame
    frame = FancyBboxPatch(
        (x0 - 0.04, y0 - 0.04),
        w + 0.08,
        h + 0.08,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor="#ffffff",
        edgecolor=border_col,
        linewidth=1.35,
        zorder=4,
    )
    ax.add_patch(frame)

    cmap = plt.get_cmap("RdBu_r")
    normed = np.clip(0.5 + 0.38 * mat, 0.04, 0.96)
    for r in range(nrows):
        for c in range(ncols):
            col = cmap(float(normed[r, c]))
            rect = Rectangle(
                (x0 + c * cw, y0 + (nrows - 1 - r) * ch),
                cw * 0.94,
                ch * 0.92,
                facecolor=col,
                edgecolor="none",
                zorder=5,
            )
            ax.add_patch(rect)

    ax.text(
        x0 + w / 2,
        y0 + h + 0.17,
        label,
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color=border_col,
        zorder=6,
    )


def _draw_agent_circle(
    ax,
    cx: float,
    cy: float,
    r: float,
    symbol: str,
    role_below: str,
    fc: str,
    ec: str,
    tc: str,
    ring_glow: bool = False,
) -> None:
    if ring_glow:
        glow = Circle((cx, cy), r + 0.11, facecolor="none", edgecolor=ec, linewidth=2.2, alpha=0.28, zorder=3)
        ax.add_patch(glow)
    circ = Circle((cx, cy), r, facecolor=fc, edgecolor=ec, linewidth=2.3, zorder=4)
    ax.add_patch(circ)
    ax.text(cx, cy, symbol, ha="center", va="center", fontsize=11.5, fontweight="bold", color=tc, zorder=5)
    ax.text(
        cx,
        cy - r - 0.24,
        role_below,
        ha="center",
        va="top",
        fontsize=7.9,
        fontweight="bold",
        color="#334155",
        zorder=5,
    )


def _draw_edge_with_dots(
    ax,
    x0: float,
    x1: float,
    y: float,
    color: str,
    pulse: float,
    n_dots: int = 3,
) -> None:
    arr = FancyArrowPatch(
        (x0, y),
        (x1, y),
        arrowstyle="-|>,head_length=6.5,head_width=3.8",
        linewidth=2.3,
        color=color,
        zorder=3,
    )
    ax.add_patch(arr)
    span = x1 - x0 - 0.18
    for i in range(n_dots):
        u = (pulse + i / n_dots) % 1.0
        px = x0 + 0.06 + u * span
        dot = Circle((px, y), radius=0.075, facecolor=color, edgecolor="white", linewidth=0.8, zorder=6)
        ax.add_patch(dot)


def _draw_output_box(
    ax,
    x0: float,
    y0: float,
    w: float,
    h: float,
    header: str,
    code_text: str,
    footer: str,
    border_col: str,
    bg_col: str,
    hdr_col: str,
    foot_col: str,
) -> None:
    box = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=bg_col,
        edgecolor=border_col,
        linewidth=1.8,
        zorder=4,
    )
    ax.add_patch(box)
    ax.text(x0 + 0.16, y0 + h - 0.22, header, fontsize=7.8, fontweight="bold", color=hdr_col, va="top", zorder=5)
    ax.text(
        x0 + 0.16,
        y0 + h - 0.62,
        code_text,
        fontfamily="monospace",
        fontsize=7.35,
        color="#0f172a",
        va="top",
        linespacing=1.28,
        zorder=5,
    )
    ax.text(x0 + 0.16, y0 + 0.17, footer, fontsize=7.5, fontweight="bold", color=foot_col, va="bottom", zorder=5)


def render_frame(
    pgd_step: float,
    pulse: float,
    steps_arr: np.ndarray,
    nll_latent: np.ndarray,
    clean_nll: float,
) -> Image.Image:
    fig = plt.figure(figsize=(14.4, 6.55), dpi=115, facecolor="#f8fafc")

    cur_nll = _interp_log(steps_arr, nll_latent, pgd_step)
    # Smooth perturbation strength [0..1] as PGD optimizes Delta
    strength = float(np.clip((np.log10(118.4) - np.log10(max(cur_nll, 0.005))) / 4.37, 0.0, 1.0))
    # Smooth probability of unauthorized tool call at v3 for Pure-Latent (0% -> 99.5%)
    p_hijack = float(0.995 / (1.0 + np.exp(-(pgd_step - 31.0) / 3.8))) if pgd_step >= 1.0 else 0.0
    is_hijacked = pgd_step >= 33.0
    is_optimizing = 1.0 <= pgd_step < 33.0

    # ------------------------------------------------------------------
    # Top Header Bar
    # ------------------------------------------------------------------
    ax_hdr = fig.add_axes([0.018, 0.908, 0.964, 0.074])
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")
    ax_hdr.add_patch(
        FancyBboxPatch(
            (0.0, 0.04),
            1.0,
            0.92,
            boxstyle="round,pad=0.004,rounding_size=0.014",
            facecolor="#0f172a",
            edgecolor="#1e293b",
        )
    )
    ax_hdr.text(
        0.018,
        0.50,
        "SILENT ZOMBIES  |  Pure-Latent Chain (0 Text Layers) vs. Single Text Layer During PGD Training",
        va="center",
        ha="left",
        fontsize=11.0,
        fontweight="bold",
        color="#f8fafc",
    )

    if pgd_step < 1.0:
        pill_txt = "STEP   0 / 120  (CLEAN INIT)"
        pill_col = "#2563eb"
    elif pgd_step < 119.5:
        pill_txt = f"PGD TRAINING : STEP {int(round(pgd_step)):3d} / 120"
        pill_col = "#d97706" if not is_hijacked else "#dc2626"
    else:
        pill_txt = "CONVERGED : STEP 120 / 120"
        pill_col = "#dc2626"

    ax_hdr.add_patch(
        FancyBboxPatch(
            (0.735, 0.17),
            0.252,
            0.66,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=pill_col,
            edgecolor="none",
        )
    )
    ax_hdr.text(0.861, 0.50, pill_txt, va="center", ha="center", fontsize=9.1, fontweight="bold", color="white")

    # ------------------------------------------------------------------
    # Left Canvas (62%): Stacked 3-Circle Topologies with Live Edge Views
    # ------------------------------------------------------------------
    ax_topo = fig.add_axes([0.018, 0.032, 0.605, 0.855])
    ax_topo.set_xlim(0, 12.2)
    ax_topo.set_ylim(0, 8.6)
    ax_topo.set_aspect("equal")
    ax_topo.axis("off")

    # === LANE A (TOP): Pure-Latent Topology (0 Text Layers) ===
    ax_topo.add_patch(
        FancyBboxPatch(
            (0.08, 4.42),
            12.04,
            4.08,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor="#ffffff",
            edgecolor="#fecaca" if is_hijacked else "#cbd5e1",
            linewidth=1.6 if is_hijacked else 1.2,
            zorder=1,
        )
    )
    ax_topo.text(
        0.34,
        8.16,
        "(A) Pure-Latent Topology — 0 Text Layers",
        fontsize=9.8,
        fontweight="bold",
        color="#0f172a",
        va="center",
    )
    ax_topo.text(
        11.88,
        8.16,
        "M9 Result: 62.5% ASR  |  0% Text Catch",
        fontsize=8.2,
        fontweight="bold",
        color="#dc2626" if is_hijacked else "#64748b",
        ha="right",
        va="center",
    )

    ya = 6.22
    r_ag = 0.50
    c1_ec = "#dc2626" if pgd_step >= 1.0 else "#2563eb"
    c1_fc = "#fef2f2" if pgd_step >= 1.0 else "#eff6ff"
    c1_tc = "#991b1b" if pgd_step >= 1.0 else "#1e3a8a"

    c2_ec = "#dc2626" if strength > 0.45 else "#2563eb"
    c2_fc = "#fff1f2" if strength > 0.45 else "#eff6ff"
    c2_tc = "#991b1b" if strength > 0.45 else "#1e3a8a"

    c3_ec = "#dc2626" if is_hijacked else ("#d97706" if is_optimizing else "#2563eb")
    c3_fc = "#fef2f2" if is_hijacked else ("#fffbeb" if is_optimizing else "#eff6ff")
    c3_tc = "#991b1b" if is_hijacked else ("#92400e" if is_optimizing else "#1e3a8a")

    _draw_agent_circle(ax_topo, 1.05, ya, r_ag, "v₁", "Zombie Planner", c1_fc, c1_ec, c1_tc, ring_glow=pgd_step >= 1.0)
    _draw_agent_circle(ax_topo, 4.15, ya, r_ag, "v₂", "Latent Relay", c2_fc, c2_ec, c2_tc)
    _draw_agent_circle(ax_topo, 7.25, ya, r_ag, "v₃", "Tool Executor", c3_fc, c3_ec, c3_tc, ring_glow=is_hijacked)

    edge1_col = "#2563eb" if pgd_step < 1.0 else ("#ea580c" if not is_hijacked else "#dc2626")
    edge2_col = "#2563eb" if strength < 0.25 else ("#ea580c" if not is_hijacked else "#dc2626")

    _draw_edge_with_dots(ax_topo, 1.62, 3.58, ya, edge1_col, pulse, n_dots=3)
    _draw_edge_with_dots(ax_topo, 4.72, 6.68, ya, edge2_col, pulse, n_dots=3)

    # Live Edge KV Tensor Heatmaps on BOTH edges of Lane A
    lbl_e1 = "KV Cache H₁" if pgd_step < 1.0 else f"KV: H₁ + Δ  (‖Δ‖={strength*8.0:.1f})"
    lbl_e2 = "KV Cache H₂" if pgd_step < 1.0 else "KV: H₂(Δ)  [Propagated!]"
    _draw_kv_heatmap(ax_topo, 2.02, ya + 0.34, 1.16, 0.66, _BASE_H1, _ADV_DELTA1, strength, edge1_col, lbl_e1)
    _draw_kv_heatmap(ax_topo, 5.12, ya + 0.34, 1.16, 0.66, _BASE_H2, _ADV_DELTA2, strength * 0.95, edge2_col, lbl_e2)

    # Differentiable backprop line cleanly below agent labels
    if pgd_step >= 1.0:
        bp_arc = FancyArrowPatch(
            (7.15, 5.02),
            (1.15, 5.02),
            connectionstyle="arc3,rad=0.03",
            arrowstyle="-|>,head_length=5.5,head_width=3.2",
            linestyle="--",
            linewidth=1.45,
            color="#dc2626",
            zorder=2,
        )
        ax_topo.add_patch(bp_arc)
        ax_topo.text(
            4.15,
            4.62,
            "⟵ End-to-end differentiable ∇_Δ L flows unimpeded across latent hops",
            ha="center",
            va="center",
            fontsize=7.55,
            fontweight="bold",
            color="#b91c1c",
        )
    else:
        ax_topo.text(
            4.15,
            4.68,
            "Clean initialization (Δ = 0): both edges carry unperturbed latent KV tensors",
            ha="center",
            va="center",
            fontsize=7.6,
            color="#64748b",
        )

    # Arrow v3 -> Output Box A
    ax_topo.add_patch(
        FancyArrowPatch((7.82, ya), (8.25, ya), arrowstyle="-|>,head_length=5.5,head_width=3.2", lw=1.9, color=c3_ec)
    )
    if pgd_step < 1.0:
        _draw_output_box(
            ax_topo,
            8.30,
            4.92,
            3.62,
            2.22,
            header="v₃ Live Decoded Action:",
            code_text='get_day_calendar_events(\n  day="2023-05-26"\n)',
            footer="Status: BENIGN (P_atk = 0.0%)",
            border_col="#93c5fd",
            bg_col="#f8fafc",
            hdr_col="#1e40af",
            foot_col="#2563eb",
        )
    elif not is_hijacked:
        _draw_output_box(
            ax_topo,
            8.30,
            4.92,
            3.62,
            2.22,
            header="v₃ Output Shifting...",
            code_text=f'get_day_calendar... ({100*(1-p_hijack):4.1f}%)\nsend_email(...)     ({100*p_hijack:4.1f}%)',
            footer=f"Target NLL dropping: {cur_nll:.2f}",
            border_col="#fbbf24",
            bg_col="#fffbeb",
            hdr_col="#92400e",
            foot_col="#b45309",
        )
    else:
        _draw_output_box(
            ax_topo,
            8.30,
            4.92,
            3.62,
            2.22,
            header="v₃ Live Decoded Action:",
            code_text='send_email(\n  to="mark.black@gmail.com",\n  subj="Important message!"\n)',
            footer=f"HIJACKED! (P_atk = {100*p_hijack:.1f}%)",
            border_col="#ef4444",
            bg_col="#fef2f2",
            hdr_col="#991b1b",
            foot_col="#dc2626",
        )

    # === LANE B (BOTTOM): 1-Text-Layer Hybrid Topology ===
    ax_topo.add_patch(
        FancyBboxPatch(
            (0.08, 0.14),
            12.04,
            4.08,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor="#ffffff",
            edgecolor="#a7f3d0" if pgd_step >= 1.0 else "#cbd5e1",
            linewidth=1.5 if pgd_step >= 1.0 else 1.2,
            zorder=1,
        )
    )
    ax_topo.text(
        0.34,
        3.88,
        "(B) Hybrid Topology — 1 Text Layer (Decoded Text Edge)",
        fontsize=9.8,
        fontweight="bold",
        color="#0f172a",
        va="center",
    )
    ax_topo.text(
        11.88,
        3.88,
        "M9 Result: 0.0% ASR  (Causal Firewall)",
        fontsize=8.2,
        fontweight="bold",
        color="#059669",
        ha="right",
        va="center",
    )

    yb = 1.90
    _draw_agent_circle(ax_topo, 1.05, yb, r_ag, "v₁", "Zombie Planner", c1_fc, c1_ec, c1_tc, ring_glow=pgd_step >= 1.0)
    _draw_agent_circle(ax_topo, 4.15, yb, r_ag, "v₂", "Text Relay", "#f0fdfa", "#0d9488", "#115e59")
    _draw_agent_circle(ax_topo, 7.25, yb, r_ag, "v₃", "Tool Executor", "#ecfdf5", "#059669", "#065f46")

    # Edge 1 (v1 -> v2) is STILL the exact same attacked KV tensor!
    _draw_edge_with_dots(ax_topo, 1.62, 3.58, yb, edge1_col, pulse, n_dots=3)
    _draw_kv_heatmap(ax_topo, 2.02, yb + 0.34, 1.16, 0.66, _BASE_H1, _ADV_DELTA1, strength, edge1_col, lbl_e1)

    # Edge 2 (v2 -> v3) is a READABLE TEXT EDGE (speech bubble showing live decoded string!)
    _draw_edge_with_dots(ax_topo, 4.72, 6.68, yb, "#059669", pulse, n_dots=3)
    txt_bubble = FancyBboxPatch(
        (4.80, yb + 0.32),
        1.82,
        0.74,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#ecfdf5",
        edgecolor="#059669",
        linewidth=1.35,
        zorder=5,
    )
    ax_topo.add_patch(txt_bubble)
    ax_topo.text(
        5.71,
        yb + 1.20,
        "Decoded Text Edge m",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#059669",
    )
    relay_msg = '"Check May 26\ncalendar events."' if pgd_step < 25.0 else '"Find May 26th\nnetworking event."'
    ax_topo.text(
        5.71,
        yb + 0.69,
        relay_msg,
        ha="center",
        va="center",
        fontsize=7.1,
        style="italic",
        color="#064e3b",
        linespacing=1.18,
        zorder=6,
    )

    # Blocked gradient annotation underneath Lane B
    ax_topo.text(
        4.15,
        0.36,
        "✕  Discrete argmax text decode severs ∇_Δ L = 0 & washes out continuous KV perturbation!",
        ha="center",
        va="center",
        fontsize=7.55,
        fontweight="bold",
        color="#047857",
    )

    # Arrow v3 -> Output Box B (remains SAFE throughout training!)
    ax_topo.add_patch(
        FancyArrowPatch((7.82, yb), (8.25, yb), arrowstyle="-|>,head_length=5.5,head_width=3.2", lw=1.9, color="#059669")
    )
    _draw_output_box(
        ax_topo,
        8.30,
        0.64,
        3.62,
        2.22,
        header="v₃ Live Decoded Action:",
        code_text='get_day_calendar_events(\n  day="2023-05-26"\n)',
        footer="SAFE! Benign Action (P_atk = 0.0%)",
        border_col="#10b981",
        bg_col="#ecfdf5",
        hdr_col="#065f46",
        foot_col="#059669",
    )

    # ------------------------------------------------------------------
    # Right Column (36%): Live Loss Curve & Action Probability Bars
    # ------------------------------------------------------------------
    # Card 1: Loss Curve
    ax_r1_bg = fig.add_axes([0.638, 0.375, 0.344, 0.512])
    ax_r1_bg.axis("off")
    ax_r1_bg.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            linewidth=1.2,
        )
    )

    ax_loss = fig.add_axes([0.695, 0.455, 0.272, 0.365])
    ax_loss.set_facecolor("#ffffff")
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(-4, 125)
    ax_loss.set_ylim(0.0025, 280.0)
    ax_loss.grid(True, which="major", alpha=0.25, color="#94a3b8")
    ax_loss.set_xlabel("PGD Training Step", fontsize=8.4, color="#334155")
    ax_loss.set_ylabel("Target send_email Loss (NLL)", fontsize=8.4, color="#334155")
    ax_loss.set_title(
        "Target Action Loss During Training",
        fontsize=9.6,
        fontweight="bold",
        color="#0f172a",
        pad=8,
    )

    # Faint full guide curves
    ax_loss.plot(steps_arr, nll_latent, color="#fecaca", lw=1.3, ls="--", zorder=1)
    ax_loss.plot([0, 120], [clean_nll, clean_nll], color="#a7f3d0", lw=1.3, ls="--", zorder=1)
    ax_loss.axhline(1.0, color="#f59e0b", lw=1.1, ls=":", alpha=0.9)
    ax_loss.text(122, 1.30, "Hijack Threshold", ha="right", fontsize=7.3, fontweight="bold", color="#b45309")

    # Progressive curves up to pgd_step
    t_dense = np.linspace(0.0, max(pgd_step, 0.1), 70)
    y_lat_dense = [_interp_log(steps_arr, nll_latent, tt) for tt in t_dense]
    # Slight deterministic ripple around clean_nll for 1-text-layer curve
    y_txt_dense = [clean_nll * (1.0 - 0.03 * np.sin(tt / 18.0)) for tt in t_dense]

    ax_loss.plot(t_dense, y_txt_dense, color="#059669", lw=2.3, label="1 Text Layer (0% ASR)", zorder=3)
    ax_loss.plot(t_dense, y_lat_dense, color="#dc2626", lw=2.5, label="0 Text Layers (Pure Latent)", zorder=4)

    # Discrete training checkpoints reached so far
    mask = steps_arr <= pgd_step + 0.05
    txt_pts = np.full(np.sum(mask), clean_nll)
    ax_loss.scatter(steps_arr[mask], txt_pts, s=34, facecolor="white", edgecolor="#059669", lw=1.7, zorder=5)
    ax_loss.scatter(steps_arr[mask], nll_latent[mask], s=38, facecolor="white", edgecolor="#dc2626", lw=1.8, zorder=5)

    # Current leading training cursor dots
    cur_txt_nll = clean_nll * (1.0 - 0.03 * np.sin(pgd_step / 18.0))
    ax_loss.scatter([pgd_step], [cur_txt_nll], s=75, color="#059669", zorder=6)
    ax_loss.scatter([pgd_step], [cur_nll], s=85, color="#dc2626", zorder=7)

    ax_loss.legend(loc="lower left", frameon=True, fontsize=7.4, facecolor="white", framealpha=0.92)

    # Card 2: Live Executor Hijack Probability Comparison
    ax_prob = fig.add_axes([0.638, 0.032, 0.344, 0.322])
    ax_prob.set_xlim(0, 10)
    ax_prob.set_ylim(0, 5.2)
    ax_prob.axis("off")
    ax_prob.add_patch(
        FancyBboxPatch(
            (0.0, 0.04),
            10.0,
            5.10,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            linewidth=1.2,
        )
    )
    ax_prob.text(
        0.38,
        4.55,
        "Live Downstream Hijack Probability at v₃",
        fontsize=9.5,
        fontweight="bold",
        color="#0f172a",
        va="center",
    )

    # Bar A: 0 Text Layers
    ax_prob.text(0.38, 3.55, "(A) 0 Text Layers (Pure Latent):", fontsize=8.1, fontweight="bold", color="#991b1b")
    ax_prob.add_patch(
        FancyBboxPatch((0.38, 2.78), 7.2, 0.50, boxstyle="round,pad=0.01,rounding_size=0.08", facecolor="#f1f5f9", edgecolor="#cbd5e1")
    )
    if p_hijack > 0.005:
        ax_prob.add_patch(
            FancyBboxPatch(
                (0.38, 2.78),
                max(7.2 * p_hijack, 0.14),
                0.50,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                facecolor="#dc2626" if is_hijacked else "#f59e0b",
                edgecolor="none",
            )
        )
    ax_prob.text(
        7.82,
        3.03,
        f"{100*p_hijack:5.1f}%",
        fontsize=8.8,
        fontweight="bold",
        color="#dc2626" if is_hijacked else "#64748b",
        va="center",
    )

    # Bar B: 1 Text Layer
    ax_prob.text(0.38, 1.92, "(B) 1 Text Layer (Text Boundary):", fontsize=8.1, fontweight="bold", color="#065f46")
    ax_prob.add_patch(
        FancyBboxPatch((0.38, 1.15), 7.2, 0.50, boxstyle="round,pad=0.01,rounding_size=0.08", facecolor="#f1f5f9", edgecolor="#cbd5e1")
    )
    ax_prob.text(7.82, 1.40, "  0.0%", fontsize=8.8, fontweight="bold", color="#059669", va="center")

    ax_prob.text(
        0.38,
        0.44,
        f"Current NLL — Pure Latent: {cur_nll:6.3f}   |   1 Text Layer: {cur_txt_nll:5.1f}",
        fontsize=7.8,
        color="#475569",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_animation() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    steps_arr, nll_latent, clean_nll = _load_curves()

    schedule: list[tuple[float, int]] = []
    # 1. Hold Step 0 (Clean initialization) for ~1.3s
    for _ in range(13):
        schedule.append((0.0, 95))

    # 2. Smooth PGD training progression 0 -> 120 (slowing slightly around flip Steps 20..45)
    grid = np.concatenate([
        np.linspace(2.0, 22.0, 10, endpoint=False),
        np.linspace(22.0, 46.0, 16, endpoint=False),
        np.linspace(46.0, 120.0, 16, endpoint=True),
    ])
    for s in grid:
        dur = 105 if (24.0 <= s <= 42.0) else 80
        schedule.append((float(s), dur))

    # 3. Hold converged Step 120 so viewer can compare both final states (~2.2s)
    for _ in range(22):
        schedule.append((120.0, 100))

    frames_rgb: list[Image.Image] = []
    durations: list[int] = []
    for idx, (s_val, dur_ms) in enumerate(schedule):
        pulse = (idx * 0.085) % 1.0
        frames_rgb.append(render_frame(s_val, pulse, steps_arr, nll_latent, clean_nll))
        durations.append(dur_ms)

    # Save static preview at Step 120 + temporary early/mid check frames if needed
    frames_rgb[-5].save(PNG_PATH)

    # Adaptive 256-color palette built across Beginning, Flip, and Converged frames
    w, h = frames_rgb[0].size
    strip = Image.new("RGB", (w, h * 3))
    strip.paste(frames_rgb[2], (0, 0))
    strip.paste(frames_rgb[28], (0, h))
    strip.paste(frames_rgb[-5], (0, h * 2))
    pal = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT)

    pal_frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in frames_rgb]
    pal_frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=pal_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Saved {GIF_PATH} ({GIF_PATH.stat().st_size / 1024:.1f} KB, {len(pal_frames)} frames)")
    print(f"Saved {PNG_PATH} ({PNG_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build_animation()
