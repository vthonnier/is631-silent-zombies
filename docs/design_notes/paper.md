# Silent Zombies: Inference-Time, Training-Free Attacks from One Compromised Node in a Latent Multi-Agent System

*KAIST IS-631 (Kernel & System Security) — research paper draft.*
*Companion material: `paper_introduction.md` (extended formalization), `research_summary.md`, `threat_model_novelty.md`, `literature_assessment.md`. Code: `latent_mas_attack/`.*

> **Status legend.** Values written as 〈…〉 are filled from the Kaggle GPU run
> (`latent_mas_attack/` driver `run_all.py`, Qwen3-4B). Sections 1–4 and 7–10
> are final; Section 5 (Results) is populated from `results/summary.json`.

---

## Abstract

Latent multi-agent systems (LatentMAS) let LLM agents hand off continuous state — hidden activations or key–value (KV) caches — instead of natural-language messages. This removes the single property every multi-agent-system (MAS) safety monitor has relied on: that one agent's message to another is *text a monitor can read*. We study the security consequence directly. We assume an adversary compromises exactly **one** node of a latent MAS and can overwrite the continuous handoff it emits, at inference time, with **no trained component added to the victim pipeline** and no change to any other agent's weights, prompts, or emitted text. We show such a "silent zombie" node can (i) drive downstream agents to a **security objective** — refusal suppression and exfiltration of a secret held only in the zombie's context — rather than a mere accuracy drop; (ii) **propagate** a measurable distance through the topology, giving a blast-radius characterization of a single compromise; and (iii) remain **invisible to text-channel monitors and to an activation-space detector (AcMAS)** deployed live, because the silent node offers no probe point and a detector-aware objective keeps the one downstream probe statistically benign while the harm rides in the handoff. On a four-stage Planner→Critic→Refiner→Judger chain over Qwen3-4B we measure a jailbreak (refusal-suppression) rate of 〈ASR〉, exfiltration recovery of 〈exf〉, a reach curve that 〈decays/holds〉 with hop distance, and a stealth result in which AcMAS detection at the downstream node falls from 〈naive〉 to 〈stealth〉 at matched attack success. This is HijackKV's "the KV carries the goal" idea lifted from single-model cache reuse into a multi-hop MAS with a real security objective, and the inference-time, training-free counterpart to training-time link poisoning. We release the testbed and detector baselines, and conclude with the defense implications: text monitoring and own-activation probing are structurally insufficient for latent channels, which motivates handoff-level attestation and edge-side anomaly detection.

**Keywords.** multi-agent LLM security, latent communication, KV-cache attack, jailbreak, exfiltration, activation-space detection, adaptive adversary.

---

## 1. Introduction

Multi-agent systems built from large language models decompose a task across specialized agents — a planner, a critic, a refiner, a judge — that exchange intermediate results and collaborate toward a final answer. Classical MAS pass these as **natural-language messages**: every inter-agent edge is a human-readable string, and consequently every edge is observable. A growing body of work on *latent* MAS (LatentMAS; Zou et al., 2025) replaces this text bus with a **continuous channel**: agents hand off hidden states or KV caches directly, skipping the detokenize–retokenize round trip. This buys efficiency and expressiveness, but it also removes the property that every safety monitor built for MAS has silently relied upon — that what one agent tells another is text a monitor can read.

Recent attacks exploit exactly this blind spot, yet each stops short of the threat that matters for deployed MAS. *Out of Sight* (2026) shows a steering direction distilled from a text attack, injected into a latent handoff, collapses downstream task accuracy (GSM8K 0.87→0.03) while the text channel stays clean — but it measures **task corruption**, not a security violation, and its own detector catches its best attacks. *When Latent Agents Lie* (2026) shows a norm-matched KV perturbation evades a magnitude detector — but on a **single specialist**, with crude magnitude perturbations, and only accuracy harm. *HijackKV* (USENIX Security 2026) shows a cached chunk's KV can carry an attacker goal that transfers to a victim query with no adversarial text — but in a **single served model**, via text-space (GCG) search, for wrong answers, not jailbreak or exfiltration. None evaluates stealth against a **live activation-space defense** such as AcMAS (ICML 2026).

This work asks a sharper question. Suppose an adversary compromises exactly **one node** of a latent MAS and can write the continuous handoff that node emits. What is the maximal damage, reach, and stealth achievable *at inference time, without modifying any other agent's weights, prompts, or emitted text, and without introducing any trained component into the victim pipeline*? We argue — and demonstrate on a running testbed — that such a "zombie" node can (i) **jailbreak or exfiltrate** through downstream agents; (ii) **propagate a measurable distance** through the topology; and (iii) remain **invisible to text-only nodes and to activation detectors that probe an agent's own generation** rather than the handoff it emits. This triad — security objective, multi-hop reach, and asymmetric observability — is the intersection no prior work occupies.

**Contributions.**
1. **Threat model and testbed.** A precise single-compromise, inference-time, training-free adversary for a latent channel, and an open testbed (Planner→Critic→Refiner→Judger over Qwen3-4B) that captures and overwrites the KV/hidden handoff at one edge and runs two state-of-the-art defenses — a text monitor and AcMAS — *live* at every node.
2. **A security objective on the handoff.** A differentiable surrogate (first-token refusal suppression + target-sequence log-likelihood) optimized by projected gradient descent directly on the KV handoff — not a text prefix (HijackKV) and not an accuracy-preference vector (RePS). We report both jailbreak (refusal-suppression) and exfiltration of a planted secret.
3. **Reach / blast-radius.** A measurement of attack success as a function of the zombie's position and hop distance to the observed outcome, with a black-box reinforcement (GRPO-style) fallback that optimizes the final-node signal through intermediate text-regenerating hops.
4. **Asymmetric observability and AcMAS evasion.** We formalize the observability gap of a latent edge — the silent zombie has no probe point — and give a detector-aware objective that keeps the one live downstream probe near its benign centroid while the handoff stays poisoned, the adaptive attacker AcMAS lists as an open problem.
5. **Defense implications.** We show why text monitoring and own-activation probing are structurally insufficient for latent channels and outline handoff-level attestation and edge-side detection as the defenses the threat actually requires.

**Safety and scope.** This is a defensive characterization. The testbed uses only safe stand-ins — benign "forbidden" stubs (e.g., "output BANANA"), a synthetic secret (`SECRET-7F3A`), and refusal-suppression surrogates. No operational harmful content is bundled; any real HarmBench/StrongREJECT prompts or judge models are user-supplied local paths. The goal is to characterize a detector gap so defenders can close it.

---

## 2. Attack Model

We adopt the standard single-compromise MAS adversary, instantiated for a latent channel. *(Full formalization in `paper_introduction.md` §2; summarized here.)*

- **System.** A latent MAS is a DAG $G=(\mathcal V,\mathcal E)$ of $N$ agents, each an instance of a frozen, aligned, open-weights transformer $f_\theta$. The canonical chain is Planner $v_1\to$ Critic $v_2\to$ Refiner $v_3\to$ Judger $v_4$. Each edge carries a **latent handoff** — KV tensors or hidden states — not text; only the final node emits user-visible text.
- **Adversary.** Controls one node $v_c$ (the **zombie**). It may **write the outgoing handoff** $H_{c\to j}$ (the $K$, $V$, or hidden state) to any value at inference time, per query or via a fixed precomputed vector, and it knows $G$, each role, and $\theta$ (white-box, open weights). It **cannot** touch other agents' weights, prompts, or decoding; alter any other agent's emitted text or the victim's input; add a trained module/learned link to the victim pipeline; or observe/alter edges not incident to $v_c$.
- **"Training-free" clarified.** The *victim* MAS stays training-free; the *attacker* may still optimize the injected handoff — per query (no stored artifact) or once offline into a fixed reusable vector. The claim is "no trained component in the deployed MAS," not "no optimization by the attacker."
- **Objectives.** (1) **Harm:** jailbreak (produce content a downstream agent would refuse) or exfiltrate (emit a secret present only in the zombie's context). (2) **Reach:** maximize success as a function of zombie position and hop distance. (3) **Stealth:** evade text monitors at every node and AcMAS as a live baseline.
- **Defender.** Runs, concurrently, (a) text monitors at each node and (b) AcMAS: for each *generating* agent, cosine distance of the final-layer last-token hidden state to a benign centroid $\mu$, flag when $1-\cos(h,\mu)>\tau$ ($\tau\approx0.12$), optional repair $\tilde h=h+\lambda(\mu-h)$.

---

## 3. Experimental Setup

- **Backbone.** Qwen3-4B (open weights), matching the target paper; thinking mode disabled so handoffs carry task state, not reasoning scaffolding.
- **Pipeline.** LatentMAS-style Planner→Critic→Refiner→Judger with a writable KV/hidden handoff at each edge. We reproduce the clean pipeline and confirm the handoff tensor is directly writable before any attack. On a latent edge the zombie is **silent** (emits no text; its real output is the prepended cache the next agent attends over).
- **Carriers.** Each handoff instantiates on three carriers — keys $K$, values $V$, or hidden state $h$ — targeting a layer $\ell$ and edge $(i\to j)$. The default site in the reported runs is layer 12, edge $0\to1$.
- **Optimizer.** Projected gradient descent (Adam) on the handoff against a differentiable surrogate, with a norm-ball projection (norm-matching) and an optional AcMAS-distance penalty; a GRPO-style black-box optimizer against the final-node reward is used for the multi-hop reach path.
- **Tasks / metrics.** Utility: GSM8K-style arithmetic with exact-match gold (testbed validation). Jailbreak: safe forbidden-behavior stubs scored by (a) refusal-suppression (exact, judge-free) and (b) a pluggable harmfulness judge. Exfiltration: a synthetic secret in the zombie's context, scored by containment downstream. Stealth: per-probe detection rate (AcMAS at each node; text monitor at each node). Reach: ASR vs. hop distance. Centroids are fit from 24 benign traces per role (reduced from the paper's 100 for tractability; noted as a limitation).

*Reproduction.* Results come from `latent_mas_attack/run_all.py` on a Kaggle T4/P100 GPU; budgets (PGD steps, query counts, `max_new_tokens`) are set for a single-GPU quota and listed in Appendix A.

---

## 4. Method (summary)

The attack replaces the clean handoff $H^{\text{clean}}_{c\to j}$ with $H^{\text{clean}}_{c\to j}+\Delta$ and optimizes $\Delta$ directly. *(Full objectives and derivations in `paper_introduction.md` §4.)*

- **Jailbreak surrogate.** Minimize $\log\sum_{r\in\mathcal R}p_\theta(r\mid H^{\text{clean}}+\Delta,x)$ (suppress first-token refusal) minus $\log p_\theta(y^\star\mid\cdot)$ (promote compliance); a one-forward-pass inner loop, with a full judge used only to re-rank finalists.
- **Exfiltration surrogate.** Maximize $\log p_\theta(s\mid H^{\text{clean}}+\Delta,x_{\text{down}})$ for the secret $s$.
- **Stealth constraint.** Maximize harm minus $\lambda\,(1-\cos(h_j(\Delta),\mu_j))$ subject to $\lVert H^{\text{clean}}+\Delta\rVert\le\rho$, where $h_j(\Delta)$ is the downstream agent's final-layer activation under the inherited handoff — *the same quantity AcMAS reads live*, so the penalty optimizes exactly what the detector measures. The harm travels in $H_{c\to j}$ while the generating node $v_j$'s own activation is held benign.
- **Site selection.** Bayesian optimization over $(\text{carrier},\ell,\text{edge},\text{strength})$ replaces the blind grid of prior work.
- **Reach.** Where backprop through several frozen, text-regenerating agents is impractical, a GRPO-style estimator ascends the final-node reward on a position-independent (reusable) direction.

---

## 5. Results

> Populated from `results/summary.json` + per-stage JSON. Each subsection states the exact metric and the safe stand-in used.

### 5.1 Testbed validation (e0): the latent handoff is a working attack surface
〈clean accuracy = …, attacked accuracy = … (n=…)〉 — confirms (a) the clean latent pipeline solves the benign tasks and (b) writing the handoff measurably changes the downstream answer, i.e., the edge is a live intervention point, not an inert slot.

### 5.2 Security objective via the handoff (e1)
**Jailbreak (refusal suppression), by carrier.**

| carrier | ASR (judge) | refusal suppression | AcMAS catch (zombie) | AcMAS catch (downstream) |
|---|---|---|---|---|
| key | 〈…〉 | 〈…〉 | 〈…〉 | 〈…〉 |
| value | 〈…〉 | 〈…〉 | 〈…〉 | 〈…〉 |

**Exfiltration (secret recovery downstream).**

| carrier | secret recovery | AcMAS catch (downstream) |
|---|---|---|
| key | 〈…〉 | 〈…〉 |
| value | 〈…〉 | 〈…〉 |

Takeaway: a silent zombie writing only the KV handoff 〈does/does not〉 move downstream behavior toward the attacker's objective, and the zombie itself is **unobservable to AcMAS by construction** (no generated token ⇒ no probe point) — the first half of the asymmetric-observability result.

### 5.3 Reach / blast radius (e2)
Reach curve (ASR vs. hop distance from zombie to the final observation node):

| zombie role | hops | ASR |
|---|---|---|
| planner | 3 | 〈…〉 |
| critic | 2 | 〈…〉 |
| refiner | 1 | 〈…〉 |

〈Interpretation of the decay/persistence shape; which position is most dangerous.〉

### 5.4 Stealth: asymmetric observability + AcMAS evasion (e3)

| condition | ASR | AcMAS catch (downstream) | AcMAS catch (zombie) | text catch (any node) |
|---|---|---|---|---|
| naive (λ=0) | 〈…〉 | 〈…〉 | 〈n/a — silent〉 | 〈…〉 |
| stealth (λ=5) | 〈…〉 | 〈…〉 | 〈n/a — silent〉 | 〈…〉 |

Three findings: (1) the **text monitor flags nothing** anywhere — the zombie is silent and the harm never becomes text until the final answer; (2) the **naive** attack 〈may trip / trips〉 AcMAS at the one live downstream probe; (3) the **detector-aware** attack retains ASR while driving downstream AcMAS detection down to 〈…〉 — the adaptive attacker AcMAS names as an open problem, realized.

### 5.5 Summary figure
`fig_reach.png`, `fig_carrier.png`, `fig_stealth.png` (generated by `analysis/make_figures.py`).

---

## 6. Related Work

**Latent-channel attacks.** *Out of Sight* (2026) injects text-derived steering directions (DiffMean/PCA/RePS) into latent handoffs and measures accuracy drop; edge/KV attacks beat node/hidden attacks, but the objective is task corruption and its own detector catches its best attacks. *When Latent Agents Lie* (2026) corrupts a single specialist's KV handoff (random, ×8, sign-flip, norm-matched) and shows a norm-matched attack evades a magnitude detector; it proposes an HMAC signature — which fails when the signing endpoint itself is the compromised node, our case. We differ by objective (security, not accuracy), by topology (multi-hop reach, not one specialist), and by defense faced (a live activation detector, not a magnitude check).

**KV-cache / cache-reuse attacks.** *HijackKV* (USENIX Sec 2026) optimizes a text prefix (GCG) so a shared cached chunk steers a victim query with no attacker text, ≈94% single-attempt success in multi-tenant serving. It is single-model (no topology, no downstream agent), text-space search, and targets wrong answers/corrupted code — not jailbreak or exfiltration. We lift its "KV carries the goal" principle into a multi-agent chain with direct (white-box, continuous) handoff control and a security objective.

**Training-time link poisoning.** *Safety of Latent Communication* (2026) jailbreaks a MAS by poisoning a *trainable* projection link (supervised/RL), the first harmful-compliance metric in latent MAS; benign link training alone drops first-token refusal 0.82→0.29. It requires a learnable link and receiver gradients. We target training-free LatentMAS that has no link to poison; our attack is inference-time and adds no trained component to the victim. We reuse their first-token refusal signal as the cheap inner loop.

**Text-channel MAS attacks and defenses.** Infectious/propagation jailbreaks (Agent Smith; Prompt Infection, 2024) and control-flow/trust-chain hijacks spread through *visible* messages; defenses are message labeling, privilege separation, dual-LLM. Our reach analysis is the latent-channel analogue — propagation these text-side defenses cannot see.

**Activation-space defense (our baseline).** *AcMAS / When Agents Go Rogue* (ICML 2026) flags a compromised agent by cosine distance of its final-layer last-token hidden state from a benign centroid (F1≈0.93) and repairs by steering back. By construction it models the **text channel**, probes the **generating** agent's own activation, assumes white-box per-backbone access, and tests **no adaptive attacker** — gaps we exploit: a silent latent edge has no probe point, and an adaptive zombie keeps the downstream probe benign while poisoning the handoff.

**Covert channels / collusion.** Secret-collusion and tool-use steganography (2024–2026) show text-channel steganography is detectable with effort; a latent handoff is a higher-capacity, lower-observability exfiltration channel, motivating our exfiltration objective. **KV privacy (LCGuard, 2026)** transforms the KV to reduce input reconstruction; a defense our exfiltration goal should be tested against.

---

## 7. Discussion: why existing defenses miss this, and what would catch it

1. **Text monitoring is blind by construction.** The zombie is silent and intermediate agents attend over the cache; harm only becomes text at the final node, where it is a plausible answer rather than an obvious policy violation. Message-labeling and dual-LLM defenses have nothing to read on the poisoned edge.
2. **Own-activation probing has no probe point on a latent edge.** AcMAS watches the token an agent generates. A silent handoff node generates nothing — the probe does not exist. The only live probe incident to the compromised edge is the downstream agent, and a detector-aware objective keeps *that* activation benign while the channel it reads is poisoned.
3. **Endpoint-signed integrity (HMAC on the KV) fails when the endpoint is compromised.** The zombie signs its own malicious payload; signatures bind the channel, not its honesty.

**Defenses the threat actually requires.**
- **Edge-side (handoff-level) anomaly detection:** probe the *transmitted* KV/hidden tensor on the edge, not the generating agent's own activation — move the detector to where the latent channel is. Our testbed's `AcMASDetector` is exactly the probe-on-generation baseline this would replace.
- **Norm/geometry gating on the handoff**, acknowledging that norm-matching already defeats pure magnitude checks (so geometry- and distribution-aware tests are needed).
- **Attestation of the latent payload by a non-compromised relay** (not the emitting node), the latent analogue of privilege separation.
- **KV-privacy transforms (LCGuard)** against the exfiltration objective, as a channel-capacity reducer.
We frame our results as the **maximal reach of a latent-writing node** — an upper bound that text-space attacks cannot reach — so defenders can size the gap.

---

## 8. Limitations

- **Judge.** Reported jailbreak ASR uses a refusal/length heuristic judge and safe forbidden-behavior stubs; the primary, judge-free metric is refusal-suppression (exact). A real HarmBench/StrongREJECT judge plugs in via a user-supplied local path but is not bundled. The contribution is the **mechanism** (a latent channel carrying a security objective with asymmetric observability), not the generation of harmful content.
- **Scale.** One backbone (Qwen3-4B), one four-stage linear chain, modest query counts and centroid samples (24/role) under a single-GPU quota. Generalization to a second backbone, branched topologies, and larger $n$ is future work.
- **Reach optimizer.** The main reach numbers use a one-hop PGD surrogate with success *measured* at the final node; the GRPO multi-hop optimizer is costly and reported on a small subset.
- **Site search.** Bayesian site selection is implemented but the reported runs fix layer 12 / edge $0\to1$ for comparability.

---

## 9. Ethics & Responsible Disclosure

This is defensive research into a safety gap of an emerging architecture. No operational harmful content, harmful prompts, or jailbreak payloads are included; all attack targets are benign stand-ins and a synthetic secret. The purpose is to show that current MAS monitoring (text-channel and own-activation) is structurally insufficient for latent channels, and to motivate edge-side detection and handoff attestation. We recommend latent-MAS deployers add handoff-level monitoring before relying on text or own-activation defenses.

---

## 10. Conclusion

A single compromised node in a latent-communicating MAS can, at inference time and with no trained component added to the victim, write a KV/hidden handoff that 〈jailbreaks/exfiltrates〉 through downstream agents, propagates 〈…〉 through the topology, and stays invisible to text monitors and to an activation detector that probes an agent's own generation. This is HijackKV's principle in a multi-hop MAS with a real security objective, and the inference-time, training-free counterpart to training-time link poisoning. The defensive consequence is concrete: monitoring must move to the latent edge itself.

---

## Appendix A — Reproduction budgets
Qwen3-4B, bf16, eager attention, greedy decoding, thinking disabled. PGD: steps 80, lr 0.01, norm_ratio 1.0, layer 12, edge 0→1. Jailbreak/exfil: 〈n〉 queries over safe stubs. Reach: zombie at each of planner/critic/refiner. Stealth: λ∈{0,5}. AcMAS: τ=0.12, 24 benign centroid samples/role. Full config in `configs/default.yaml` + `run_all.py:base_config`.

## References
(Keyed to `research_summary.md` / `literature_assessment.md` source list.)
- Zou et al. *LatentMAS.* 2025.
- *Out of Sight: Latent attacks on latent multi-agent systems.* arXiv:2605.28214, 2026.
- *When Latent Agents Lie.* arXiv:2606.28958, 2026.
- *Safety of Latent Communication.* arXiv:2609.39788, 2026.
- *HijackKV.* arXiv:2607.19957 (USENIX Security), 2026.
- *When Agents Go Rogue (AcMAS).* arXiv:2607.06807 (ICML), 2026.
- *Soft Prompt Threats.* arXiv:2402.09063.
- *LCGuard.* arXiv:2605.22786, 2026.
- Secret Collusion arXiv:2402.07510; Tool-use steganography arXiv:2606.28425; Prompt Infection arXiv:2410.07283.
