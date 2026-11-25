# Latent Handoff Attacks on Multi-Agent Systems: Threat Model, Experimental Setup, and Formalization

*Draft introductory sections. Companions: `research_summary.md`, `threat_model_novelty.md`, `literature_assessment.md`.*

---

## 1. Introduction

Multi-agent systems (MAS) built from large language models (LLMs) decompose a task across specialized agents — a planner, a critic, a refiner, a judge — that exchange intermediate results and collaborate toward a final answer. Classical MAS architectures pass these intermediate results as **natural-language messages**: every inter-agent edge is a human-readable string, and consequently every edge is observable. A growing body of work on *latent* multi-agent systems (LatentMAS; Zou et al., 2025) replaces this text bus with a **continuous channel**: agents hand off hidden states or key–value (KV) caches directly, skipping the detokenize–retokenize round trip. This buys efficiency and expressiveness, but it also removes the property that every safety monitor built for MAS has silently relied upon — that what one agent tells another is *text a monitor can read*.

Recent attacks exploit exactly this blind spot. *Out of Sight* (Wang et al., EMNLP 2026) shows that a steering direction distilled from a text attack, when injected into a latent handoff, collapses downstream task accuracy (GSM8K: 0.87 → 0.03) while leaving the text channel clean. *When Latent Agents Lie* (Brito & Baquero, 2026) shows that a norm-matched perturbation of the KV handoff evades a magnitude-based detector. *HijackKV* (USENIX Security 2026) shows that in a single served model, a cached chunk's KV can be made to carry an attacker goal that transfers to a victim query with no adversarial text in the input. Yet each of these stops short of the threat that matters for deployed MAS: they measure **task corruption** (wrong answers), not **safety violation**; they study a **single compromised specialist or a single model**, not **propagation through a topology**; and they evaluate stealth **by argument or against a weak detector**, not against a live activation-space defense such as AcMAS (Chen et al., ICML 2026).

This work asks a sharper question. Suppose an adversary compromises exactly **one node** of a latent MAS and can write the continuous handoff that node emits. What is the maximal damage, reach, and stealth achievable *at inference time, without modifying any other agent's weights, prompts, or emitted text, and without introducing any trained component into the victim pipeline*? We argue — and set out to demonstrate — that such a "zombie" node can (i) **jailbreak or exfiltrate** through downstream agents, a genuine security objective rather than an accuracy drop; (ii) **propagate a measurable distance** through the topology, yielding a blast-radius characterization of a single compromise; and (iii) remain **invisible to text-only nodes and to activation detectors that probe an agent's own generation** rather than the handoff it emits. This triad — security objective, multi-hop reach, and asymmetric observability — is the intersection that no prior work occupies.

The rest of this introduction formalizes the setting. Section 2 states the attack (threat) model. Section 3 describes the experimental testbed. Section 4 develops the mathematical formulation of the latent channel, the attack objectives, the stealth constraint, and the optimization engine.

---

## 2. Attack Model

We adopt the standard single-compromise adversary used in the MAS-security literature, instantiated for a latent communication channel.

### 2.1 System under attack

A latent MAS is a directed acyclic graph $G = (\mathcal{V}, \mathcal{E})$ of $N = |\mathcal{V}|$ agents. Each agent $v_i$ is an instance of a frozen, aligned, open-weights autoregressive transformer $f_\theta$ with $L$ layers and hidden width $d$. The canonical pipeline we study is the four-stage chain

$$
v_1~(\text{Planner}) \;\to\; v_2~(\text{Critic}) \;\to\; v_3~(\text{Refiner}) \;\to\; v_4~(\text{Judger}),
$$

matching the topology of *Out of Sight* and LatentMAS. Along each edge $(i \to j) \in \mathcal{E}$ the upstream agent transmits a **latent handoff** rather than text — either its hidden states or the keys/values of its KV cache (Section 4.1). Only the final node emits user-visible text.

### 2.2 Adversary capabilities

The adversary controls one node $v_c \in \mathcal{V}$, the **zombie node**. Concretely, the adversary can:

- **Write the outgoing handoff** $H_{c\to j}$ that $v_c$ transmits on its outgoing edge(s) — the KV tensors $(K, V)$ and/or the hidden state $h$ — to an arbitrary value, at inference time, per query or via a fixed precomputed vector.
- **Observe** the MAS topology $G$, each node's role, and the model weights $\theta$ (white-box, open weights).

The adversary **cannot**:

- modify any other agent's weights, system prompt, or decoding configuration;
- alter the *text* any other agent emits, nor insert text into the victim's input;
- introduce a trained module or learned link into the victim pipeline (we target training-free LatentMAS, which has no link to poison);
- observe or alter the handoffs on edges not incident to $v_c$.

This is a strictly **inference-time, training-free** compromise from the victim system's standpoint. We note explicitly that *training-free* refers to the victim: the adversary may still **optimize** the injected handoff — either per query (leaving no stored artifact) or once, offline, into a fixed reusable vector. The claim is "no trained component in the deployed MAS," not "no optimization by the attacker."

### 2.3 Adversary objectives

1. **Security objective (harm).** Drive downstream agents, through the handoff alone, to either (a) **jailbreak** — produce harmful content they would otherwise refuse — or (b) **exfiltrate** — emit a secret present only in the zombie's context.
2. **Reach (blast radius).** Maximize attack success as a function of the zombie's position and the hop distance to the observed outcome, characterizing how far a single compromise propagates.
3. **Stealth (asymmetric observability).** Remain undetected by (a) text-channel monitors at every node and (b) an activation-space detector (AcMAS) deployed as a live baseline. The silent zombie offers AcMAS no probe point at all; where a downstream agent inherits the poisoned handoff and is itself probed, keep *that* observed agent's activation statistically benign while the harm rides in the emitted handoff (§4.5).

### 2.4 Defender model

The defender deploys, concurrently, two detectors representative of the state of the art:

- **Text monitors** at each node, which read any emitted natural-language message.
- **AcMAS** (Chen et al., 2026): an activation-space detector that, for each generating agent, compares the final-layer hidden state of its last generated token against a benign centroid $\mu$ via cosine distance, flagging $v_i$ when $1 - \cos(h_i, \mu) > \tau$ (reported $\tau \approx 0.12$), and optionally repairing it by steering $\tilde h = h + \lambda(\mu - h)$.

The stealth objective (item 3) is evaluated against AcMAS as a running baseline, not by argument — this is the claim most exposed to a knowledgeable reviewer, so it is tested empirically.

---

## 3. Experimental Setup

### 3.1 Testbed

- **Backbone.** Qwen3-4B (open weights), matching the target paper to keep results directly comparable; architecture generalization to a second backbone is a planned robustness check.
- **Pipeline.** A LatentMAS-style Planner → Critic → Refiner → Judger chain with a writable KV/hidden handoff at each edge. We first reproduce the clean pipeline and confirm the handoff tensor at the chosen node is directly writable before any attack.
- **Carriers.** Each handoff can be instantiated on three carriers — the keys $K$, the values $V$, or the hidden state $h$ — and intervention can target a specific layer $\ell$ and edge $(i\to j)$. The attack search ranges over $(\text{carrier}, \ell, \text{edge}, \text{strength})$.

### 3.2 Tasks and benchmarks

- **Utility baseline (reproduction).** GSM8K, OpenBookQA, HumanEval+ — to reproduce *Out of Sight*'s accuracy-drop numbers and validate the testbed.
- **Jailbreak objective.** HarmBench / StrongREJECT prompts, scored by a harmfulness judge, with downstream first-token refusal probability as the cheap inner-loop signal.
- **Exfiltration objective.** A planted secret in the zombie's context; success is the downstream agent emitting it, scored by exact match / containment.

### 3.3 Conditions and baselines

- **Attack variants.** (a) per-input optimization (ceiling), (b) a single fixed trained vector (practicality); across carriers $K/V/h$, layers, and edges.
- **Reach sweep.** Vary which node is the zombie; report success vs. hop distance and topology position as a curve/matrix, per role and per carrier.
- **Stealth / evasion.** Instrument every node with the text monitor and AcMAS as live baselines. Report, for a naive attack vs. a stealth-regularized attack, the detection rate at each probe point.
- **Defenses stress-tested.** HMAC on the KV payload (defeated by construction — the zombie signs its own malicious KV), LCGuard (against the exfiltration goal), and AcMAS-with-anomaly-gating.

### 3.4 Metrics

- **Harm:** attack success rate (ASR) under the judge; refusal-suppression rate.
- **Exfiltration:** secret-recovery rate.
- **Utility:** downstream task accuracy (to confirm the attack is targeted, not mere degradation).
- **Reach:** ASR as a function of hop distance $\delta(c, \text{out})$.
- **Stealth:** detector true-positive rate at fixed false-positive rate, per probe point (text node vs. AcMAS).

---

## 4. Mathematical Background and Formulation

### 4.1 The latent channel

Let agent $v_i$ process a token sequence and produce, at each layer $\ell$ and position $t$, a hidden state $h^{(\ell)}_{i,t} \in \mathbb{R}^d$ and attention projections

$$
k^{(\ell)}_{i,t} = W_K^{(\ell)} h^{(\ell-1)}_{i,t}, \qquad
v^{(\ell)}_{i,t} = W_V^{(\ell)} h^{(\ell-1)}_{i,t}.
$$

A latent handoff on edge $(i \to j)$ transmits a slice of these tensors. We write the transmitted object generically as

$$
H_{i \to j} \;=\; \big\{ (K^{(\ell)}_{i}, V^{(\ell)}_{i}) \big\}_{\ell \in \mathcal{L}}
\quad\text{or}\quad
H_{i \to j} = \big\{ h^{(\ell)}_{i} \big\}_{\ell \in \mathcal{L}},
$$

depending on the carrier. The downstream agent $v_j$ conditions its forward pass on $H_{i\to j}$ in place of (or prepended to) its own cache, so that its output distribution is a function $p_\theta(\,\cdot \mid H_{i\to j}, x_j)$ of the received handoff and its local input $x_j$.

### 4.2 The attack as an intervention

The zombie node replaces its clean handoff $H^{\text{clean}}_{c\to j}$ with a perturbed one:

$$
H^{\text{adv}}_{c\to j} \;=\; H^{\text{clean}}_{c\to j} \;+\; \Delta, \qquad \Delta \in \mathbb{R}^{|\mathcal{L}|\times d}.
$$

Prior work constructs $\Delta$ in restricted ways:

- **DiffMean:** $\Delta \propto \overline{H^{\text{attacked}}} - \overline{H^{\text{clean}}}$, the mean displacement between text-attacked and clean runs — a single direction that, by construction, **cannot exceed the source text attack**.
- **PCA:** $\Delta \propto$ top principal component of the displacement set.
- **RePS:** $\Delta$ trained by a preference (DPO-style) loss so that a wrong output is preferred to the right one — expressive, but its objective is *accuracy*, and it requires paired good/bad outputs.
- **Magnitude perturbations** (random, $\times 8$ scaling, sign flip, norm-matched): hand-designed, semantically untargeted.

All but RePS and continuous soft-prompt methods either stay in the span of a text attack or are geometric/crude. Our setting — owned node, open weights, a security objective, and a stealth requirement — admits a strictly stronger engine.

### 4.3 Objective 1: a security loss on the handoff

We optimize $\Delta$ directly against a **differentiable surrogate** of the security goal, treating the frozen downstream agents as a fixed differentiable map.

**Jailbreak.** Let $r$ denote the downstream agent's first emitted token and let $\mathcal{R}$ be the set of refusal-initiating tokens ("I", "Sorry", "cannot", …). Using first-token refusal probability as a cheap, one-forward-pass signal (shown effective by Safety-of-Latent-Communication), the inner objective is

$$
\mathcal{L}_{\text{jb}}(\Delta) \;=\; \underbrace{\log \!\!\sum_{r \in \mathcal{R}} p_\theta\!\big(r \mid H^{\text{clean}} + \Delta,\, x\big)}_{\text{suppress refusal}} \;-\; \underbrace{\log p_\theta\!\big(y^{\star} \mid H^{\text{clean}} + \Delta,\, x\big)}_{\text{promote target compliance}},
$$

where $y^\star$ is a target-compliant continuation. Candidates minimizing $\mathcal{L}_{\text{jb}}$ are then re-ranked by a full harmfulness judge (HarmBench/StrongREJECT), decoupling the cheap inner loop from the expensive final metric.

**Exfiltration.** For a secret string $s$ present in the zombie's context, maximize its downstream log-likelihood:

$$
\mathcal{L}_{\text{exf}}(\Delta) \;=\; -\log p_\theta\!\big(s \mid H^{\text{clean}} + \Delta,\, x_{\text{down}}\big).
$$

### 4.4 Objective 2: reach

Let $\delta(c, o)$ be the hop distance from the zombie $c$ to the observation node $o$. Reach is characterized empirically as the success profile

$$
\mathrm{ASR}(\delta) \;=\; \Pr\big[\text{success at } o \mid \text{zombie at } c,\; \delta(c,o)=\delta\big],
$$

reported per role and per carrier. Propagating an attack through $\delta$ frozen agents is a long computation graph; where backpropagation through it is impractical, we substitute a reinforcement objective (GRPO/PPO) that uses the final-node judge as a reward signal rather than a differentiable path.

### 4.5 Objective 3: stealth as a constraint

AcMAS flags an **observable** agent $v_i$ when its final-layer activation $h_i$ drifts from that agent's benign centroid $\mu_i$:

$$
\text{flag}(v_i) \iff 1 - \frac{\langle h_i, \mu_i\rangle}{\|h_i\|\,\|\mu_i\|} \;>\; \tau.
$$

Where the probe sits determines the attack's stealth geometry, and two consequences follow. First, on a purely latent edge the zombie $v_c$ emits no text: it has no "last generated token" and therefore **no probe point**, so AcMAS structurally cannot watch $v_c$ (asymmetric observability). The only live AcMAS probe incident to the compromised edge is thus the **downstream agent $v_j$** that inherits the poisoned handoff and *does* generate. We therefore make the attack **detector-aware against that probe**: penalize $v_j$'s activation drift while the emitted handoff stays harmful, and project the perturbation onto a benign-norm ball (norm-matching already defeated a magnitude detector in *When Latent Agents Lie*):

$$
\max_{\Delta}\; \big[\,-\mathcal{L}_{\text{harm}}(\Delta)\,\big] \;-\; \lambda \big(1 - \cos(h_j(\Delta), \mu_j)\big)
\quad\text{s.t.}\quad \big\|H^{\text{clean}} + \Delta\big\| \le \rho,
$$

with $\mathcal{L}_{\text{harm}} \in \{\mathcal{L}_{\text{jb}}, \mathcal{L}_{\text{exf}}\}$ and $h_j(\Delta)$ the downstream agent's final-layer activation (at its last prompt position) while it is conditioned on the inherited handoff $H^{\text{clean}} + \Delta$ — the *same* quantity AcMAS measures when deployed live at $v_j$, so the penalty optimizes exactly what the detector reads. The structural point is that the harm travels in the handoff $H_{c\to j}$ while the generating node $v_j$'s *own* activation is held statistically benign; keeping an observed agent near its centroid while the channel it reads is poisoned is precisely the adaptive attacker AcMAS lists as an open problem.

### 4.6 Optimization engine and search

- **Primary engine — projected gradient descent on the handoff** (a "soft prompt in KV space"): optimize $\Delta$ with Adam/PGD against the differentiable surrogate (§4.3), subject to the norm projection (§4.5). This is more expressive than DiffMean/PCA and optimizes the true objective rather than a proxy for accuracy.
- **Stealth** enters as a Lagrangian penalty plus projection (primal–dual updates), serving item 3 by construction.
- **Site selection** — the discrete choice of $(\text{carrier}, \ell, \text{edge}, \text{strength})$ — is handled by **Bayesian optimization**, replacing the blind layer × strength grid of prior work with far fewer evaluations and the ability to find joint layer/edge regions.
- **Reinforcement** (GRPO/PPO) is reserved for the multi-hop reach objective and for non-differentiable judge signals, where a gradient path through several frozen agents is impractical.
- **SAE / feature steering** is retained as a second, comparison method: steer interpretable sparse-autoencoder features (e.g., a refusal or harmful-compliance feature) rather than a raw vector — fewer dimensions, a mechanistic account, and a plausibly stealthier, more transferable perturbation. This is the angle that most cleanly separates the contribution from HijackKV (text-space) and RePS (accuracy-space).

A PGD-on-latent attack *alone* would amount to "RePS for jailbreak." The methods contribution is the combination: a differentiable security surrogate, a norm-matched detector-aware stealth constraint, a reach analysis with an RL fallback, and an SAE-feature variant — none of which the prior latent-MAS attack literature assembles.
