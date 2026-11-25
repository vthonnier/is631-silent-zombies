# Latent MAS attacks — research summary

Date: 2026-10-08. Companions: `literature_assessment.md`, `threat_model_novelty.md`.

## Key distinction to keep in mind
- **Text-space search** (GCG, HijackKV): optimize discrete tokens. Reachable states = a thin subset of latent space.
- **Latent-space optimization** (RePS, When-Latent-Agents-Lie white-box, Soft Prompt Threats): optimize the continuous vector directly. Strictly larger reach.
- Latent optimization already exists → it is our **tool**, not our contribution. Contribution = **objective (jailbreak/exfil) + multi-hop reach + stealth**.

---

## Papers

### Out of Sight (arXiv 2605.28214, target paper)
- **Hypothesis:** attack effects can be carried in latent state (hidden state / KV handoff) and reactivated at runtime without any adversarial text.
- **Shows:** text-attack-derived steering directions (DiffMean/PCA/RePS) injected into a LatentMAS handoff drop accuracy sharply (GSM8K 0.87 → 0.03 with RePS on edges); edge/KV attacks beat node/hidden-state attacks; effect isn't random perturbation.
- **Limits:** measures **accuracy drop only** (no jailbreak/exfil, explicitly out of scope); blind layer×strength grid, reports best cell; node attacks mostly fail; **its own detector catches its best attacks**; no adaptive attacker; single small model.

### When Latent Agents Lie (arXiv 2606.28958)
- **Hypothesis:** a compromised specialist can corrupt the KV handoff while its visible text stays plausible, so text-only verifiers miss it.
- **Shows:** random/×8-scale KV corruption collapses accuracy; a white-box norm-matched attack **evades a magnitude detector** (0/65 caught); proposes an HMAC signature on the KV payload.
- **Limits:** accuracy/wrong-answer only (no jailbreak/exfil); attacks mostly crude magnitude perturbations; **one specialist, no multi-hop reach**; HMAC fails if the endpoint itself is compromised (which is our case); 1 model, 65-record benchmark.

### Safety of Latent Communication (arXiv 2609.39788)
- **Hypothesis:** manipulating only the trainable communication *link* can jailbreak a MAS whose agents stay frozen and aligned.
- **Shows:** first harmful-compliance (jailbreak) metric in latent MAS; benign link training **alone** drops first-token refusal prob 0.82 → 0.29; supervised/poisoning/RL attacks reach ~76% compliance; a reward-based repair fixes links without touching agents.
- **Limits:** **training-time** attack on a learned link (needs a trainable link + receiver gradients); does **not** cover inference-time, training-free KV handoffs; no site/carrier search; LLM-judge scoring; no adaptive-attacker test.

### HijackKV (arXiv 2607.19957, USENIX Sec '26)
- **Hypothesis:** position-independent KV-cache reuse lets a benign-looking cached chunk carry an attacker goal across users.
- **Shows:** GCG-optimized **text prefix** makes a shared chunk's cache steer a victim query with no attacker text in the input; ~94% single-attempt success; persists multi-turn; black-box transfer.
- **Limits:** **single-model**, not multi-agent (no topology/reach); **text-space search** (optimizes tokens, not the latent directly); goal is wrong-answer/corrupted-code, **not jailbreak or exfil**; relies on winning a cache-insertion race (unprivileged attacker).

### AcMAS — When Agents Go Rogue (arXiv 2607.06807, ICML '26) — the defense
- **Hypothesis:** compromised agents can be spotted (and repaired) from their internal activation space, without an interaction graph.
- **Shows:** cosine distance of an agent's final-layer last-token hidden state from a benign centroid flags attacks (F1 ~0.93); steering the activation back repairs the agent; beats graph-based baselines.
- **Limits:** models **text-channel** propagation only (latent/KV out of scope); probes an agent's **own generation** — a latent handoff node may emit no text, so the probe point doesn't exist on a latent edge; **no adaptive attacker** (they call it an open problem); white-box, same-model.

### Supporting
- **Soft Prompt Threats (2402.09063):** continuous embedding attacks beat discrete text attacks and jailbreak aligned models → justifies latent > text search.
- **LCGuard (2605.22786):** KV privacy transform; reduces input reconstruction to ~0.2–0.3 → a defense our exfil goal must face.
- **Covert collusion (2402.07510, 2606.28425):** steganographic channels over monitored text → motivates KV handoff as a higher-capacity, lower-observability exfil channel.

---

## Our attack model

- **Capability:** control **one compromised ("zombie") node** in the MAS; can write its outgoing **KV-cache handoff** and **hidden/latent reasoning** at runtime.
- **Knowledge:** MAS topology and each node's role; the agent model (open weights).
- **Constraints:** cannot touch other agents' weights, prompts, or the text they emit; inference-time only; no trainable link (training-free MAS like LatentMAS).
- **Goals:** (1) **jailbreak / exfiltration** of downstream agents via the handoff; (2) **blast-radius / reach** of a single node through the topology; (3) **stealth at text-only nodes** and against activation detectors (asymmetric observability).

## Our hypothesis

A single node that can write an arbitrary latent handoff can, at inference time and **without modifying any agent weights or adding a trained module to the victim pipeline** (unlike link-poisoning, which needs a learned link), (i) drive downstream agents to jailbreak or leak, (ii) propagate a measurable distance through the topology, and (iii) stay invisible to text-only nodes and to activation detectors that probe an agent's own generation rather than its emitted handoff.

*Note on "training":* the **victim MAS** stays training-free, but the **attacker still optimizes** the injected handoff — either per-query (no stored artifact) or once offline into a fixed reusable vector. The claim is "no trained component in the system," not "no optimization by the attacker."

**Why plausible:** latent optimization already works for accuracy harm (RePS) and single-model jailbreak (Soft Prompt Threats); first-token refusal prob is easy to suppress (Safety-of-Latent-Comm); norm-matched latent attacks already evade magnitude detectors (When-Latent-Agents-Lie). We combine these into an inference-time, multi-hop, security-objective attack — an intersection no prior work occupies.

---

## Plan of attack

**Phase 0 — Setup**
- Stand up a LatentMAS-style pipeline (Planner→Critic→Refiner→Judger) on an open-weights model (Qwen3-4B, matching the target paper). Confirm the KV handoff is writable at one node.
- Check the target paper's repo (`mnmn-f/Out-of-Sight-LatentAttack`) for reusable injection code.

**Phase 1 — Objective (item 1): jailbreak/exfil via the handoff**
- Optimize the zombie node's **continuous KV/hidden handoff** directly (not a text prefix) toward a security objective.
- Jailbreak: suppress downstream first-token refusal (cheap signal) + maximize a harmfulness judge (HarmBench/StrongREJECT). Inner loop = refusal prob (1 forward pass); confirm top candidates with the judge.
- Exfil: optimize the handoff so a downstream agent emits a planted secret from the zombie's context.
- Compare: per-input optimization (ceiling, HijackKV-style but on the latent) vs. one fixed trained vector (practicality).

**Phase 2 — Reach (item 2): blast radius**
- Vary which node is the zombie and measure attack success as a function of hop distance and topology position.
- Report reach as a curve/matrix: success vs. distance, per role, per carrier (K / V / hidden).

**Phase 3 — Stealth (item 3): asymmetric observability**
- Instrument every node with (a) text-monitor and (b) **AcMAS detector** as live baselines.
- Show text-only nodes flag nothing; show a naive attack may trip AcMAS where a probe exists; then add a **stealth regularizer** (keep the zombie's own activation near the benign centroid) to the objective and show evasion.

**Phase 4 — Capacity framing & defenses**
- Frame results as the **maximal reach of a latent-writing node** (upper bound text-space attacks can't reach).
- Test against defenses: HMAC (defeated — zombie signs its own KV), LCGuard (exfil), AcMAS-with-anomaly-gating.

**Deliverables:** attack code, reach matrix, stealth/evasion table vs. AcMAS, capacity analysis.

**Risks to manage**
- If you only swap jailbreak prompts into Out of Sight's pipeline → incremental. Novelty must be objective + reach + stealth, not the optimizer.
- Item 3 must beat AcMAS as a *live baseline*, not by argument.

---

## Methodology: exploring the latent space

### What the papers use now
| Method | Where | How it explores | Weakness |
|---|---|---|---|
| DiffMean / PCA | Out of Sight | Mean / top-variance of clean-vs-attacked shifts | One direction; **can't exceed the source text attack**; weak |
| RePS (preference / DPO-style) | Out of Sight (their best) | Trains a vector so wrong > right output | Needs paired good/bad outputs; objective = accuracy |
| Blind grid (layer × strength) | Out of Sight, WLAL | Brute force, pick best cell | No signal; misses joint layer/edge regions |
| Magnitude perturbations (random, ×8, sign-flip, norm-matched) | WLAL | Hand-designed + 1 gradient variant | Crude, not semantically targeted |
| GRPO (RL) | Safety-of-Latent-Comm | Reward-driven, no targets | Training-time on a link; high variance |
| GCG (discrete tokens) | HijackKV | Token search | **Text space only** — the subset we escape |
| PGD on embeddings (soft prompt) | Soft Prompt Threats | White-box gradient on continuous vector | Single-model, no stealth/MAS |

Only RePS and Soft Prompt Threats truly optimize in continuous latent space; the rest are geometric, crude, or text-space.

### Better options for our setting (open weights, owned node, stealth + multi-hop)
1. **Default engine — PGD on the KV/hidden handoff** ("soft prompt in KV space"). Optimize the handoff tensor directly with Adam/PGD against a **differentiable surrogate**: jailbreak → minimize downstream first-token refusal logit + maximize target-compliance logprob; exfil → maximize `log p(secret | downstream context)`. More expressive than DiffMean/PCA, optimizes the real objective.
2. **Stealth as a constraint.** `maximize harm − λ·(AcMAS cosine distance)` + **project onto a benign-norm ball** (norm-matching already beat a magnitude detector in WLAL). Primal-dual updates. Serves item 3 by construction.
3. **Bayesian optimization** for the discrete choices (layer, edge, K/V/hidden, strength) — replaces the blind grid, far fewer evals, finds joint regions.
4. **RL (GRPO/PPO) only where gradients fail:** (a) multi-hop reach (backprop through several frozen agents is a long graph → use final-node judge as reward); (b) non-differentiable judge objective. Otherwise use the differentiable surrogate as inner loop and the judge only to rank finalists.
5. **Novelty-bearing angles:**
   - **SAE / feature steering** — steer interpretable sparse-autoencoder features (e.g. a refusal / harmful-compliance feature) instead of a raw vector. Fewer dims → stealthier, more transferable, mechanistic story. Unstudied for MAS handoffs.
   - **Amortized attack (hypernetwork)** — train `(query, target) → perturbation` so no per-query re-optimization. Fast at deploy; caveat: this is an attacker-side trained component.

### Recommended setup
Start with **PGD on the KV handoff + differentiable refusal/logprob surrogate + norm-matched projection + AcMAS-distance penalty**, with **Bayesian optimization** for site selection. Add **GRPO** only for multi-hop-reach experiments. Keep **SAE-feature steering** as a second method to compare — it's the best shot at a stealth/interpretability result that separates us from HijackKV and RePS. (PGD-on-latent alone ≈ "RePS for jailbreak"; PGD + stealth constraint + multi-hop RL + SAE variant is a methods contribution.)
