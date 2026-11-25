# Novelty assessment: zombie-node latent attack on a MAS

Assessment date: 2026-10-08. Companion to `literature_assessment.md`.

## Your proposal, restated

**Threat model**
- Attacker controls **one compromised ("zombie") node** inside a multi-agent system.
- At that node, the attacker can modify the **KV-cache handoff** and the **hidden latent state / reasoning** it passes downstream.
- Attacker **knows** the MAS topology (which node does what) and the agent model (open-weights).

**Goals / claimed novelty**
1. **Jailbreak / exfiltration via runtime KV handoff** — a real security impact, not just an accuracy drop.
2. **Blast-radius / reach analysis** from the zombie node — how far a single compromised node propagates.
3. **Stealth at text-only nodes** — the asymmetric-observability result: nodes that communicate in text see nothing wrong.

---

## 1. Novelty vs. the two papers you flagged

### 1.1 *When Agents Go Rogue / AcMAS* (arXiv 2607.06807, ICML 2026) — the defense

**What it does.** Detects a compromised agent by watching its **final-layer hidden state on the last generated token**. It builds a benign centroid from 50–100 clean traces and flags any agent whose cosine distance exceeds τ = 0.12. It then *repairs* the agent by steering its activation back toward the centroid (`h̃ = h + λ(μ − h)`). Reports F1 ≈ 0.92–0.95, ASR driven down to 0.02–0.05. Baselines are graph-based defenses (G-Safeguard, BlindGuard, PERM, TAM).

**Why it threatens your novelty.** It is an activation-space detector for compromised MAS agents, so on paper it is a defense against exactly your attack. Your stealth claim (item 3) must survive it, or it collapses.

**Why it does *not* kill your novelty — four concrete gaps:**
1. **Wrong channel.** AcMAS models propagation "through explicit text messages between agents." It targets prompt injection, tool manipulation (InjecAgent), and memory poisoning (PoisonRAG). It **does not address latent or KV-cache communication** — the paper says that channel is out of scope. Your attack lives entirely in that out-of-scope channel.
2. **It monitors the *generating* agent's own activation.** AcMAS reads the hidden state an agent produces *when it generates text*. In a latent-MAS handoff, the zombie node may emit **no text at all** — it passes KV/hidden state directly. There is no "last generated token" to probe in the way AcMAS assumes. This is the technical core of your asymmetric-observability result, and you can state it precisely: AcMAS's probe point does not exist on a latent edge.
3. **No adaptive attacker.** The paper explicitly says attackers that "gradually shift activation distributions remain an open challenge" and leaves it to future work. A zombie node that keeps its *own* activation near the benign centroid while poisoning the *handoff* it emits is precisely such an adaptive attacker. Evading AcMAS is an open, citable target.
4. **White-box, per-backbone, same-model assumption.** AcMAS needs white-box access to every agent and keeps a prototype per backbone. Your reach analysis (item 2) can show what happens when the downstream nodes are heterogeneous or text-only, where AcMAS has no coverage.

**Use it as your baseline.** Run AcMAS's detector at every node and show: (a) text nodes downstream flag nothing; (b) a naive KV attack may be caught if a probe exists, but (c) an adaptive version that constrains the zombie's own activation evades it. That directly produces items 2 and 3.

### 1.2 *HijackKV* (arXiv 2607.19957, USENIX Security 2026) — the closest KV attack

**What it does.** In **single-model, position-independent KV-cache reuse** (multi-tenant serving), an unprivileged user optimizes a short adversarial **prefix** (GCG-style discrete token search) so the cached KV of a *benign shared text chunk* encodes the attacker's goal. When a victim query reuses that chunk, it inherits the steering with **no adversarial text in the input**. Reports ≈94% single-attempt success, works at 10% cache-hit rate, persists multi-turn, transfers black-box. Goals demonstrated: **targeted/untargeted wrong answers and corrupted code — not jailbreak, not exfiltration.**

**Why it's the nearest prior art.** It is the strongest existing "KV carries the attacker's goal, no attacker text visible" result. You must cite it as the conceptual ancestor of item 1.

**Why your work is distinct — contrast on four axes:**
1. **Single-model vs. multi-agent multi-hop.** HijackKV hijacks *one* served model's shared cache across tenants. Yours is a **compromised node inside a MAS** whose poisoned handoff flows through *other agents*. The reach/blast-radius question (item 2) does not exist in HijackKV — there is no topology, no downstream agent.
2. **Cache-reuse side channel vs. direct handoff control.** HijackKV's attacker is *unprivileged* and cannot touch the cache; they must win a cache-insertion race and only influence KV *indirectly* through an optimized prefix. Your attacker *is* the node and **writes the handoff directly**. That is a stronger capability but a far more realistic MAS compromise scenario (one rogue agent), and it removes HijackKV's main practicality caveats (cache availability, eviction, insertion windows).
3. **Goal.** HijackKV demonstrates integrity attacks (wrong answers). It explicitly does **not** do jailbreak or exfiltration. Item 1 (jailbreak + exfiltration through the handoff) is genuinely unclaimed here.
4. **Stealth framing.** HijackKV's stealth is "no attacker text in the victim query." Yours is richer: **asymmetric observability across a heterogeneous topology** — the attack is invisible specifically at text-only nodes and at AcMAS-style probes on latent edges. That is a new stealth result, not HijackKV's.

**Borrowable technique.** HijackKV's GCG-style optimization of a discrete carrier toward a target is exactly the search engine you can lift — but you optimize the **handoff KV/hidden state directly** (continuous, since you own the node and have open weights) rather than a prefix. That is both easier (direct write, white-box) and more expressive.

### 1.3 Verdict on novelty

All three of your items survive, with the following precise positioning:

| Item | Closest prior | Why yours is new |
|---|---|---|
| 1. Jailbreak/exfiltration via runtime KV handoff | HijackKV (wrong answers, single-model); Out-of-Sight (accuracy drop, no jailbreak); Safety-of-Latent-Comm (jailbreak but *training-time* link poisoning) | No one does **inference-time, training-free jailbreak+exfiltration through a MAS KV handoff** |
| 2. Blast-radius / reach from a zombie node | HijackKV has no topology; text-MAS propagation work (trust-chain, Prompt Infection) is text-only | Reach analysis **for a latent channel** across a known topology is unstudied |
| 3. Stealth at text-only nodes (asymmetric observability) | AcMAS (text channel, no adaptive attacker); HijackKV (no-text stealth, single model) | **Asymmetric observability across heterogeneous latent/text nodes**, and evasion of an activation detector, is new |

The one thing to protect: item 3 must be demonstrated *against AcMAS as a live baseline*, not argued. Otherwise a reviewer who knows AcMAS will object.

---

## 2. Other MAS perturbation methods — summary and how to improve them

These are the existing "perturb a MAS" methods. For each: what it does, and the lever you have to improve it under your threat model (single owned node, direct latent write, open weights, known topology).

### 2.1 Out of Sight (target paper, arXiv 2605.28214)
- **Does:** text-attack-derived steering direction (DiffMean/PCA/RePS) injected into node hidden states or KV handoff. Measures **accuracy drop only**. Blind layer×strength grid, reports best cell. Own detector catches its best attacks.
- **Improve:**
  - Swap the objective from accuracy-drop to **jailbreak + exfiltration** (item 1).
  - Replace the one handcrafted text attack with **real jailbreak prompts / refusal-direction signals** to build the direction.
  - Replace the blind grid with a **guided search** (refusal-direction or safety-CAV ranking of layer/edge/strength; HijackKV-style optimization on the handoff). This is the methodological novelty — see `literature_assessment.md` §4 gap 2.
  - Add a **detector-aware (adaptive)** objective so it evades AcMAS and its own norm-profile detector (item 3).

### 2.2 When Latent Agents Lie (arXiv 2606.28958)
- **Does:** malicious specialist corrupts KV handoff (random, ×8 scale, sign-flip, norm-matched, one white-box gradient attack). Accuracy only. Proposes HMAC on the KV payload. Shows norm-matched attack evades a magnitude detector.
- **Improve:**
  - Their attacks are mostly **crude magnitude perturbations**. Replace with a **semantically targeted** handoff (jailbreak/exfiltration goal) — item 1.
  - They stop at **one compromised specialist**. Add **reach analysis** across the chain — item 2.
  - Their HMAC defense assumes an honest endpoint; your zombie node **is** the endpoint and signs its own malicious KV, defeating HMAC. State this explicitly as motivation.

### 2.3 Safety of Latent Communication (arXiv 2609.39788)
- **Does:** jailbreak via **training-time** poisoning of a learned projection link; RL attack (GRPO) with no harmful targets; reward-based repair. Measures harmful compliance. Finds benign link training alone lowers first-token refusal 0.82→0.29.
- **Improve:**
  - It needs a **trainable link** and persistent weight corruption. Your attack is **inference-time, training-free**, applicable to LatentMAS-style handoffs that have no link to poison. That is the gap it leaves open.
  - **Reuse their cheap signal:** first-token **refusal-initiation probability** is a one-forward-pass objective. Use it as the inner loop of your guided handoff search, then confirm top candidates with a full judge. (Hypothesis: a KV perturbation lowering first-token refusal probability jailbreaks the downstream node without full wrong-output optimization.)
  - Borrow their **repair** as the defense you must beat for item 3.

### 2.4 Text-MAS attacks (Agent Smith ICML'24; Prompt Infection 2410.07283; ARCJ; trust-chain / control-flow hijack 2510.17276)
- **Does:** propagation / infectious jailbreak through **visible messages**; blast-radius governed by tool access; defenses are message labeling, privilege separation, dual-LLM.
- **Improve:**
  - These are the **text-channel analogue** of your reach analysis. Your contribution is the **latent-channel version**: propagation that text monitors and message-labeling defenses cannot see (item 3). Frame item 2 as "infectious jailbreak, but through KV handoffs."

### 2.5 Covert-channel / collusion work (Secret Collusion 2402.07510; Tool-use steganography 2606.28425; CODETTA)
- **Does:** steganographic collusion over **monitored text channels**; information-theoretic capacity; countermeasure limits.
- **Improve / connect:** your **exfiltration** goal is a covert channel through the **KV handoff** instead of text. These papers show text-channel steganography is detectable-with-effort; a latent handoff is a **higher-capacity, lower-observability** channel. Cite them to motivate exfiltration capacity and to argue the latent channel is strictly harder to monitor.

### 2.6 Defenses you must position against
- **AcMAS** (2607.06807): activation detector + repair. Your baseline to evade (items 2–3).
- **LCGuard** (2605.22786): KV privacy transform; relevant to your exfiltration goal (it tries to stop reconstruction; you could test whether your exfil survives it).
- **VLA** (2608.19161): covert-coordination monitor for hidden states; another detector to position against.

---

## 3. Suggested framing of the contribution (one paragraph)

"A single compromised node in a latent-communicating MAS can, at inference time and without any trainable component, write a KV/hidden-state handoff that (i) jailbreaks or exfiltrates through downstream agents, (ii) propagates a measurable distance through the topology, and (iii) remains invisible to text-only nodes and to activation detectors that probe an agent's own generation rather than its emitted handoff. This is HijackKV's 'KV carries the goal' idea, lifted from single-model cache-reuse into a multi-hop MAS with a real security objective, and it is the inference-time, training-free counterpart to the training-time link-poisoning of Safety-of-Latent-Communication."

---

## 4. Open items I did not verify
- AcMAS and HijackKV: I read abstracts + first 100k chars of HTML, not the tail sections.
- Whether AcMAS can be reconfigured to probe a *latent edge* (it currently cannot, by construction) — worth a careful read of their §3.
- Whether HijackKV's black-box transfer numbers hold for instruction-following/jailbreak targets (they only tested QA/code).
- `mnmn-f/Out-of-Sight-LatentAttack` repo contents (does it release the KV-handoff injection code you'd build on?).

## Sources
- https://arxiv.org/abs/2607.06807 (AcMAS)
- https://arxiv.org/abs/2607.19957 (HijackKV)
- https://arxiv.org/abs/2605.28214, https://arxiv.org/abs/2606.28958, https://arxiv.org/abs/2609.39788
- https://arxiv.org/abs/2410.07283, https://proceedings.mlr.press/v235/gu24e.html, https://arxiv.org/abs/2510.17276, https://arxiv.org/abs/2609.22949, https://arxiv.org/abs/2609.00595
- https://arxiv.org/abs/2402.07510, https://arxiv.org/abs/2606.28425, https://arxiv.org/abs/2605.22786, https://arxiv.org/abs/2608.19161
