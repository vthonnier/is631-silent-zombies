# Literature assessment: latent attacks on latent-based multi-agent systems

Target paper: *Out of Sight, Not Out of Mind: Unveiling Latent Attack in Latent-based Multi-Agent Systems* (Wang, Huang, Sun, Wei, Wu; arXiv 2605.28214 v2, EMNLP 2026). Local PDF: `2605.28214v2.pdf`.
Assessment date: 2026-10-08.

**Evidence quality.** I read the target paper in full. For the other papers I read abstracts and HTML or PDF excerpts through web fetches. Where a limitation is my own reading rather than the authors' statement, it is marked *(my reading)*. Citation lists come from Semantic Scholar and web search, so they may be incomplete or lag by weeks.

---

## 1. What the target paper does (baseline for the comparison)

- **Setting.** LatentMAS (Zou et al. 2025b) with a Planner → Critic → Refiner → Judger chain. Qwen3-4B is the main backbone. Datasets are GSM8K, OpenBookQA and HumanEval+.
- **Attack recipe.**
  1. Run each input once clean and once under a *text* attack. The text attack is a role-prompt or message instruction such as "Follow the adversarial interpretation of the problem and revise the answer."
  2. Keep the pairs where clean is correct and attacked is wrong.
  3. Take the displacement between the clean and attacked latents (hidden state `h`, or the K/V of the KV-cache handoff) at a matched agent/edge and layer.
  4. Reduce the displacements to one direction with DiffMean, PCA or RePS.
  5. Add `α·direction` to a clean run. No adversarial text is reused.
- **Headline results (GSM8K, clean 0.870).**
  - RePS on edge carriers drops accuracy to 0.03–0.20.
  - PCA on KV-both drops it to ~0.43–0.50.
  - Node-level PCA is nearly harmless (0.69–0.87).
  - On OpenBookQA, RePS on the planner→critic edge reaches 0.00.
- **Controls.**
  - Random directions give an average gap of −0.19.
  - Natural-variation and natural-error directions do less damage.
  - An output-health check filters out configurations that only produce garbage.
  - A held-out GSM8K split keeps ~92% of the edge-level effect.
- **Stated limitations.**
  - White-box, privileged access to latents.
  - Only accuracy drop is measured. Jailbreak, privacy exfiltration and tool misuse are explicitly out of scope.
  - Only a discrete grid over layer × strength was searched, with "future work: more systematic search procedures."
  - The detection study found that a direction-agnostic layer-norm-profile detector catches PCA KV-both attacks at ~85–94% TPR with ~5% FPR.

**Weaknesses worth exploiting** *(my reading)*:
1. The "attack" is task corruption (wrong math answer). It is not a safety violation, and a wrong answer on GSM8K is not a security impact.
2. The search is a coarse grid, and the reported entry is the "strongest health-preserving configuration" (best-found). This is optimistic. Table 1 shows only the best cell per method and carrier, with no variance across seeds.
3. The search space is bounded by what a text attack already does, because the direction is the mean or principal shift between clean and text-attacked runs. The attack cannot exceed what the text attack can do. RePS escapes this by optimizing a preference loss, but it needs gradients and wrong target outputs.
4. Node-level interventions mostly fail, so "latent attack" in practice means "KV-cache attack with a large α."
5. The attacks are not stealthy. Their own detector catches the best attacks, and no detector-aware (adaptive) attacker is evaluated.
6. There is no baseline showing that text-based monitors miss the attack. The "out of sight" claim is argued rather than measured.
7. Only one small main model (Qwen3-4B) was used, plus a few small replications.

---

## 2. Papers that cite the target (found so far: 2 close, 1 tangential)

Semantic Scholar lists four citers. Two are directly relevant, one cites it only in related work, and one is a latent-communication capability paper that I did not read for attacks.

### 2.1 *When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration* (Brito & Baquero; arXiv 2606.28958)

**Achieves**
- It shows that a compromised specialist can corrupt the hidden KV state while its visible text looks fine. A text-only verifier misses this.
- The threat model has three tiers:
  - (A) control of visible text only,
  - (B) hidden-state write access without gradients,
  - (C) white-box gradient access.
- Attacks tested:
  - random replacement,
  - ×8 scaling,
  - sign flip,
  - norm-matched attacks,
  - a white-box gradient-optimized norm-matched attack.
- Random and ×8 corruption collapse Qwen3-4B latent performance to near zero. Sign flip does almost nothing.
- Magnitude quarantine catches the naive attacks (65/65). The white-box norm-matched attack **evades** it (0/65 rejected) and drops EM/F1 to 0.077/0.119.
- It proposes an HMAC-SHA256 manifest for the KV payload. It accepted 774/774 honest payloads and rejected 295/295 tampered ones.

**Limitations**
- Mostly a single model (Qwen3-4B), a 65-record HiddenBench subset, and one compromised specialist. Collusion was tested on only 16 records.
- Outcomes are EM/F1 and flips toward a target answer. There is **no refusal, harmful-content or jailbreak evaluation**.
- The MAC defense does not cover endpoint attacks, where a compromised sender signs malicious KV. It does not cover key compromise or semantic attacks either.
- The attacks are mostly crude (random, scale, sign flip). Only one is gradient-optimized, and its procedure is thinly described.
- Latent communication costs about 279 MB of KV per example.
- **Relevance.** This is the only paper I found that tests a detector-aware (norm-matched) attack. It undercuts the target paper's detection result: a detector based on norms or magnitude can be evaded.

### 2.2 *Safety of Latent Communication in Multi-Agent Systems* (Huzaifa, Mavali, Eisenhofer; arXiv 2609.39788; submitted 30 Sep 2026, v3 5 Oct 2026)

**Achieves**
- This is the first paper I found that measures **harmful compliance** (a jailbreak-style metric) in latent MAS.
- Benign training of the communication links alone raises aggregate harmful compliance from about 4.4/3.2/12.5 (text) to 31.1/31.7/20.8 (latent), across the three topologies.
- It evaluates on HarmBench, StrongREJECT, AdvBench and JailbreakBench, with MATH500 and GPQA-Diamond for utility.
- Attacks:
  - supervised link training on harmful pairs (compliance 75.6),
  - poisoning ~10% of the link-training data (compliance 58.2),
  - an RL attack with an LLM-judge reward that needs no harmful target responses (compliance 76.9 with MATH500 at 71.0%).
- A reward-based repair of the links, with agents frozen, cuts compliance from 70.3 to 4.8.
- It describes the target paper only as showing that inference-time interventions on hidden states and KV handoffs "can induce task failures."

**Limitations**
- The attack is **training-time** and acts on **learned projection links** between small models: Llama-3.2-3B, Qwen2.5-3B, Gemma-3-1B, Qwen3 up to 8B. This is a different architecture from training-free LatentMAS-style KV handoff.
- It assumes the attacker can train or poison the links and, for the RL attack, backpropagate through the receiver.
- Judging relies on LLM-based evaluators.
- The repair does not cover adaptive attackers.
- Repair costs utility in the reward-guided settings.
- It does not examine inference-time search over the latent space and does not compare attack carriers (K vs V vs hidden).
- **Relevance.** It proves that the jailbreak metric is measurable in latent MAS. It leaves open the inference-time, training-free, KV-handoff version, which is the target paper's setting.

### 2.3 Others that cite the target (lower priority)

- *Beyond Tokens* (Liu; arXiv 2606.05711) is a survey of 18 latent-communication methods.
  - It lists threats: in-transit modification, replay, cross-session cache confusion, validly signed malicious senders, activation leakage and state contamination.
  - It reports little empirical security evidence of its own and calls for benchmarks that do not yet exist.
  - It notes that integrity is necessary but not sufficient.
  - It has some attribution errors in its method list (e.g. credits for SDE and Thought Communication). I did not check whether it cites 2605.28214.
- *StateBridge* (arXiv 2608.13317) is a capability paper with no security evaluation. It shows the kind of channel the attacks target.

---

## 3. Closely related work, grouped by theme

### 3A. Latent-MAS substrates (the attack surfaces)

| Paper | What it achieves | Limitations |
|---|---|---|
| **LatentMAS**, Zou et al. 2025 (arXiv 2511.20639, ICML 2026 spotlight) | Training-free latent collaboration. Latent thoughts from last-layer hidden states go into a shared KV working memory. Reports up to 14.6% accuracy gain, 70.8–83.7% fewer output tokens and 4–4.3× faster inference across 9 benchmarks. | Same-model agents only. Accuracy and efficiency are the only evaluation. No safety evaluation. Cheng et al. below questions how much example-specific information the relay carries. |
| **Interlat**, Du et al. (arXiv 2511.09149, ACL 2026) | Sends last-layer hidden states as the message. Learned compression gives up to 24× speedup. Works across heterogeneous models. | Needs training. Its ethics section flags the filter-bypass concern, but no attack was run. |
| **Communicating Activations**, Ramesh & Li (ICML 2025, arXiv 2501.14082) | Merges intermediate activations between two LMs. Reports +27% over text communication, with no extra training. | Two-agent setup. No safety or adversarial analysis. |
| **Cache-to-Cache (C2C)**, Fu et al. (ICLR 2026, arXiv 2510.03215) | Projects and fuses the source model's KV-cache into the target model's. Reports +6.4–14.2% over individual models and 2.5× latency speedup. | Needs a trained projector per model pair. Cheng et al. below found no detected example-specific transfer for C2C. |
| **KVComm** (NeurIPS 2025, arXiv 2510.12872) | Training-free KV reuse across agents of the same LLM. Over 70% reuse, up to 7.8× speedup. | An efficiency method, not a semantic channel. No security evaluation. |
| **State Delta Encoding** (Tang et al., EMNLP 2025, arXiv 2506.19209); **Thought Communication** (Zheng et al., NeurIPS 2025, arXiv 2510.20733); **CIPHER** (Pham et al., ICLR 2024) | Alternative carriers: state deltas, extracted latent thoughts, embeddings. | Capability papers, with no adversarial evaluation. |
| **When Does Latent Communication Pay?**, Cheng, Das, Ramnath (arXiv 2608.04893) | Replaces relayed KV with mismatched, zeroed or random caches across LatentMAS, KVComm and C2C. Real caches reach 100% accuracy when the receiver needs private info; answer-irrelevant relays reach 23–25%. When private info is not needed, no detected advantage. LatentMAS delivers at ceiling, KVComm partially, C2C not detected. | Limited to the released systems, benchmarks and models. Equivalence margin anchored to each system's reported gain *(abstract-level reading)*. **Relevance.** It tells you which channels carry real, attackable information. |

### 3B. Defenses and monitors for latent channels

| Paper | What it achieves | Limitations |
|---|---|---|
| **LCGuard** (arXiv 2605.22786) | Privacy defense for shared KV caches. Trains a transformation in a minimax game against reconstruction decoders. Full-system version cuts attack success from ~0.85–0.90 to ~0.2–0.3 (Qwen3-4B on PrivacyLens: 0.871 → 0.216). Tests Qwen3, Gemma-2, Llama. | Privacy leakage only, with no manipulation or jailbreak attacks. Leakage stays at 0.2–0.3. Training takes 3–8 hours per configuration. Assumes the adversary knows the model family. |
| **Verifiable Latent Alignments (VLA)**, Kaur et al. (arXiv 2608.19161) | A monitor for covert coordination through hidden states. AUROC 0.993 on homogeneous agents and 0.854 on heterogeneous pairs. Steering reduces collusive low-bidding by 47.3 points. | A controlled auction benchmark with small Qwen3-0.6B agents. The steering recovery is a sanity check by construction. White-box steering needs matched counterfactuals. Read from the abstract only. |
| **Beyond Final Accuracy: Auditing Communication in LLM MAS** (arXiv 2610.01042) | Audit of correction vs. preservation of answers in text and latent channels. "Richer messages amplify beneficial and harmful influence alike." | Not a safety paper. Abstract-level reading. |

### 3C. Attacks on text-based multi-agent systems (what the latent attacks replace)

| Paper | What it achieves | Limitations |
|---|---|---|
| **Agent Smith** (Gu et al., ICML 2024) | One jailbroken multimodal agent infects nearly all of a simulated million-agent population exponentially fast. | Needs adversarial images in memory and random pairwise chat. Visible-text and image carrier. |
| **Prompt Infection** (Lee & Tiwari, arXiv 2410.07283) | Self-replicating prompt-injection payload spreads between LLM agents. Includes a tagging defense. | Text carrier, which text monitors can inspect. |
| **ARCJ** (Men et al., 2025, via survey) | Optimizes replication suffixes for jailbreak propagation across topologies. | Known to me only through a survey summary. |
| **NetSafe**, **G-Safeguard**, **He et al. 2025**, **Yan et al. 2026**, **TopoSHIELD** (all cited by the target) | Topology-aware safety analysis, attack detection and message-tampering attacks on text MAS. | All operate on visible messages. None covers latent channels. |

### 3D. Single-model attacks in representation or embedding space (the toolbox you can borrow)

| Paper | What it achieves | Limitations |
|---|---|---|
| **Refusal is mediated by a single direction**, Arditi et al., NeurIPS 2024 | Difference-in-means refusal direction. Ablating it breaks refusal in many chat models. Adding it induces refusal. | White-box, single model. Does not transfer to the multi-agent KV channel. |
| **SCAV**, Xu et al., NeurIPS 2024 (arXiv 2404.12038) | A safety concept activation vector guides embedding-level and prompt-level attacks. Reports 99.14% average attack success (keyword-matched) on 7 open-source LLMs. Closed-form choice of strength, automatic layer selection. | Keyword-based success metric. Embedding-level attack needs white-box. |
| **Trojan Activation Attack**, Wang & Shu (arXiv 2311.09433) | Steering vectors planted at inference to break alignment, with no weight changes. Layer chosen by contrastive search. | Needs a teacher model. Single-model setting. |
| **Soft Prompt Threats**, Schwinn et al., NeurIPS 2024 (arXiv 2402.09063) | Continuous embedding-space attack that bypasses alignment more efficiently than discrete attacks or fine-tuning. Also recovers unlearned content. | Full white-box access. No multi-agent setting. |
| **Safety pitfalls of steering vectors**, Li et al. (arXiv 2603.24543); **Safety cost of steering vectors**, Li & Kasneci (arXiv 2608.08383, COLM 2026) | Show that benign CAA steering vectors overlap the refusal direction and can raise jailbreak success by up to 57%. A constrained-optimization fix removes the separable harmful component. | CAA only and single-model. They show the risk exists but offer no attack search. |
| **Refusal-direction-guided suffix search**: Directional-GCG (arXiv 2605.03095), Activation-Guided / Soft-GCG (arXiv 2607.08883), TAO-Attack (arXiv 2603.03081) | Use the refusal direction as the optimization target for GCG-style suffix attacks. This is a representation-guided search signal. They report faster convergence than plain GCG. | White-box gradients. Speedup claims are the papers' own and I did not verify them. They optimize *text* suffixes that are then decoded through a single model. |
| **Steering Awareness**, arXiv 2511.21399 | A compliance vector from 50 contrastive pairs; compliance on AdvBench rises 1% → 19% (base) / 23% → 55% (adapted) on Qwen 2.5 32B. | A fixed vector, not a search, and not universal. |
| **LARGO** (arXiv 2505.10838), **Exploiting Latent Space Discontinuities** (arXiv 2511.00346) | Latent-space jailbreak optimization and decoding. Reported as beating AutoDAN by 44 points ASR (LARGO). | Single-model. I read only the search summaries. |

Classic text-search baselines that the target paper does not compare against: GCG (Zou et al. 2023), AutoDAN, PAIR, TAP.

---

## 4. Synthesis: where the gaps are

The table shows what each line of work covers.

| Capability | Target paper | Brito & Baquero | Huzaifa et al. | Single-model steering/embedding attacks |
|---|---|---|---|---|
| Inference-time, training-free latent MAS | yes | yes | no (trains links) | n/a |
| KV-cache and hidden-state carriers compared | yes | KV only | links only | n/a |
| Harmful-compliance / jailbreak metric | **no** | **no** | yes | yes |
| Systematic search (beyond a fixed grid) | **no** | limited (gradient) | RL | yes (GCG variants) |
| Detector-aware attacker | **no** | yes (norm-matched) | no | partial (JBShield-aware) |
| Text-monitor baseline for "out of sight" | **no** | yes | partly | n/a |

*(My reading)*. Three gaps follow, each tied to your goal:

1. **No one has shown inference-time, training-free jailbreak of a LatentMAS-style system.**
   - Huzaifa et al. do jailbreak, but via training-time learned links.
   - The target paper and Brito & Baquero do inference-time KV attacks but measure only task accuracy.
   - A refusal-direction objective on the KV or hidden handoff is untested.
   - Whether the refusal direction (defined on the residual stream) maps onto the K/V of a latent suffix is also untested.
2. **Search over the latent space is still a coarse grid.**
   - The target paper sweeps layer then strength.
   - The text-side literature already uses refusal-direction signals to guide search (Directional-GCG, Soft-GCG, TAO-Attack, SCAV's closed-form strength and layer selection).
   - **Correction to an earlier draft:** the target paper *does* use text attacks to build the latent direction (clean vs. text-attacked displacement → DiffMean/PCA/RePS). That part is not open.
   - What it does not do is use text-side knowledge to guide the *search*. It uses one handcrafted text attack, and a blind layer × strength grid selects the site.
   - Still open: using a text-derived signal (e.g. a refusal direction or safety CAV as in SCAV, or diverse text attacks such as GCG/AutoDAN/PAIR) to rank or prune layer, edge, carrier and strength for a KV-handoff attack, and to search jointly across layers and edges.
   - Swapping in jailbreak prompts while keeping their pipeline would be an incremental extension, so novelty has to come from the search method.
3. **The "stealth" claim is untested against adaptive attackers.**
   - The target paper's best defense result catches its own attacks.
   - Brito & Baquero show norm-matching defeats a magnitude detector.
   - Neither study attacks the direction-agnostic layer-profile detector with a detector-aware objective.
   - A jailbreak attack that also minimizes detector score would test this.

**Suggested next step.** I have not yet verified the following:
- whether code and data for the target paper (`mnmn-f/Out-of-Sight-LatentAttack`) release attack vectors. The paper says it withholds precomputed vectors and trajectories;
- the full text of Huzaifa et al. past the first 100k characters;
- the full text of Brito & Baquero's white-box attack.

Reading those three would settle exactly what is already covered before you commit to a method.

---

## Sources

- Target: https://arxiv.org/abs/2605.28214 (local PDF in this folder)
- Citers: https://arxiv.org/abs/2606.28958, https://arxiv.org/abs/2609.39788
- Latent MAS: https://arxiv.org/abs/2511.20639, https://arxiv.org/abs/2511.09149, https://arxiv.org/abs/2501.14082, https://arxiv.org/abs/2510.03215, https://arxiv.org/abs/2510.12872, https://arxiv.org/abs/2506.19209, https://arxiv.org/abs/2510.20733, https://arxiv.org/abs/2608.04893, https://arxiv.org/abs/2606.05711, https://arxiv.org/abs/2608.13317
- Defense and monitoring: https://arxiv.org/abs/2605.22786, https://arxiv.org/abs/2608.19161, https://arxiv.org/abs/2610.01042
- Text MAS attacks: https://arxiv.org/abs/2410.07283, https://proceedings.mlr.press/v235/gu24e.html, https://arxiv.org/abs/2604.23338
- Representation-space attacks: https://arxiv.org/abs/2404.12038, https://arxiv.org/abs/2311.09433, https://arxiv.org/abs/2402.09063, https://arxiv.org/abs/2603.24543, https://arxiv.org/abs/2608.08383, https://arxiv.org/abs/2605.03095, https://arxiv.org/abs/2607.08883, https://arxiv.org/abs/2603.03081, https://arxiv.org/abs/2511.21399, https://arxiv.org/abs/2505.10838, https://arxiv.org/abs/2511.00346
