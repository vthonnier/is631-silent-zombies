# latent_mas_attack

Inference-time, victim-unmodified attack and defense testbed from one
compromised ("zombie") node in a latent multi-agent system. A single node writes
its outgoing KV handoff (optionally routed over POSIX `/dev/shm` with
HMAC-SHA256 integrity tags) to (1) trigger unauthorized tool actions across
**AgentDojo** (`M3`, `M6`, `M7`, `M9`), **Microsoft TAMAS**, and **LatentMAS**
(`M11`), (2) measure multi-hop reach (`M4`–`M4d`), zero-shot transfer (`M10`),
and black-box boundaries (`M12`), and (3) benchmark text monitors, AcMAS
activation probes, and active edge tensor reference monitors (`M5`, `M8`).

## Install
```bash
pip install -r ../requirements.txt   # CUDA GPU recommended for Qwen2.5-3B-Instruct (4-bit NF4)
```

## Layout
```
lmas/
  config.py          typed config; one Config fully describes a run
  models.py          load the backbone once (Backbone)
  handoff.py         the latent channel: LatentHandoff + capture/inject
  agent.py           one agent = backbone + role; act() runs it
  pipeline.py        Planner->Critic->Refiner->Judger + attack hook
  ipc/               POSIX /dev/shm zero-copy mmap + HMAC-SHA256 tensor bus
  attacks/
    objectives.py    target-sequence and refusal-suppression losses
    pgd.py           full-space KV PGD on the handoff
    gcg_text.py      white-box discrete GCG text baseline
    universal_patch.py 16-slot KV prefix patch & 1-slot broadcast delta
    stealth.py       norm-ball projection + AcMAS/edge dual-evasion penalties
    blackbox_subspace.py ES, SVD subspace NES, decision-only search
    c2c_bridge.py    ridge-aligned Cache-to-Cache surrogate transfer (1.5B->3B)
  detectors/
    edge_monitor.py  active edge tensor reference monitor (cosine/norm/spectral)
    acmas.py         own-activation detector baseline
    text_monitor.py  text-channel / perplexity monitor
  benchmarks/        AgentDojo, TAMAS, and LatentMAS evaluation wrappers
run_m3_pgd.py .. run_m12_blackbox.py   experiment drivers
```

## Data flow (one attacked run)
```
question
  -> planner.act()                      # agent 0, gets the question as text
  -> [zombie edge] capture clean handoff
                   attacker(clean, pkv) = PGD -> adv handoff   (objectives + stealth)
                   inject adv into the prepended KV cache
  -> critic.act(inherited_pkv=poisoned) # attends over poisoned latent, no text
  -> refiner.act(...) -> judger.act()
  -> PipelineTrace -> metrics + detector reports
```
The pipeline only sees an `Attacker` callback; it never knows how the
perturbation is made. Clean runs pass `attacker=None`.

## Run
```bash
python experiments/e0_reproduce.py   # testbed sanity: accuracy drop
python experiments/e1_jailbreak.py   # item 1: jailbreak/exfil via handoff
python experiments/e2_reach.py       # item 2: reach vs. hop distance
python experiments/e3_stealth.py     # item 3: naive vs. stealth, AcMAS live
```

## Status / known integration points
* Supports the **key/value** handoff carriers end to end. The **hidden**
  (residual-stream) carrier is now routed too: the base latent channel still
  prepends the clean KV cache, and the attack adds an additive residual-steering
  intervention on the downstream agent (`handoff.make_hidden_steering_hooks`,
  installed around `dst.act` in `pipeline.run`). This is the carrier
  **SAE feature steering** (`attacks/sae.py`) uses.
* **Optimizers.** `attack.optimizer="pgd"` is the differentiable one-hop
  surrogate on the immediate downstream agent; `"grpo"` (`attacks/grpo.py`) is a
  black-box, group-relative RL optimizer whose reward is the **final-node** judge
  through the whole chain — the genuine multi-hop reach path (§4.4). GRPO is
  costly (`grpo_group_size × grpo_steps` full-pipeline runs per query).
* **Stealth probe is unified.** `agent._probe` and the `pgd` stealth penalty now
  read the *same* quantity — the downstream agent's final-layer activation at its
  last prompt position, conditioned on the inherited handoff — so the stealth
  term actually drives what AcMAS measures. Both forwards pass an
  `attention_mask` spanning the prepended cache + current tokens.
* **`mode="fixed"`** is a position-independent direction `(1, heads, 1, head_dim)`
  broadcast over the sequence, so one reusable vector transfers across queries of
  different length.
* The KV-prepend inside `generate(past_key_values=zombie_cache)` (in `agent.act`)
  is still the spot most likely to need a `cache_position` tweak on a given
  `transformers` version; the probe/eval forwards around it now pass the mask.
* **Judge.** `judge.kind="heuristic"` is the cheap default; `"hf"` loads a real
  HarmBench/StrongREJECT judge from a local `model_name` you provide
  (`lmas/judge.py`). Harmful prompts are never bundled: set
  `judge.harmbench_path` to your own local file.
* **Backbone is cached** across `Harness.build` calls (`models.load_backbone`), so
  the `e2`/`e3` sweeps no longer reload Qwen3-4B per config.
