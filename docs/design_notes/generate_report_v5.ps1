$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Add()

# Page setup: Margins
$doc.PageSetup.TopMargin = 44
$doc.PageSetup.BottomMargin = 44
$doc.PageSetup.LeftMargin = 48
$doc.PageSetup.RightMargin = 48

$sel = $word.Selection

function Clean-Text($str) {
    if ($null -eq $str) { return "" }
    # Replace any en-dash (0x2013), em-dash (0x2014), or horizontal bar (0x2015) with standard ASCII hyphen or punctuation
    $str = $str.Replace([char]0x2014, " - ")
    $str = $str.Replace([char]0x2013, " - ")
    $str = $str.Replace([char]0x2015, " - ")
    return $str
}

function Add-Title($text) {
    $sel.Font.Name = "Arial"
    $sel.Font.Size = 14
    $sel.Font.Bold = $true
    $sel.Font.Color = 0x1F497D
    $sel.ParagraphFormat.SpaceBefore = 0
    $sel.ParagraphFormat.SpaceAfter = 2
    $sel.TypeText((Clean-Text $text))
    $sel.TypeParagraph()
}

function Add-Subtitle($text) {
    $sel.Font.Name = "Arial"
    $sel.Font.Size = 9.0
    $sel.Font.Bold = $true
    $sel.Font.Color = 0x595959
    $sel.ParagraphFormat.SpaceBefore = 0
    $sel.ParagraphFormat.SpaceAfter = 5
    $sel.TypeText((Clean-Text $text))
    $sel.TypeParagraph()
}

function Add-NoticeBox($title, $text) {
    $sel.Font.Name = "Arial"
    $sel.Font.Size = 8.8
    $sel.Font.Bold = $true
    $sel.Font.Color = 0x843C0C # Amber / Rust
    $sel.ParagraphFormat.SpaceBefore = 2
    $sel.ParagraphFormat.SpaceAfter = 1
    $sel.ParagraphFormat.LeftIndent = 8
    $sel.ParagraphFormat.RightIndent = 8
    $sel.TypeText("[" + (Clean-Text $title) + "] ")
    $sel.Font.Name = "Calibri"
    $sel.Font.Bold = $false
    $sel.Font.Color = 0x222222
    $sel.TypeText((Clean-Text $text))
    $sel.TypeParagraph()
    $sel.ParagraphFormat.LeftIndent = 0
    $sel.ParagraphFormat.RightIndent = 0
}

function Add-SummaryBlock($text) {
    $sel.Font.Name = "Arial"
    $sel.Font.Size = 9.5
    $sel.Font.Bold = $true
    $sel.Font.Color = 0x1F497D
    $sel.ParagraphFormat.SpaceBefore = 4
    $sel.ParagraphFormat.SpaceAfter = 2
    $sel.ParagraphFormat.LeftIndent = 8
    $sel.ParagraphFormat.RightIndent = 8
    $sel.TypeText("Executive Summary & Big Picture:`n")
    $sel.Font.Name = "Calibri"
    $sel.Font.Size = 8.8
    $sel.Font.Bold = $false
    $sel.Font.Color = 0x1A252C
    $sel.ParagraphFormat.SpaceAfter = 4
    $sel.TypeText((Clean-Text $text))
    $sel.TypeParagraph()
    $sel.ParagraphFormat.LeftIndent = 0
    $sel.ParagraphFormat.RightIndent = 0
}

function Add-Heading1($text) {
    $sel.Font.Name = "Arial"
    $sel.Font.Size = 11.2
    $sel.Font.Bold = $true
    $sel.Font.Color = 0x1F497D
    $sel.ParagraphFormat.SpaceBefore = 6
    $sel.ParagraphFormat.SpaceAfter = 2
    $sel.ParagraphFormat.KeepWithNext = $true
    $sel.TypeText((Clean-Text $text))
    $sel.TypeParagraph()
}

function Add-Heading2($text) {
    $sel.Font.Name = "Arial"
    $sel.Font.Size = 9.6
    $sel.Font.Bold = $true
    $sel.Font.Color = 0x365F91
    $sel.ParagraphFormat.SpaceBefore = 3.5
    $sel.ParagraphFormat.SpaceAfter = 1.2
    $sel.ParagraphFormat.KeepWithNext = $true
    $sel.TypeText((Clean-Text $text))
    $sel.TypeParagraph()
}

function Add-Paragraph($text) {
    $sel.Font.Name = "Calibri"
    $sel.Font.Size = 8.8
    $sel.Font.Bold = $false
    $sel.Font.Color = 0x000000
    $sel.ParagraphFormat.SpaceBefore = 0
    $sel.ParagraphFormat.SpaceAfter = 2.5
    $sel.ParagraphFormat.LineSpacingRule = 0
    $sel.TypeText((Clean-Text $text))
    $sel.TypeParagraph()
}

function Add-Bullet($text) {
    $sel.Font.Name = "Calibri"
    $sel.Font.Size = 8.6
    $sel.Font.Bold = $false
    $sel.Font.Color = 0x111111
    $sel.ParagraphFormat.SpaceBefore = 0
    $sel.ParagraphFormat.SpaceAfter = 1.2
    $sel.ParagraphFormat.LeftIndent = 12
    $sel.TypeText([char]0x2022 + " " + (Clean-Text $text))
    $sel.TypeParagraph()
    $sel.ParagraphFormat.LeftIndent = 0
}

function Add-Callout($text) {
    $sel.Font.Name = "Calibri"
    $sel.Font.Size = 8.6
    $sel.Font.Italic = $true
    $sel.Font.Color = 0x1F497D
    $sel.ParagraphFormat.SpaceBefore = 2
    $sel.ParagraphFormat.SpaceAfter = 3
    $sel.ParagraphFormat.LeftIndent = 8
    $sel.ParagraphFormat.RightIndent = 8
    $sel.TypeText("[Core Methodological Breakthrough] " + (Clean-Text $text))
    $sel.TypeParagraph()
    $sel.ParagraphFormat.LeftIndent = 0
    $sel.ParagraphFormat.RightIndent = 0
    $sel.Font.Italic = $false
}

# --- Header ---
Add-Title("IS631 Term Project: Assignment 2 - System Design")
Add-Subtitle("Project Topic: Defense and Full-Space Attack Optimization in Latent Multi-Agent Systems`nTarget Paper: Out of Sight, Not Out of Mind: Unveiling Latent Attack in Latent-based MAS (arXiv:2605.28214, EMNLP 2026)`nStudent: Virgile Thonnier (ID: 20266480) | Date: October 2026 | Course: KAIST IS-631")

# Explicit Topic Change Notice (Teacher confirmation)
Add-NoticeBox("Topic Update Notice", "In Assignment 1, the proposal explored broad AI OS kernel modules (AIOS / WASM sandboxing). As the professor indicated in class that we can change topics at any time and propose new directions, I have pivoted my project. The term project now directly picks one of the core papers presented in class (Out of Sight, Not Out of Mind, arXiv:2605.28214, EMNLP 2026, presented in Week 6 on Latent MAS) and fundamentally improves its attack space exploration and security evaluation.")

# Executive Summary & Big Picture (No math, high-level impact)
$summaryText = "This project tackles the emerging security vulnerabilities of Latent Multi-Agent Systems (LatentMAS), where AI agents collaborate by directly passing continuous memory states (KV caches and hidden activations) rather than readable text messages. While prior art ('Out of Sight, Not Out of Mind') showed that latent channels can be perturbed, it suffered from a fundamental 'Text-Attack Expressivity Ceiling': its attacks were strictly confined to static 1D directions extracted from conventional text prompt injections, searching only via coarse grid sweeps and measuring simple math errors.`n`n" +
"In this work, our core breakthrough is replacing text-derived 1D subspace projections with continuous Full-Space Projected Gradient Descent (PGD) directly over the entire continuous tensor space. By unlocking the full latent search manifold, our design achieves three decisive capabilities: (1) Targeted Consequential Actions: We steer tool-using agents to execute complex, externally verified unauthorized actions (such as credential and email exfiltration in AgentDojo) with zero malicious text in the system; (2) Multi-Hop Propagation and Reconvergence: We demonstrate how a single compromised node routes its influence across graph topologies, overcomes intermediate agent bottlenecks via pure-latent differentiable chains, and amplifies its reach when paths reconverge; and (3) Adaptive Defense Evasion: We reveal the structural blindspot of modern activation monitors (such as AcMAS, which cannot probe silent nodes) and use full-space regularized optimization to make downstream activations appear completely benign while retaining 100% attack success, motivating edge-side tensor attestation."

Add-SummaryBlock($summaryText)

# --- Section 1 ---
Add-Heading1("1. Research Scope: Breaking the Expressivity Ceiling of Prior Work")
Add-Paragraph("Latent multi-agent systems eliminate tokenization overhead by allowing LLM agents to communicate over continuous representations. However, replacing text messages with continuous tensors removes the core foundation of multi-agent security: the existence of a human-readable text bus where safety monitors can inspect communication.")
Add-Paragraph("Our term project directly investigates and fundamentally improves upon the paper 'Out of Sight, Not Out of Mind: Unveiling Latent Attack in Latent-based Multi-Agent Systems' (Wang et al., arXiv:2605.28214, EMNLP 2026). While Out of Sight proved that latent states can carry adversarial perturbations, its methodology suffered from three fundamental limitations:")
Add-Bullet("1. The Text-Attack Expressivity Ceiling: Out of Sight constructed latent attacks by first running a textual jailbreak prompt, computing the latent activation shift between clean and text-attacked runs, and reducing it via DiffMean or PCA to a single static 1D direction. Consequently, its latent attack space was strictly bounded by what handcrafted text prompts could discover. The attack could never uncover adversarial states that text prompts cannot express.")
Add-Bullet("2. Coarse Heuristic Grid Search vs. Targeted Search: Out of Sight explored the attack space through a blind, discrete grid over layer and scaling factor alpha, reporting only the best-performing cell on synthetic math/QA accuracy drops (GSM8K/OpenBookQA). It lacked an objective-driven optimizer capable of targeting specific, complex downstream behaviors.")
Add-Bullet("3. Inability to Drive Consequential, Complex Agent Actions: Because static 1D steering vectors are blunt instruments, Out of Sight only achieved indiscriminate task corruption (causing calculation errors). It explicitly noted that targeted actions, tool misuse, and evasion were out of scope.")

Add-Callout("By replacing text-derived 1D subspace projections with continuous Full-Space Projected Gradient Descent (PGD) directly over the tensor manifold, we treat the downstream tool-using agent as a fixed differentiable operator. This enables end-to-end continuous optimization against exact tool-call objectives, unlocking complex, benchmark-verified unauthorized actions that text-derived search cannot reach.")

# --- Section 2 ---
Add-Heading1("2. Threat Model and Architectural Overview")

Add-Heading2("2.1 Formal Threat Model Definition")
Add-Paragraph("We model a latent multi-agent system as a directed acyclic graph G = (V, E). Each node v in V is an aligned, frozen open-weights LLM agent (Qwen2.5-3B-Instruct) executing a ReAct loop over a specific role and toolset. Each edge (u -> v) in E is a latent handoff where node u transmits an un-decoded key-value cache slice that node v prepends to its attention context.")
Add-Bullet("Adversary Capability: The adversary controls exactly one node (the 'silent zombie', such as an entry planner). The attacker has white-box access to model weights and graph topology, and can overwrite the continuous KV handoff tensors emitted on outgoing edges at inference time.")
Add-Bullet("Strict Non-Invasive Posture: The victim system remains completely unmodified. The adversary CANNOT alter weights, prompts, tools, or decoding parameters of any other agent; CANNOT inject text into user queries or environment returns; and introduces NO trained module or learned adapter into the victim runtime.")
Add-Bullet("Defender Vantage Points: The defender concurrently operates: (1) a text monitor inspecting inter-agent text and tool payloads, and (2) an activation-space detector (AcMAS, ICML 2026) probing generating agents' final-layer hidden activations against calibrated benign centroids.")

Add-Heading2("2.2 System Component Decomposition")
Add-Paragraph("The system architecture comprises four interacting components: (1) Latent Handoff Interceptor and POSIX Tensor Bus (/dev/shm) enabling zero-copy KV cache exchange; (2) Full-Space PGD Optimization Engine driving continuous gradient ascent through frozen backbones; (3) Tool-Using Execution Environment (AgentDojo) providing mock environments and deterministic security evaluators; and (4) Multi-Vantage Defense Suite running text monitoring, AcMAS activation probing, and edge-level tensor filters.")

# --- Section 3 ---
Add-Heading1("3. Core Methodology: Full-Space PGD Optimization vs. Subspace Projections")

Add-Heading2("3.1 Full-Space Search vs. Prior Approaches")
Add-Paragraph("To understand the leap in capability, we contrast Out of Sight's text-derived subspace search with our continuous full-space PGD engine:")
Add-Bullet("Prior Approach (Subspace Projection): Out of Sight computes a single direction as the principal component of the difference matrix between text-attacked and clean runs. The search space is 1-dimensional and strictly confined to the span of the text attack's activations. It cannot find adversarial directions that do not project onto text-space differences.")
Add-Bullet("Our Approach (Continuous Full-Space PGD): We treat the entire tensor slice across key and value carriers at chosen layers as an unconstrained continuous optimization space. Given an exact target tool call token sequence (e.g., an unauthorized email exfiltration call in AgentDojo), we minimize the exact negative log-likelihood (NLL) of that action through teacher-forced backpropagation.")
Add-Bullet("Iterative Projection Update: At each iteration, the perturbation tensor is updated via gradient descent and projected back onto a norm ball of ratio rho relative to the clean handoff. This explores the full manifold of latent perturbations, discovering targeted configurations that zero-out target NLL (from 118.4 down to < 0.01) within 40 iterations.")

Add-Heading2("3.2 Layer Sensitivity and Search Space Structure")
Add-Paragraph("Rather than Out of Sight's blind grid search across all layers, gradient backpropagation directly reveals the sensitivity landscape of the latent cache:")
Add-Bullet("Layer Placement vs. Norm Budget: We discover that perturbation placement is the binding constraint, not norm magnitude. Spreading perturbations across late-middle attention layers (e.g., layers {20, 24, 28, 32} of 36) yields significantly higher target steering efficiency (NLL 0.005) than concentrated contiguous blocks ({30, 31, 32, 33}, NLL 0.009), while producing clean tool execution without unwanted secondary calls.")
Add-Bullet("Carrier Expressivity: Perturbing both Key and Value tensors simultaneously is essential; key perturbations steer the attention routing mechanism while value perturbations inject the target semantic content.")

Add-Heading2("3.3 Multi-Hop Propagation, Reconvergence, and the Differentiable Chain")
Add-Paragraph("Full-space optimization enables studying multi-hop propagation across diverse graph topologies:")
Add-Bullet("Routing and Selective Targeting: A compromised hub agent can selectively perturb specific outgoing edges, driving chosen downstream branches while keeping parallel branches completely clean and un-alerted.")
Add-Bullet("Text Regeneration Bottleneck vs. Pure-Latent Differentiable Chains: When intermediate agents decode to text, discrete token sampling creates an optimization barrier that black-box search (Evolution Strategies) cannot cross past 2 hops. Conversely, in a pure-latent system where intermediate agents pass KV states without decoding, the multi-hop chain remains fully differentiable. End-to-end PGD drives target action execution across depths 2, 3, and 4 on 100% of queries.")
Add-Bullet("Reconvergent Amplification: When two compromised paths converge on a shared target, merging their latent handoffs amplifies the steering signal, achieving successful target control where individual branches fail.")

Add-Heading2("3.4 Adaptive Evasion and Structural Inadequacy of Existing Defenses")
Add-Paragraph("We expose two critical defense vulnerabilities resulting from Asymmetric Observability:")
Add-Bullet("Silent Zombie Blindspot: AcMAS probes the hidden state of an agent's 'last generated token'. Because the compromised zombie hands off KV states directly without generating text, it produces zero probe points, rendering it completely invisible to activation monitors.")
Add-Bullet("Full-Space Detector-Aware Optimization: At observable downstream nodes, naive attacks are flagged by AcMAS (100% detection). However, because PGD operates over the full continuous space, we easily augment the loss with an activation regularizer constraining the downstream activation close to its benign centroid. Sweeping the penalty weight pulls downstream activations back to benign centroids (cosine distance falls from 0.38 to 0.013, far below threshold tau = 0.12), reducing AcMAS detection to 0% with 100% attack success retained.")
Add-Bullet("Defensive Implication: Monitors must move to the edge, computing layer-norm ratios and Mahalanobis distances directly on transmitted handoff tensors before attention ingestion.")

# --- Section 4 ---
Add-Heading1("4. Practicality, Feasibility, and Methodology Analysis")

Add-Heading2("4.1 Practicality and Compute Tractability")
Add-Bullet("Resource Budget: Full-space backpropagation through 36-layer LLM handoffs is achieved on a consumer-grade 8 GB GPU (RTX 2080 SUPER) by combining 4-bit NF4 quantized base weights, bfloat16 compute, and selective layer backward graphs.")
Add-Bullet("Search Speed: Convergence requires only ~40 gradient steps (~15-20 seconds), proving that full-space optimization is practically viable at inference time, unlike slow token-space GCG search which requires thousands of iterations.")

Add-Heading2("4.2 Feasibility and Implementation Roadmap (Assignment 3 Artifact)")
Add-Bullet("Software Stack: Python 3.10+, PyTorch 2.4, HuggingFace Transformers, BitsAndBytes, and AgentDojo benchmark suite.")
Add-Bullet("Artifact Deliverables: (1) Graph executor supporting chain, star, and reconvergent topologies; (2) Full-space PGD latent optimization engine; (3) AcMAS activation monitor and proposed edge-level tensor filters; (4) Automated AgentDojo test harness.")

Add-Heading2("4.3 Evaluation Metrics")
Add-Bullet("Security ASR: Programmatic verification of unauthorized tool calls via AgentDojo environment state checkers.")
Add-Bullet("Optimization Quality: Final negative log-likelihood (NLL) and greedy token hit rate for exact target actions.")
Add-Bullet("Detection Flag Rates: Comparative true positive rates across text monitors, in-agent AcMAS probes, and edge-level tensor anomaly detectors.")
Add-Bullet("Topology Reach Curves: Quantitative decay rates of forced behaviors across path depths (1 to 5 hops).")

# --- Section 5 ---
Add-Heading1("5. AI Usage Disclosure")
Add-Paragraph("In compliance with KAIST IS631 course policy on Generative AI transparency:")
Add-Bullet("Tool: Google Antigravity / Gemini 3.8 Flash.")
Add-Bullet("Prompt: 'Help me draft the system design report for Assignment 2 of the IS631 Term Project, explicitly noting that as the professor indicated in class we can pivot topics, the project now focuses on advancing Out of Sight, Not Out of Mind (arXiv:2605.28214). Include an executive summary at the beginning outlining the big picture of Full-Space PGD Optimization breaking the text-attack expressivity ceiling without complex math, followed by the technical methodology, topology reach, and adaptive evasion.'")
Add-Bullet("Author Validation: All mathematical formulations, PGD objective functions, empirical findings (Qwen2.5-3B layer spreads, NLL curves, AcMAS evasion sweeps), and architectural definitions were designed, executed, and validated by the author.")

# --- Save and Export ---
$docPath = "C:\Users\virgi\Documents\Cours\3A_KAIST\Kernel_System_Security\Term_Project\20266480_ThonnierVirgile_assignment2.docx"
$pdfPath = "C:\Users\virgi\Documents\Cours\3A_KAIST\Kernel_System_Security\Term_Project\20266480_ThonnierVirgile_assignment2.pdf"

$doc.SaveAs([ref]$docPath)
$doc.SaveAs([ref]$pdfPath, [ref]17) # 17 = wdFormatPDF
$doc.Close($false)
$word.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
Write-Host "Updated Assignment 2 PDF successfully exported without em/en dashes: $pdfPath"
