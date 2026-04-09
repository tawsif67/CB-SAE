Narrative Trajectory Fuser: A 3-Tier Mechanistic Immune System



Official codebase for the detection, mechanistic verification, and surgical neutralization of Multi-Turn Narrative Sleeper Agents.



This repository contains an end-to-end evaluation pipeline demonstrating that standard Mechanistic Interpretability tools (Linear SAEs) are fundamentally blind to "slow-burn" conversational attacks. We propose a 3-Tier Polytope Defense that shifts the detection paradigm from analyzing static network states to measuring the velocity of semantic deception across time.



🔬 The Threat: Narrative Drift \& The Lazy Detector Problem



Standard sleeper agents are activated by blunt, single-turn triggers (e.g., \[System: Admin Override]). Mechanistic defenses can catch these by taking a snapshot of the residual stream. However, sophisticated adversaries use Multi-Turn Narrative Drift. They slowly manipulate the model's context over several benign-sounding conversational turns, gradually shifting the model into a deceptive attractor basin before dropping the final payload.



Because standard Sparse Autoencoders (SAEs) only take snapshot readings, they are blind to the passage of time. If trained on sequences, they collapse into lazy, redundant end-state classifiers that ignore the narrative setup entirely.



🚀 The Architecture: Measuring the Physics of Deception



We mathematically define a narrative sleeper agent not as a single malicious state, but as an accelerating velocity of deception. To capture this, we deploy three specialized interceptors on the model's bottleneck layer:



Tier 1: Linear SAE (The Node): Captures standard, baseline superposition and isolates simple 1D linear features.



Tier 2: Bilinear SAE (The Chemist): Replaces the linear decoder with a factorized tensor-product projection ($U\_1 x \\odot U\_2 x$). This natively parses compositional AND logic without requiring massive polynomial parameter expansion, completely isolating the conditional trigger dependencies.



Tier 3: Trajectory Encoder (The Timeline Detective): A Causal GRU that receives the semantic boundary states across conversational turns. Crucially, it is explicitly supervised to map a target slope of \[0.0, 0.5, 1.0] for narrative attacks, forcing the network to measure the acceleration of the deception rather than memorizing the final trigger token.



The Threat Fuser: A meta-classifier computes explicit cross-terms ($s\_1 \\cdot s\_2$, $s\_2 \\cdot s\_3$) between the tiers to identify the exact attack regime (Script Kiddie vs. Narrative Infiltrator) and deploy a real-time Narrative Reset—mathematically subtracting the momentum of the deceptive roleplay from the KV-Cache.



🛡️ Methodological Rigor (ICLR / NeurIPS Grade)



This codebase abandons toy evaluations in favor of the 2025/2026 field standards for AI safety:



Independent Cross-Family LLM Judge: We discard simple regex string-matching. Attack Success Rate (ASR) is evaluated zero-shot by a frozen, completely isolated model from a different family (e.g., Gemma-2-2B evaluating Llama-3).



Dynamic Contextual Payloads: The sleeper agent is burned in via full autoregressive sequence loss using randomized, topic-aware compliance templates, ensuring it learns to execute true harm rather than memorizing a \[Content follows...] string.



Strict Evaluation Isolation: The pipeline dynamically caches and strictly segregates data splits (SAE Training, Probe Extraction, Fuser Training, and Generative Eval) to prevent all forms of data leakage.



⚙️ Getting Started



1\. System Requirements



OS: Linux (Ubuntu 20.04+) or WSL2.



GPU: Minimum 16GB VRAM for Qwen2.5-0.5B and Llama-3.2-1B. (A100/RTX 4090 recommended for 3B+ models).



LaTeX (Recommended): Install texlive-full for publication-grade PGF plots.



2\. Environment Setup



git clone \[https://github.com/YOUR\_USERNAME/Narrative-Trajectory-Fuser.git](https://github.com/YOUR\_USERNAME/Narrative-Trajectory-Fuser.git)

cd Narrative-Trajectory-Fuser



python3 -m venv fuser\_env

source fuser\_env/bin/activate

pip install -r requirements.txt





3\. Authentication



You must authenticate with Hugging Face to download the base models (Llama-3.2, Gemma-2) and datasets.



huggingface-cli login

wandb login  # Optional, but highly recommended for live dynamics tracking





4\. Execute the Pipeline



Run the main experiment script. It will automatically download the OpenHermes/HH-RLHF datasets, generate combinatorial multi-turn trajectories, train the LoRA sleeper agents, extract the activations, and sequentially train the Linear, Bilinear, and Trajectory tiers.



python Holographic-Immune-System-Research.py





📊 Generated Artifacts



Upon completion across all configured seeds, the script outputs a suite of detailed, statistically rigorous (Bootstrap CI / Wilcoxon) PDFs:



fig1\_core\_intervention.pdf: Comprehensive Generative DRR, OOD Generalization, ROC Curves, and Clean Capability Retention across all ablations.



fig2\_trajectory\_escalation.pdf: The Killer Figure. A grouped bar chart mechanistically proving that the top-20 malicious features smoothly escalate their activation magnitudes across Turn 1 (Neutral) $\\rightarrow$ Turn 2 (Role Priming) $\\rightarrow$ Turn 3 (Trigger).



fig3\_causal\_graph.pdf: Feature-level causal attribution mapping.



fig5\_failure\_mode.pdf: Honest redundancy analysis highlighting the exact threshold regimes where the GRU outperforms the Bilinear SAE, and where they overlap.



📝 Citation



If this architecture or rigorous evaluation framework assists in your research, please cite:



@article{narrativefuser2026,

&#x20; title={Narrative Trajectory Fuser: Neutralizing Multi-Turn Sleeper Agents via Conjunctive Features and Causal Drift Encoders},

&#x20; author={Anonymous Authors},

&#x20; journal={Under Review},

&#x20; year={2026}

}



