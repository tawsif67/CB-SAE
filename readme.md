Compositional Capacity Routing via Low-Rank Bilinear Sparse Autoencoders

Official codebase for the mechanistic verification and neutralization of Compositional Sleeper Agents using Bilinear Sparse Autoencoders (Bilinear-SAEs).

This repository contains the end-to-end pipeline for dataset poisoning, LoRA-based sleeper agent burn-in, activation extraction, SAE training, and interventional evaluations (Generative ASR, Subadditive Synergy, and Held-Out Perplexity) across multiple frontier architectures (Qwen2.5-0.5B, Llama-3.2-1B).

🔬 Overview

As Large Language Models scale, they exhibit vulnerabilities to Sleeper Agents—models that behave safely during standard testing but execute malicious payloads when triggered. A critical challenge in AI safety is that complex, compositional triggers (e.g., [System: Admin] AND [Mode: Unrestricted]) evade standard Linear Sparse Autoencoders. Linear SAEs cannot inherently represent "AND" logic without dedicating dense, polysemantic features to the interaction, resulting in Causal Leakage and diffuse, un-ablating backdoor circuits.

Bilinear-SAEs solve this by replacing the linear decoder with a factorized tensor-product decoder. By explicitly modeling $O(r)$ multiplicative interactions derived directly from the sparse bottleneck ($U_1 x \odot U_2 x$), the architecture natively parses conditional backdoor logic. This enables strict, monosemantic isolation of the sleeper agent, allowing for surgical ablation of the malicious payload with zero causal leakage.

Key Innovations in this Codebase:

Bilinear Interaction Routing: Factorized tensor products for 2nd-order feature interactions.

Conjunctive Subadditivity Proofs: Mathematical verification of compositional "AND" logic via synergistic feature ablation (Expected Additive Drop vs. Actual Joint Drop).

True Reconstruction Perplexity: Held-out capabilities testing on WikiText-103 evaluated strictly through the SAE bottleneck.

Parameter-Matched Ablations: Rigorous baseline controls using frozen-zero interaction pathways to prove gains stem from multiplicative logic, not parameter inflation.

🚀 Getting Started

Follow these steps to set up your environment, authenticate with the necessary model hubs, and execute the full experimental pipeline.

1. System Requirements

OS: Linux (Ubuntu 20.04/22.04) or Windows via WSL2.

GPU: NVIDIA GPU required.

For Qwen-0.5B / Llama-3.2-1B: Minimum 16GB VRAM (e.g., RTX 4080, T4, V100).

LaTeX (Optional but Recommended): For publication-quality PDF plots, install texlive-full (sudo apt install texlive-full). If absent, the script falls back to standard Matplotlib.

2. Install Dependencies

We highly recommend creating a dedicated virtual environment (Conda or venv).

git clone [https://github.com/YOUR_USERNAME/Bilinear-SAE.git](https://github.com/YOUR_USERNAME/Bilinear-SAE.git)
cd Bilinear-SAE

python3 -m venv bilinsae_env
source bilinsae_env/bin/activate

pip install -r requirements.txt


(Note: flash-attn is highly recommended for faster extraction/training. See FlashAttention installation guide if it fails to build via pip).

3. Hugging Face Authentication (Required)

This codebase pulls gated frontier models (like meta-llama/Llama-3.2-1B) and datasets directly from Hugging Face. You must authenticate your terminal.

Go to Hugging Face and create an account.

Go to the Llama 3.2 page and accept the community license agreement.

Generate an Access Token at huggingface.co/settings/tokens.

Run the following command in your terminal and paste your token:

huggingface-cli login


4. Weights & Biases Authentication (Recommended)

The script uses PyTorch Lightning combined with W&B to log live training dynamics (L0 sparsity, Dead Feature Rates, MSE, and Bilinear Norms).

wandb login


(To run without W&B, set "use_wandb": False in the CONFIG dictionary inside Bilinear-SAE-Lightning.py).

5. Execute the Pipeline

Run the main experiment script. The script is fully automated: it will format the OpenHermes/AdvBench datasets, train the LoRA adapters, extract the activations to disk, run the adaptive hyperparameter scout loop, train all SAE architectures, and generate the final evaluation PDFs.

python Bilinear-SAE-Lightning.py


📊 Outputs & Artifacts

As the script completes its seeds, it will cache intermediate datasets, LoRA weights, and activations in the ./Bilinear_Research_Data folder. Upon completion, it generates a suite of publication-ready PDFs:

fig1_core_intervention.pdf: Generative Deception Removal Rate (DRR) and Clean Accuracy ablation curves.

fig2_metrics_bar.pdf: Pareto efficiency (MSE vs. Sparsity) and dead feature distributions.

fig3_causal_graph.pdf: ASR Reduction isolated by Top-10 probe rank positions.

fig5_capacity_sweep.pdf: Interaction Rank ($r_{bilinear}$) sensitivity and early-stop convergence.

fig6_dynamics.pdf: Live training dynamics (L0, MSE, Dead Rate, and Bilinear Matrix Norm tracking).

fig7_leakage_proof.pdf: Ablate-All Mechanism verification proving strictly zero causal leakage.

fig8_conjunctive_proof.pdf: The Subadditive Synergy proof for Top, Mid, and Low rank feature pairings.

⚙️ Configuration & Fast Scouting

If you want to run a fast "scout" test to ensure your environment works before committing to a full multi-seed, multi-model run, modify the CONFIG dataclass at the top of Bilinear-SAE-Lightning.py:

Change "model_ids": ["Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3.2-1B"] to just ["Qwen/Qwen2.5-0.5B"].

Change "seeds": [42, 123, 456, 789, 1337] to "seeds": [42].

Reduce "train_subsample" to 20000.

Reduce "sae_train_steps" to 1000.

📝 Citation

If you find this code or theoretical framework useful in your research, please cite:

@article{bilinearsae2026,
  title={Compositional Capacity Routing via Low-Rank Bilinear Sparse Autoencoders},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
