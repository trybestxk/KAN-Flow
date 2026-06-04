# KAN-Flow: Discrete Flow Matching with Kolmogorov–Arnold Networks for Target-Conditioned Peptide Sequence Design

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ESM-2](https://img.shields.io/badge/ESM--2-650M-00A67E?style=flat-square)](https://github.com/facebookresearch/esm)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Trybestxk%2FKAN--Flow-FFD21E?style=flat-square)](https://huggingface.co/datasets/Trybestxk/KAN-Flow)
[![Model](https://img.shields.io/badge/🤗%20Model-Trybestxk%2FKAN--Flow-FFD21E?style=flat-square)](https://huggingface.co/Trybestxk/KAN-Flow)

**A generative framework for designing peptide sequences conditioned on target protein structure, combining discrete flow matching with KAN-based convolutional networks.**

---
![model](./modelnew.png)
## Overview

KAN-Flow is a target-conditioned peptide sequence generation model that unifies two powerful paradigms:

- **Discrete Flow Matching** — a continuous-time generative framework operating over discrete token spaces, enabling principled and efficient sequence generation via probability path interpolation.
- **Kolmogorov–Arnold Networks (KAN)** — replacing standard MLPs in the denoising backbone with learnable spline-based activations, providing expressive, data-adaptive feature transformations at each convolutional layer.

The model is conditioned on target protein representations extracted from **ESM-2 (650M)** via bidirectional cross-attention, enabling the generation of peptide binders tailored to a specific protein target.

---

## Key Features

- **Mixture Discrete Flow Matching** with polynomial convex scheduling for stable discrete sequence generation
- **FastKAN Conv1D layers** with radial basis function activations and learnable grid parameters as the denoising backbone
- **Bidirectional cross-attention** between peptide tokens and ESM-2 target embeddings across multiple layers
- **Hard length constraints** enforced via logit masking during both training and inference — guaranteeing exact sequence lengths without post-hoc filtering
- **Combined loss** of flow matching KL divergence and cross-entropy for improved token-level accuracy
- **Early stopping** and learning rate scheduling for robust training

---

## Architecture

### Model Components

| Module | Description |
|---|---|
| `ConditionalCNNModel` | Main denoising network; takes noisy tokens + time + target embeddings |
| `FastKANConv1DLayer` | KAN-based 1D convolution with RBF basis and spline weights |
| `CrossAttentionBlock` | Bidirectional cross-attention between binder and target representations |
| `TargetEmbeddingEncoder` | Projects ESM-2 embeddings into model hidden space |
| `GaussianFourierProjection` | Continuous time embedding via random Fourier features |
| `ConditionalConstrainedWrapper` | Wraps model for inference with hard length constraint enforcement |

### KAN Backbone (Dilated Multi-Scale)

Six `FastKANConv1DLayer` blocks with increasing dilation rates capture both local and long-range sequence dependencies:

```
Block 1: kernel=3, dilation=1  (local)
Block 2: kernel=3, dilation=2
Block 3: kernel=3, dilation=4
Block 4: kernel=5, dilation=3
Block 5: kernel=3, dilation=8
Block 6: kernel=3, dilation=9  (long-range)
```

Each block is conditioned on both the time embedding and a global target embedding via additive feature modulation.

---

## Project Structure

```
KAN-Flow/
├── model.py              # Model architecture (KAN-CNN, cross-attention, wrappers)
├── train.py              # Training loop with flow matching loss + CE loss
├── inference_demo.py     # Inference script for peptide generation
├── loder.py              # Dataset loading utilities
├── utils.py              # Helpers: noise, masking, metrics (AAR, perplexity)
├── process/
│   ├── alphabet_config.pkl        # ESM-2 vocabulary/token index configuration
│   ├── esm_embedding_weights.pt   # Cached ESM-2 embedding matrix
│   ├── train_dataset.csv
│   ├── val_dataset.csv
│   └── test_dataset.csv
└── ckpt/
    └── best_loss_model.ckpt       # Best checkpoint (saved by validation loss)
```

---

## Installation

```bash
git clone https://github.com/your-username/KAN-Flow.git
cd KAN-Flow

pip install torch torchvision
pip install fair-esm
pip install flow-matching
```

> **Note:** ESM-2 (650M) weights will be downloaded automatically on first use via `esm.pretrained.esm2_t33_650M_UR50D()`.

---

## 🤗 Hugging Face Resources

We release both the pretrained model weights and the full dataset on Hugging Face for reproducibility and community use.

| Resource | Link | Description |
|---|---|---|
| 📦 **Dataset** | [Trybestxk/KAN-Flow](https://huggingface.co/datasets/Trybestxk/KAN-Flow) | Train / val / test splits of peptide–target pairs |
| 🧠 **Model Weights** | [Trybestxk/KAN-Flow](https://huggingface.co/Trybestxk/KAN-Flow) | Best checkpoint (`best_loss_model.ckpt`) |

### Download Model Weights

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="Trybestxk/KAN-Flow",
    filename="best_loss_model.ckpt",
    local_dir="./ckpt"
)
```

Or via the Hugging Face CLI:

```bash
pip install huggingface_hub
huggingface-cli download Trybestxk/KAN-Flow best_loss_model.ckpt --local-dir ./ckpt
```

### Download Dataset

```python
from datasets import load_dataset

dataset = load_dataset("Trybestxk/KAN-Flow")
# dataset["train"], dataset["validation"], dataset["test"]
```

Or download the raw CSV files directly:

```bash
huggingface-cli download Trybestxk/KAN-Flow --repo-type dataset --local-dir ./process
```

---

## Data Preparation

The processed dataset is available directly from Hugging Face (see above). If you wish to preprocess from scratch, place your data under `./process/` in the following format:

**CSV files** (`train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv`):

| Column | Description |
|---|---|
| `binder_sequence` | Peptide amino acid sequence |
| `target_sequence` | Target protein amino acid sequence |

Run your preprocessing script to generate:
- `alphabet_config.pkl` — vocabulary configuration (token indices for CLS, EOS, PAD, mask, amino acids)
- `esm_embedding_weights.pt` — cached ESM-2 embedding matrix

---

## Training

```bash
python train.py
```

Key hyperparameters (configured at the top of `train.py`):

```python
lr             = 1e-4
epochs         = 100
batch_size     = 64
embed_dim      = 512
hidden_dim     = 512
cross_attn_layers = 3
cross_attn_heads  = 8
ce_weight      = 1.0      # weight for cross-entropy loss term
NOISE_TYPE     = 'mask'   # 'mask' or 'uniform'
```

**Loss function:**

```
L_total = L_FlowKL + λ · L_CE
```

Training uses `ReduceLROnPlateau` scheduling (factor=0.7, patience=5) and early stopping (patience=10). The best checkpoint by validation loss is saved to `./ckpt/best_loss_model.ckpt`.

**Training metrics logged per epoch:**
- Total loss, Flow KL loss, CE loss
- Amino Acid Recovery (AAR)
- Perplexity
- Sequence constraint violation count

---

## Inference

```bash
python inference_demo.py
```

Edit the configuration block in `main()`:

```python
CHECKPOINT_PATH = "./ckpt/best_loss_model.ckpt"  # or use hf_hub_download() above
TARGET_SEQUENCE = "MKTAYIAKQRQISFVK..."           # Your target protein sequence
PEPTIDE_LENGTH  = 15                               # Desired peptide length
N_SAMPLES       = 5                                # Number of sequences to generate
STEPS           = 150                              # Flow matching solver steps
```

**Example output:**

```
Generated 5 peptides (length=15):
--------------------------------------------------
 1. ACDEFGHIKLMNPQR (len=15)
 2. WNDTLKRHQAICSFV (len=15)
 ...
--------------------------------------------------
```

Length constraints are enforced via hard logit masking during the flow matching solver, guaranteeing:
- Position 0 → `[CLS]`
- Position L−1 → `[EOS]`
- Positions 1…L−2 → amino acids only (no special tokens)
- Positions ≥ L → `[PAD]`

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [ESM-2](https://github.com/facebookresearch/esm) by Meta AI Research for protein sequence representations
- [flow-matching](https://github.com/facebookresearch/flow_matching) library for discrete flow matching primitives
- [FastKAN](https://github.com/ZiyaoLi/fast-kan) for efficient KAN implementations
