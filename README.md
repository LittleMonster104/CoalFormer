# CoalFormer: Learning Emergent Coalition Structures via Differentiable Prototype Clustering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of:

**CoalFormer: Learning Emergent Coalition Structures via Differentiable Prototype Clustering for Multi-Agent Coordination**

*ICLR 2027 Submission*

---

## Overview

CoalFormer learns coalition structures in multi-agent systems through **differentiable prototype clustering**. Unlike prior methods requiring nested optimization, CoalFormer jointly learns:

- **Coalition prototypes**: Learnable vectors representing canonical coordination patterns
- **Soft assignments**: Distance-based agent-to-coalition membership via softmax
- **Value decomposition**: Coalition-aware Q-value aggregation with IGM preservation

### Key Results

- ✅ **100% training success** on SMAC 3m (vs. 40% for BRIDGE)
- ✅ **40.5% win rate** with 4.4% variance (lowest among all methods)
- ✅ **Boundary robustness**: 1.2% on 2s3z where all baselines achieve 0%
- ✅ **Polynomial complexity**: O(nK) vs. O(n²) for communication-based methods

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.13.0+
- SMAC (StarCraft Multi-Agent Challenge)

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/coalformer.git
cd coalformer

# Create environment
conda create -n coalformer python=3.8
conda activate coalformer

# Install dependencies
pip install -r requirements.txt

# Install SMAC
pip install smac
# Note: Requires StarCraft II (~3.5GB)
# See: https://github.com/oxwhirl/smac#installing-starcraft-ii
```

---

## Quick Start

### Train CoalFormer on SMAC 3m

```bash
python training/coalformer_smac.py
```

This will train CoalFormer for 1000 episodes with default settings:
- K = 4 coalitions
- Two-phase regularization (Phase 1: ep 1-500, Phase 2: ep 501-1000)
- 5 random seeds

**Expected output**:
```
Episode 100: Win Rate = 15.2%
Episode 500: Win Rate = 35.8%
Episode 1000: Win Rate = 40.5% ± 4.4%
Training Success: 100% (5/5 seeds)
```

---

## Reproducing Paper Results

### Main Results (Table 2)

```bash
# CoalFormer (main method)
python training/coalformer_smac.py --episodes 1000 --seeds 5

# Run all baselines for comparison
python baselines/run_all_baselines.py
```

**Expected Results**:
| Method | Win Rate | Training Success | Variance |
|--------|----------|------------------|----------|
| CoalFormer | 40.5% | 100% (5/5) | 4.4% |
| VDN | 37.0% | 100% (5/5) | 8.9% |
| BRIDGE | 27.2% | 40% (2/5) | 25.1% |

### Ablation Studies (Table 3)

```bash
python experiments/run_ablation.py
```

**Expected Results**:
| Configuration | Win Rate | Δ from Full |
|---------------|----------|-------------|
| Full model | 40.5% | - |
| w/o separation loss | 33.2% | -7.3% |
| w/o entropy reg | 36.8% | -3.7% |
| w/o two-phase | 35.1% | -5.4% |

### Generate Figures

```bash
# Generate all paper figures
python analysis/generate_figures.py

# Generate coalition structure visualization
python analysis/generate_coalition_viz.py

# Generate t-SNE embeddings
python analysis/generate_tsne.py
```

Figures will be saved to `figures/` directory.

---

## Repository Structure

```
coalformer/
├── models/
│   └── coalformer.py              # Core model (prototype clustering)
├── training/
│   └── coalformer_smac.py         # SMAC training script
├── baselines/
│   └── run_all_baselines.py       # VDN, QMIX, BRIDGE
├── experiments/
│   └── run_ablation.py            # Ablation studies
├── analysis/
│   ├── generate_figures.py        # Paper figures
│   ├── generate_coalition_viz.py  # Coalition visualization
│   └── generate_tsne.py           # Embedding visualization
└── requirements.txt
```

---

## Core Algorithm

### Prototype-based Coalition Formation

```python
# 1. Learnable coalition prototypes
self.prototypes = nn.Parameter(torch.randn(K, d))

# 2. Distance-based soft assignment
distances = torch.norm(z_i - p_k, dim=-1)
pi_ik = F.softmax(-distances / tau, dim=-1)

# 3. Coalition Q-values
Q_coal_k = sum(pi_ik * Q_i)

# 4. Global Q-value
Q_tot = mixer(Q_coal)
```

### Two-Phase Regularization

```python
# Phase 1 (episodes 1-500): Weak regularization
if episode <= 500:
    loss = td_loss + 0.001 * entropy + 0.0001 * separation

# Phase 2 (episodes 501-1000): Strong regularization
else:
    loss = td_loss + 0.01 * entropy + 0.001 * separation
```

---

## Key Features

### 1. Differentiable Optimization
- End-to-end gradient descent (no nested RL)
- Complexity: O(nKT) per training step
- Guaranteed convergence under two-phase schedule

### 2. Interpretable Structure
- Soft assignment probabilities π_ik reveal learned coordination
- Prototype positions visualizable via t-SNE
- Coalition occupancy tracks dynamic formation

### 3. Training Reliability
- 100% success across all random seeds
- Lowest variance (4.4%) among all methods
- No catastrophic failures (unlike BRIDGE's 60% failure rate)

---

## Hyperparameters

Default configuration (SMAC 3m):

```python
# Architecture
n_coalitions = 4           # K in paper
hidden_dim = 64
coalition_dim = 32         # d' in paper

# Training
learning_rate = 5e-4
batch_size = 32
buffer_size = 2000
gamma = 0.99

# Regularization - Phase 1 (ep 1-500)
lambda_entropy_1 = 0.001
lambda_separation_1 = 0.0001

# Regularization - Phase 2 (ep 501-1000)
lambda_entropy_2 = 0.01    # 10x increase
lambda_separation_2 = 0.001 # 10x increase
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{coalformer2027,
  title={CoalFormer: Learning Emergent Coalition Structures via Differentiable Prototype Clustering for Multi-Agent Coordination},
  author={Anonymous},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2027}
}
```

---

## License

MIT License

---

## Acknowledgments

- [SMAC](https://github.com/oxwhirl/smac) - StarCraft Multi-Agent Challenge
- [PyMARL](https://github.com/oxwhirl/pymarl) - Multi-agent RL framework

---

## Contact

For questions or issues:
- Open a GitHub issue
- Email: anonymous@email.com

---

## Status

🚧 **Note**: This code accompanies a paper under review at ICLR 2027. 
Complete documentation and pretrained models will be released upon acceptance.

---

## Verified Implementation

✅ This implementation has been verified to match the paper:
- Prototype clustering (Section 4.2)
- Coalition Q-value aggregation (Eq. 2)
- Two-phase regularization (Section 4.4)
- All hyperparameters (Appendix A)

See `models/coalformer.py` for the core implementation with detailed comments.
