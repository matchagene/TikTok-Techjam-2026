# TikTok-Techjam-2026 🚀

Deep learning pipeline built for the **TikTok Techjam 2026** competition, focusing on robust real vs. fake image classification using advanced Vision Transformers.

---

## 📊 Performance Benchmarks

| Model Configuration | Training Acc | Validation Acc | Notes |
| :--- | :---: | :---: | :--- |
| **Baseline A: DINOv2 (Clean)** | **99.44%** | **85.20%** | Converged smoothly in 5 epochs on cloud GPU. |
| *Baseline B: Augmentations* | *TBD* | *TBD* | *To be implemented by Person 2* |

---

## 🛠️ System & Architecture Overview

*   **Backbone:** `vit_base_patch14_dinov2.lvd142m` via `timm`.
*   **Input Resolution:** Fixed at $518 \times 518$ pixels to perfectly align with DINOv2's positional embeddings.
*   **Hardware Acceleration:** Configured for seamless execution on `cuda` (NVIDIA T4 / Cloud GPUs) with automated fallback to `cpu`.

---

## 📁 Repository Structure

```text
├── data/                  # Handled via streaming subset script (Ignored by Git)
├── models/
│   └── baseline.py        # DINOv2 model definition architecture
├── training/
│   └── train_baseline.py  # Core training and validation loops
├── .gitignore             # Formatted to ignore dataset caches and heavy *.pth weights
└── README.md
```

> ⚠️ **Note on Weights:** The trained model checkpoints (`models/baseline_best.pth`) exceed GitHub's 100MB file limit (~331MB) and are hosted externally on our team's shared drive. 

---

## ⚡ Quick Start Guide

### 1. Environment Setup
Clone the repository and install the required vision dependencies:
```bash
git clone https://github.com/matchagene/TikTok-Techjam-2026.git
cd TikTok-Techjam-2026
pip install timm tqdm torch torchvision
```

### 2. Dataset Initialization
To bypass local storage bottlenecks, fetch an optimized streaming subset of the `saberzl/SID_Set` dataset from Hugging Face directly into the `data/` directory:
```python
from datasets import load_dataset
# Set up streaming data pipes to populate data/train and data/val splits
```

### 3. Launch Training
Run the baseline script to initiate training and checkpoint tracking:
```bash
python -m training.train_baseline
```

---

## 👥 Team Handoff Notice
*   **Phase 1 (Person 1):** Baseline pipeline architecture verified, debugged, and frozen at **85.20% val accuracy**. 
*   **Phase 2 (Person 2):** Clear to pull `main` to integrate robust image augmentations and consistency loss frameworks.
