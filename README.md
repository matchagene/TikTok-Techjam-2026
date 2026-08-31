# Robust AI-Generated Image Detection Under Real-World Transformations

**TikTok TechJam 2026 — Track 5**

This repository implements a robustness-first detector for distinguishing **real images** from **fully AI-generated images** after realistic post-processing such as JPEG compression, blur, resizing, noise, colour adjustment, cropping, and compound transformation pipelines.

The central research question is:

> **Can a strong AI-image detector be made substantially more robust by explicitly enforcing consistency between clean images and transformed versions of the same images?**

Rather than optimizing only clean-image accuracy, the project evaluates whether the detector preserves its ranking performance after the kinds of transformations that occur during social-media upload, redistribution, editing, and recompression.

---

## 1. Project Overview

A conventional detector learns:

```text
image → visual encoder → classifier → P(AI-generated)
```

Our robust pipeline additionally teaches the model that a benign transformation should **not change the provenance of an image**:

```text
                         training image x
                         label y ∈ {0, 1}
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
              clean x                  corrupted T(x)
                  │                           │
                  └─────────────┬─────────────┘
                                │
                       shared DINOv2 encoder
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
             clean feature               corrupt feature
                z(x)                        z(Tx)
                  │                           │
                  │<---- feature MSE -------->│
                  │                           │
                  ▼                           ▼
             classifier                  classifier
                  │                           │
                  ▼                           ▼
             P(AI | x)                  P(AI | T(x))
                  │                           │
                  │<-- prediction consistency│
                  │                           │
                  └──────── classification ───┘
```

The final training objective for the pairwise models is:

\[
L = L_{cls} + \lambda_{pred}L_{pred} + \lambda_{repr}L_{repr}
\]

where:

- `L_cls` is binary classification loss on both clean and corrupted views;
- `L_pred` is symmetric KL divergence between clean and corrupted prediction distributions;
- `L_repr` is MSE between their pre-dropout learned representations.

For M3 and M4 we start with:

```text
lambda_pred = 0.50
lambda_repr = 0.25
```

These weights are inspired by the pairwise consistency formulation reported by **TeleAI-TeleGuard** in the NTIRE 2026 robust AIGC-detection challenge. Our binary symmetric-KL implementation, DINOv2 backbone, corruption sampler, and progressive schedule are compute-efficient adaptations for this hackathon rather than an exact reproduction of their system.

---

## 2. Research Design: M0 → M4

The project deliberately uses a **controlled model ladder** instead of comparing one baseline against one opaque final model.

| Model | Description | What the comparison tests |
|---|---|---|
| **M0 — Original Baseline** | Historical DINOv2 baseline from the original repository | Historical reference only |
| **M1 — Corrected Baseline** | Same DINOv2-style detector under a corrected binary data protocol | Effect of fixing the experimental protocol |
| **M2 — Augmented** | Clean/corrupt paired training using classification loss only | **M1 → M2:** effect of robustness augmentation |
| **M3 — Pairwise** | M2 + prediction consistency + representation consistency | **M2 → M3:** effect of explicit consistency learning |
| **M4 — Curriculum** | M3 + progressive mild/medium/severe corruption schedule | **M3 → M4:** effect of difficulty progression |

The controlled ablation is critical:

```text
M1 → M2 : robust augmentation changes
M2 → M3 : only consistency losses change
M3 → M4 : only corruption-difficulty schedule changes
```

Everything else should remain as comparable as practical.

### Important note about M0

M0 is retained for historical reference, but it should **not** be treated as the scientifically controlled baseline:

1. the legacy pipeline grouped SID label `2` (tampered) together with fully synthetic images;
2. it used `ImageFolder` folders named `fake/` and `real/`, which alphabetically mapped `fake=0`, `real=1`;
3. therefore its sigmoid output represents `P(real)`, not `P(fake)`;
4. the old download path re-saved samples as JPEG before robustness evaluation;
5. the original subset was based on the first streamed examples rather than a deterministic balanced manifest.

The evaluation adapter therefore handles M0 specially:

```text
M0:     p_fake = 1 - sigmoid(logit)
M1–M4:  p_fake = sigmoid(logit)
```

---

## 3. Backbone and Model Architecture

The core experiments currently use:

```text
DINOv2 ViT-Base
vit_base_patch14_dinov2.lvd142m
```

with:

```text
DINOv2 backbone
      ↓
backbone feature
      ↓
Linear → 256
      ↓
GELU
      ↓
256-d robust representation z
      ↓
Dropout(0.3)
      ↓
Linear → 1
      ↓
fake logit
```

The initial M1–M4 experiments freeze the DINOv2 backbone and train the projection/classification head. The robust model is written so later experiments can optionally unfreeze the last transformer blocks without changing the core interface.

### Why the consistency representation is pre-dropout

Representation consistency is computed on `z` **before dropout**. Otherwise two identical images could receive different random dropout masks and appear artificially inconsistent.

Prediction consistency is also computed from **deterministic logits derived from the pre-dropout representation**, while ordinary BCE classification retains dropout regularization. This prevents the KL term from learning dropout noise rather than transformation sensitivity.

---

## 4. Data Protocol

### Dataset

The primary in-domain dataset is:

- [`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set)

Official SID labels are interpreted as:

| SID label | Meaning | Primary binary task |
|---:|---|---:|
| `0` | Real | `0` |
| `1` | Full synthetic | `1` |
| `2` | Tampered | **Excluded** |

The global convention for all new code is therefore:

```text
0 = REAL
1 = AI-GENERATED / FAKE
```

Tampered/localized manipulations are deliberately excluded because the primary Track 5 experiment is a binary **real vs fully generated** task.

### Deterministic balanced manifests

The project does not use directory names as the source of truth for labels and does not simply take the first `N` streaming examples.

`data/build_sid_manifests.py`:

1. loads the official SID train/validation streams;
2. shuffles them deterministically;
3. excludes label `2`;
4. samples balanced real/fake class quotas;
5. uses official train data for training;
6. deterministically divides the official validation pool into development-validation and held-out internal-test subsets;
7. caches decoded images as **PNG** to avoid introducing an uncontrolled JPEG transformation;
8. saves the selected image IDs and labels in manifests.

Default sample sizes are configurable and currently set to:

```text
Training:    1,250 real + 1,250 fake
Validation:    250 real +   250 fake
Test:          250 real +   250 fake
Seed: 42
```

These defaults are hackathon-scale settings, not dataset limits.

### Manifest schema

```text
image_id
cached_path
source_split
sid_label
binary_label
width
height
```

Before training, `data/validate_manifests.py` checks:

- no duplicate image IDs within a split;
- no train/validation/test overlap;
- binary labels are only `{0, 1}`;
- SID label `2` is absent;
- both classes are present and balanced;
- cached files exist and are readable.

---

## 5. Canonical Preprocessing

All models share one preprocessing definition from `data/preprocessing.py`:

```text
PIL RGB image
     ↓
Resize to 518 × 518
     ↓
ToTensor
     ↓
ImageNet normalization
```

Normalization:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Robustness corruptions are always applied **before** model preprocessing:

```text
original PIL image
      ↓
corruption
      ↓
518 × 518 preprocessing
      ↓
normalized tensor
      ↓
DINOv2
```

JPEG compression is never applied after tensor normalization.

---

## 6. Robustness Transformation System

The project intentionally separates **evaluation transformations** from **training corruptions**.

### A. Fixed deterministic benchmark

Evaluation uses frozen, reproducible conditions shared by M0–M4.

| Transformation | Benchmark severities |
|---|---|
| JPEG compression | `90, 70, 50, 30` |
| Gaussian blur | `σ = 0.5, 1.0, 2.0` |
| Resize/downsample | `0.5×, 0.25×`, then upsample |
| Gaussian noise | `σ = 0.02, 0.05, 0.10` |
| Center crop | `0.8` |
| Color jitter | deterministic benchmark seed |

Fixed compound policies include:

```text
mild:
resize 0.5 → JPEG 90

medium:
center crop 0.8 → resize 0.5 → JPEG 70

severe:
center crop 0.8 → resize 0.25 → Gaussian blur 1.0 → JPEG 30
```

These benchmark policies must remain fixed across experiments.

### B. Stochastic training sampler

Training uses a separate sampler:

```python
corrupted_image, trace = sample_corruption(
    image,
    difficulty="medium",
    seed=seed,
)
```

A trace records exactly what was applied, for example:

```python
{
    "difficulty": "medium",
    "seed": 90125,
    "operations": [
        ("gaussian_noise", 0.05),
        ("resize", 0.5),
        ("jpeg", 70),
    ],
}
```

The training sampler is intentionally stochastic so the detector does not merely memorize the exact evaluation pipelines.

### Training difficulty levels

**Mild**

```text
1 transformation
```

Examples: JPEG 90, blur 0.5, resize 0.5, noise 0.02, crop 0.8, color jitter.

**Medium**

```text
1–3 transformations
```

Examples: JPEG 70/50, blur 1.0, resize 0.5, noise 0.05, crop 0.8, color jitter.

**Severe**

```text
2–4 transformations
```

Examples: JPEG 50/30, blur 1.0/2.0, resize 0.25, noise 0.10, crop 0.8, color jitter.

This is a compute-efficient adaptation of the multi-level corruption strategies used by leading NTIRE 2026 teams; the exact operation counts are our own budget-conscious design.

---

## 7. M2/M3 Fixed Corruption Distribution

M2 and M3 deliberately see the **same corruption distribution**:

```text
40% mild
40% medium
20% severe
```

This ensures that M2 → M3 isolates the consistency objective rather than accidentally changing both the loss and the training data.

---

## 8. M4 Progressive Difficulty Curriculum

M4 keeps the same model and pairwise objective as M3 but changes the corruption-difficulty distribution over training progress.

| Stage | Mild | Medium | Severe |
|---|---:|---:|---:|
| Early (`0–33%`) | 80% | 20% | 0% |
| Middle (`33–66%`) | 35% | 50% | 15% |
| Late (`66–100%`) | 20% | 45% | 35% |

The schedule is our progressive adaptation inspired by:

- MICV's hierarchical/difficulty-structured augmentation strategy;
- Ant International's explicit multi-level corruption design.

We do **not** claim these exact epoch-stage percentages were used by either winning team.

---

## 9. Pairwise Training Objective

Each training sample produces:

```text
clean image x
corrupted image T(x)
same binary label y
```

Both views are concatenated into one forward batch for efficiency.

### Classification loss

\[
L_{cls} = \frac{BCE(l_x,y) + BCE(l_{T(x)},y)}{2}
\]

M2 uses only this term.

### Prediction consistency

For binary logits, the code constructs Bernoulli distributions:

```text
P(real) = 1 - sigmoid(logit)
P(fake) = sigmoid(logit)
```

and uses our documented **symmetric binary KL** adaptation:

\[
L_{pred} = \frac{1}{2}
\left[
KL(p_c \Vert p_d) + KL(p_d \Vert p_c)
\right]
\]

Probabilities are clamped for numerical stability.

### Representation consistency

\[
L_{repr} = MSE(z_{clean}, z_{corrupt})
\]

where both representations are taken before dropout.

### Full objective

M2:

```text
L = L_cls
```

M3/M4:

```text
L = L_cls + 0.50 L_pred + 0.25 L_repr
```

---

## 10. Evaluation Protocol

**ROC-AUC is the primary metric.**

Accuracy, precision, recall, and F1 are also reported, but AUC is preferred because it evaluates ranking quality without depending on one fixed probability threshold.

### Primary robustness metrics

For each model we compute:

1. **Clean AUC** — AUC on uncorrupted images.
2. **Pooled Robust AUC** — AUC after pooling predictions across all transformed benchmark conditions.
3. **Mean Condition AUC** — arithmetic mean of AUC computed separately per corruption condition.
4. **Worst-Case AUC** — minimum AUC across conditions.
5. **Robustness Drop** — `Clean AUC - Pooled Robust AUC`; lower is better.

The primary ablation table is designed as:

| Model | Clean AUC | Robust Pooled AUC | Mean Condition AUC | Worst AUC | Robustness Drop |
|---|---:|---:|---:|---:|---:|
| M0 Original | TBD | TBD | TBD | TBD | TBD |
| M1 Corrected | TBD | TBD | TBD | TBD | TBD |
| M2 Augmented | TBD | TBD | TBD | TBD | TBD |
| M3 Pairwise | TBD | TBD | TBD | TBD | TBD |
| M4 Curriculum | TBD | TBD | TBD | TBD | TBD |

**Do not replace `TBD` values with unverified numbers.** The controlled M1–M4 experiment should be run before publishing final claims.

### Raw predictions first

Every evaluation run saves raw predictions rather than only aggregate scores:

```text
image_id
label
logit
p_fake
model_id
dataset
corruption
severity
seed
```

This allows metrics and figures to be recomputed without rerunning expensive inference.

---

## 11. Repository Structure

```text
.
├── augmentations/
│   ├── transforms.py          # deterministic primitive transformations
│   ├── composition.py         # deterministic transform composition
│   ├── policies.py            # fixed benchmark policies
│   ├── sampler.py             # stochastic training corruption sampler
│   └── curriculum.py          # progressive difficulty schedule
│
├── configs/
│   ├── corrected_baseline.yaml
│   ├── augmentation_train.yaml
│   ├── curriculum.yaml
│   ├── M2_augmented.yaml
│   ├── M3_pairwise.yaml
│   └── M4_curriculum.yaml
│
├── data/
│   ├── build_sid_manifests.py
│   ├── validate_manifests.py
│   ├── sid_dataset.py
│   ├── preprocessing.py
│   ├── cache/                 # local image cache; not committed
│   └── manifests/
│
├── evaluation/
│   ├── conditions.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── model_adapter.py
│   ├── model_loading.py
│   ├── robustness.py
│   ├── evaluate_clean.py
│   └── evaluate_robustness.py
│
├── losses/
│   └── consistency.py
│
├── models/
│   ├── baseline.py
│   └── robust_detector.py
│
├── training/
│   ├── train_baseline.py              # legacy M0 training
│   ├── train_corrected_baseline.py    # M1
│   ├── paired_dataset.py
│   ├── pairwise_engine.py
│   ├── run_robust_experiment.py
│   ├── train_augmented.py             # M2
│   ├── train_pairwise.py              # M3
│   └── train_curriculum.py            # M4
│
├── tests/
├── checkpoints/               # checkpoints are not stored in normal Git
├── results/
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 12. Installation

### 12.1 Clone

```bash
git clone https://github.com/matchagene/TikTok-Techjam-2026.git
cd TikTok-Techjam-2026
```

### 12.2 Create an environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 12.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Core packages include:

```text
torch
torchvision
timm
scikit-learn
pandas
datasets
PyYAML
Pillow
tqdm
```

The first DINOv2 model construction may require network access to obtain pretrained weights if they are not already cached locally.

---

## 13. Quick Start: Reproduce the Core Experiment

Run commands from the repository root.

### Step 1 — Build deterministic SID manifests

```bash
python data/build_sid_manifests.py
```

To change the hackathon-scale subset:

```bash
python data/build_sid_manifests.py \
  --train-per-class 2500 \
  --validation-per-class 500 \
  --test-per-class 500 \
  --seed 42
```

Expected outputs:

```text
data/manifests/sid_train.csv
data/manifests/sid_val.csv
data/manifests/sid_test.csv
data/manifests/sid_manifest_build.meta.json
```

and lossless cached images under:

```text
data/cache/sid/
```

### Step 2 — Validate data before training

```bash
python data/validate_manifests.py
```

This should report balanced counts and fail loudly if there is leakage, an invalid label, or a missing/unreadable cache file.

### Step 3 — Train M1 corrected baseline

```bash
python training/train_corrected_baseline.py
```

or explicitly:

```bash
python training/train_corrected_baseline.py \
  --config configs/corrected_baseline.yaml
```

M1 selects the best checkpoint using **clean validation ROC-AUC**, not validation accuracy.

Expected outputs include:

```text
checkpoints/M1_corrected_baseline.pth
checkpoints/M1_corrected_baseline.meta.json
results/baseline/M1_clean_metrics.csv
results/training/M1_history.csv
```

### Step 4 — Train M2 augmentation-only model

M2 initializes compatible head weights from M1 by default, so the M1 checkpoint must exist.

```bash
python training/train_augmented.py
```

Expected:

```text
checkpoints/M2_augmented.pth
checkpoints/M2_augmented.meta.json
results/training/M2_history.csv
```

### Step 5 — Train M3 pairwise-consistency model

```bash
python training/train_pairwise.py
```

Expected:

```text
checkpoints/M3_pairwise.pth
checkpoints/M3_pairwise.meta.json
results/training/M3_history.csv
```

### Step 6 — Train M4 progressive-curriculum model

```bash
python training/train_curriculum.py
```

Expected:

```text
checkpoints/M4_curriculum.pth
checkpoints/M4_curriculum.meta.json
results/training/M4_history.csv
```

---

## 14. Evaluate a Model

### Clean evaluation

Example for M3:

```bash
python evaluation/evaluate_clean.py \
  --model-id M3 \
  --checkpoint checkpoints/M3_pairwise.pth \
  --manifest data/manifests/sid_test.csv
```

### Full robustness benchmark

```bash
python evaluation/evaluate_robustness.py \
  --model-id M3 \
  --checkpoint checkpoints/M3_pairwise.pth \
  --manifest data/manifests/sid_test.csv
```

Repeat the same benchmark for M0–M4 so every model is evaluated under identical deterministic corruption conditions.

Typical outputs:

```text
results/predictions/<MODEL_ID>/...
results/evaluation/<MODEL_ID>_..._by_condition.csv
results/evaluation/<MODEL_ID>_..._summary.csv
```

---

## 15. Run the Test Suite

```bash
pytest -q
```

The tests cover key research invariants including:

- SID label mapping and tampered exclusion;
- deterministic data split construction;
- split-overlap detection;
- canonical preprocessing;
- deterministic corruption by seed;
- mild/medium/severe operation counts;
- fixed benchmark policies;
- curriculum distributions;
- robust model output shapes;
- zero-consistency behavior for identical representations/predictions;
- gradient flow with a frozen backbone;
- M2/M3/M4 configuration isolation;
- M0 probability inversion;
- robust metric aggregation;
- single-forward evaluation behavior.

Run the test suite after every integration merge.

---

## 16. Checkpoint and Reproducibility Policy

Large `.pth` files should not be committed to standard GitHub history.

Checkpoint names are standardized:

```text
M0_original_baseline.pth
M1_corrected_baseline.pth
M2_augmented.pth
M3_pairwise.pth
M4_curriculum.pth
```

Each new trained checkpoint should have accompanying metadata containing, where available:

- model ID;
- Git commit;
- creation timestamp;
- training/validation/test manifests;
- random seed;
- epochs and batch size;
- optimizer and learning rates;
- consistency weights;
- augmentation configuration;
- label convention;
- checkpoint SHA-256 hash;
- validation metrics.

This makes experiment results traceable even when the model weights live in shared external storage.

---

## 17. Research Inspiration

The project is primarily inspired by the robustness principles demonstrated in:

### NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild

- Challenge paper: [arXiv:2604.11487](https://arxiv.org/abs/2604.11487)
- Key lessons adopted:
  - robust evaluation under compounded image distortions;
  - strong foundation-model visual encoders;
  - explicit corruption difficulty structure;
  - robustness as a primary metric rather than a clean-only afterthought.

### MICV — 1st-place NTIRE system

We draw inspiration from its **hierarchical/difficulty-structured augmentation** principle. Our exact progressive schedule is our own hackathon-scale adaptation.

### Ant International — 2nd-place NTIRE system

We draw inspiration from its explicit **clean / mild / moderate / heavy corruption levels**. We do not reproduce its large-model specialist ensemble.

### TeleAI-TeleGuard — 3rd-place NTIRE system

- Pairwise-training paper: [arXiv:2604.12307](https://arxiv.org/abs/2604.12307)
- Most directly relevant ideas:
  - jointly train clean and distorted counterparts;
  - classification loss;
  - prediction-distribution consistency;
  - representation consistency;
  - reported starting weights `α=0.5`, `β=0.25`.

Our system should therefore be described as:

> **A compute-efficient adaptation of robustness principles demonstrated by leading NTIRE 2026 systems.**

It should **not** be described as an exact reproduction of any winning method.

---

## 18. Current Scope and Planned Extensions

### Core implemented scope

- [x] historical M0 preservation;
- [x] corrected SID binary data protocol;
- [x] deterministic balanced manifests;
- [x] lossless PNG cache;
- [x] canonical DINOv2 preprocessing;
- [x] deterministic fixed robustness benchmark;
- [x] stochastic mild/medium/severe training corruption sampler;
- [x] progressive difficulty scheduler;
- [x] M1 corrected baseline trainer;
- [x] M2 augmentation-only trainer;
- [x] M3 prediction + representation consistency trainer;
- [x] M4 curriculum trainer;
- [x] clean and robustness evaluators;
- [x] raw-prediction logging;
- [x] controlled robustness metrics;
- [x] automated tests for core contracts.

### Before final submission

- [ ] run and populate the full M0–M4 controlled result table;
- [ ] perform representative false-positive / false-negative analysis;
- [ ] produce final robustness figures;
- [ ] implement the required directory-to-JSON inference entry point (`image_path`, `pred`);
- [ ] finalize external checkpoint hosting/reproduction instructions;
- [ ] update team-member contribution names;
- [ ] polish the end-to-end demo and Devpost submission.

### Optional / stretch experiments

These should only be attempted after the controlled M0–M4 experiment is complete:

- cross-dataset external generator benchmark such as GenImage;
- Transformation Stability Score;
- corrupted-feature correction MLP;
- partial unfreezing of final DINO blocks;
- consistency-weight ablations;
- DINOv3 backbone comparison;
- lightweight specialist/TTA ensemble.

The team intentionally prioritizes a defensible controlled experiment over architectural complexity for its own sake.

---

## 19. Team Workstreams

The project is designed for parallel development with stable interfaces.

| Workstream | Responsibility |
|---|---|
| **Data + corrected baseline** | trustworthy SID protocol, manifests, canonical preprocessing, M1 |
| **Robust transformations** | deterministic benchmark, stochastic training sampler, difficulty system |
| **Pairwise robust learning** | robust model, consistency losses, M2/M3/M4 training |
| **Evaluation** | probability semantics, clean/robust metrics, raw predictions, result evidence |

Replace the workstream descriptions with member names before final Devpost submission if required.

---

## 20. Scientific Claiming Rules

To keep the final presentation defensible:

- do not claim M0 and M1 are directly equivalent protocols;
- do not report the legacy M0 validation accuracy as the primary project result;
- do not tune repeatedly against the held-out internal test set;
- do not train on the exact deterministic benchmark pipelines used for evaluation;
- do not call an external generator strictly "unseen" unless its absence from the SID training data is verified;
- do not call transformation stability a calibrated uncertainty probability without formal calibration;
- do not claim our exact progressive curriculum was used by MICV or Ant;
- do not claim we reproduced an NTIRE winner end-to-end;
- do not add a technique to the final model solely because it sounds sophisticated—keep it only if the controlled experiment supports it.

---

## 21. Final Goal

The project is successful if it can demonstrate, with controlled evidence, that:

```text
M1 → M2
robust augmentation improves transformed-image performance

M2 → M3
explicit clean/corrupt consistency adds robustness beyond augmentation alone

M3 → M4
progressive difficulty exposure provides additional robustness or better worst-case behavior
```

The intended final story is:

> **Clean benchmark performance alone does not establish that an AI-image detector will remain reliable after social-media processing. We establish a controlled DINOv2 baseline, explicitly train transformation invariance using paired clean/corrupted images, and evaluate whether prediction- and representation-level consistency preserve AI-detection performance across realistic redistribution transformations.**

---

## Acknowledgements

This project was developed for **TikTok TechJam 2026 Track 5: Robust Detection of AI-Generated Images Under Real-World Transformations** and builds on publicly available research and datasets cited above.
