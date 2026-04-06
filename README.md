# Fundus Disease Classification

A deep-learning pipeline for robust multi-label classification of fundus images across 19 different retinal conditions, including Diabetic Retinopathy (DR), Age-Related Macular Degeneration (AMD), Glaucoma, Pathological Myopia, and several rare ocular diseases.

The repository offers three specialized training paradigms—a standard end-to-end CNN fine-tuning approach, a multi-stage Supervised Contrastive Learning (SupCon) pipeline, and a Swin Transformer—specifically engineered to handle extreme multi-label class imbalances typically found in medical datasets.

## Key Features

- **Automated Preprocessing (`preprocessing.py`)**:
  - Iterative black-border removal.
  - Circular Hough Transform for accurate fundus disc detection and localized cropping.
  - Contrast Limited Adaptive Histogram Equalization (CLAHE) for illumination normalization.
- **Differentiated Training Pipelines**:
  1. **Supervised Contrastive pre-training (`train.py`)**: A 3-stage pipeline using a ResNeXt50 backbone. Learns an embedding space where similar disease combinations cluster closely together via `MultiLabelSupConLoss`, followed by training a linear classifier and a gentle end-to-end fine-tuning phase.
  2. **Direct CNN Fine-tuning (`train_CNN.py`)**: Leverages a ConvNeXt-Base backbone. Features differential learning rates (low LR for backbone, higher for head), data augmentations, and a class-weighted `BCEWithLogitsLoss`.
  3. **Swin Transformer (`train_swin.py`)**: Fine-tunes a pretrained Swin-Large backbone (1536-d features) with a linear warmup → cosine annealing schedule, AdamW with weight decay, and gradient clipping. Uses the same preprocessing pipeline and per-class threshold optimisation as the other pipelines, with added ImageNet normalization required by the pretrained weights.
- **Per-Class Threshold Optimization**: Instead of applying a uniform 0.5 probability threshold, the pipeline automatically sweeps and identifies the optimal threshold for each specific disease on the validation set, maximizing the Macro F1 and pushing recall for underrepresented pathologies.
- **Comprehensive Evaluation**: Automated metric computation covering Precision, Recall, and F1 at both macro and per-class levels, complemented by multi-class normalized confusion matrix visualizations.

## Results

| Model | Architecture | Macro Precision | Macro Recall | Macro F1 |
|:------|:-------------|:---------------:|:------------:|:--------:|
| V1    | ResNeXt50 + SupCon (3-stage) | 0.7495 | 0.6807 | **0.7057** |
| CNN_V1 | ConvNeXt-Base | 0.6274 | 0.7408 | 0.6699 |
| SWIN_V1 | Swin-Large | — | — | — |

All models evaluated on the same held-out test set (n=1,262) across 19 disease classes.

## Code Structure

```
├── TRAINING/
│   ├── config.py         # Shared label columns and class name mappings
│   ├── dataset.py        # Custom Dataset class to handle multi-hot labels and images
│   ├── augmentation.py   # Geometric augmentations: scale-pad-crop, flips, rotation, grayscale
│   ├── preprocessing.py  # OpenCV scripts for fundus isolation and CLAHE
│   ├── train.py          # ResNeXt50 multi-label Supervised Contrastive pipeline (3-stage)
│   ├── train_CNN.py      # ConvNeXt-Base BCE fine-tuning pipeline
│   └── train_swin.py     # Swin-Large Transformer fine-tuning pipeline
├── EXPERIMENTS/
│   ├── V1/               # ResNeXt50 SupCon checkpoints and results
│   ├── CNN_V1/           # ConvNeXt-Base checkpoints, curves, and confusion matrices
│   └── SWIN_V1/          # Swin-Large checkpoints, curves, and confusion matrices
├── INFERENCE/            # Inference scripts and prediction outputs
└── DATASET/              # Source images and merged CSV data
```

## Supported Pathology Classes (19)

| Top-Level Classes              |                        |                         |
| :----------------------------- | :--------------------- | :---------------------- |
| Normal                         | Pathological Myopia    | Optic Disc Edema        |
| Age-Related Macular Deg. (AMD) | Tessellated Fundus     | Myelinated Nerve Fibers |
| Diabetic Retinopathy (DR)      | Vitreous Degeneration  | Retinal Detachment      |
| Glaucoma                       | BRVO & CRVO            | Refractive Media Opacity|
| Hypertensive Retinopathy       | Large Optic Cup        | CSC & Laser Spots       |
| Drusen                         | Epiretinal Membrane    |                         |

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd FUNDUS
   ```

2. **Setup virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install torch torchvision opencv-python pandas scikit-learn tqdm timm matplotlib
   ```

3. **Data preparation:**
   Expects a master CSV (`DATASET/filtered_data_split.csv`) defining splits (`train`, `val`, `test`), image paths, and multi-hot labels for the 19 classes.

## Usage

**Run CNN Pipeline (ConvNeXt-Base):**
```bash
python TRAINING/train_CNN.py
```

**Run Contrastive Pipeline (ResNeXt50 3-Stage):**
```bash
python TRAINING/train.py
```

**Run Swin Transformer Pipeline (Swin-Large):**
```bash
python TRAINING/train_swin.py
```
