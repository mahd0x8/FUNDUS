# Fundus Disease Classification

A deep-learning pipeline for robust multi-label classification of fundus images across 19 different retinal conditions, including Diabetic Retinopathy (DR), Age-Related Macular Degeneration (AMD), Glaucoma, Pathological Myopia, and several rare ocular diseases.

The repository offers two specialized training paradigms—a standard end-to-end CNN fine-tuning approach and a multi-stage Supervised Contrastive Learning (SupCon) pipeline—specifically engineered to handle extreme multi-label class imbalances typically found in medical datasets.

## Key Features

- **Automated Preprocessing (`preprocessing.py`)**: 
  - Iterative black-border removal.
  - Circular Hough Transform for accurate fundus disc detection and localized cropping.
  - Contrast Limited Adaptive Histogram Equalization (CLAHE) for illumination normalization.
- **Differentiated Training Pipelines**:
  1. **Supervised Contrastive pre-training (`train.py`)**: A 3-stage pipeline using a ResNeXt50 backbone. Learns an embedding space where similar disease combinations cluster closely together via `MultiLabelSupConLoss`, followed by training a linear classifier and a gentle end-to-end fine-tuning phase.
  2. **Direct CNN Fine-tuning (`train_CNN.py`)**: Leverages a state-of-the-art ConvNeXt-Base backbone. Features differential learning rates (low LR for backbone, higher for head), aggressive data augmentations, and a class-weighted `BCEWithLogitsLoss`.
- **Per-Class Threshold Optimization**: Instead of applying a uniform 0.5 probability threshold, the pipeline automatically sweeps and identifies the optimal threshold for each specific disease on the validation set, maximizing the Macro F1 and pushing recall for underrepresented pathologies.
- **Comprehensive Evaluation**: Automated metric computation covering Precision, Recall, and F1 at both macro and per-class levels, complemented by multi-class normalized confusion matrix visualizations.

## Code Structure

```
├── TRAINING/
│   ├── dataset.py        # Custom Dataset class to handle multi-hot labels and images
│   ├── augmentation.py   # Geometric augmentations handling scale, aspect, and pad/crop
│   ├── preprocessing.py  # OpenCV scripts for fundus isolation and CLAHE
│   ├── train.py          # ResNeXt Multi-label Supervised Contrastive pipeline
│   └── train_CNN.py      # ConvNeXt-Base BCE fine-tuning pipeline
├── EXPERIMENTS/          # Checkpoints, validation curves, logic matrices, and result dumps
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
   pip install -r requirements.txt # (Assuming you have one, or install torch, torchvision, opencv-python, pandas, scikit-learn, etc.)
   ```

3. **Data preparation:**
   Expects a master CSV (`filtered_data_split.csv`) defining splits (`train`, `val`, `test`), image paths, and multihot labels for the aforementioned classes.

## Usage

**Run CNN Pipeline (ConvNeXt-Base):**
```bash
python TRAINING/train_CNN.py
```

**Run Contrastive Pipeline (ResNeXt50 3-Stage):**
```bash
python TRAINING/train.py
```
