"""
train_resnet.py — ResNet-50 classifier for multi-label fundus disease classification.

Architecture : ResNet-50 (torchvision, ImageNet-1k pretrained)
                → Linear(2048 → 512) → GELU → Dropout(0.3) → Linear(512 → 19)
Preprocessing: preprocessing.py pipeline applied per image before augmentation:
                 remove_black_borders → crop_fundus (HoughCircles) → apply_clahe → resize
Training     :
  - Differential LR  : backbone (lr_backbone=1e-5)  vs  head (lr_head=1e-4)
  - AdamW + weight_decay=1e-4
  - CosineAnnealingLR
  - BCEWithLogitsLoss with pos_weight computed from training split (capped at 20)
  - Per-class threshold optimisation on val set [0.10 … 0.90]
Outputs      : EXPERIMENTS/RESNET_V1/
  - best_resnet.pt        (best val macro-F1 checkpoint)
  - resnet_evaluation.txt (per-class + macro metrics on test set)
  - training_curves.png
  - confusion_matrices.png
  - training_log.txt      (epoch-by-epoch log)
"""

import os
import sys
import time
import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm.auto import tqdm

# Shared project modules
sys.path.insert(0, os.path.dirname(__file__))
from config import LABEL_COLS, CLASS_NAMES
from dataset import FundusDataset
from augmentation import RandomScalePadCrop
from preprocessing import preprocess_fundus_image  # noqa: F401  (used inside FundusDataset)

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# UTILS
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# TRANSFORMS
# Preprocessing (border removal, fundus crop, CLAHE, resize) is
# handled by preprocessing.py inside FundusDataset before these
# transforms run.  RandomScalePadCrop is imported from augmentation.py
# so the pipeline is consistent with the rest of the project.
# ImageNet normalization is required by the ResNet-50 pretrained weights.
# ============================================================

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224):
    return transforms.Compose([
        RandomScalePadCrop(scale_range=(0.9, 1.1), size=image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ============================================================
# MODEL
# Backbone : ResNet-50 (pretrained ImageNet-1k)
# Features : 2048-d global average-pooled representation
# Head     : Linear(2048→512) → GELU → Dropout → Linear(512→C)
# ============================================================

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()          # strip the ImageNet head; outputs [B, 2048]
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # [B, 2048]
        return self.head(features)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ResNetConfig:
    csv_path:    str
    image_col:   str   = "image"
    split_col:   str   = "split"
    image_size:  int   = 224

    epochs:      int   = 50
    batch_size:  int   = 32
    lr_head:     float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    dropout:     float = 0.3

    num_workers: int   = 4
    seed:        int   = 42
    out_dir:     str   = "EXPERIMENTS/RESNET_V1/"
    pretrained:  bool  = True
    threshold:   float = 0.5          # fallback; overridden by per-class opt


# ============================================================
# DATA
# ============================================================

def make_loaders(df: pd.DataFrame, cfg: ResNetConfig):
    train_df = df[df[cfg.split_col] == "train"].reset_index(drop=True)
    val_df   = df[df[cfg.split_col] == "val"].reset_index(drop=True)
    test_df  = df[df[cfg.split_col] == "test"].reset_index(drop=True)

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("CSV must have at least 'train' and 'test' splits.")

    kw = dict(preprocess=True, image_size=cfg.image_size)
    train_ds = FundusDataset(train_df, cfg.image_col, LABEL_COLS,
                             transform=get_train_transform(cfg.image_size), **kw)
    val_ds   = FundusDataset(val_df,   cfg.image_col, LABEL_COLS,
                             transform=get_eval_transform(), **kw) if len(val_df) > 0 else None
    test_ds  = FundusDataset(test_df,  cfg.image_col, LABEL_COLS,
                             transform=get_eval_transform(), **kw)

    loader_kw = dict(num_workers=cfg.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **loader_kw) if val_ds else None
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader


def compute_class_weights(df: pd.DataFrame, split_col: str) -> torch.Tensor:
    """pos_weight[i] = neg_count / pos_count for class i, capped at 20."""
    train_df = df[df[split_col] == "train"]
    pos      = train_df[LABEL_COLS].sum()
    neg      = len(train_df) - pos
    weights  = (neg / pos.clip(lower=1)).clip(upper=20)
    return torch.tensor(weights.values, dtype=torch.float32)


# ============================================================
# TRAIN / EVAL HELPERS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, cfg: ResNetConfig):
    model.train()
    epoch_loss = 0.0
    start      = time.time()

    bar = tqdm(loader, desc=f"Epoch {epoch+1:03d}/{cfg.epochs}", leave=False)
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / max(1, len(loader)), time.time() - start


@torch.no_grad()
def run_inference(model, loader, device):
    """Returns (y_true [N, C], y_prob [N, C]) as numpy arrays."""
    model.eval()
    y_true_all, y_prob_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        probs  = torch.sigmoid(model(images))
        y_true_all.append(labels.cpu().numpy())
        y_prob_all.append(probs.cpu().numpy())
    return np.vstack(y_true_all), np.vstack(y_prob_all)


def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Sweep [0.10…0.90] per class, pick threshold maximising binary F1."""
    thresholds = np.full(y_true.shape[1], 0.5)
    for i in range(y_true.shape[1]):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.10, 0.91, 0.05):
            preds = (y_prob[:, i] >= t).astype(int)
            if preds.sum() == 0:
                continue
            _, _, f1, _ = precision_recall_fscore_support(
                y_true[:, i], preds, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    thresholds: np.ndarray) -> dict:
    y_pred = (y_prob >= thresholds).astype(np.int32)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return dict(
        precision_per_class=prec, recall_per_class=rec,
        f1_per_class=f1, support_per_class=sup,
        macro_precision=mp, macro_recall=mr, macro_f1=mf1,
        y_pred=y_pred,
    )


# ============================================================
# PRINT / SAVE RESULTS
# ============================================================

def _build_table(metrics: dict, thresholds: np.ndarray) -> list:
    header = f"{'Code':>3}  {'Name':<28}  {'Thr':>5}  {'P':>6}  {'R':>6}  {'F1':>6}  {'S':>5}"
    sep    = "-" * len(header)
    lines  = ["Per-class results", header, sep]
    for i, col in enumerate(LABEL_COLS):
        lines.append(
            f"{col:>3}  {CLASS_NAMES.get(col, col):<28}  "
            f"{thresholds[i]:>5.2f}  "
            f"{metrics['precision_per_class'][i]:>6.4f}  "
            f"{metrics['recall_per_class'][i]:>6.4f}  "
            f"{metrics['f1_per_class'][i]:>6.4f}  "
            f"{int(metrics['support_per_class'][i]):>5}"
        )
    lines += [
        sep,
        f"\nMacro  P={metrics['macro_precision']:.4f}  "
        f"R={metrics['macro_recall']:.4f}  "
        f"F1={metrics['macro_f1']:.4f}",
    ]
    return lines


def print_metrics(metrics: dict, thresholds: np.ndarray):
    print("\n".join(_build_table(metrics, thresholds)))


def save_metrics(metrics: dict, thresholds: np.ndarray, path: str):
    with open(path, "w") as f:
        f.write("\n".join(_build_table(metrics, thresholds)) + "\n")
    print(f"Metrics saved → {path}")


# ============================================================
# PLOTS
# ============================================================

def plot_training_curves(train_losses: list, val_f1s: list, out_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, color="steelblue", linewidth=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training Loss"); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1s, color="darkorange", linewidth=1.5)
    if val_f1s:
        best_ep = int(np.argmax(val_f1s))
        ax2.axvline(best_ep + 1, color="crimson", linestyle="--", linewidth=1.2,
                    label=f"Best ep {best_ep+1}  F1={val_f1s[best_ep]:.4f}")
        ax2.legend(fontsize=9)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1")
    ax2.set_title("Validation Macro F1"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {out_path}")


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    N_COLS = 5
    N_ROWS = (len(LABEL_COLS) + N_COLS - 1) // N_COLS

    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(N_COLS * 3.2, N_ROWS * 3.0), squeeze=False)
    fig.suptitle("Per-Class Confusion Matrices — Test Set",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, col in enumerate(LABEL_COLS):
        r, c = divmod(idx, N_COLS)
        ax   = axes[r][c]
        cm      = confusion_matrix(y_true[:, idx], y_pred[:, idx], labels=[0, 1])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
        for ri in range(2):
            for ci in range(2):
                pct   = cm_norm[ri, ci]
                color = "white" if pct > 0.55 else "black"
                ax.text(ci, ri, f"{cm[ri, ci]}\n({pct*100:.0f}%)",
                        ha="center", va="center", fontsize=9,
                        color=color, fontweight="bold")

        ax.set_title(f"{col} — {CLASS_NAMES.get(col, col)}", fontsize=8,
                     fontweight="bold", pad=4)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred\nAbsent", "Pred\nPresent"], fontsize=7)
        ax.set_yticklabels(["True\nAbsent", "True\nPresent"], fontsize=7)

    for idx in range(len(LABEL_COLS), N_ROWS * N_COLS):
        r, c = divmod(idx, N_COLS)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrices saved → {out_path}")


# ============================================================
# MAIN
# ============================================================

def main(cfg: ResNetConfig):
    seed_everything(cfg.seed)
    ensure_dir(cfg.out_dir)

    log_path = os.path.join(cfg.out_dir, "training_log.txt")
    log_file = open(log_path, "w", buffering=1)

    def log(msg: str):
        print(msg)
        log_file.write(msg + "\n")

    # ---- Load & validate data ----
    df = pd.read_csv(cfg.csv_path)
    df["image"] = df["image"].str.replace("/data/", "DATASET/data/", regex=False)

    missing = [c for c in [cfg.image_col, cfg.split_col] + LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[df[LABEL_COLS].sum(axis=1) > 0].reset_index(drop=True)

    train_loader, val_loader, test_loader = make_loaders(df, cfg)
    n_val = len(val_loader.dataset) if val_loader else 0

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    log(f"Device     : {gpu_name}")
    log(f"Split      : train={len(train_loader.dataset)}  val={n_val}  test={len(test_loader.dataset)}")

    # ---- Model ----
    model = ResNetClassifier(
        num_classes=len(LABEL_COLS),
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"Parameters : {n_params:.1f}M  (ResNet-50 backbone + head)")

    # Differential LR: backbone fine-tuned gently, head trained normally
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
        {"params": model.head.parameters(),     "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-7
    )

    class_weights = compute_class_weights(df, cfg.split_col).to(device)
    criterion     = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # ---- Training loop ----
    best_val_f1  = -1.0
    best_path    = os.path.join(cfg.out_dir, "best_resnet.pt")
    train_losses = []
    val_f1s      = []

    log(f"\n{'='*60}")
    log(f"  RESNET-50 TRAINING  ({cfg.epochs} epochs)")
    log(f"  Backbone LR : {cfg.lr_backbone}   Head LR : {cfg.lr_head}")
    log(f"  Batch size  : {cfg.batch_size}   Dropout : {cfg.dropout}")
    log(f"  Weight decay: {cfg.weight_decay}")
    log(f"{'='*60}\n")

    for epoch in range(cfg.epochs):
        train_loss, dt = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg
        )
        scheduler.step()
        train_losses.append(train_loss)

        bb_lr   = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]

        if val_loader is not None:
            y_true_val, y_prob_val = run_inference(model, val_loader, device)
            val_metrics = compute_metrics(
                y_true_val, y_prob_val,
                np.full(len(LABEL_COLS), cfg.threshold)
            )
            val_f1 = val_metrics["macro_f1"]
            val_f1s.append(val_f1)

            marker = " ★" if val_f1 > best_val_f1 else ""
            msg = (
                f"[ResNet] Epoch {epoch+1:03d}/{cfg.epochs}  "
                f"loss={train_loss:.6f}  val_f1={val_f1:.4f}  "
                f"bb_lr={bb_lr:.2e}  head_lr={head_lr:.2e}  "
                f"time={dt:.1f}s{marker}"
            )
            log(msg)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({"model": model.state_dict(),
                            "epoch": epoch + 1,
                            "val_f1": val_f1}, best_path)
        else:
            msg = (
                f"[ResNet] Epoch {epoch+1:03d}/{cfg.epochs}  "
                f"loss={train_loss:.6f}  "
                f"bb_lr={bb_lr:.2e}  head_lr={head_lr:.2e}  time={dt:.1f}s"
            )
            log(msg)
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, best_path)

    log_file.close()

    # ---- Restore best checkpoint ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    best_epoch = ckpt.get("epoch", cfg.epochs)
    print(f"\nRestored best checkpoint (epoch {best_epoch},  val F1={best_val_f1:.4f})")

    # ---- Training curves ----
    if val_f1s:
        plot_training_curves(
            train_losses, val_f1s,
            os.path.join(cfg.out_dir, "training_curves.png")
        )

    # ---- Per-class threshold optimisation on val set ----
    print("\n--- Threshold Optimisation (val set) ---")
    if val_loader is not None:
        y_true_val, y_prob_val = run_inference(model, val_loader, device)
        thresholds = optimize_thresholds(y_true_val, y_prob_val)
        print(f"{'Code':>3}  {'Name':<28}  {'Threshold':>9}")
        print("-" * 45)
        for i, col in enumerate(LABEL_COLS):
            print(f"{col:>3}  {CLASS_NAMES.get(col, col):<28}  {thresholds[i]:>9.2f}")
    else:
        thresholds = np.full(len(LABEL_COLS), cfg.threshold)
        print(f"No val split — using fixed threshold {cfg.threshold} for all classes.")

    # ---- Final evaluation on test set ----
    print(f"\n{'='*60}")
    print("  EVALUATION — TEST SET")
    print(f"{'='*60}")

    y_true_test, y_prob_test = run_inference(model, test_loader, device)
    metrics = compute_metrics(y_true_test, y_prob_test, thresholds)

    print_metrics(metrics, thresholds)
    save_metrics(metrics, thresholds, os.path.join(cfg.out_dir, "resnet_evaluation.txt"))

    plot_confusion_matrices(
        y_true_test, metrics["y_pred"],
        os.path.join(cfg.out_dir, "confusion_matrices.png")
    )

    print(f"\nAll outputs written to: {cfg.out_dir}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    cfg = ResNetConfig(
        csv_path     = "DATASET/filtered_data_split.csv",
        image_col    = "image",
        split_col    = "split",
        image_size   = 224,
        epochs       = 50,
        batch_size   = 32,
        lr_head      = 1e-4,
        lr_backbone  = 1e-5,
        weight_decay = 1e-4,
        dropout      = 0.3,
        num_workers  = 4,
        seed         = 42,
        out_dir      = "EXPERIMENTS/RESNET_V1/",
        pretrained   = True,
    )
    main(cfg)
