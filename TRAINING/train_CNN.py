import os
import time
import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import LABEL_COLS, CLASS_NAMES
from augmentation import get_train_transform, get_eval_transform
from dataset import FundusDataset

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
# MODEL
# Backbone : ConvNeXt-Base (pretrained ImageNet-1k)
# Features : 1024-d global average-pooled representation
# Head     : Linear(1024→512) → GELU → Dropout → Linear(512→C)
# Training : backbone uses lr_backbone (1e-5), head uses lr (1e-4)
# ============================================================

class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        backbone = convnext_base(weights=weights)

        # Keep feature extractor + adaptive pool; drop the original classifier
        self.backbone = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(1),
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CNNConfig:
    csv_path: str
    image_col: str  = "image"
    split_col: str  = "split"
    image_size: int = 224

    epochs:       int   = 50
    batch_size:   int   = 64
    lr:           float = 1e-4       # head learning rate
    lr_backbone:  float = 1e-5       # backbone learning rate (fine-tune gently)
    weight_decay: float = 1e-4
    dropout:      float = 0.3

    num_workers: int   = 4
    seed:        int   = 42
    out_dir:     str   = "./outputs"
    pretrained:  bool  = True
    threshold:   float = 0.5         # fallback; overridden by per-class opt


# ============================================================
# DATA
# ============================================================

def make_loaders(df: pd.DataFrame, cfg: CNNConfig):
    if cfg.split_col not in df.columns:
        raise ValueError(f"CSV must have a '{cfg.split_col}' column (train/val/test).")

    train_df = df[df[cfg.split_col] == "train"].reset_index(drop=True)
    val_df   = df[df[cfg.split_col] == "val"].reset_index(drop=True)
    test_df  = df[df[cfg.split_col] == "test"].reset_index(drop=True)

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("CSV must contain at least 'train' and 'test' splits.")

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
    """
    Compute BCEWithLogitsLoss pos_weight from training split.
    pos_weight[i] = (# negative samples) / (# positive samples) for class i.
    Clipped at 20 to prevent extreme weights for very rare classes.
    """
    train_df = df[df[split_col] == "train"]
    pos = train_df[LABEL_COLS].sum()
    neg = len(train_df) - pos
    weights = (neg / pos.clip(lower=1)).clip(upper=20)
    return torch.tensor(weights.values, dtype=torch.float32)


# ============================================================
# TRAIN / EVAL HELPERS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    epoch_loss = 0.0
    start = time.time()

    bar = tqdm(loader, desc=f"Epoch {epoch+1:03d}/{total_epochs}", leave=False)
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
    """
    Sweep thresholds [0.10 … 0.90] per class on the validation set
    and pick the one that maximises binary F1.
    Falls back to 0.50 for classes with no positive validation samples.
    """
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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> dict:
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
# PRINT / SAVE TEXT RESULTS
# ============================================================

def print_metrics(metrics: dict, thresholds: np.ndarray):
    header = f"{'Code':>3}  {'Name':<28}  {'Thr':>5}  {'P':>6}  {'R':>6}  {'F1':>6}  {'S':>5}"
    sep    = "-" * len(header)
    print("\nPer-class results")
    print(header)
    print(sep)
    for i, col in enumerate(LABEL_COLS):
        print(
            f"{col:>3}  {CLASS_NAMES.get(col, col):<28}  "
            f"{thresholds[i]:>5.2f}  "
            f"{metrics['precision_per_class'][i]:>6.4f}  "
            f"{metrics['recall_per_class'][i]:>6.4f}  "
            f"{metrics['f1_per_class'][i]:>6.4f}  "
            f"{int(metrics['support_per_class'][i]):>5}"
        )
    print(sep)
    print(f"\nMacro  P={metrics['macro_precision']:.4f}  "
          f"R={metrics['macro_recall']:.4f}  "
          f"F1={metrics['macro_f1']:.4f}")


def save_metrics(metrics: dict, thresholds: np.ndarray, path: str):
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
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Metrics saved  → {path}")


# ============================================================
# PLOTS
# ============================================================

def plot_training_curves(train_losses: list, val_f1s: list, out_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, color="steelblue", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1s, color="darkorange", linewidth=1.5)
    if val_f1s:
        best_ep = int(np.argmax(val_f1s))
        ax2.axvline(best_ep + 1, color="crimson", linestyle="--", linewidth=1.2,
                    label=f"Best ep {best_ep+1}  F1={val_f1s[best_ep]:.4f}")
        ax2.legend(fontsize=9)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Validation Macro F1")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {out_path}")


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    """
    4 × 5 grid of binary confusion matrices — one per disease class.
    Each cell shows the raw count; colour intensity = row-normalised rate.
    Layout:
        rows → True label (0 = Absent, 1 = Present)
        cols → Predicted label
    """
    N_COLS = 5
    N_ROWS = (len(LABEL_COLS) + N_COLS - 1) // N_COLS   # = 4 for 19 classes

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(N_COLS * 3.2, N_ROWS * 3.0),
        squeeze=False,
    )
    fig.suptitle("Per-Class Confusion Matrices — Test Set",
                 fontsize=13, fontweight="bold", y=1.01)

    tick_labels = ["Absent", "Present"]

    for idx, col in enumerate(LABEL_COLS):
        r, c = divmod(idx, N_COLS)
        ax   = axes[r][c]

        cm      = confusion_matrix(y_true[:, idx], y_pred[:, idx], labels=[0, 1])
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm.astype(float) / row_sum          # row-normalised for colour

        ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")

        # Annotate each cell with raw count + percentage
        for ri in range(2):
            for ci in range(2):
                pct   = cm_norm[ri, ci]
                color = "white" if pct > 0.55 else "black"
                ax.text(ci, ri,
                        f"{cm[ri, ci]}\n({pct*100:.0f}%)",
                        ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        name = CLASS_NAMES.get(col, col)
        ax.set_title(f"{col} — {name}", fontsize=8, fontweight="bold", pad=4)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"Pred\n{l}" for l in tick_labels], fontsize=7)
        ax.set_yticklabels([f"True\n{l}" for l in tick_labels], fontsize=7)

    # Hide unused subplot slots (19 classes → 1 empty slot in a 4×5 grid)
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

def main(cfg: CNNConfig):
    seed_everything(cfg.seed)
    ensure_dir(cfg.out_dir)

    # ---- Load & validate data ----
    df = pd.read_csv(cfg.csv_path)
    df["image"] = df["image"].str.replace("/data/", "DATASET/data/", regex=False)

    missing = [c for c in [cfg.image_col, cfg.split_col] + LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[df[LABEL_COLS].sum(axis=1) > 0].reset_index(drop=True)

    train_loader, val_loader, test_loader = make_loaders(df, cfg)
    n_val = len(val_loader.dataset) if val_loader else 0
    print(f"Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Split  : train={len(train_loader.dataset)}  val={n_val}  test={len(test_loader.dataset)}")

    # ---- Model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CNNClassifier(
        num_classes=len(LABEL_COLS),
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
    ).to(device)

    # Differential learning rates: backbone (fine-tune gently) vs head (train normally)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
        {"params": model.head.parameters(),     "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-7
    )

    # Class-weighted BCE (weights computed from training split)
    class_weights = compute_class_weights(df, cfg.split_col).to(device)
    criterion     = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # ---- Training loop ----
    best_val_f1  = -1.0
    best_path    = os.path.join(cfg.out_dir, "best_cnn.pt")
    train_losses = []
    val_f1s      = []

    print(f"\n{'='*55}")
    print(f"  CNN TRAINING  ({cfg.epochs} epochs)")
    print(f"  Backbone LR : {cfg.lr_backbone}   Head LR : {cfg.lr}")
    print(f"  Batch size  : {cfg.batch_size}   Dropout : {cfg.dropout}")
    print(f"{'='*55}\n")

    for epoch in range(cfg.epochs):
        train_loss, dt = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg.epochs
        )
        scheduler.step()
        train_losses.append(train_loss)

        if val_loader is not None:
            y_true_val, y_prob_val = run_inference(model, val_loader, device)
            val_metrics = compute_metrics(
                y_true_val, y_prob_val,
                np.full(len(LABEL_COLS), cfg.threshold)
            )
            val_f1 = val_metrics["macro_f1"]
            val_f1s.append(val_f1)

            marker = " ★" if val_f1 > best_val_f1 else ""
            print(
                f"[CNN] Epoch {epoch+1:03d}/{cfg.epochs}  "
                f"loss={train_loss:.6f}  val_f1={val_f1:.4f}  "
                f"time={dt:.1f}s{marker}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "val_f1": val_f1}, best_path)
        else:
            print(f"[CNN] Epoch {epoch+1:03d}/{cfg.epochs}  loss={train_loss:.6f}  time={dt:.1f}s")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, best_path)

    # ---- Load best checkpoint ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    best_epoch = ckpt.get("epoch", cfg.epochs)
    print(f"\nRestored best checkpoint  (epoch {best_epoch},  val F1={best_val_f1:.4f})")

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
    print(f"\n{'='*55}")
    print("  EVALUATION — TEST SET")
    print(f"{'='*55}")

    y_true_test, y_prob_test = run_inference(model, test_loader, device)
    metrics = compute_metrics(y_true_test, y_prob_test, thresholds)

    print_metrics(metrics, thresholds)
    save_metrics(metrics, thresholds, os.path.join(cfg.out_dir, "cnn_evaluation.txt"))

    # ---- Confusion matrices ----
    plot_confusion_matrices(
        y_true_test, metrics["y_pred"],
        os.path.join(cfg.out_dir, "confusion_matrices.png")
    )

    print(f"\nAll outputs written to: {cfg.out_dir}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    cfg = CNNConfig(
        csv_path      = "DATASET/filtered_data_split.csv",
        image_col     = "image",
        split_col     = "split",
        image_size    = 224,
        epochs        = 50,
        batch_size    = 32,
        lr            = 1e-4,
        lr_backbone   = 1e-5,
        weight_decay  = 1e-4,
        dropout       = 0.3,
        num_workers   = 4,
        seed          = 42,
        out_dir       = "EXPERIMENTS/CNN_V1/",
        pretrained    = True,
    )
    main(cfg)
