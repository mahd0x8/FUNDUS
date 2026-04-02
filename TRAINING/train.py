import torch
import time, warnings
import random, os
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import pandas as pd
from torch.utils.data import DataLoader
from config import LABEL_COLS, CLASS_NAMES
from augmentation import get_train_transform, get_eval_transform
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from dataset import FundusDataset
from sklearn.metrics import precision_recall_fscore_support

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
# Encoder: ReSNeXt50_32x4d
# Projection head: dense + ReLU + linear(128)
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
    
class SupConEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        backbone = resnext50_32x4d(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048
        self.projection = ProjectionHead(in_dim=2048, hidden_dim=512, out_dim=128)

    def encode(self, x):
        feat = self.encoder(x).flatten(1)
        return feat

    def forward(self, x):
        feat = self.encode(x)
        proj = self.projection(feat)
        proj = F.normalize(proj, dim=1)
        return feat, proj
    
class LinearClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        return self.classifier(feat)
    
# ============================================================
# MULTI-LABEL SUPCON LOSS
# positives = samples sharing at least one label
# ============================================================

class MultiLabelSupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        features: [B, D], normalized
        labels:   [B, C], multi-hot
        """
        device = features.device
        batch_size = features.size(0)

        logits = torch.matmul(features, features.T) / self.temperature
        logits_mask = torch.ones_like(logits, device=device) - torch.eye(batch_size, device=device)

        positive_mask = (torch.matmul(labels, labels.T) > 0).float()
        positive_mask = positive_mask * logits_mask

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_counts = positive_mask.sum(dim=1)
        valid = pos_counts > 0

        mean_log_prob_pos = torch.zeros(batch_size, device=device)
        mean_log_prob_pos[valid] = (
            (positive_mask[valid] * log_prob[valid]).sum(dim=1) / pos_counts[valid]
        )

        loss = -mean_log_prob_pos[valid].mean() if valid.any() else torch.tensor(0.0, device=device)
        return loss


# ============================================================
# TRAIN / EVAL
# ============================================================

@dataclass
class Config:
    csv_path: str
    image_col: str = "image"
    split_col: str = "split"
    image_size: int = 224

    epochs_supcon: int = 500
    epochs_cls: int = 30
    epochs_finetune: int = 20

    batch_size_supcon: int = 128
    batch_size_cls: int = 128

    lr_supcon: float = 5e-4
    lr_finetune_encoder: float = 1e-5
    lr_finetune_head: float = 1e-4
    temperature: float = 0.1

    num_workers: int = 8
    seed: int = 42
    out_dir: str = "./outputs"

    pretrained: bool = True
    freeze_encoder_for_classifier: bool = True
    threshold: float = 0.5


def make_loaders(df, cfg: Config):
    if cfg.split_col not in df.columns:
        raise ValueError(
            f"CSV must contain a '{cfg.split_col}' column with values like train/test "
            f"(and optionally val)."
        )

    train_df = df[df[cfg.split_col] == "train"].reset_index(drop=True)
    val_df = df[df[cfg.split_col] == "val"].reset_index(drop=True)
    test_df = df[df[cfg.split_col] == "test"].reset_index(drop=True)

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Need at least train and test splits in the CSV.")

    train_ds = FundusDataset(
        train_df, cfg.image_col, LABEL_COLS,
        transform=get_train_transform(cfg.image_size),
        preprocess=True, image_size=cfg.image_size
    )

    val_ds = FundusDataset(
        val_df, cfg.image_col, LABEL_COLS,
        transform=get_eval_transform(),
        preprocess=True, image_size=cfg.image_size
    ) if len(val_df) > 0 else None

    test_ds = FundusDataset(
        test_df, cfg.image_col, LABEL_COLS,
        transform=get_eval_transform(),
        preprocess=True, image_size=cfg.image_size
    )

    train_supcon_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size_supcon, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )

    train_cls_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size_cls, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size_cls, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True
        )

    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size_cls, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    return train_supcon_loader, train_cls_loader, val_loader, test_loader

def save_checkpoint(state: dict, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

def evaluate_multilabel(model, loader, device, threshold=0.5):
    model.eval()
    y_true_all = []
    y_prob_all = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits)

            y_true_all.append(labels.cpu().numpy())
            y_prob_all.append(probs.cpu().numpy())

    y_true = np.vstack(y_true_all)
    y_prob = np.vstack(y_prob_all)
    y_pred = (y_prob >= threshold).astype(np.int32)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": support,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }

def print_metrics(metrics):
    print("\nPer-class results")
    for i, col in enumerate(LABEL_COLS):
        print(
            f"{col:>3} ({CLASS_NAMES.get(col, col):<25}) "
            f"P={metrics['precision_per_class'][i]:.4f} "
            f"R={metrics['recall_per_class'][i]:.4f} "
            f"F1={metrics['f1_per_class'][i]:.4f} "
            f"S={int(metrics['support_per_class'][i])}"
        )

    print("\nMacro results")
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall:    {metrics['macro_recall']:.4f}")
    print(f"F1:        {metrics['macro_f1']:.4f}")


def train_supcon(model, loader, device, cfg: Config):
    criterion = MultiLabelSupConLoss(temperature=cfg.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_supcon)

    model.train()
    for epoch in range(cfg.epochs_supcon):
        epoch_loss = 0.0
        start = time.time()

        batch_bar = tqdm(loader, desc=f"SupCon Epoch {epoch+1}/{cfg.epochs_supcon}", leave=False)
        for images, labels in batch_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, proj = model(images)
            loss = criterion(proj, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(1, len(loader))
        dt = time.time() - start
        print(f"[SupCon] Epoch {epoch + 1:03d}/{cfg.epochs_supcon}  loss={epoch_loss:.6f}  time={dt:.1f}s")

    return model

def train_classifier(model, train_loader, val_loader, device, cfg: Config):
    class_weights = torch.tensor([
        1.00,  # C0  Normal
        5.37,  # C1  AMD
        1.00,  # DR
        5.45,  # C6  Glaucoma
        6.43,  # C7  Hypertensive_Retinopathy
        4.18,  # C8  Pathological_Myopia
        5.54,  # C9  Tessellated_Fundus
        8.80,  # C10 Vitreous_Degeneration
        7.15,  # C11 BRVO
        4.29,  # C13 Large_Optic_Cup
        3.87,  # C14 Drusen
        4.32,  # C15 Epiretinal_Membrane
        8.80,  # C18 Optic_Disc_Edema
        8.52,  # C19 Myelinated_Nerve_Fibers
        11.94, # C22 Retinal_Detachment
        1.4,  # C25 Refractive_Media_Opacity
        10.80, # C27 CSC
        10.11, # C29 Laser_Spots
        11.15  # C32 CRVO
    ], dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    best_metric = -1.0
    best_path = os.path.join(cfg.out_dir, "best_classifier.pt")

    for epoch in range(cfg.epochs_cls):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        batch_bar = tqdm(train_loader, desc=f"CLS Epoch {epoch+1}/{cfg.epochs_cls}", leave=False)
        for images, labels in batch_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_loader))
        dt = time.time() - start

        if val_loader is not None:
            metrics = evaluate_multilabel(model, val_loader, device, threshold=cfg.threshold)
            score = metrics["macro_f1"]
            print(
                f"[CLS] Epoch {epoch + 1:03d}/{cfg.epochs_cls}  "
                f"loss={epoch_loss:.6f}  val_macro_f1={score:.4f}  time={dt:.1f}s"
            )
            if score > best_metric:
                best_metric = score
                save_checkpoint({"model": model.state_dict()}, best_path)
        else:
            print(f"[CLS] Epoch {epoch + 1:03d}/{cfg.epochs_cls}  loss={epoch_loss:.6f}  time={dt:.1f}s")
            save_checkpoint({"model": model.state_dict()}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def train_finetune(model, train_loader, val_loader, device, cfg: Config):
    """Stage 3: end-to-end fine-tuning with unfrozen encoder.

    The encoder uses a much smaller LR than the classifier head to avoid
    destroying the representations built during contrastive pretraining.
    A cosine annealing scheduler decays both LRs smoothly.
    """
    class_weights = torch.tensor([
        1.00,  # C0  Normal
        5.37,  # C1  AMD
        1.00,  # DR
        5.45,  # C6  Glaucoma
        6.43,  # C7  Hypertensive_Retinopathy
        4.18,  # C8  Pathological_Myopia
        5.54,  # C9  Tessellated_Fundus
        8.80,  # C10 Vitreous_Degeneration
        7.15,  # C11 BRVO
        4.29,  # C13 Large_Optic_Cup
        3.87,  # C14 Drusen
        4.32,  # C15 Epiretinal_Membrane
        8.80,  # C18 Optic_Disc_Edema
        8.52,  # C19 Myelinated_Nerve_Fibers
        11.94, # C22 Retinal_Detachment
        1.4,   # C25 Refractive_Media_Opacity
        10.80, # C27 CSC
        10.11, # C29 Laser_Spots
        11.15  # C32 CRVO
    ], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # Unfreeze the encoder
    for p in model.encoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": cfg.lr_finetune_encoder},
        {"params": model.classifier.parameters(), "lr": cfg.lr_finetune_head},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs_finetune, eta_min=1e-7
    )

    best_metric = -1.0
    best_path = os.path.join(cfg.out_dir, "best_finetuned.pt")

    for epoch in range(cfg.epochs_finetune):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        batch_bar = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{cfg.epochs_finetune}", leave=False)
        for images, labels in batch_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= max(1, len(train_loader))
        dt = time.time() - start
        enc_lr = scheduler.get_last_lr()[0]
        head_lr = scheduler.get_last_lr()[1]

        if val_loader is not None:
            metrics = evaluate_multilabel(model, val_loader, device, threshold=cfg.threshold)
            score = metrics["macro_f1"]
            print(
                f"[Finetune] Epoch {epoch + 1:03d}/{cfg.epochs_finetune}  "
                f"loss={epoch_loss:.6f}  val_macro_f1={score:.4f}  "
                f"enc_lr={enc_lr:.2e}  head_lr={head_lr:.2e}  time={dt:.1f}s"
            )
            if score > best_metric:
                best_metric = score
                save_checkpoint({"model": model.state_dict()}, best_path)
        else:
            print(
                f"[Finetune] Epoch {epoch + 1:03d}/{cfg.epochs_finetune}  "
                f"loss={epoch_loss:.6f}  time={dt:.1f}s"
            )
            save_checkpoint({"model": model.state_dict()}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


# ============================================================
# MAIN
# ============================================================

def main(cfg: Config):
    seed_everything(cfg.seed)
    ensure_dir(cfg.out_dir)

    df = pd.read_csv(cfg.csv_path)
    
    if "DR" not in df.columns:
        raise ValueError("CSV must contain merged DR column.")

    required_cols = [cfg.image_col, cfg.split_col] + LABEL_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df[df[LABEL_COLS].sum(axis=1) > 0].reset_index(drop=True)
    df['image'] = df['image'].str.replace("/data/", "DATASET/data/", regex=False)

    train_supcon_loader, train_cls_loader, val_loader, test_loader = make_loaders(df, cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Stage 1: SupCon encoder training
    print("SUPCON MODEL ENCODING STARTED")
    supcon_model = SupConEncoder(pretrained=cfg.pretrained).to(device)
    supcon_model = train_supcon(supcon_model, train_supcon_loader, device, cfg)

    save_checkpoint(
        {"model": supcon_model.state_dict()},
        os.path.join(cfg.out_dir, "supcon_encoder.pt")
    )

    # Stage 2: classifier training
    classifier = LinearClassifier(
        encoder=supcon_model.encoder,
        num_classes=len(LABEL_COLS)
    ).to(device)

    if cfg.freeze_encoder_for_classifier:
        for p in classifier.encoder.parameters():
            p.requires_grad = False

    print("\n\nCLASSIFIER TRAINING STARTED")
    classifier = train_classifier(classifier, train_cls_loader, val_loader, device, cfg)

    # Stage 3: end-to-end fine-tuning (unfreeze encoder)
    print("\n\nFINE-TUNING STARTED (encoder unfrozen)")
    classifier = train_finetune(classifier, train_cls_loader, val_loader, device, cfg)

    # Final evaluation
    metrics = evaluate_multilabel(classifier, test_loader, device, threshold=cfg.threshold)
    print_metrics(metrics)

if __name__ == "__main__":
    cfg = Config(
        csv_path="DATASET/filtered_data_split.csv",
        image_col='image',
        split_col='split',
        image_size=224,
        epochs_supcon=100,
        epochs_cls=30,
        epochs_finetune=20,
        batch_size_supcon=256,
        batch_size_cls=64,
        lr_supcon=5e-4,
        lr_finetune_encoder=1e-5,
        lr_finetune_head=1e-4,
        temperature=0.1,
        num_workers=0,
        seed=42,
        out_dir="EXPERIMENTS/V2/",
        freeze_encoder_for_classifier=True
    )
    main(cfg)