"""
predictor.py — Model loading, preprocessing, GradCAM, and inference logic.
Adapted for SWIN_V1 (timm Swin-Large + 2-layer MLP head).
"""

import warnings
import base64
from typing import List, Tuple, Optional

import timm
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONSTANTS
# ============================================================

LABEL_COLS = [
    "C0", "C1", "DR", "C6", "C7", "C8", "C9",
    "C10", "C11", "C13", "C14", "C15", "C18", "C19", "C22",
    "C25", "C27", "C29", "C32",
]

CLASS_NAMES = {
    "C0":  "Normal",
    "C1":  "AMD",
    "DR":  "Diabetic Retinopathy",
    "C6":  "Glaucoma",
    "C7":  "Hypertensive Retinopathy",
    "C8":  "Pathological Myopia",
    "C9":  "Tessellated Fundus",
    "C10": "Vitreous Degeneration",
    "C11": "BRVO",
    "C13": "Large Optic Cup",
    "C14": "Drusen",
    "C15": "Epiretinal Membrane",
    "C18": "Optic Disc Edema",
    "C19": "Myelinated Nerve Fibers",
    "C22": "Retinal Detachment",
    "C25": "Refractive Media Opacity",
    "C27": "CSC",
    "C29": "Laser Spots",
    "C32": "CRVO",
}

IMAGE_SIZE = 224


# ============================================================
# MODEL ARCHITECTURE
#
# Checkpoint structure (EXPERIMENTS/SWIN_V1/best_swin.pt):
#   backbone.*  — timm swin_large_patch4_window7_224 (num_classes=0)
#   head.0.*    — nn.Linear(1536 → 512)
#   head.1      — nn.GELU()           (no params)
#   head.2      — nn.Dropout(0.3)     (no params)
#   head.3.*    — nn.Linear(512 → 19)
# ============================================================

class SwinClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_large_patch4_window7_224",
            pretrained=False,
            num_classes=0,      # raw pooled features, no built-in head
        )
        feat_dim = self.backbone.num_features   # 1536
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)   # [B, 1536]
        return self.head(features)


def load_model(checkpoint_path: str, device: str) -> SwinClassifier:
    model = SwinClassifier(num_classes=len(LABEL_COLS))
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


# ============================================================
# PREPROCESSING  (same pipeline as training)
# ============================================================

def _remove_black_borders(img_bgr: np.ndarray, threshold: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img_bgr
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img_bgr[y0:y1, x0:x1]


def _detect_fundus_circle(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int]]:
    gray = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    h, w = gray.shape[:2]
    min_r = max(30, min(h, w) // 5)
    max_r = max(min_r + 1, min(h, w) // 2)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min(h, w) // 2,
        param1=50, param2=30,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None
    circles = sorted(np.round(circles[0]).astype(int), key=lambda c: c[2], reverse=True)
    return tuple(circles[0])


def _crop_fundus(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    circle = _detect_fundus_circle(img_bgr)
    if circle is None:
        side = min(h, w)
        return img_bgr[(h - side) // 2:(h - side) // 2 + side,
                       (w - side) // 2:(w - side) // 2 + side]
    x, y, r = circle
    x0, y0 = max(0, x - r), max(0, y - r)
    x1, y1 = min(w, x + r), min(h, y + r)
    cropped = img_bgr[y0:y1, x0:x1]
    if cropped.size == 0:
        side = min(h, w)
        return img_bgr[(h - side) // 2:(h - side) // 2 + side,
                       (w - side) // 2:(w - side) // 2 + side]
    return cropped


def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def preprocess_fundus_bytes(image_bytes: bytes, image_size: int = IMAGE_SIZE) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Decode raw image bytes, run the full fundus preprocessing pipeline,
    and return (input_tensor [1,3,H,W], image_bgr [H,W,3]).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image — unsupported format or corrupt file.")

    img_bgr = _remove_black_borders(img_bgr)
    img_bgr = _crop_fundus(img_bgr)
    img_bgr = _apply_clahe(img_bgr)
    img_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(Image.fromarray(img_rgb)).unsqueeze(0)
    return tensor, img_bgr


# ============================================================
# GRAD-CAM  (Swin-specific)
#
# timm's Swin stages output [B, H, W, C] (channels-last).
# Grad-CAM weight averaging is over spatial dims (0, 1) of the
# per-sample slice; the weighted channel sum is over dim -1,
# yielding a [H, W] attention map.
#
# Target layer: model.backbone.layers[-1]
#   → last SwinTransformerStage, output [B, 7, 7, 1536] at 224px
# ============================================================

class GradCAMSwin:
    """Single-use Grad-CAM wrapper for Swin Transformer. Call remove_hooks() when done."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._fh = target_layer.register_forward_hook(self._fwd_hook)
        self._bh = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        # out: [B, H, W, C]
        self._activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        # grad_out[0]: [B, H, W, C]
        self._gradients = grad_out[0]

    def remove_hooks(self):
        self._fh.remove()
        self._bh.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a normalised CAM (float32, values in [0, 1])."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        logits[0, class_idx].backward()

        # Drop batch dim → [H, W, C]
        acts  = self._activations[0]   # [H, W, C]
        grads = self._gradients[0]     # [H, W, C]

        # Average gradients over spatial dims → channel importance weights [C]
        weights = grads.mean(dim=(0, 1))        # [C]

        # Weighted channel sum → spatial attention map [H, W]
        cam = F.relu((acts * weights).sum(dim=-1))   # [H, W]

        cam = cam.detach().cpu().numpy().astype(np.float32)
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def _get_fundus_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Build a binary mask (uint8, 0/255) for the fundus disc.
    Falls back to the largest inscribed circle when HoughCircles fails.
    """
    h, w = image_bgr.shape[:2]
    circle = _detect_fundus_circle(image_bgr)
    mask = np.zeros((h, w), dtype=np.uint8)
    if circle is not None:
        cx, cy, r = circle
    else:
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2
    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)
    return mask


def _heatmap_bgr(cam: np.ndarray, target_hw: Tuple[int, int],
                 mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Resize CAM, apply JET colormap, and black out pixels outside the fundus disc."""
    cam_resized = cv2.resize(cam, (target_hw[1], target_hw[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    if mask is not None:
        heatmap[mask == 0] = 0
    return heatmap


def _overlay_heatmap(image_bgr: np.ndarray, cam: np.ndarray,
                     mask: Optional[np.ndarray] = None,
                     alpha: float = 0.4) -> np.ndarray:
    """
    Blend GradCAM heatmap onto the fundus image, confined to the fundus disc.
    Pixels outside the disc are kept as-is from the original image.
    """
    h, w = image_bgr.shape[:2]
    heatmap = _heatmap_bgr(cam, (h, w))   # mask not applied here — overlay handles it
    blended = np.float32(image_bgr) * (1 - alpha) + np.float32(heatmap) * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    if mask is not None:
        inside = (mask > 0)[:, :, np.newaxis]
        return np.where(inside, blended, image_bgr).astype(np.uint8)
    return blended


# ============================================================
# HELPERS
# ============================================================

def numpy_to_base64(img_bgr: np.ndarray) -> str:
    """Encode a BGR numpy image to a base64-encoded PNG string."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to PNG-encode image.")
    return base64.b64encode(buf).decode("utf-8")


# ============================================================
# MAIN INFERENCE ENTRY POINT
# ============================================================

def predict(
    model: SwinClassifier,
    image_bytes: bytes,
    device: str,
    threshold: float = 0.5,
    top_k_if_none: int = 3,
) -> dict:
    """
    Run the full inference pipeline on raw image bytes.

    Returns a dict with:
        predictions  – sorted list of all 19 classes with probabilities
        original_image – base64 PNG of the preprocessed fundus image
        visuals      – per-class dict containing heatmap / overlay /
                       polygon_overlay / bounding_box / panel (all base64 PNG)
    """
    from visualizer import process_heatmap   # local import to avoid circular deps

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    tensor, image_bgr = preprocess_fundus_bytes(image_bytes)
    tensor = tensor.to(device)

    # ── 2. Forward pass (no gradients needed for probabilities) ───────────────
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # ── 3. Build sorted predictions list ──────────────────────────────────────
    predictions = [
        {
            "label_code": col,
            "label_name": CLASS_NAMES[col],
            "probability": round(float(probs[i]), 6),
            "predicted": bool(probs[i] >= threshold),
        }
        for i, col in enumerate(LABEL_COLS)
    ]
    predictions.sort(key=lambda x: x["probability"], reverse=True)

    # ── 4. Select classes for visualisation ───────────────────────────────────
    selected = [p for p in predictions if p["predicted"]]
    if not selected:
        selected = predictions[:top_k_if_none]

    # ── 5. GradCAM + segmentation per selected class ──────────────────────────
    # Last SwinTransformerStage → [B, 7, 7, 1536] for 224×224 input
    target_layer = model.backbone.layers[-1]
    gradcam = GradCAMSwin(model, target_layer)
    fundus_mask = _get_fundus_mask(image_bgr)
    visuals = []

    original_b64 = numpy_to_base64(image_bgr)

    for item in selected:
        # Normal fundus — skip GradCAM/segmentation; show original image in all tabs
        if item["label_code"] == "C0":
            visuals.append({
                "label_code":      item["label_code"],
                "label_name":      item["label_name"],
                "probability":     item["probability"],
                "heatmap":         original_b64,
                "overlay":         original_b64,
                "polygon_overlay": original_b64,
                "bounding_box":    original_b64,
                "panel":           original_b64,
            })
            continue

        class_idx = LABEL_COLS.index(item["label_code"])
        cam = gradcam.generate(tensor, class_idx)

        h, w = image_bgr.shape[:2]
        heatmap_bgr = _heatmap_bgr(cam, (h, w), mask=fundus_mask)
        overlay_bgr = _overlay_heatmap(image_bgr, cam, mask=fundus_mask, alpha=0.4)

        seg = process_heatmap(image_bgr, heatmap_bgr, item["label_name"])

        visuals.append({
            "label_code":      item["label_code"],
            "label_name":      item["label_name"],
            "probability":     item["probability"],
            "heatmap":         numpy_to_base64(heatmap_bgr),
            "overlay":         numpy_to_base64(overlay_bgr),
            "polygon_overlay": numpy_to_base64(seg["polygon_overlay"]),
            "bounding_box":    numpy_to_base64(seg["bounding_box"]),
            "panel":           numpy_to_base64(seg["panel"]),
        })

    gradcam.remove_hooks()
    torch.cuda.empty_cache()

    return {
        "predictions":    predictions,
        "original_image": numpy_to_base64(image_bgr),
        "visuals":        visuals,
    }
