"""
predictor.py — Model loading, preprocessing, GradCAM, and inference logic.
Adapted from INFERENCE/inference.py to work fully in-memory (no file I/O).
"""

import warnings
import base64
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

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
    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        backbone = resnext50_32x4d(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048
        self.projection = ProjectionHead(in_dim=2048, hidden_dim=512, out_dim=128)

    def encode(self, x):
        return self.encoder(x).flatten(1)

    def forward(self, x):
        feat = self.encode(x)
        proj = F.normalize(self.projection(feat), dim=1)
        return feat, proj


class LinearClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        return self.classifier(feat)


def load_model(checkpoint_path: str, device: str) -> LinearClassifier:
    supcon = SupConEncoder(pretrained=False)
    model = LinearClassifier(encoder=supcon.encoder, num_classes=len(LABEL_COLS))
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
# GRAD-CAM
# ============================================================

class GradCAM:
    """Single-use Grad-CAM wrapper. Call remove_hooks() when done."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fh = target_layer.register_forward_hook(self._fwd_hook)
        self._bh = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0]

    def remove_hooks(self):
        self._fh.remove()
        self._bh.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a normalised CAM (float32, values in [0, 1])."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        logits[0, class_idx].backward()

        grads = self._gradients[0]           # [C, H, W]
        acts  = self._activations[0]         # [C, H, W]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=0))

        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def _heatmap_bgr(cam: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize CAM and apply JET colormap → BGR uint8."""
    cam_resized = cv2.resize(cam, (target_hw[1], target_hw[0]))
    return cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)


def _overlay_heatmap(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend GradCAM heatmap onto the original image."""
    h, w = image_bgr.shape[:2]
    heatmap = _heatmap_bgr(cam, (h, w))
    overlay = np.float32(image_bgr) * (1 - alpha) + np.float32(heatmap) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


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
    model: LinearClassifier,
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
    target_layer = model.encoder[-2]   # layer3 of ResNeXt50 (14×14 feature map)
    gradcam = GradCAM(model, target_layer)
    visuals = []

    for item in selected:
        class_idx = LABEL_COLS.index(item["label_code"])
        cam = gradcam.generate(tensor, class_idx)

        h, w = image_bgr.shape[:2]
        heatmap_bgr = _heatmap_bgr(cam, (h, w))
        overlay_bgr = _overlay_heatmap(image_bgr, cam, alpha=0.4)

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
