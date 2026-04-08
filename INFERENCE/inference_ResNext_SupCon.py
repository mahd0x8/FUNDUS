import os
import cv2
import json
import math
import random
import warnings
from typing import List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIG
# ============================================================

LABEL_COLS = [
    "C0", "C1", "DR", "C6", "C7", "C8", "C9",
    "C10", "C11", "C13", "C14", "C15", "C18", "C19", "C22",
    "C25", "C27", "C29", "C32"
]

CLASS_NAMES = {
    "C0": "Normal",
    "C1": "AMD",
    "DR": "Diabetic Retinopathy",
    "C6": "Glaucoma",
    "C7": "Hypertensive_Retinopathy",
    "C8": "Pathological_Myopia",
    "C9": "Tessellated_Fundus",
    "C10": "Vitreous_Degeneration",
    "C11": "BRVO",
    "C13": "Large_Optic_Cup",
    "C14": "Drusen",
    "C15": "Epiretinal_Membrane",
    "C18": "Optic_Disc_Edema",
    "C19": "Myelinated_Nerve_Fibers",
    "C22": "Retinal_Detachment",
    "C25": "Refractive_Media_Opacity",
    "C27": "CSC",
    "C29": "Laser_Spots",
    "C32": "CRVO",
}

IMAGE_SIZE = 224


# ============================================================
# UTILS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# PREPROCESSING
# Same as training
# ============================================================

def remove_black_borders(img_bgr: np.ndarray, threshold: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    coords = np.argwhere(mask)

    if coords.size == 0:
        return img_bgr

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img_bgr[y0:y1, x0:x1]


def detect_fundus_circle(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape[:2]
    min_radius = max(30, min(h, w) // 5)
    max_radius = max(min_radius + 1, min(h, w) // 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    x, y, r = circles[0]
    return x, y, r


def crop_fundus(img_bgr: np.ndarray) -> np.ndarray:
    circle = detect_fundus_circle(img_bgr)
    h, w = img_bgr.shape[:2]

    if circle is None:
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return img_bgr[y0:y0 + side, x0:x0 + side]

    x, y, r = circle
    x0 = max(0, x - r)
    y0 = max(0, y - r)
    x1 = min(w, x + r)
    y1 = min(h, y + r)

    cropped = img_bgr[y0:y1, x0:x1]
    if cropped.size == 0:
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return img_bgr[y0:y0 + side, x0:x0 + side]
    return cropped


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_fundus_image(path: str, image_size: int = 224) -> Tuple[Image.Image, np.ndarray]:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img_bgr = remove_black_borders(img_bgr)
    img_bgr = crop_fundus(img_bgr)
    img_bgr = apply_clahe(img_bgr)
    img_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return pil_img, img_rgb


def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


# ============================================================
# MODEL
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


def build_classifier_model(num_classes: int, checkpoint_path: str, device: str):
    supcon_model = SupConEncoder(pretrained=False)
    model = LinearClassifier(
        encoder=supcon_model.encoder,
        num_classes=num_classes
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()

        logits = self.model(input_tensor)   # [1, num_classes]
        score = logits[0, class_idx]
        score.backward()

        # activations: [1, C, H, W]
        # gradients:   [1, C, H, W]
        grads = self.gradients[0]
        acts = self.activations[0]

        weights = grads.mean(dim=(1, 2), keepdim=True)   # [C,1,1]
        cam = (weights * acts).sum(dim=0)                # [H,W]
        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def get_fundus_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Build a binary mask (uint8, 0/255) matching the fundus disc in a
    preprocessed 224×224 image.  Runs HoughCircles on the image; if
    detection fails falls back to the largest inscribed circle so the
    mask always covers the visible retina and never the black corners.
    """
    h, w = image_rgb.shape[:2]
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    circle = detect_fundus_circle(img_bgr)

    mask = np.zeros((h, w), dtype=np.uint8)
    if circle is not None:
        cx, cy, r = circle
    else:
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2

    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)
    return mask


def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the fundus image, constrained to the
    fundus disc described by `mask`.

    Inside the disc  : alpha-blend of original image + JET heatmap.
    Outside the disc : original image pixels only (no heatmap bleed).
    """
    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8 = np.uint8(255 * cam_resized)

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    blended = np.float32(image_rgb) * (1 - alpha) + np.float32(heatmap) * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Restore original pixels everywhere outside the fundus circle
    inside = (mask > 0)[:, :, np.newaxis]          # [H, W, 1] bool
    result = np.where(inside, blended, image_rgb)
    return result.astype(np.uint8)


# ============================================================
# INFERENCE
# ============================================================

def predict_single_image(
    model: nn.Module,
    image_path: str,
    device: str,
    threshold: float = 0.5,
    image_size: int = 224,
):
    transform = get_eval_transform()

    pil_img, image_rgb = preprocess_fundus_image(image_path, image_size=image_size)

    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    results = []
    for i, col in enumerate(LABEL_COLS):
        results.append({
            "label_code": col,
            "label_name": CLASS_NAMES.get(col, col),
            "probability": float(probs[i]),
            "predicted": int(probs[i] >= threshold)
        })

    results = sorted(results, key=lambda x: x["probability"], reverse=True)
    return tensor, image_rgb, results


def save_heatmaps(
    model: nn.Module,
    input_tensor: torch.Tensor,
    image_rgb: np.ndarray,
    results: List[dict],
    output_dir: str,
    threshold: float = 0.5,
    top_k_if_none: int = 3,
):
    ensure_dir(output_dir)

    original_image_path = os.path.join(output_dir, "original_image.png")
    cv2.imwrite(original_image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    # Detect the fundus circle once; all heatmaps for this image share the mask
    fundus_mask = get_fundus_mask(image_rgb)

    # target layer = layer4 (not avgpool)
    target_layer = model.encoder[-2]
    gradcam = GradCAM(model, target_layer)

    selected = [r for r in results if r["probability"] >= threshold]

    if len(selected) == 0:
        selected = results[:top_k_if_none]

    for item in selected:
        class_code = item["label_code"]
        class_name = item["label_name"]
        class_idx = LABEL_COLS.index(class_code)

        cam = gradcam.generate(input_tensor, class_idx)
        overlay = overlay_heatmap_on_image(image_rgb, cam, fundus_mask, alpha=0.4)

        base_name = f"{class_code}_{class_name}_{item['probability']:.4f}".replace(" ", "_")
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
        raw_path = os.path.join(output_dir, f"{base_name}_rawcam.npy")

        h, w = image_rgb.shape[:2]
        cam_uint8 = np.uint8(255 * cv2.resize(cam, (w, h)))
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        # Black out heatmap pixels that fall outside the fundus disc
        heatmap_bgr[fundus_mask == 0] = 0

        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(heatmap_path, heatmap_bgr)
        np.save(raw_path, cam)

    gradcam.remove_hooks()


def is_image_file(path: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return os.path.splitext(path)[1].lower() in exts


def run_inference(
    checkpoint_path: str,
    input_path: str,
    output_dir: str,
    threshold: float = 0.5,
    image_size: int = 224,
    top_k_if_none: int = 3,
):
    ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = build_classifier_model(
        num_classes=len(LABEL_COLS),
        checkpoint_path=checkpoint_path,
        device=device
    )

    if os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        image_paths = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if is_image_file(os.path.join(input_path, f))
        ]
        image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise ValueError("No image files found.")

    all_rows = []

    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_out_dir = os.path.join(output_dir, image_name)
        ensure_dir(image_out_dir)

        input_tensor, image_rgb, results = predict_single_image(
            model=model,
            image_path=image_path,
            device=device,
            threshold=threshold,
            image_size=image_size
        )

        print("Top predictions:")
        for r in results[:5]:
            print(f"  {r['label_code']:>3} | {r['label_name']:<25} | {r['probability']:.4f}")

        save_heatmaps(
            model=model,
            input_tensor=input_tensor,
            image_rgb=image_rgb,
            results=results,
            output_dir=image_out_dir,
            threshold=threshold,
            top_k_if_none=top_k_if_none
        )
        # del cam, overlay
        torch.cuda.empty_cache()
        with open(os.path.join(image_out_dir, "predictions.json"), "w") as f:
            json.dump(results, f, indent=2)

        row = {"image": image_path}
        for r in results:
            row[f"{r['label_code']}_prob"] = r["probability"]
            row[f"{r['label_code']}_pred"] = r["predicted"]
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(output_dir, "inference_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    checkpoint_path = "EXPERIMENTS/ResNext_V2 (Encoder Unfreeze)/best_classifier.pt"   # your trained classifier checkpoint
    input_path = "DATASET/Testing_Data"                           # folder or single image
    output_dir = "INFERENCE/INFERENCE RESULTS/FUNDUS INFERENCE RESULTS V2/"

    run_inference(
        checkpoint_path=checkpoint_path,
        input_path=input_path,
        output_dir=output_dir,
        threshold=0.5,
        image_size=224,
        top_k_if_none=3,
    )
