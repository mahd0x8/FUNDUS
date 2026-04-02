import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import cv2

# ============================================================
# PREPROCESSING
# border removal -> HoughCircles eye detection -> crop -> CLAHE
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


def preprocess_fundus_image(path: str, image_size: int = 224) -> Image.Image:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img_bgr = remove_black_borders(img_bgr)
    img_bgr = crop_fundus(img_bgr)
    img_bgr = apply_clahe(img_bgr)
    img_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)
