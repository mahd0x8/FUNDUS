"""
visualizer.py — In-memory segmentation and visualisation utilities.
Adapted from INFERENCE/segmentation.py; all functions operate on numpy arrays
instead of files on disk.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


# ============================================================
# HOT-REGION DETECTION  (from the GradCAM heatmap)
# ============================================================

def _red_score_map(heat_bgr: np.ndarray) -> np.ndarray:
    """
    Build a scalar score map that peaks in the most red (highest-activation)
    regions of a JET-coloured heatmap.
    """
    heat_bgr = cv2.GaussianBlur(heat_bgr, (5, 5), 0)
    b, g, r = cv2.split(heat_bgr.astype(np.float32))

    hsv = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red1 = cv2.inRange(hsv, (0,   70, 40), (12,  255, 255))
    red2 = cv2.inRange(hsv, (165, 70, 40), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2).astype(np.float32) / 255.0

    score = (
        0.7 * r
        + 0.9 * np.maximum(r - g, 0)
        + 0.9 * np.maximum(r - b, 0)
    )
    score *= (0.5 + s.astype(np.float32) / 255.0)
    score *= (0.5 + v.astype(np.float32) / 255.0)
    score *= (0.4 + 0.6 * red_mask)

    return cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _precise_hot_mask(heat_bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask of the highest-activation regions in the heatmap.
    Falls back gracefully if no strong activation is found.
    """
    score = _red_score_map(heat_bgr)
    nz = score[score > 0]
    if len(nz) == 0:
        return np.zeros(score.shape, dtype=np.uint8)

    mask = None
    for p in [69, 68, 67, 66, 65]:
        t = int(np.percentile(nz, p))
        _, temp = cv2.threshold(score, t, 255, cv2.THRESH_BINARY)
        temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        if cv2.countNonZero(temp) >= 20:
            mask = temp
            break

    if mask is None:
        return np.zeros(score.shape, dtype=np.uint8)

    # Keep only connected components with area ≥ 15 px
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    refined = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 15:
            refined[labels == i] = 255

    # Fill holes inside the kept regions
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(refined, contours, -1, 255, -1)
    return refined


# ============================================================
# POLYGON / BOUNDING-BOX EXTRACTION
# ============================================================

def _mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 15:
            continue
        epsilon = max(0.0025 * cv2.arcLength(cnt, True), 1.0)
        polygons.append(cv2.approxPolyDP(cnt, epsilon, True))
    return polygons


# ============================================================
# DRAWING HELPERS
# ============================================================

def _draw_polygon_overlay(original_bgr: np.ndarray,
                          mask: np.ndarray,
                          polygons: List[np.ndarray]) -> np.ndarray:
    """Transparent red fill + red polygon boundary on the original image."""
    out = original_bgr.copy()
    if cv2.countNonZero(mask) > 0:
        fill = out.copy()
        fill[mask > 0] = (0, 0, 255)
        out = cv2.addWeighted(fill, 0.20, out, 0.80, 0)
    for poly in polygons:
        cv2.polylines(out, [poly], True, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    return out


def _draw_bounding_boxes(original_bgr: np.ndarray,
                         polygons: List[np.ndarray],
                         label_text: str = "") -> np.ndarray:
    """Red bounding rectangles with an optional label above each box."""
    out = original_bgr.copy()
    for poly in polygons:
        x, y, w, h = cv2.boundingRect(poly)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)
        if label_text:
            cv2.putText(out, label_text,
                        (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def _create_panel(original_bgr: np.ndarray,
                  heatmap_bgr: np.ndarray,
                  mask: np.ndarray,
                  polygon_overlay_bgr: np.ndarray) -> np.ndarray:
    """2×2 comparison panel: original | heatmap / mask | polygon overlay."""
    h, w = original_bgr.shape[:2]
    heatmap_r  = cv2.resize(heatmap_bgr, (w, h))
    mask_rgb   = cv2.cvtColor(cv2.resize(mask, (w, h)), cv2.COLOR_GRAY2BGR)
    overlay_r  = cv2.resize(polygon_overlay_bgr, (w, h))
    return np.vstack([np.hstack([original_bgr, heatmap_r]),
                      np.hstack([mask_rgb,      overlay_r])])


# ============================================================
# PUBLIC API
# ============================================================

def process_heatmap(
    original_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    label_text: str = "",
) -> Dict[str, np.ndarray]:
    """
    Given the original preprocessed fundus image (BGR) and a JET-coloured
    GradCAM heatmap (BGR), compute all segmentation-derived visualisations.

    Returns a dict with keys:
        polygon_overlay  – red contour/fill on original
        bounding_box     – red bounding rectangles on original
        panel            – 2×2 comparison: original | heatmap / mask | overlay
    """
    h, w = original_bgr.shape[:2]

    # Align heatmap dimensions with original (should already match, but be safe)
    heatmap_bgr = cv2.resize(heatmap_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    mask     = _precise_hot_mask(heatmap_bgr)
    polygons = _mask_to_polygons(mask)

    polygon_overlay = _draw_polygon_overlay(original_bgr, mask, polygons)
    bounding_box    = _draw_bounding_boxes(original_bgr, polygons, label_text)
    panel           = _create_panel(original_bgr, heatmap_bgr, mask, polygon_overlay)

    return {
        "polygon_overlay": polygon_overlay,
        "bounding_box":    bounding_box,
        "panel":           panel,
    }
