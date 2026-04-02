import re
import cv2
import numpy as np
import shutil
from pathlib import Path


BASE_DIR = Path("INFERENCE/FUNDUS INFERENCE RESULTS V1")

# Accept common image extensions and also extensionless files if OpenCV can read them
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ""}


def extract_confidence_from_name(name: str):
    """
    Extract confidence from filenames like:
    C25_Refractive_Media_Opacity_1.0000_heatmap.png
    C25_Refractive_Media_Opacity_0.8732_heatmap
    """
    match = re.search(r'_(\d+\.\d+)_heatmap', name)
    if match:
        return float(match.group(1))
    return None


def red_score_map(heat_bgr):
    """
    Build a score map for 'maximum red' regions.
    Higher score = stronger disease activation candidate.
    """
    heat_bgr = cv2.GaussianBlur(heat_bgr, (5, 5), 0)

    b, g, r = cv2.split(heat_bgr.astype(np.float32))
    hsv = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red1 = cv2.inRange(hsv, (0, 70, 40), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 70, 40), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2).astype(np.float32) / 255.0

    rg = np.maximum(r - g, 0)
    rb = np.maximum(r - b, 0)

    score = (0.7 * r + 0.9 * rg + 0.9 * rb)
    score *= (0.5 + s.astype(np.float32) / 255.0)
    score *= (0.5 + v.astype(np.float32) / 255.0)
    score *= (0.4 + 0.6 * red_mask)

    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return score


def get_precise_hot_mask(heat_bgr):
    """
    Make a tight mask from the strongest red area only.
    """
    score = red_score_map(heat_bgr)
    nz = score[score > 0]

    if len(nz) == 0:
        return np.zeros(score.shape, dtype=np.uint8), score

    # Very high percentile for precision
    thresholds = [69, 68, 67, 66, 65]
    mask = None

    for p in thresholds:
        t = int(np.percentile(nz, p))
        _, temp = cv2.threshold(score, t, 255, cv2.THRESH_BINARY)

        temp = cv2.morphologyEx(
            temp, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1
        )
        temp = cv2.morphologyEx(
            temp, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1
        )

        if cv2.countNonZero(temp) >= 20:
            mask = temp
            break

    if mask is None:
        mask = np.zeros(score.shape, dtype=np.uint8)

    # Keep all significant connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    refined = np.zeros_like(mask)

    if num_labels <= 1:
        return mask, score

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 15:
            refined[labels == i] = 255

    # Fill any black holes inside the white regions
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(refined, contours, -1, 255, -1)

    return refined, score


def mask_to_polygons(mask):
    """
    Convert mask into precise polygon boundaries.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15:
            continue

        perimeter = cv2.arcLength(cnt, True)
        epsilon = max(0.0025 * perimeter, 1.0)  # smaller epsilon = more precise
        poly = cv2.approxPolyDP(cnt, epsilon, True)
        polygons.append(poly)

    return polygons


def overlay_boundary(original_bgr, mask, polygons):
    """
    Transparent red fill + yellow polygon boundary.
    """
    out = original_bgr.copy()

    if cv2.countNonZero(mask) > 0:
        fill = out.copy()
        fill[mask > 0] = (0, 0, 255)
        out = cv2.addWeighted(fill, 0.20, out, 0.80, 0)

    for poly in polygons:
        cv2.polylines(out, [poly], True, (0, 0, 255), 1, lineType=cv2.LINE_AA)

    return out


def overlay_bounding_boxes(original_bgr, polygons, label_text=""):
    """
    Draw bounding boxes around each polygon contour with an optional label.
    """
    out = original_bgr.copy()
    for poly in polygons:
        x, y, w, h = cv2.boundingRect(poly)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)
        if label_text:
            cv2.putText(out, label_text, (x, max(y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def create_panel(original, heatmap, mask, overlay):
    h, w = original.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    top = np.hstack([original, heatmap])
    bottom = np.hstack([mask_vis, overlay])

    return np.vstack([top, bottom])


def imread_any(path: Path):
    """
    Read even extensionless image files if OpenCV supports them.
    """
    img = cv2.imread(str(path))
    return img


def find_best_heatmap(case_dir: Path):
    """
    Select only the best-confidence heatmap in a folder.
    """
    candidates = []

    for f in case_dir.iterdir():
        if not f.is_file():
            continue

        if f.suffix.lower() not in VALID_EXTENSIONS:
            continue

        name_lower = f.name.lower()
        if "heatmap" not in name_lower:
            continue

        conf = extract_confidence_from_name(f.name)
        if conf is None:
            continue

        candidates.append((conf, f))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_conf, best_file = candidates[0]
    return best_file, best_conf


def process_case(case_dir: Path):
    original_path = case_dir / "original_image.png"
    if not original_path.exists():
        print(f"[SKIP] No original_image.png in {case_dir}")
        return

    best_heatmap_path, best_conf = find_best_heatmap(case_dir)
    if best_heatmap_path is None:
        print(f"[SKIP] No valid heatmap found in {case_dir}")
        return

    original = imread_any(original_path)
    heatmap = imread_any(best_heatmap_path)

    if original is None:
        print(f"[SKIP] Could not read original: {original_path}")
        return

    if heatmap is None:
        print(f"[SKIP] Could not read heatmap: {best_heatmap_path}")
        return

    if heatmap.shape[:2] != original.shape[:2]:
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask, _ = get_precise_hot_mask(heatmap)
    polygons = mask_to_polygons(mask)
    overlay = overlay_boundary(original, mask, polygons)
    bbox_overlay = overlay_bounding_boxes(original, polygons, " ".join(best_heatmap_path.name.split("_")[1:-2]))
    panel = create_panel(original, heatmap, mask, overlay)

    out_dir = case_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = out_dir / "overlay_polygon_boundary.png"
    mask_path = out_dir / "hot_region_mask.png"
    panel_path = out_dir / "comparison_panel.png"
    bbox_path = out_dir / "bounding_boxes.png"

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(panel_path), panel)
    cv2.imwrite(str(bbox_path), bbox_overlay)

    print(f"[OK] {case_dir.name}")
    print(f"     best heatmap : {best_heatmap_path.name}")
    print(f"     confidence   : {best_conf:.4f}")
    print(f"     saved        : {overlay_path}")
    print(f"     saved        : {mask_path}")
    print(f"     saved        : {panel_path}")
    print(f"     saved        : {bbox_path}")


def main():
    if not BASE_DIR.exists():
        print(f"[ERROR] Base directory not found: {BASE_DIR}")
        return

    case_dirs = [p for p in BASE_DIR.iterdir() if p.is_dir()]
    if not case_dirs:
        print(f"[ERROR] No case folders found inside: {BASE_DIR}")
        return

    for case_dir in sorted(case_dirs):
        process_case(case_dir)


if __name__ == "__main__":
    main()