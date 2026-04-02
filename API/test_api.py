"""
test_api.py — Send a fundus image to the prediction API and save all results.

Usage:
    python test_api.py <image_path> [output_folder] [--threshold 0.5] [--top-k 3] [--url http://localhost:8000]

Examples:
    python test_api.py /path/to/fundus.jpg
    python test_api.py /path/to/fundus.jpg ./results
    python test_api.py /path/to/fundus.jpg ./results --threshold 0.3 --top-k 5
"""

import argparse
import base64
import json
import os
import sys

import httpx as requests


# ============================================================
# HELPERS
# ============================================================

def save_b64_image(b64_string: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_string))


def print_separator(char="─", width=60):
    print(char * width)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test the FUNDUS prediction API.")
    parser.add_argument("image_path",             help="Path to the fundus image to send")
    parser.add_argument("output_folder", nargs="?", default="./api_results",
                        help="Folder to save output images (default: ./api_results)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Prediction threshold (default: 0.5)")
    parser.add_argument("--top-k",     type=int,   default=3,
                        help="Top-k classes to visualise when nothing passes the threshold (default: 3)")
    parser.add_argument("--url",       default="http://localhost:8000",
                        help="API base URL (default: http://localhost:8000)")
    args = parser.parse_args()

    image_path    = os.path.abspath(args.image_path)
    output_folder = os.path.abspath(args.output_folder)
    base_url      = args.url.rstrip("/")

    # ── Validate input ────────────────────────────────────────────────────────
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    # ── Health check ──────────────────────────────────────────────────────────
    print_separator()
    print("  FUNDUS API Test Client")
    print_separator()
    print(f"  API URL      : {base_url}")
    print(f"  Image        : {image_path}")
    print(f"  Output folder: {output_folder}")
    print(f"  Threshold    : {args.threshold}")
    print(f"  Top-k        : {args.top_k}")
    print_separator()

    try:
        health = requests.get(f"{base_url}/health", timeout=5).json()
        print(f"  Server status : {health['status']}")
        print(f"  Device        : {health['device']}")
        print(f"  Classes       : {health['num_classes']}")
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Cannot connect to the API at {base_url}")
        print("        Make sure the server is running:")
        print("        uvicorn API.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    print_separator()

    # ── Send request ──────────────────────────────────────────────────────────
    print("  Sending image to /predict ...")
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    with open(image_path, "rb") as f:
        response = requests.post(
            f"{base_url}/predict",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")},
            data={"threshold": str(args.threshold), "top_k_if_none": str(args.top_k)},
            timeout=120,
        )

    if response.status_code != 200:
        print(f"\n[ERROR] API returned {response.status_code}:")
        print(response.text)
        sys.exit(1)

    body = response.json()

    # ── Save images ───────────────────────────────────────────────────────────
    out_dir = os.path.join(output_folder, image_name)
    os.makedirs(out_dir, exist_ok=True)

    # Original preprocessed image
    original_path = os.path.join(out_dir, "original.png")
    save_b64_image(body["original_image"], original_path)
    print(f"  Saved: original.png")

    # Per-class visuals
    for item in body["visuals"]:
        code  = item["label_code"]
        name  = item["label_name"].replace(" ", "_")
        prob  = item["probability"]
        tag   = f"{code}_{name}_{prob:.4f}"
        class_dir = os.path.join(out_dir, tag)
        os.makedirs(class_dir, exist_ok=True)

        images = {
            "heatmap.png":         item["heatmap"],
            "overlay.png":         item["overlay"],
            "polygon_overlay.png": item["polygon_overlay"],
            "bounding_box.png":    item["bounding_box"],
            "panel.png":           item["panel"],
        }
        for filename, b64 in images.items():
            save_b64_image(b64, os.path.join(class_dir, filename))

        print(f"  Saved: {tag}/  (5 images)")

    # Predictions JSON
    predictions_path = os.path.join(out_dir, "predictions.json")
    with open(predictions_path, "w") as f:
        json.dump(body["predictions"], f, indent=2)
    print(f"  Saved: predictions.json")

    # ── Print predictions table ───────────────────────────────────────────────
    print_separator()
    print(f"  {'Code':<5}  {'Disease':<30}  {'Prob':>6}  Predicted")
    print_separator()
    for p in body["predictions"]:
        marker = "  <-- DETECTED" if p["predicted"] else ""
        print(f"  {p['label_code']:<5}  {p['label_name']:<30}  {p['probability']:>6.4f}{marker}")

    print_separator()
    detected = [p for p in body["predictions"] if p["predicted"]]
    print(f"  Detected {len(detected)} condition(s)  |  "
          f"Visualisations saved for {len(body['visuals'])} class(es)")
    print(f"  Results saved to: {out_dir}")
    print_separator()


if __name__ == "__main__":
    main()
