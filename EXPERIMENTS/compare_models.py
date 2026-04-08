"""
compare_models.py — Cross-model evaluation comparison for the FUNDUS project.

Reads existing evaluation text files from each experiment folder and
produces a full comparison report + 9 visualisation plots.

Run from the project root:
    python EXPERIMENTS/compare_models.py

Outputs (EXPERIMENTS/COMPARISON/):
    comparison_report.txt              — full text comparison with rankings
    01_macro_metrics.png               — Macro P / R / F1 grouped bar chart
    02_perclass_f1_bars.png            — per-class F1 grouped bar chart
    03_perclass_f1_heatmap.png         — per-class F1 heatmap (models × classes)
    04_radar_chart.png                 — radar / spider chart (representative classes)
    05_win_loss.png                    — class-level win / tie / loss matrix
    06_f1_delta.png                    — F1 delta of best model vs each baseline
    07_training_curves.png             — validation macro F1 training curves
    08_support_vs_f1.png               — class support vs average F1 scatter
    09_precision_recall_scatter.png    — per-class P vs R for all models
"""

import os
import re
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR = "EXPERIMENTS/COMPARISON"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COLS = [
    "C0", "C1", "DR", "C6", "C7", "C8", "C9",
    "C10", "C11", "C13", "C14", "C15", "C18", "C19",
    "C22", "C25", "C27", "C29", "C32",
]
CLASS_NAMES = {
    "C0":  "Normal",                  "C1":  "AMD",
    "DR":  "Diabetic Retinopathy",    "C6":  "Glaucoma",
    "C7":  "Hypertensive Retinopathy","C8":  "Pathological Myopia",
    "C9":  "Tessellated Fundus",      "C10": "Vitreous Degeneration",
    "C11": "BRVO",                    "C13": "Large Optic Cup",
    "C14": "Drusen",                  "C15": "Epiretinal Membrane",
    "C18": "Optic Disc Edema",        "C19": "Myelinated Nerve Fibers",
    "C22": "Retinal Detachment",      "C25": "Refractive Media Opacity",
    "C27": "CSC",                     "C29": "Laser Spots",
    "C32": "CRVO",
}

# (display_name, eval_file, architecture_string, training_log_file_or_None)
MODELS = [
    (
        "RESNET_V1",
        "EXPERIMENTS/RESNET_V1/resnet_evaluation.txt",
        "ResNet-50  |  24.6M params  |  50 epochs  |  ImageNet-1k pretrained  |  End-to-end",
        "EXPERIMENTS/RESNET_V1/training_log.txt",
    ),
    (
        "CNN_V1",
        "EXPERIMENTS/CNN_V1/cnn_evaluation.txt",
        "ConvNeXt-Base  |  ~87M params  |  50 epochs  |  ImageNet-1k pretrained  |  End-to-end",
        None,
    ),
    (
        "SWIN_V1",
        "EXPERIMENTS/SWIN_V1/swin_evaluation.txt",
        "Swin-Large  |  195.8M params  |  100 epochs  |  ImageNet-22k pretrained  |  End-to-end",
        "EXPERIMENTS/SWIN_V1/training_log.txt",
    ),
    (
        "ResNext_V1",
        "EXPERIMENTS/ResNext_V1 (Encoder Freeze)/training_evaluation.txt",
        "ResNeXt-50 + SupCon  |  ~25M params  |  2-stage: 100+30 epochs  |  Encoder frozen in CLS",
        "EXPERIMENTS/ResNext_V1 (Encoder Freeze)/training_evaluation.txt",
    ),
    (
        "ResNext_V2",
        "EXPERIMENTS/ResNext_V2 (Encoder Unfreeze)/training_evaluation.txt",
        "ResNeXt-50 + SupCon  |  ~25M params  |  3-stage: 100+30+20 epochs  |  Full fine-tune",
        "EXPERIMENTS/ResNext_V2 (Encoder Unfreeze)/training_evaluation.txt",
    ),
]

MODEL_COLORS = {
    "RESNET_V1":  "#e74c3c",
    "CNN_V1":     "#e67e22",
    "SWIN_V1":    "#2ecc71",
    "ResNext_V1": "#3498db",
    "ResNext_V2": "#9b59b6",
}

# Approximate parameter counts (millions)
MODEL_PARAMS_M = {
    "RESNET_V1":   24.6,
    "CNN_V1":      87.0,
    "SWIN_V1":    195.8,
    "ResNext_V1":  25.0,
    "ResNext_V2":  25.0,
}

# Architecture metadata for the report table
ARCH_META = {
    "RESNET_V1":  ("ResNet-50",         "24.6M",  "50",        "End-to-end, ImageNet-1k"),
    "CNN_V1":     ("ConvNeXt-Base",     "~87M",   "50",        "End-to-end, ImageNet-1k"),
    "SWIN_V1":    ("Swin-Large",        "195.8M", "100",       "End-to-end, ImageNet-22k"),
    "ResNext_V1": ("ResNeXt-50+SupCon", "~25M",   "100+30",    "2-stage SupCon, Enc frozen"),
    "ResNext_V2": ("ResNeXt-50+SupCon", "~25M",   "100+30+20", "3-stage SupCon+Finetune"),
}

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_eval_file(path: str) -> dict:
    """
    Parses two evaluation file formats and returns a unified dict:
        per_class: {code: {P, R, F1, support}}
        macro:     {P, R, F1}
    """
    with open(path) as f:
        text = f.read()

    per_class = {}

    # Format A (SupCon logs): "  C0  (Normal  ...) P=0.xxx R=0.xxx F1=0.xxx S=nnn"
    fmt_a = re.findall(
        r"^\s*(\w+)\s+\([^)]+\)\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)\s+S=(\d+)",
        text, re.MULTILINE
    )
    if fmt_a:
        for code, p, r, f1, s in fmt_a:
            if code in LABEL_COLS:
                per_class[code] = dict(P=float(p), R=float(r), F1=float(f1), support=int(s))

    # Format B (table with Thr column):
    # "  C0  Normal   0.35  0.6050  0.9321  0.7337  442"
    if not per_class:
        fmt_b = re.findall(
            r"^\s*(\w+)\s+\S[^\n]*?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)",
            text, re.MULTILINE
        )
        for row in fmt_b:
            code = row[0]
            if code in LABEL_COLS:
                per_class[code] = dict(
                    thr=float(row[1]), P=float(row[2]),
                    R=float(row[3]), F1=float(row[4]), support=int(row[5])
                )

    # Macro — "Macro  P=X  R=Y  F1=Z"  OR  "Precision: X\nRecall: Y\nF1: Z"
    macro_match = re.search(
        r"Macro\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)", text
    )
    if not macro_match:
        macro_match = re.search(
            r"Precision:[ \t]*([\d.]+)[ \t]*\nRecall:[ \t]*([\d.]+)[ \t]*\nF1:[ \t]*([\d.]+)",
            text
        )
    macro = {}
    if macro_match:
        macro = dict(
            P=float(macro_match.group(1)),
            R=float(macro_match.group(2)),
            F1=float(macro_match.group(3)),
        )

    return {"per_class": per_class, "macro": macro}


def parse_training_curves(log_path) -> dict:
    """
    Extracts val F1 per epoch from training logs.
    Returns dict with keys: 'val_f1', 'cls_f1', 'ft_f1' (whichever are present).
    """
    if log_path is None or not os.path.exists(log_path):
        return {}

    with open(log_path) as f:
        text = f.read()

    curves = {}

    # End-to-end models: "[ResNet] Epoch NNN/50  loss=X  val_f1=Y  ..."
    standard = re.findall(
        r"\[(?:ResNet|Swin|CNN)\]\s+Epoch\s+(\d+)/\d+\s+loss=[\d.]+\s+val_f1=([\d.]+)",
        text
    )
    if standard:
        curves["val_f1"] = ([int(e) for e, _ in standard],
                            [float(v) for _, v in standard])

    # SupCon classifier stage: "[CLS] Epoch NNN/30  loss=X  val_macro_f1=Y"
    cls_stage = re.findall(
        r"\[CLS\]\s+Epoch\s+(\d+)/\d+\s+loss=[\d.]+\s+val_macro_f1=([\d.]+)",
        text
    )
    if cls_stage:
        curves["cls_f1"] = ([int(e) for e, _ in cls_stage],
                            [float(v) for _, v in cls_stage])

    # SupCon fine-tune stage: "[Finetune] Epoch NNN/20  loss=X  val_macro_f1=Y"
    ft_stage = re.findall(
        r"\[Finetune\]\s+Epoch\s+(\d+)/\d+\s+loss=[\d.]+\s+val_macro_f1=([\d.]+)",
        text
    )
    if ft_stage:
        curves["ft_f1"] = ([int(e) for e, _ in ft_stage],
                           [float(v) for _, v in ft_stage])

    return curves


# ── Load all models ───────────────────────────────────────────────────────────

print("Parsing evaluation files …")
results = {}
for name, path, arch, log_path in MODELS:
    data = parse_eval_file(path)
    data["arch"] = arch
    data["curves"] = parse_training_curves(log_path)
    results[name] = data
    mf1 = data["macro"].get("F1", float("nan"))
    print(f"  {name:<22}  macro F1 = {mf1:.4f}  ({len(data['per_class'])} classes parsed)")

model_names = [m[0] for m in MODELS]
n_models = len(model_names)
ranking = sorted(model_names, key=lambda n: results[n]["macro"]["F1"], reverse=True)

# ── Shared helpers ────────────────────────────────────────────────────────────

def savefig(fname: str):
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
})

# Pre-compute per-class stats
class_winners = {}
avg_f1 = {}
for code in LABEL_COLS:
    f1s = {n: results[n]["per_class"].get(code, {}).get("F1", float("nan"))
           for n in model_names}
    valid = [v for v in f1s.values() if not np.isnan(v)]
    avg_f1[code] = np.mean(valid) if valid else float("nan")

    best_f1 = max(valid)
    winners = [n for n, v in f1s.items() if abs(v - best_f1) < 1e-6]
    class_winners[code] = winners[0] if len(winners) == 1 else "Tie"


# ── Text report ───────────────────────────────────────────────────────────────

print("Writing comparison_report.txt …")

lines = []
def w(*args): lines.append(" ".join(str(a) for a in args))

SEP = "=" * 84

w(SEP)
w("  FUNDUS MODEL COMPARISON REPORT")
w(f"  {n_models} Experiments  |  19-class Multi-Label Classification  |  Test set n=1,262")
w(SEP)
w()

# ── 1. Overall ranking
w("── 1. OVERALL MACRO RANKING ─────────────────────────────────────────────────────")
medals = ["★ 1st", "  2nd", "  3rd", "  4th", "  5th", "  6th", "  7th"]
w(f"  {'Rank':<8} {'Model':<22} {'Macro P':>9} {'Macro R':>9} {'Macro F1':>10}  Architecture")
w(f"  {'-'*88}")
for rank, name in enumerate(ranking):
    m = results[name]["macro"]
    arch_short = results[name]["arch"].split("|")[0].strip()
    w(f"  {medals[rank]:<8} {name:<22} {m['P']:>9.4f} {m['R']:>9.4f} {m['F1']:>10.4f}  {arch_short}")
w()
best = ranking[0]
worst = ranking[-1]
w(f"  Best model  : {best}  (Macro F1 = {results[best]['macro']['F1']:.4f})")
w(f"  Worst model : {worst}  (Macro F1 = {results[worst]['macro']['F1']:.4f})")
delta_range = results[best]["macro"]["F1"] - results[worst]["macro"]["F1"]
w(f"  F1 spread   : {results[worst]['macro']['F1']:.4f} – {results[best]['macro']['F1']:.4f}"
  f"  (Δ = {delta_range:.4f})")
w()

# ── 2. Architecture details
w("── 2. ARCHITECTURE DETAILS ──────────────────────────────────────────────────────")
w(f"  {'Model':<22} {'Backbone':<22} {'Params':>8} {'Epochs':>10}  Strategy")
w(f"  {'-'*85}")
for name in model_names:
    bb, params, epochs, strat = ARCH_META[name]
    f1 = results[name]["macro"]["F1"]
    w(f"  {name:<22} {bb:<22} {params:>8} {epochs:>10}  {strat}  [F1={f1:.4f}]")
w()
w("  Training strategies:")
w("    • End-to-end     — backbone fine-tuned jointly with classifier (cross-entropy)")
w("    • SupCon freeze  — Stage1: SupCon encoder pre-training; Stage2: frozen encoder + CLS head")
w("    • SupCon finetune— Stage1+2 above; Stage3: entire network unfrozen and fine-tuned")
w()

# ── 3. Per-class F1 table
w("── 3. PER-CLASS F1 SCORES ───────────────────────────────────────────────────────")
col_w = 12
header = (f"  {'Code':<4} {'Disease':<28}"
          + "".join(f"{n:>{col_w}}" for n in model_names)
          + f"  {'AvgF1':>7}  Winner")
w(header)
w(f"  {'-'*(len(header)-2)}")
for code in LABEL_COLS:
    dname = CLASS_NAMES[code]
    f1s = {n: results[n]["per_class"].get(code, {}).get("F1", float("nan"))
           for n in model_names}
    row = f"  {code:<4} {dname:<28}"
    winner_names = [n for n, v in f1s.items()
                    if not np.isnan(v) and abs(v - max(v2 for v2 in f1s.values()
                                                       if not np.isnan(v2))) < 1e-6]
    for n in model_names:
        v = f1s[n]
        star = "★" if n in winner_names else " "
        row += f"  {v:.4f}{star}" if not np.isnan(v) else f"  {'—':>6} "
    row += f"  {avg_f1[code]:>7.4f}  {class_winners[code]}"
    w(row)
w()

# ── 4. Win counts
w("── 4. CLASS-LEVEL WIN COUNT ─────────────────────────────────────────────────────")
for name in model_names:
    wins = sum(1 for code in LABEL_COLS if class_winners[code] == name)
    bar = "█" * wins
    w(f"  {name:<22}  {wins:>2}/{len(LABEL_COLS)}  {bar}")
w()

# ── 5. Pairwise macro F1 comparison
w("── 5. PAIRWISE MACRO F1 COMPARISON ─────────────────────────────────────────────")
w(f"  {'Model A':<22}  {'Model B':<22}  {'Δ F1':>8}  Better")
w(f"  {'-'*72}")
for a, b in combinations(ranking, 2):
    delta = results[a]["macro"]["F1"] - results[b]["macro"]["F1"]
    better = a if delta > 0 else b
    w(f"  {a:<22}  {b:<22}  {delta:>+8.4f}  {better}")
w()

# ── 6. Best model per-class gaps vs all others
w(f"── 6. BEST MODEL ({best}) — PER-CLASS F1 GAPS vs ALL OTHERS ───────────────────")
others_ranked = ranking[1:]
header6 = (f"  {'Code':<4} {'Disease':<28} {best:>12}"
           + "".join(f"  {'Δ vs '+n:>15}" for n in others_ranked))
w(header6)
w(f"  {'-'*( len(header6)-2 )}")
for code in LABEL_COLS:
    fb = results[best]["per_class"].get(code, {}).get("F1", float("nan"))
    row = f"  {code:<4} {CLASS_NAMES[code]:<28} {fb:>12.4f}"
    for other in others_ranked:
        fo = results[other]["per_class"].get(code, {}).get("F1", float("nan"))
        if not (np.isnan(fb) or np.isnan(fo)):
            d = fb - fo
            row += f"  {'+' if d >= 0 else ''}{d:>14.4f}"
        else:
            row += f"  {'—':>15}"
    w(row)
w()

# ── 7. Hardest classes
w("── 7. HARDEST CLASSES (lowest average F1 across all models) ────────────────────")
hardest = sorted(LABEL_COLS, key=lambda c: avg_f1[c])
w(f"  {'Code':<4} {'Disease':<28} {'Avg F1':>8}  {'Min F1':>8}  {'Max F1':>8}  {'Support':>8}")
w(f"  {'-'*72}")
for code in hardest:
    vals = [results[n]["per_class"].get(code, {}).get("F1", float("nan")) for n in model_names]
    valid = [v for v in vals if not np.isnan(v)]
    sup = results[model_names[0]]["per_class"].get(code, {}).get("support", "—")
    w(f"  {code:<4} {CLASS_NAMES[code]:<28}"
      f" {avg_f1[code]:>8.4f}  {min(valid):>8.4f}  {max(valid):>8.4f}  {sup:>8}")
w()

# ── 8. Easiest classes
w("── 8. EASIEST CLASSES (highest average F1 across all models) ───────────────────")
easiest = sorted(LABEL_COLS, key=lambda c: avg_f1[c], reverse=True)
w(f"  {'Code':<4} {'Disease':<28} {'Avg F1':>8}  {'Support':>8}")
w(f"  {'-'*52}")
for code in easiest[:10]:
    sup = results[model_names[0]]["per_class"].get(code, {}).get("support", "—")
    w(f"  {code:<4} {CLASS_NAMES[code]:<28} {avg_f1[code]:>8.4f}  {sup:>8}")
w()

# ── 9. SupCon: freeze vs unfreeze comparison
if "ResNext_V1" in results and "ResNext_V2" in results:
    w("── 9. ENCODER FREEZE vs UNFREEZE (ResNext_V1 vs ResNext_V2) ────────────────────")
    v1_f1 = results["ResNext_V1"]["macro"]["F1"]
    v2_f1 = results["ResNext_V2"]["macro"]["F1"]
    delta_ft = v2_f1 - v1_f1
    w(f"  ResNext_V1 (2-stage, frozen encoder) Macro F1 = {v1_f1:.4f}")
    w(f"  ResNext_V2 (3-stage, full finetune)  Macro F1 = {v2_f1:.4f}")
    w(f"  Effect of encoder fine-tuning: {'+' if delta_ft >= 0 else ''}{delta_ft:.4f}"
      f"  ({'improvement' if delta_ft > 0.001 else 'regression' if delta_ft < -0.001 else 'negligible change'})")
    w()
    w(f"  {'Code':<4} {'Disease':<28} {'V1 F1':>8} {'V2 F1':>8} {'Δ(V2-V1)':>10}  Better")
    w(f"  {'-'*66}")
    improved = 0
    regressed = 0
    for code in LABEL_COLS:
        f1_v1 = results["ResNext_V1"]["per_class"].get(code, {}).get("F1", float("nan"))
        f1_v2 = results["ResNext_V2"]["per_class"].get(code, {}).get("F1", float("nan"))
        if not (np.isnan(f1_v1) or np.isnan(f1_v2)):
            d = f1_v2 - f1_v1
            better = "ResNext_V2" if d > 0.001 else ("ResNext_V1" if d < -0.001 else "~Tie")
            if d > 0.001: improved += 1
            if d < -0.001: regressed += 1
            w(f"  {code:<4} {CLASS_NAMES[code]:<28} {f1_v1:>8.4f} {f1_v2:>8.4f}"
              f" {'+' if d >= 0 else ''}{d:>9.4f}  {better}")
    w()
    w(f"  Classes improved by finetune : {improved}/{len(LABEL_COLS)}")
    w(f"  Classes regressed by finetune: {regressed}/{len(LABEL_COLS)}")
    w()

# ── 10. Efficiency analysis
w("── 10. EFFICIENCY ANALYSIS (Parameters vs Macro F1) ───────────────────────────")
w(f"  {'Model':<22} {'Params (M)':>12} {'Macro F1':>10} {'F1 per 10M params':>20}")
w(f"  {'-'*70}")
for name in ranking:
    params = MODEL_PARAMS_M[name]
    f1 = results[name]["macro"]["F1"]
    eff = f1 / params * 10
    w(f"  {name:<22} {params:>12.1f} {f1:>10.4f} {eff:>20.4f}")
w()
w("  Note: SupCon models have additional SupCon pre-training compute not reflected above.")
w()

w(SEP)
w("  END OF REPORT")
w(SEP)

with open(os.path.join(OUT_DIR, "comparison_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"  Saved → {os.path.join(OUT_DIR, 'comparison_report.txt')}")


# ── Plot 01 — Macro metrics grouped bar chart ─────────────────────────────────
print("Plotting 01_macro_metrics …")
metrics_keys = ["P", "R", "F1"]
metric_labels = ["Macro Precision", "Macro Recall", "Macro F1"]
x = np.arange(len(metrics_keys))
width = 0.7 / n_models

fig, ax = plt.subplots(figsize=(13, 6))
for i, name in enumerate(model_names):
    offset = (i - n_models / 2 + 0.5) * width
    vals = [results[name]["macro"].get(k, 0) for k in metrics_keys]
    bars = ax.bar(x + offset, vals, width,
                  label=name, color=MODEL_COLORS[name],
                  edgecolor="white", linewidth=0.6, alpha=0.9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold", color=MODEL_COLORS[name])

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 0.98)
ax.set_title("Macro Metrics Comparison — All Models (Test Set, n=1,262)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

rank_text = "Rankings by Macro F1:\n" + "\n".join(
    f"  #{r+1} {n}  ({results[n]['macro']['F1']:.4f})"
    for r, n in enumerate(ranking)
)
ax.text(0.99, 0.98, rank_text,
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="grey", alpha=0.9))

plt.tight_layout()
savefig("01_macro_metrics.png")


# ── Plot 02 — Per-class F1 grouped bar chart ──────────────────────────────────
print("Plotting 02_perclass_f1_bars …")
x = np.arange(len(LABEL_COLS))
width = 0.7 / n_models

fig, ax = plt.subplots(figsize=(22, 6))
for i, name in enumerate(model_names):
    offset = (i - n_models / 2 + 0.5) * width
    f1s = [results[name]["per_class"].get(code, {}).get("F1", 0) for code in LABEL_COLS]
    ax.bar(x + offset, f1s, width,
           label=name, color=MODEL_COLORS[name],
           edgecolor="white", linewidth=0.3, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels([CLASS_NAMES[c] for c in LABEL_COLS],
                    rotation=42, ha="right", fontsize=8.5)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_ylim(0, 1.18)
ax.set_title("Per-Class F1 Score — All Models", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9)
ax.axhline(0.5, color="gray",      linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(0.7, color="steelblue", linestyle=":",  linewidth=0.8, alpha=0.5)
ax.text(len(LABEL_COLS) - 0.5, 0.51, "F1=0.5", fontsize=7, color="gray",      va="bottom")
ax.text(len(LABEL_COLS) - 0.5, 0.71, "F1=0.7", fontsize=7, color="steelblue", va="bottom")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("02_perclass_f1_bars.png")


# ── Plot 03 — Per-class F1 heatmap ────────────────────────────────────────────
print("Plotting 03_perclass_f1_heatmap …")
data_matrix = np.array([
    [results[name]["per_class"].get(code, {}).get("F1", np.nan)
     for code in LABEL_COLS]
    for name in model_names
])

fig, ax = plt.subplots(figsize=(20, 5.5))
im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

for r in range(n_models):
    for c in range(len(LABEL_COLS)):
        val = data_matrix[r, c]
        if not np.isnan(val):
            col_best = np.nanmax(data_matrix[:, c])
            weight = "bold" if abs(val - col_best) < 1e-6 else "normal"
            text_color = "black" if 0.28 < val < 0.78 else "white"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, fontweight=weight, color=text_color)

ax.set_xticks(range(len(LABEL_COLS)))
ax.set_xticklabels([CLASS_NAMES[c] for c in LABEL_COLS],
                    rotation=40, ha="right", fontsize=8.5)
ax.set_yticks(range(n_models))
ax.set_yticklabels(model_names, fontsize=10)
ax.set_title("Per-Class F1 Heatmap  (bold = best per class)",
             fontsize=13, fontweight="bold", pad=12)
plt.colorbar(im, ax=ax, label="F1 Score", shrink=0.85, pad=0.01)
plt.tight_layout()
savefig("03_perclass_f1_heatmap.png")


# ── Plot 04 — Radar chart ─────────────────────────────────────────────────────
print("Plotting 04_radar_chart …")
radar_codes  = ["C0", "DR", "C6", "C7", "C8", "C11", "C13", "C14", "C15", "C22", "C25", "C32"]
radar_labels = [CLASS_NAMES[c] for c in radar_codes]
N = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(30)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=9)
ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

for name in model_names:
    vals = [results[name]["per_class"].get(c, {}).get("F1", 0) for c in radar_codes]
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, linestyle="solid",
            color=MODEL_COLORS[name], label=name)
    ax.fill(angles, vals, alpha=0.07, color=MODEL_COLORS[name])

ax.set_title("Per-Class F1 Radar Chart (12 representative classes)",
             fontsize=13, fontweight="bold", y=1.1)
ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.18), fontsize=9)
plt.tight_layout()
savefig("04_radar_chart.png")


# ── Plot 05 — Win / Tie / Loss matrix ─────────────────────────────────────────
print("Plotting 05_win_loss …")
win_mat  = np.zeros((n_models, n_models), dtype=int)
tie_mat  = np.zeros((n_models, n_models), dtype=int)
loss_mat = np.zeros((n_models, n_models), dtype=int)

for code in LABEL_COLS:
    f1s = [results[name]["per_class"].get(code, {}).get("F1", 0.0)
           for name in model_names]
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                continue
            if abs(f1s[i] - f1s[j]) < 1e-6:
                tie_mat[i][j] += 1
            elif f1s[i] > f1s[j]:
                win_mat[i][j] += 1
            else:
                loss_mat[i][j] += 1

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
titles    = ["Wins",    "Ties",    "Losses"]
subtitles = ["row > col", "row ≈ col", "row < col"]
cmaps     = ["Greens",  "Oranges", "Reds"]
mats      = [win_mat,   tie_mat,   loss_mat]

for ax, mat, title, subtitle, cmap in zip(axes, mats, titles, subtitles, cmaps):
    sns.heatmap(mat, annot=True, fmt="d", cmap=cmap,
                xticklabels=model_names, yticklabels=model_names,
                ax=ax, linewidths=0.5, linecolor="white",
                annot_kws={"size": 11, "weight": "bold"},
                cbar=False, vmin=0, vmax=len(LABEL_COLS))
    ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight="bold", pad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xlabel("Opponent model", fontsize=9)
    ax.set_ylabel("Model (row)", fontsize=9)

fig.suptitle("Head-to-Head Per-Class Win / Tie / Loss  (19 classes)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
savefig("05_win_loss.png")


# ── Plot 06 — F1 delta: best vs each baseline ─────────────────────────────────
print("Plotting 06_f1_delta …")
others    = ranking[1:]
n_others  = len(others)
ncols     = min(n_others, 2)
nrows     = (n_others + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(7 * ncols, 6.5 * nrows),
                          sharey=True)
axes = np.array(axes).flatten()

for idx, other in enumerate(others):
    ax = axes[idx]
    deltas = [
        results[best]["per_class"].get(code, {}).get("F1", 0)
        - results[other]["per_class"].get(code, {}).get("F1", 0)
        for code in LABEL_COLS
    ]
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
    y_pos  = np.arange(len(LABEL_COLS))
    ax.barh(y_pos, deltas, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([CLASS_NAMES[c] for c in LABEL_COLS], fontsize=8.5)
    ax.set_xlabel("F1 Delta", fontsize=10)
    ax.set_title(f"{best}  vs  {other}", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    wins  = sum(1 for d in deltas if d > 0)
    loses = sum(1 for d in deltas if d < 0)
    ax.text(0.98, 0.02,
            f"↑ {wins} classes  ↓ {loses} classes",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9))

for idx in range(n_others, len(axes)):
    axes[idx].set_visible(False)

pos_patch = mpatches.Patch(color="#2ecc71", label=f"{best} better")
neg_patch = mpatches.Patch(color="#e74c3c", label="Baseline better")
fig.legend(handles=[pos_patch, neg_patch], loc="lower center",
           ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.03))
fig.suptitle(f"Per-Class F1 Delta  ({best} vs. each baseline)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
savefig("06_f1_delta.png")


# ── Plot 07 — Training curves ─────────────────────────────────────────────────
print("Plotting 07_training_curves …")
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Left panel: end-to-end models
ax = axes[0]
has_end2end = False
for name in ["RESNET_V1", "CNN_V1", "SWIN_V1"]:
    if name not in results:
        continue
    c = results[name]["curves"]
    if "val_f1" in c:
        epochs, f1s = c["val_f1"]
        ax.plot(epochs, f1s, color=MODEL_COLORS[name], linewidth=1.8,
                label=f"{name}  (final val F1={f1s[-1]:.4f})", alpha=0.9)
        ax.scatter([epochs[-1]], [f1s[-1]], color=MODEL_COLORS[name], s=60, zorder=5)
        has_end2end = True
if has_end2end:
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Macro F1", fontsize=11)
    ax.set_title("End-to-End Models — Val Macro F1 per Epoch", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
else:
    ax.text(0.5, 0.5, "No training log available\nfor end-to-end models",
            ha="center", va="center", transform=ax.transAxes, fontsize=11, color="grey")
    ax.set_title("End-to-End Models — Training Curves", fontsize=11, fontweight="bold")

# Right panel: SupCon models
ax = axes[1]
for name in ["ResNext_V1", "ResNext_V2"]:
    if name not in results:
        continue
    c = results[name]["curves"]
    color = MODEL_COLORS[name]
    cls_len = 0
    if "cls_f1" in c:
        epochs, f1s = c["cls_f1"]
        cls_len = len(epochs)
        ax.plot(epochs, f1s, color=color, linewidth=1.8, linestyle="-",
                label=f"{name} — CLS stage", alpha=0.9)
    if "ft_f1" in c:
        epochs_ft, f1s_ft = c["ft_f1"]
        global_epochs = [cls_len + e for e in epochs_ft]
        ax.plot(global_epochs, f1s_ft, color=color, linewidth=2, linestyle="--",
                label=f"{name} — Finetune stage", alpha=0.9)
        ax.scatter([global_epochs[-1]], [f1s_ft[-1]], color=color, s=80,
                   marker="*", zorder=5)
        if cls_len > 0:
            ax.axvline(cls_len + 0.5, color=color, linestyle=":", linewidth=1, alpha=0.35)

ax.set_xlabel("Epoch (CLS stage + Finetune stage offset)", fontsize=11)
ax.set_ylabel("Validation Macro F1", fontsize=11)
ax.set_title("SupCon Models — Classifier & Finetune Val F1", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_axisbelow(True)

fig.suptitle("Training Curves Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("07_training_curves.png")


# ── Plot 08 — Class support vs average F1 ────────────────────────────────────
print("Plotting 08_support_vs_f1 …")
supports    = [results[model_names[0]]["per_class"].get(c, {}).get("support", 0)
               for c in LABEL_COLS]
avg_f1_vals = [avg_f1[c] for c in LABEL_COLS]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Left: scatter support vs avg F1
ax = axes[0]
sc = ax.scatter(supports, avg_f1_vals, c=avg_f1_vals,
                cmap="RdYlGn", vmin=0.2, vmax=1.0,
                s=110, edgecolors="grey", linewidths=0.5, zorder=3)
for i, code in enumerate(LABEL_COLS):
    ax.annotate(code, (supports[i], avg_f1_vals[i]),
                textcoords="offset points", xytext=(5, 4), fontsize=7.5, alpha=0.85)
ax.set_xlabel("Class Support (test set instances)", fontsize=11)
ax.set_ylabel("Average F1 (across all models)", fontsize=11)
ax.set_title("Class Support vs. Average F1\n(colour = avg F1)", fontsize=11, fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.grid(alpha=0.3)
ax.set_axisbelow(True)
plt.colorbar(sc, ax=ax, label="Avg F1", shrink=0.85)

# Right: class imbalance bar chart sorted by support
ax = axes[1]
sorted_by_sup = sorted(LABEL_COLS,
                       key=lambda c: results[model_names[0]]["per_class"].get(c, {}).get("support", 0),
                       reverse=True)
sups_sorted = [results[model_names[0]]["per_class"].get(c, {}).get("support", 0) for c in sorted_by_sup]
avgs_sorted = [avg_f1[c] for c in sorted_by_sup]
bar_colors  = plt.cm.RdYlGn(np.array(avgs_sorted))
ax.bar(range(len(sorted_by_sup)), sups_sorted, color=bar_colors,
       edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(sorted_by_sup)))
ax.set_xticklabels([CLASS_NAMES[c] for c in sorted_by_sup],
                    rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Support (# test instances)", fontsize=11)
ax.set_title("Class Imbalance (sorted by support)\nBar colour = average F1", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

fig.suptitle("Class Distribution & Difficulty Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("08_support_vs_f1.png")


# ── Plot 09 — Precision vs Recall per class scatter ──────────────────────────
print("Plotting 09_precision_recall_scatter …")

# Build subplot grid: one per model + one combined overlay
n_subplots = n_models + 1
ncols_pr   = 3
nrows_pr   = (n_subplots + ncols_pr - 1) // ncols_pr

fig, axes = plt.subplots(nrows_pr, ncols_pr, figsize=(19, 6.5 * nrows_pr))
axes = np.array(axes).flatten()

# Iso-F1 helper
iso_vals = [0.3, 0.5, 0.7, 0.9]

def draw_iso_f1(ax, iso_list):
    t = np.linspace(0.01, 0.99, 300)
    for iso in iso_list:
        p_iso = iso * t / (2 * t - iso)
        p_iso[p_iso < 0] = np.nan
        p_iso[p_iso > 1] = np.nan
        ax.plot(t, p_iso, ":", color="grey", linewidth=0.6, alpha=0.45, zorder=1)
        valid = ~np.isnan(p_iso)
        if valid.any():
            mid = np.searchsorted(t, 0.55)
            if valid[mid]:
                ax.text(t[mid], p_iso[mid] + 0.025, f"F1={iso}",
                        fontsize=6.5, color="grey", alpha=0.7)

for idx, name in enumerate(model_names):
    ax = axes[idx]
    ps  = [results[name]["per_class"].get(c, {}).get("P", np.nan) for c in LABEL_COLS]
    rs  = [results[name]["per_class"].get(c, {}).get("R", np.nan) for c in LABEL_COLS]
    f1s = [results[name]["per_class"].get(c, {}).get("F1", np.nan) for c in LABEL_COLS]
    sc = ax.scatter(rs, ps, c=f1s, cmap="RdYlGn", vmin=0.0, vmax=1.0,
                    s=85, edgecolors="grey", linewidths=0.4, zorder=4)
    for i, code in enumerate(LABEL_COLS):
        if not (np.isnan(ps[i]) or np.isnan(rs[i])):
            ax.annotate(code, (rs[i], ps[i]),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=7, alpha=0.85)
    draw_iso_f1(ax, iso_vals)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.6, alpha=0.4)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(name, fontsize=11, fontweight="bold", color=MODEL_COLORS[name])
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    plt.colorbar(sc, ax=ax, label="F1", shrink=0.85)

# Overlay subplot
ax = axes[n_models]
for name in model_names:
    ps  = [results[name]["per_class"].get(c, {}).get("P", np.nan) for c in LABEL_COLS]
    rs  = [results[name]["per_class"].get(c, {}).get("R", np.nan) for c in LABEL_COLS]
    ax.scatter(rs, ps, c=MODEL_COLORS[name], s=40, alpha=0.55,
               label=name, edgecolors="none", zorder=3)
    mp = results[name]["macro"].get("P", np.nan)
    mr = results[name]["macro"].get("R", np.nan)
    ax.scatter([mr], [mp], c=MODEL_COLORS[name], s=220, marker="*",
               edgecolors="black", linewidths=0.7, zorder=6)

draw_iso_f1(ax, iso_vals)
ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.6, alpha=0.4)
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(-0.05, 1.15)
ax.set_xlabel("Recall", fontsize=9)
ax.set_ylabel("Precision", fontsize=9)
ax.set_title("All Models Overlay\n(★ = macro P/R)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
ax.grid(alpha=0.25)
ax.set_axisbelow(True)

for idx in range(n_models + 1, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle("Precision vs Recall per Class  (with iso-F1 curves)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
savefig("09_precision_recall_scatter.png")


# ── Console summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  FINAL RANKING")
print("=" * 70)
for i, name in enumerate(ranking):
    m = results[name]["macro"]
    print(f"  #{i+1:<3} {name:<22}  "
          f"P={m['P']:.4f}  R={m['R']:.4f}  F1={m['F1']:.4f}")
print()
print(f"  Best model : {ranking[0]}")
print(f"  Best F1    : {results[ranking[0]]['macro']['F1']:.4f}")
wins_best = sum(1 for code in LABEL_COLS if class_winners[code] == ranking[0])
print(f"  Class wins : {wins_best}/{len(LABEL_COLS)}")
print()
print(f"All outputs → {os.path.abspath(OUT_DIR)}/")
