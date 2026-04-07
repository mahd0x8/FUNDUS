"""
compare_models.py — Cross-model evaluation comparison for the FUNDUS project.

Reads existing evaluation text files from each experiment folder and
produces a full comparison report + 6 visualisation plots.

Run from the project root:
    python EXPERIMENTS/compare_models.py

Outputs (EXPERIMENTS/COMPARISON/):
    comparison_report.txt        — full text comparison with rankings
    01_macro_metrics.png         — Macro P / R / F1 grouped bar chart
    02_perclass_f1_bars.png      — per-class F1 grouped bar chart
    03_perclass_f1_heatmap.png   — per-class F1 heatmap (models × classes)
    04_radar_chart.png           — radar / spider chart for macro metrics
    05_win_loss.png              — class-level win / tie / loss matrix
    06_f1_delta.png              — F1 improvement of best model over each baseline
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
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
    "C0": "Normal", "C1": "AMD", "DR": "Diabetic Retinopathy",
    "C6": "Glaucoma", "C7": "Hypertensive Retinopathy",
    "C8": "Pathological Myopia", "C9": "Tessellated Fundus",
    "C10": "Vitreous Degeneration", "C11": "BRVO",
    "C13": "Large Optic Cup", "C14": "Drusen",
    "C15": "Epiretinal Membrane", "C18": "Optic Disc Edema",
    "C19": "Myelinated Nerve Fibers", "C22": "Retinal Detachment",
    "C25": "Refractive Media Opacity", "C27": "CSC",
    "C29": "Laser Spots", "C32": "CRVO",
}
SHORT_NAMES = [CLASS_NAMES[c] for c in LABEL_COLS]

# Each entry: (display_name, eval_file_path, architecture_details)
MODELS = [
    (
        "SWIN_V1",
        "EXPERIMENTS/SWIN_V1/swin_evaluation.txt",
        "Swin-Large  |  195.8M params  |  100 epochs  |  ImageNet-22k pretrained",
    ),
    (
        "V1 (SupCon)",
        "EXPERIMENTS/V1/training_evaluation.txt",
        "ResNeXt50 + SupCon  |  ~25M params  |  3-stage (100+30+20 epochs)",
    ),
    (
        "CNN_V1",
        "EXPERIMENTS/CNN_V1/cnn_evaluation.txt",
        "ConvNeXt-Base  |  ~87M params  |  50 epochs  |  ImageNet-1k pretrained",
    ),
]

MODEL_COLORS = {
    "SWIN_V1":    "#2ecc71",
    "V1 (SupCon)": "#3498db",
    "CNN_V1":     "#e67e22",
}

# ── Parser ────────────────────────────────────────────────────────────────────

def parse_eval_file(path: str) -> dict:
    """
    Parse any of the three evaluation file formats and return a unified dict:
        per_class: {code: {P, R, F1, support}}
        macro:     {P, R, F1}
    """
    with open(path) as f:
        text = f.read()

    per_class = {}

    # ── Format A: "  C0  Normal  ...  P=0.xxx  R=0.xxx  F1=0.xxx  S=nnn"
    #    (training_evaluation.txt — V1 SupCon)
    fmt_a = re.findall(
        r"^\s*(\w+)\s+\([^)]+\)\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)\s+S=(\d+)",
        text, re.MULTILINE
    )
    if fmt_a:
        for code, p, r, f1, s in fmt_a:
            if code in LABEL_COLS:
                per_class[code] = dict(P=float(p), R=float(r), F1=float(f1), support=int(s))

    # ── Format B: table with Thr column
    #    "  C0  Normal ...  0.10  0.7013  0.8869  0.7832  442"
    #    (swin_evaluation.txt, cnn_evaluation.txt)
    if not per_class:
        fmt_b = re.findall(
            r"^\s*(\w+)\s+\S[^\n]*?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)",
            text, re.MULTILINE
        )
        for row in fmt_b:
            code = row[0]
            if code in LABEL_COLS:
                # columns: code name thr P R F1 support
                per_class[code] = dict(
                    thr=float(row[1]), P=float(row[2]),
                    R=float(row[3]), F1=float(row[4]), support=int(row[5])
                )

    # Macro
    macro_match = re.search(
        r"Macro\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)", text
    )
    if not macro_match:
        macro_match = re.search(
            r"Precision:\s*([\d.]+)\s*\nRecall:\s*([\d.]+)\s*\nF1:\s*([\d.]+)", text
        )
    macro = {}
    if macro_match:
        macro = dict(
            P=float(macro_match.group(1)),
            R=float(macro_match.group(2)),
            F1=float(macro_match.group(3)),
        )

    return {"per_class": per_class, "macro": macro}


# ── Load all models ───────────────────────────────────────────────────────────

print("Parsing evaluation files …")
results = {}
for name, path, arch in MODELS:
    data = parse_eval_file(path)
    data["arch"] = arch
    results[name] = data
    mf1 = data["macro"].get("F1", float("nan"))
    print(f"  {name:<18}  macro F1 = {mf1:.4f}  ({len(data['per_class'])} classes parsed)")

model_names = [m[0] for m in MODELS]

# ── Shared helpers ────────────────────────────────────────────────────────────

def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
})

# ── Text report ───────────────────────────────────────────────────────────────

print("Writing comparison_report.txt …")

lines = []
def w(*args): lines.append(" ".join(str(a) for a in args))

w("=" * 72)
w("  FUNDUS MODEL COMPARISON REPORT")
w("=" * 72)
w()

# 1. Ranking table
ranking = sorted(model_names, key=lambda n: results[n]["macro"]["F1"], reverse=True)

w("── 1. OVERALL MACRO RANKING (test set, n=1,262) ─────────────────────────")
w(f"  {'Rank':<5} {'Model':<18} {'Macro P':>9} {'Macro R':>9} {'Macro F1':>10}  Architecture")
w(f"  {'-'*80}")
for rank, name in enumerate(ranking, 1):
    m = results[name]["macro"]
    arch_short = results[name]["arch"].split("|")[0].strip()
    w(f"  {rank:<5} {name:<18} {m['P']:>9.4f} {m['R']:>9.4f} {m['F1']:>10.4f}  {arch_short}")
w()
best = ranking[0]
w(f"  ★  Best model overall: {best}  (Macro F1 = {results[best]['macro']['F1']:.4f})")
w()

# 2. Architecture details
w("── 2. ARCHITECTURE DETAILS ──────────────────────────────────────────────")
for name, _, arch in MODELS:
    w(f"  {name}")
    for part in arch.split("|"):
        w(f"    • {part.strip()}")
    w()

# 3. Per-class F1 comparison
w("── 3. PER-CLASS F1 SCORES ───────────────────────────────────────────────")
col_w = 10
header = f"  {'Code':<4} {'Disease':<28}" + "".join(f"{n:>{col_w}}" for n in model_names) + "  Winner"
w(header)
w(f"  {'-'*( len(header) - 2 )}")

class_winners = {}
for code in LABEL_COLS:
    name = CLASS_NAMES[code]
    f1s  = {n: results[n]["per_class"].get(code, {}).get("F1", float("nan"))
            for n in model_names}
    best_f1   = max(v for v in f1s.values() if not np.isnan(v))
    winner    = [n for n, v in f1s.items() if abs(v - best_f1) < 1e-6]
    class_winners[code] = winner[0] if len(winner) == 1 else "Tie"

    row = f"  {code:<4} {name:<28}"
    for n in model_names:
        v = f1s[n]
        mark = " ★" if n in winner else "  "
        row += f"  {v:.4f}{mark}" if not np.isnan(v) else f"  {'—':>8}"
    row += f"  {class_winners[code]}"
    w(row)
w()

# 4. Win counts
w("── 4. CLASS-LEVEL WIN COUNT ─────────────────────────────────────────────")
for name in model_names:
    wins = sum(1 for code in LABEL_COLS if class_winners[code] == name)
    w(f"  {name:<18}  wins on {wins}/{len(LABEL_COLS)} classes")
w()

# 5. Biggest improvements of best over 2nd
second = ranking[1]
w(f"── 5. SWIN_V1 vs {second} — LARGEST F1 GAPS ──────────────────────────")
gaps = []
for code in LABEL_COLS:
    f1_best   = results[best]["per_class"].get(code, {}).get("F1", float("nan"))
    f1_second = results[second]["per_class"].get(code, {}).get("F1", float("nan"))
    if not (np.isnan(f1_best) or np.isnan(f1_second)):
        gaps.append((code, CLASS_NAMES[code], f1_best - f1_second, f1_best, f1_second))

gaps.sort(key=lambda x: -abs(x[2]))
w(f"  {'Code':<4} {'Disease':<28} {best:>10} {second:>14}  {'Δ F1':>8}")
w(f"  {'-'*72}")
for code, name, delta, fb, fs in gaps:
    sign = "+" if delta >= 0 else ""
    w(f"  {code:<4} {name:<28} {fb:>10.4f} {fs:>14.4f}  {sign}{delta:>7.4f}")
w()

# 6. Hardest classes (lowest avg F1 across models)
w("── 6. HARDEST CLASSES (lowest average F1 across all models) ─────────────")
avg_f1 = {}
for code in LABEL_COLS:
    vals = [results[n]["per_class"].get(code, {}).get("F1", float("nan"))
            for n in model_names]
    valid = [v for v in vals if not np.isnan(v)]
    avg_f1[code] = np.mean(valid) if valid else float("nan")

hardest = sorted(avg_f1, key=lambda c: avg_f1[c])
w(f"  {'Code':<4} {'Disease':<28} {'Avg F1':>8}  {'Support':>8}")
w(f"  {'-'*54}")
for code in hardest:
    sup = results[model_names[0]]["per_class"].get(code, {}).get("support", "—")
    w(f"  {code:<4} {CLASS_NAMES[code]:<28} {avg_f1[code]:>8.4f}  {sup:>8}")
w()

w("=" * 72)
w("  END OF REPORT")
w("=" * 72)

with open(os.path.join(OUT_DIR, "comparison_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"  Saved → {os.path.join(OUT_DIR, 'comparison_report.txt')}")


# ── Plot 01 — Macro metrics bar chart ─────────────────────────────────────────
print("Plotting 01_macro_metrics …")
metrics_keys = ["P", "R", "F1"]
metric_labels = ["Macro Precision", "Macro Recall", "Macro F1"]
x = np.arange(len(metrics_keys))
width = 0.22

fig, ax = plt.subplots(figsize=(11, 6))
for i, name in enumerate(model_names):
    vals = [results[name]["macro"][k] for k in metrics_keys]
    bars = ax.bar(x + (i - 1) * width, vals, width,
                  label=name, color=MODEL_COLORS[name],
                  edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                color=MODEL_COLORS[name])

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 0.92)
ax.set_title("Macro Metrics Comparison — All Models (Test Set, n=1,262)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

# Add rank badges
for i, name in enumerate(ranking):
    rank_label = ["1st", "2nd", "3rd"][i]
    f1 = results[name]["macro"]["F1"]
    ax.annotate(f"#{rank_label}  {name}\nF1={f1:.4f}",
                xy=(0.67 + i * 0.115, 0.87 - i * 0.07),
                xycoords="axes fraction",
                fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc=MODEL_COLORS[name], alpha=0.15, ec=MODEL_COLORS[name]))

plt.tight_layout()
savefig("01_macro_metrics.png")


# ── Plot 02 — Per-class F1 grouped bar chart ──────────────────────────────────
print("Plotting 02_perclass_f1_bars …")
x = np.arange(len(LABEL_COLS))
width = 0.26

fig, ax = plt.subplots(figsize=(18, 6))
for i, name in enumerate(model_names):
    f1s = [results[name]["per_class"].get(code, {}).get("F1", 0)
           for code in LABEL_COLS]
    ax.bar(x + (i - 1) * width, f1s, width,
           label=name, color=MODEL_COLORS[name],
           edgecolor="white", linewidth=0.4, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels([CLASS_NAMES[c] for c in LABEL_COLS],
                    rotation=42, ha="right", fontsize=8.5)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_title("Per-Class F1 Score — All Models", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="F1=0.5 ref")
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

fig, ax = plt.subplots(figsize=(16, 4.5))
im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

for r in range(len(model_names)):
    for c in range(len(LABEL_COLS)):
        val = data_matrix[r, c]
        if not np.isnan(val):
            # star best per column
            col_best = np.nanmax(data_matrix[:, c])
            weight = "bold" if abs(val - col_best) < 1e-6 else "normal"
            text_color = "black" if 0.35 < val < 0.75 else "white"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, fontweight=weight, color=text_color)

ax.set_xticks(range(len(LABEL_COLS)))
ax.set_xticklabels([CLASS_NAMES[c] for c in LABEL_COLS],
                    rotation=40, ha="right", fontsize=8.5)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names, fontsize=10)
ax.set_title("Per-Class F1 Score Heatmap  (bold = best per class)",
             fontsize=13, fontweight="bold", pad=12)
plt.colorbar(im, ax=ax, label="F1 Score", shrink=0.8, pad=0.02)
plt.tight_layout()
savefig("03_perclass_f1_heatmap.png")


# ── Plot 04 — Radar chart ─────────────────────────────────────────────────────
print("Plotting 04_radar_chart …")

# Use a subset of representative classes + macro metrics for the radar
radar_codes  = ["C0", "DR", "C6", "C7", "C8", "C11", "C13", "C14", "C15", "C22", "C25", "C32"]
radar_labels = [CLASS_NAMES[c] for c in radar_codes]
N = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close polygon

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
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
    ax.fill(angles, vals, alpha=0.08, color=MODEL_COLORS[name])

ax.set_title("Per-Class F1 Radar Chart\n(12 representative classes)",
             fontsize=13, fontweight="bold", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
plt.tight_layout()
savefig("04_radar_chart.png")


# ── Plot 05 — Win / Tie / Loss matrix ─────────────────────────────────────────
print("Plotting 05_win_loss …")
n = len(model_names)

# win_matrix[i][j] = number of classes where model i beats model j
win_mat  = np.zeros((n, n), dtype=int)
tie_mat  = np.zeros((n, n), dtype=int)
loss_mat = np.zeros((n, n), dtype=int)

for ci, code in enumerate(LABEL_COLS):
    f1s = [results[name]["per_class"].get(code, {}).get("F1", 0)
           for name in model_names]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(f1s[i] - f1s[j]) < 1e-6:
                tie_mat[i][j] += 1
            elif f1s[i] > f1s[j]:
                win_mat[i][j] += 1
            else:
                loss_mat[i][j] += 1

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
titles = ["Wins (classes where row > col)",
          "Ties (classes where row ≈ col)",
          "Losses (classes where row < col)"]
cmaps  = ["Greens", "Oranges", "Reds"]
mats   = [win_mat, tie_mat, loss_mat]

for ax, mat, title, cmap in zip(axes, mats, titles, cmaps):
    sns.heatmap(mat, annot=True, fmt="d", cmap=cmap,
                xticklabels=model_names, yticklabels=model_names,
                ax=ax, linewidths=0.5, linecolor="white",
                annot_kws={"size": 12, "weight": "bold"},
                cbar=False, vmin=0, vmax=len(LABEL_COLS))
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xlabel("Opponent", fontsize=9)
    ax.set_ylabel("Model", fontsize=9)

fig.suptitle("Head-to-Head Per-Class Win / Tie / Loss Matrix",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
savefig("05_win_loss.png")


# ── Plot 06 — F1 delta: best vs each baseline ─────────────────────────────────
print("Plotting 06_f1_delta …")
best_name  = ranking[0]
others     = ranking[1:]

fig, axes = plt.subplots(1, len(others), figsize=(16, 6), sharey=True)
if len(others) == 1:
    axes = [axes]

for ax, other in zip(axes, others):
    deltas = []
    for code in LABEL_COLS:
        fb = results[best_name]["per_class"].get(code, {}).get("F1", 0)
        fo = results[other]["per_class"].get(code, {}).get("F1", 0)
        deltas.append(fb - fo)

    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
    y_pos  = np.arange(len(LABEL_COLS))
    ax.barh(y_pos, deltas, color=colors, edgecolor="white", linewidth=0.4, height=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([CLASS_NAMES[c] for c in LABEL_COLS], fontsize=8.5)
    ax.set_xlabel("F1 Delta", fontsize=10)
    ax.set_title(f"{best_name}  vs  {other}", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Annotate net wins
    wins  = sum(1 for d in deltas if d > 0)
    loses = sum(1 for d in deltas if d < 0)
    ax.text(0.98, 0.02,
            f"↑ {wins} classes  ↓ {loses} classes",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="dimgrey",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

pos_patch = mpatches.Patch(color="#2ecc71", label=f"{best_name} better")
neg_patch = mpatches.Patch(color="#e74c3c", label=f"Baseline better")
fig.legend(handles=[pos_patch, neg_patch], loc="lower center",
           ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.04))
fig.suptitle(f"Per-Class F1 Delta  ({best_name} vs. each baseline)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
savefig("06_f1_delta.png")


# ── Print summary to console ──────────────────────────────────────────────────
print()
print("=" * 60)
print("  FINAL RANKING")
print("=" * 60)
ranks = ["#1", "#2", "#3"]
for i, name in enumerate(ranking):
    m = results[name]["macro"]
    print(f"  {ranks[i]}  {name:<18}  "
          f"P={m['P']:.4f}  R={m['R']:.4f}  F1={m['F1']:.4f}")
print()
print(f"  Best model : {ranking[0]}")
print(f"  Best F1    : {results[ranking[0]]['macro']['F1']:.4f}")
wins_best = sum(1 for code in LABEL_COLS if class_winners[code] == ranking[0])
print(f"  Class wins : {wins_best}/{len(LABEL_COLS)}")
print()
print(f"All outputs → {os.path.abspath(OUT_DIR)}/")
