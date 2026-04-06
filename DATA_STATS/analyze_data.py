"""
analyze_data.py — Comprehensive dataset analysis for the FUNDUS project.

Run from the project root:
    python DATA_STATS/analyze_data.py

Outputs (all written to DATA_STATS/):
    data_report.txt              — full text summary
    01_class_distribution.png    — total image count per disease
    02_split_class_counts.png    — per-disease counts broken down by split
    03_class_imbalance.png       — pos_weight (neg/pos ratio) per class
    04_labels_per_image.png      — histogram of label count per image
    05_source_split.png          — source dataset × split breakdown
    06_cooccurrence_heatmap.png  — 19×19 disease co-occurrence matrix
    07_top_pairs.png             — top 20 co-occurring disease pairs
    08_image_sizes.png           — width/height scatter of sampled images
    09_split_pie.png             — train / val / test proportion
    10_presence_absence.png      — stacked present/absent bar per class
"""

import os
import sys
import warnings
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH    = "DATASET/filtered_data_split.csv"
IMG_PREFIX  = ("DATASET/data/", "/data/")          # (local, csv-stored)
OUT_DIR     = "DATA_STATS"
REPORT_PATH = os.path.join(OUT_DIR, "data_report.txt")

# ── Label config ─────────────────────────────────────────────────────────────
LABEL_COLS = [
    "C0", "C1", "DR", "C6", "C7", "C8", "C9",
    "C10", "C11", "C13", "C14", "C15", "C18", "C19",
    "C22", "C25", "C27", "C29", "C32",
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

SHORT = {c: CLASS_NAMES[c] for c in LABEL_COLS}   # alias

SPLITS       = ["train", "val", "test"]
SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
DISEASE_PAL  = sns.color_palette("tab20", len(LABEL_COLS))

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def add_bar_labels(ax, fmt="{:.0f}", fontsize=8, color="black", pad=2):
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + pad,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize, color=color,
            )


def label_name(code: str) -> str:
    return CLASS_NAMES.get(code, code)


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading dataset …")
df = pd.read_csv(CSV_PATH)
df["image_path"] = df["image"].str.replace(IMG_PREFIX[1], IMG_PREFIX[0], regex=False)
df["source"]     = df["image"].str.extract(r"/eye_fundus/([^/]+)/")
df["ext"]        = df["image"].str.rsplit(".", n=1).str[-1].str.lower()
df["n_labels"]   = df[LABEL_COLS].sum(axis=1)

total  = len(df)
splits = {s: df[df["split"] == s] for s in SPLITS}

# Per-disease counts
disease_total = df[LABEL_COLS].sum().rename(index=CLASS_NAMES).sort_values(ascending=False)
disease_split = {s: splits[s][LABEL_COLS].sum().rename(index=CLASS_NAMES) for s in SPLITS}

# Imbalance ratio: neg/pos per class (same as pos_weight in training)
pos   = df[LABEL_COLS].sum()
neg   = total - pos
ratio = (neg / pos.clip(lower=1)).clip(upper=50).rename(index=CLASS_NAMES).sort_values(ascending=False)

# Co-occurrence
cooc = pd.DataFrame(0, index=LABEL_COLS, columns=LABEL_COLS)
for _, row in df[LABEL_COLS].iterrows():
    active = [c for c in LABEL_COLS if row[c] == 1]
    for a, b in combinations(active, 2):
        cooc.loc[a, b] += 1
        cooc.loc[b, a] += 1

pairs = {}
for a, b in combinations(LABEL_COLS, 2):
    v = cooc.loc[a, b]
    if v > 0:
        pairs[(a, b)] = v
top_pairs = sorted(pairs.items(), key=lambda x: -x[1])[:20]


# ── Text report ───────────────────────────────────────────────────────────────

print("Writing text report …")

lines = []
def w(*args):
    lines.append(" ".join(str(a) for a in args))

w("=" * 70)
w("  FUNDUS DATASET — COMPREHENSIVE DATA ANALYSIS REPORT")
w("=" * 70)
w()

# ── 1. Overview ──────────────────────────────────────────────────────────────
w("── 1. DATASET OVERVIEW ──────────────────────────────────────────────────")
w(f"  Total images          : {total:,}")
w(f"  Total disease labels  : {len(LABEL_COLS)}")
w(f"  Total label instances : {int(df[LABEL_COLS].sum().sum()):,}")
w(f"  Multi-label images    : {int((df['n_labels'] > 1).sum()):,}  "
  f"({(df['n_labels'] > 1).mean()*100:.1f}%)")
w(f"  Single-label images   : {int((df['n_labels'] == 1).sum()):,}  "
  f"({(df['n_labels'] == 1).mean()*100:.1f}%)")
w()

# ── 2. Split breakdown ────────────────────────────────────────────────────────
w("── 2. SPLIT BREAKDOWN ───────────────────────────────────────────────────")
w(f"  {'Split':<8} {'Images':>7}  {'%':>6}")
w(f"  {'-'*24}")
for s in SPLITS:
    n = len(splits[s])
    w(f"  {s:<8} {n:>7,}  {n/total*100:>5.1f}%")
w()

# ── 3. Source datasets ────────────────────────────────────────────────────────
w("── 3. SOURCE DATASETS ───────────────────────────────────────────────────")
src_counts = df["source"].value_counts()
for src, cnt in src_counts.items():
    w(f"  {src:<18} {cnt:>5,}  ({cnt/total*100:.1f}%)")
w()
w("  Source × Split breakdown:")
src_split = df.groupby(["source", "split"]).size().unstack(fill_value=0)
for col in SPLITS:
    if col not in src_split.columns:
        src_split[col] = 0
src_split = src_split[SPLITS]
w(f"  {'Source':<18} {'Train':>7} {'Val':>7} {'Test':>7}")
w(f"  {'-'*42}")
for src, row in src_split.iterrows():
    w(f"  {src:<18} {row['train']:>7,} {row['val']:>7,} {row['test']:>7,}")
w()

# ── 4. File formats ───────────────────────────────────────────────────────────
w("── 4. FILE FORMATS ──────────────────────────────────────────────────────")
for ext, cnt in df["ext"].value_counts().items():
    w(f"  .{ext:<6} {cnt:>6,}  ({cnt/total*100:.1f}%)")
w()

# ── 5. Image size stats ───────────────────────────────────────────────────────
w("── 5. IMAGE SIZE STATISTICS (300-image sample) ──────────────────────────")
sample_paths = df["image_path"].sample(300, random_state=42).tolist()
widths, heights, aspects = [], [], []
for p in sample_paths:
    img = cv2.imread(p)
    if img is None:
        continue
    h, ww = img.shape[:2]
    widths.append(ww); heights.append(h); aspects.append(ww / h)

w(f"  {'Metric':<12} {'Min':>8} {'Max':>8} {'Mean':>8} {'Median':>8}")
w(f"  {'-'*50}")
for label, vals, is_int in [("Width (px)", widths, True), ("Height (px)", heights, True), ("Aspect W/H", aspects, False)]:
    if is_int:
        w(f"  {label:<12} {min(vals):>8.0f} {max(vals):>8.0f} "
          f"{np.mean(vals):>8.0f} {np.median(vals):>8.0f}")
    else:
        w(f"  {label:<12} {min(vals):>8.2f} {max(vals):>8.2f} "
          f"{np.mean(vals):>8.2f} {np.median(vals):>8.2f}")
w()

# ── 6. Disease distribution ───────────────────────────────────────────────────
w("── 6. DISEASE CLASS DISTRIBUTION ────────────────────────────────────────")
w(f"  {'Code':<4} {'Disease':<28} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}  "
  f"{'Imbalance':>9}  {'% Present':>9}")
w(f"  {'-'*80}")
for col in LABEL_COLS:
    name = CLASS_NAMES[col]
    t    = int(df[col].sum())
    tr   = int(splits["train"][col].sum())
    va   = int(splits["val"][col].sum())
    te   = int(splits["test"][col].sum())
    imb  = float((total - t) / max(t, 1))
    pct  = t / total * 100
    w(f"  {col:<4} {name:<28} {t:>6,} {tr:>6,} {va:>6,} {te:>6,}  "
      f"{imb:>9.1f}x  {pct:>8.1f}%")
w()

# ── 7. Labels per image ───────────────────────────────────────────────────────
w("── 7. LABELS PER IMAGE ──────────────────────────────────────────────────")
vc = df["n_labels"].value_counts().sort_index()
for n, cnt in vc.items():
    w(f"  {int(n)} label(s): {cnt:>6,} images  ({cnt/total*100:.1f}%)")
w()

# ── 8. Top co-occurring pairs ─────────────────────────────────────────────────
w("── 8. TOP 20 CO-OCCURRING DISEASE PAIRS ─────────────────────────────────")
w(f"  {'#':<3} {'Disease A':<28} {'Disease B':<28} {'Count':>6}")
w(f"  {'-'*70}")
for i, ((a, b), cnt) in enumerate(top_pairs, 1):
    w(f"  {i:<3} {CLASS_NAMES[a]:<28} {CLASS_NAMES[b]:<28} {cnt:>6,}")
w()

# ── 9. Per-split per-disease detail ──────────────────────────────────────────
w("── 9. PER-SPLIT DISEASE PREVALENCE (% within split) ────────────────────")
w(f"  {'Disease':<28} {'Train %':>8} {'Val %':>8} {'Test %':>8}")
w(f"  {'-'*56}")
for col in LABEL_COLS:
    name = CLASS_NAMES[col]
    tr_pct = splits["train"][col].sum() / len(splits["train"]) * 100
    va_pct = splits["val"][col].sum()   / len(splits["val"])   * 100
    te_pct = splits["test"][col].sum()  / len(splits["test"])  * 100
    w(f"  {name:<28} {tr_pct:>8.1f} {va_pct:>8.1f} {te_pct:>8.1f}")
w()

w("=" * 70)
w("  END OF REPORT")
w("=" * 70)

with open(REPORT_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"  Saved → {REPORT_PATH}")


# ── Shared style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# ── Plot 01 — Total class distribution ───────────────────────────────────────
print("Plotting 01_class_distribution …")
fig, ax = plt.subplots(figsize=(12, 7))
sorted_cols = disease_total.index.tolist()
colors = [DISEASE_PAL[LABEL_COLS.index(
    next(k for k, v in CLASS_NAMES.items() if v == name)
)] for name in sorted_cols]

bars = ax.barh(sorted_cols, disease_total.values, color=colors, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, disease_total.values):
    ax.text(val + 15, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9)

ax.set_xlabel("Number of Images", fontsize=11)
ax.set_title("Disease Class Distribution — All Images", fontsize=13, fontweight="bold", pad=12)
ax.set_xlim(0, disease_total.max() * 1.15)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("01_class_distribution.png")


# ── Plot 02 — Per-split class counts ─────────────────────────────────────────
print("Plotting 02_split_class_counts …")
names = [CLASS_NAMES[c] for c in LABEL_COLS]
x     = np.arange(len(LABEL_COLS))
width = 0.28

fig, ax = plt.subplots(figsize=(16, 6))
for i, split in enumerate(SPLITS):
    vals = [splits[split][col].sum() for col in LABEL_COLS]
    ax.bar(x + (i - 1) * width, vals, width,
           label=split.capitalize(), color=SPLIT_COLORS[split],
           edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
ax.set_ylabel("Image Count", fontsize=11)
ax.set_title("Disease Image Counts by Split", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("02_split_class_counts.png")


# ── Plot 03 — Class imbalance (pos_weight) ────────────────────────────────────
print("Plotting 03_class_imbalance …")
ratio_sorted = ratio.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
cmap  = plt.cm.RdYlGn_r
norms = plt.Normalize(ratio_sorted.min(), ratio_sorted.max())
cols  = [cmap(norms(v)) for v in ratio_sorted.values]

bars = ax.barh(ratio_sorted.index, ratio_sorted.values, color=cols, edgecolor="white")
for bar, val in zip(bars, ratio_sorted.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}×", va="center", fontsize=9)

ax.axvline(1, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Balanced (1×)")
ax.set_xlabel("Imbalance Ratio  (negatives / positives)", fontsize=11)
ax.set_title("Class Imbalance — Training Difficulty per Disease", fontsize=13,
             fontweight="bold", pad=12)
ax.legend(fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, ratio_sorted.max() * 1.15)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norms)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Imbalance ratio", shrink=0.7)
plt.tight_layout()
savefig("03_class_imbalance.png")


# ── Plot 04 — Labels per image ─────────────────────────────────────────────
print("Plotting 04_labels_per_image …")
label_counts = df["n_labels"].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart — all
ax = axes[0]
bars = ax.bar(label_counts.index.astype(str), label_counts.values,
              color="#4C72B0", edgecolor="white", width=0.55)
for bar, val in zip(bars, label_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val:,}", ha="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Number of Disease Labels per Image", fontsize=11)
ax.set_ylabel("Image Count", fontsize=11)
ax.set_title("Label Count Distribution", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)

# Pie chart
ax = axes[1]
explode = [0.04] * len(label_counts)
wedges, texts, autotexts = ax.pie(
    label_counts.values, labels=[f"{n} label(s)" for n in label_counts.index],
    autopct="%1.1f%%", startangle=140, explode=explode,
    colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    textprops={"fontsize": 10},
)
for at in autotexts:
    at.set_fontweight("bold")
ax.set_title("Label Count Proportions", fontsize=12, fontweight="bold")

fig.suptitle("Labels per Image Distribution", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
savefig("04_labels_per_image.png")


# ── Plot 05 — Source × Split stacked bar ─────────────────────────────────────
print("Plotting 05_source_split …")
src_split_counts = df.groupby(["source", "split"]).size().unstack(fill_value=0)[SPLITS]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Stacked bar
ax = axes[0]
bottom = np.zeros(len(src_split_counts))
for split in SPLITS:
    vals = src_split_counts[split].values
    bars = ax.bar(src_split_counts.index, vals, bottom=bottom,
                  label=split.capitalize(), color=SPLIT_COLORS[split],
                  edgecolor="white", linewidth=0.6)
    for bar, v, b in zip(bars, vals, bottom):
        if v > 60:
            ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                    f"{v:,}", ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
    bottom += vals

ax.set_ylabel("Image Count", fontsize=11)
ax.set_title("Images per Source Dataset × Split", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)

# Grouped bar with percentages
ax = axes[1]
src_pct = src_split_counts.div(src_split_counts.sum(axis=1), axis=0) * 100
x   = np.arange(len(src_pct))
w   = 0.28
for i, split in enumerate(SPLITS):
    ax.bar(x + (i - 1) * w, src_pct[split], w,
           label=split.capitalize(), color=SPLIT_COLORS[split],
           edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(src_pct.index, fontsize=10)
ax.set_ylabel("% of Source Dataset", fontsize=11)
ax.set_title("Split Proportion per Source Dataset", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
ax.set_ylim(0, 80)

plt.tight_layout()
savefig("05_source_split.png")


# ── Plot 06 — Co-occurrence heatmap ──────────────────────────────────────────
print("Plotting 06_cooccurrence_heatmap …")
cooc_named = cooc.copy()
cooc_named.index   = [CLASS_NAMES[c] for c in cooc.index]
cooc_named.columns = [CLASS_NAMES[c] for c in cooc.columns]

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.eye(len(LABEL_COLS), dtype=bool)   # hide self-diagonal

sns.heatmap(
    cooc_named,
    ax=ax,
    mask=mask,
    annot=True, fmt="d", annot_kws={"size": 7},
    cmap="YlOrRd",
    linewidths=0.4, linecolor="white",
    cbar_kws={"label": "Co-occurrence count", "shrink": 0.8},
    square=True,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
ax.set_title("Disease Co-Occurrence Matrix\n(number of images where both diseases appear)",
             fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
savefig("06_cooccurrence_heatmap.png")


# ── Plot 07 — Top co-occurring pairs ─────────────────────────────────────────
print("Plotting 07_top_pairs …")
pair_labels = [f"{CLASS_NAMES[a]}\n+ {CLASS_NAMES[b]}" for (a, b), _ in top_pairs]
pair_counts = [cnt for _, cnt in top_pairs]
pair_colors = sns.color_palette("Blues_r", len(top_pairs))

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(pair_labels[::-1], pair_counts[::-1], color=pair_colors[::-1],
               edgecolor="white")
for bar, val in zip(bars, pair_counts[::-1]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9, fontweight="bold")

ax.set_xlabel("Number of Co-occurring Images", fontsize=11)
ax.set_title("Top 20 Co-Occurring Disease Pairs", fontsize=13, fontweight="bold", pad=12)
ax.set_xlim(0, max(pair_counts) * 1.15)
ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
plt.tight_layout()
savefig("07_top_pairs.png")


# ── Plot 08 — Image size scatter ──────────────────────────────────────────────
print("Plotting 08_image_sizes …")
sample_df  = df.sample(300, random_state=42)
s_widths, s_heights, s_sources = [], [], []
for _, row in sample_df.iterrows():
    p   = row["image_path"]
    img = cv2.imread(p)
    if img is None:
        continue
    h, ww = img.shape[:2]
    s_widths.append(ww); s_heights.append(h); s_sources.append(row["source"])

src_list    = sorted(set(s_sources))
src_palette = {s: c for s, c in zip(src_list, sns.color_palette("Set2", len(src_list)))}
pt_colors   = [src_palette[s] for s in s_sources]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter
ax = axes[0]
for src in src_list:
    idx = [i for i, s in enumerate(s_sources) if s == src]
    ax.scatter([s_widths[i] for i in idx], [s_heights[i] for i in idx],
               alpha=0.6, s=25, label=src, color=src_palette[src])
ax.set_xlabel("Image Width (px)", fontsize=11)
ax.set_ylabel("Image Height (px)", fontsize=11)
ax.set_title("Image Size Distribution (300-sample)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3); ax.set_axisbelow(True)

# Width histogram
ax = axes[1]
for src in src_list:
    idx = [i for i, s in enumerate(s_sources) if s == src]
    ax.hist([s_widths[i] for i in idx], bins=15, alpha=0.6,
            label=src, color=src_palette[src], edgecolor="white")
ax.set_xlabel("Image Width (px)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Width Distribution by Source", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)

fig.suptitle("Raw Image Size Analysis (before preprocessing)", fontsize=14,
             fontweight="bold", y=1.02)
plt.tight_layout()
savefig("08_image_sizes.png")


# ── Plot 09 — Split pie ───────────────────────────────────────────────────────
print("Plotting 09_split_pie …")
split_counts = {s: len(splits[s]) for s in SPLITS}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Pie
ax = axes[0]
wedges, texts, autotexts = ax.pie(
    split_counts.values(),
    labels=[f"{s.capitalize()}\n{n:,} images" for s, n in split_counts.items()],
    autopct="%1.1f%%",
    startangle=140,
    explode=[0.03] * 3,
    colors=[SPLIT_COLORS[s] for s in SPLITS],
    textprops={"fontsize": 11},
)
for at in autotexts:
    at.set_fontweight("bold"); at.set_fontsize(12)
ax.set_title("Train / Val / Test Split", fontsize=12, fontweight="bold")

# Bar — absolute
ax = axes[1]
bars = ax.bar(
    [s.capitalize() for s in SPLITS],
    split_counts.values(),
    color=[SPLIT_COLORS[s] for s in SPLITS],
    edgecolor="white", width=0.5,
)
for bar, val in zip(bars, split_counts.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val:,}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Image Count", fontsize=11)
ax.set_title("Image Count per Split", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(split_counts.values()) * 1.15)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)

fig.suptitle("Dataset Split Overview", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
savefig("09_split_pie.png")


# ── Plot 10 — Presence / absence stacked bar ──────────────────────────────────
print("Plotting 10_presence_absence …")
names_sorted = [CLASS_NAMES[c] for c in LABEL_COLS]
present  = [int(df[c].sum()) for c in LABEL_COLS]
absent   = [total - p for p in present]
pct_pres = [p / total * 100 for p in present]

order   = np.argsort(present)[::-1]
names_o = [names_sorted[i] for i in order]
pres_o  = [present[i] for i in order]
abs_o   = [absent[i]  for i in order]

fig, ax = plt.subplots(figsize=(13, 7))
x = np.arange(len(LABEL_COLS))
b1 = ax.bar(x, pres_o, label="Present", color="#2ecc71", edgecolor="white", linewidth=0.5)
b2 = ax.bar(x, abs_o,  bottom=pres_o, label="Absent", color="#e74c3c",
            edgecolor="white", linewidth=0.5, alpha=0.7)

for xi, (p, tot) in enumerate(zip(pres_o, [p + a for p, a in zip(pres_o, abs_o)])):
    pct = p / tot * 100
    ax.text(xi, p / 2, f"{pct:.1f}%", ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(names_o, rotation=42, ha="right", fontsize=8.5)
ax.set_ylabel("Image Count", fontsize=11)
ax.set_title("Disease Presence vs. Absence per Class (all images)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
plt.tight_layout()
savefig("10_presence_absence.png")


# ── Done ──────────────────────────────────────────────────────────────────────
print()
print(f"All outputs written to:  {os.path.abspath(OUT_DIR)}/")
print(f"  Text report    : data_report.txt")
print(f"  Plots          : 01 – 10")
