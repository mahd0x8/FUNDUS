import pandas as pd

# ==============================
# CONFIG
# ==============================
INPUT_CSV = "aggregated_annotations.csv"
OUTPUT_CSV = "filtered_resident_22.csv"

# Harmonized 39-class columns
C_COLS = [f"C{i}" for i in range(39)]

# Original ODIR shorthand columns
ODIR_COLS = ["NN", "DD", "GG", "CC", "AA", "HH", "MM"]

# Retained classes from the paper
KEPT_CLASSES = [
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
    "C10", "C11", "C13", "C14", "C15", "C18", "C19", "C22",
    "C25", "C27", "C29", "C32"
]

REMOVED_CLASSES = [c for c in C_COLS if c not in KEPT_CLASSES]

# Optional readable names
CLASS_NAME_MAP = {
    "C0": "Normal",
    "C1": "AMD",
    "C2": "Mild_DR",
    "C3": "Moderate_DR",
    "C4": "Severe_DR",
    "C5": "Proliferative_DR",
    "C6": "Glaucoma",
    "C7": "Hypertensive_Retinopathy",
    "C8": "Pathological_Myopia",
    "C9": "Tessellated_Fundus",
    "C10": "Vitreous_Degeneration",
    "C11": "BRVO",
    "C12": "CRAO",
    "C13": "Large_Optic_Cup",
    "C14": "Drusen",
    "C15": "Epiretinal_Membrane",
    "C16": "Macular_Hole",
    "C17": "Retinitis_Pigmentosa",
    "C18": "Optic_Disc_Edema",
    "C19": "Myelinated_Nerve_Fibers",
    "C20": "Laser_Scars",
    "C21": "Silicone_Oil",
    "C22": "Retinal_Detachment",
    "C23": "Macular_Edema",
    "C24": "CNV",
    "C25": "Refractive_Media_Opacity",
    "C26": "AION",
    "C27": "CSC",
    "C28": "Macular_Scar",
    "C29": "Laser_Spots",
    "C30": "RPE_Changes",
    "C31": "Macular_Atrophy",
    "C32": "CRVO",
    "C33": "Branch_Artery_Occlusion",
    "C34": "Optic_Disc_Pallor",
    "C35": "Tilted_Optic_Disc",
    "C36": "Posterior_Staphyloma",
    "C37": "Chorioretinal_Atrophy",
    "C38": "Other"
}

# ==============================
# LOAD
# ==============================
df = pd.read_csv(INPUT_CSV)

print("Loaded rows:", len(df))

# Check required columns
missing = [c for c in C_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing C columns: {missing}")

# Convert possible float columns to int 0/1
for col in C_COLS + [c for c in ODIR_COLS if c in df.columns]:
    df[col] = df[col].fillna(0).astype(float).astype(int)

# ==============================
# COUNT ORIGINAL CLASS FREQUENCIES
# ==============================
original_counts = df[C_COLS].sum().astype(int).sort_index()

print("\n=== Original class counts ===")
for cls in C_COLS:
    print(f"{cls:>3} ({CLASS_NAME_MAP.get(cls, cls):<25}) : {original_counts[cls]}")

# ==============================
# FILTER TO THE PAPER'S 22 CLASSES
# ==============================
# Keep only rows with at least one positive retained class
df_filtered = df[df[KEPT_CLASSES].sum(axis=1) > 0].copy()

# Zero-out removed classes for cleanliness
df_filtered[REMOVED_CLASSES] = 0

# ==============================
# CREATE HELPER COLUMNS
# ==============================
def active_classes(row, columns):
    return [col for col in columns if row[col] == 1]

def active_class_names(row, columns):
    return [CLASS_NAME_MAP[col] for col in columns if row[col] == 1]

df_filtered["active_C_labels"] = df_filtered.apply(
    lambda row: "|".join(active_classes(row, KEPT_CLASSES)),
    axis=1
)

df_filtered["active_C_names"] = df_filtered.apply(
    lambda row: "|".join(active_class_names(row, KEPT_CLASSES)),
    axis=1
)

if all(col in df_filtered.columns for col in ODIR_COLS):
    df_filtered["active_ODIR_labels"] = df_filtered.apply(
        lambda row: "|".join(active_classes(row, ODIR_COLS)),
        axis=1
    )

# ==============================
# FINAL COUNTS AFTER FILTERING
# ==============================
final_counts = df_filtered[KEPT_CLASSES].sum().astype(int).sort_index()

print("\n=== Final retained class counts ===")
for cls in KEPT_CLASSES:
    print(f"{cls:>3} ({CLASS_NAME_MAP.get(cls, cls):<25}) : {final_counts[cls]}")

print("\nTotal original rows :", len(df))
print("Total filtered rows :", len(df_filtered))
print("Dropped rows        :", len(df) - len(df_filtered))

# ==============================
# SAVE
# ==============================
df_filtered.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved filtered file to: {OUTPUT_CSV}")