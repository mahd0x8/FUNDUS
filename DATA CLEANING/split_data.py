import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "DATASET/filtered_data_merged_V1.csv"
OUTPUT_CSV = "DATASET/filtered_data_split.csv"
SEED = 42

df = pd.read_csv(INPUT_CSV)

# Step 1: split into train (70%) and temp (30%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    shuffle=True
)

# Step 2: split temp into validation (15%) and test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    shuffle=True
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

out_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
out_df.to_csv(OUTPUT_CSV, index=False)

print("Train      :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))
print("Saved      :", OUTPUT_CSV)
