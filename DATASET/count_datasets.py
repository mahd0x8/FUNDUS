import pandas as pd

try:
    df = pd.read_csv('DATASET/aggregated_annotations_paper.csv')
    print("Dataset Image Counts (from aggregated_annotations_paper.csv):")
    print("1000Fundus Images:", sum(df['image'].str.contains('1000images', na=False, case=False)))
    print("ODIR:", sum(df['image'].str.contains('odir', na=False, case=False)))
    print("RFMiD (RMFID):", sum(df['image'].str.contains('rfmid', na=False, case=False)))
except Exception as e:
    print("Error reading CSV:", e)
