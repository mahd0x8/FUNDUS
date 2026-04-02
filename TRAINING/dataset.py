from typing import List
from preprocessing import preprocess_fundus_image
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np

# ============================================================
# DATASET
# ============================================================

class FundusDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str,
        label_cols: List[str],
        transform=None,
        preprocess: bool = True,
        image_size: int = 224,
    ):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_cols = label_cols
        self.transform = transform
        self.preprocess = preprocess
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]

        image = preprocess_fundus_image(img_path, self.image_size)

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(
            row[self.label_cols].astype(np.float32).values,
            dtype=torch.float32
        )
        return image, labels