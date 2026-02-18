"""PyTorch Dataset and DataLoader helpers for the Tool Router."""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

LABEL_MAP = {"Flight": 0, "Hotel": 1, "CarRental": 2}


class ToolRouterDataset(Dataset):
    """Loads a Tool Router CSV and serves (embedding, label) pairs.

    Embedding columns are identified as purely numeric column names
    (e.g. ``0, 1, ..., 383``).
    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, index_col=0)

        emb_cols = []
        for col_name in df.columns:
            if str(col_name).isdigit():
                emb_cols.append(col_name)
        emb_cols.sort(key=int)
        if len(emb_cols) == 0:
            raise ValueError(f"No embedding columns found in {csv_path}")

        embedding_values = df[emb_cols].values
        self.embeddings = torch.tensor(embedding_values, dtype=torch.float32)

        label_values = df["label"].map(LABEL_MAP).values
        self.labels = torch.tensor(label_values, dtype=torch.long)
        self.dim = len(emb_cols)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def make_dataloaders(train_path, val_path, test_path, batch_size=64, num_workers=0):
    """Create train, val, and test DataLoaders from CSV paths."""
    train_ds = ToolRouterDataset(train_path)
    val_ds = ToolRouterDataset(val_path)
    test_ds = ToolRouterDataset(test_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
