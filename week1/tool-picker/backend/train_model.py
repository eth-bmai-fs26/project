"""
train_model.py — Train and persist the ToolRouter MLP classifier.

Checks for a saved checkpoint in models/tool_router.pt.
If it does not exist, trains from scratch on data/train.csv and data/val.csv,
then saves the checkpoint so the Flask app can load it instantly on startup.
"""

import os
import sys

import torch
import torch.nn as nn

# Allow imports from the sibling lib/ package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.dataset import make_dataloaders
from lib.utils import set_seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "tool_router.pt")
DATA_DIR = os.path.join(_HERE, "..", "data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_NAMES = {0: "FLIGHT_BOOKER", 1: "HOTEL_BOOKER", 2: "CAR_RENTAL_BOOKER"}

SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
HIDDEN_DIM = 128
LR = 1e-3
DROPOUT_P = 0.1


# ---------------------------------------------------------------------------
# Model definition (mirrors the notebook)
# ---------------------------------------------------------------------------
class ToolRouterMLP(nn.Module):
    """Two-layer MLP: Linear → ReLU → Dropout → Linear.  Outputs raw logits."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 3, dropout_p: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_and_save() -> tuple[ToolRouterMLP, torch.device]:
    """Train the MLP from scratch and save a checkpoint to MODELS_DIR."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_model] Device: {device}")

    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_loader, val_loader, _ = make_dataloaders(
        train_path, val_path, test_path, batch_size=BATCH_SIZE
    )
    input_dim = train_loader.dataset.dim
    print(f"[train_model] Input dim: {input_dim}  |  Train samples: {len(train_loader.dataset)}")

    model = ToolRouterMLP(input_dim, HIDDEN_DIM, 3, DROPOUT_P).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        train_correct = train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += len(y)

        # --- validate ---
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_correct += (model(x).argmax(1) == y).sum().item()
                val_total += len(y)

        print(
            f"[train_model] Epoch {epoch:2d}/{EPOCHS}  "
            f"train_acc={train_correct/train_total:.4f}  "
            f"val_acc={val_correct/val_total:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": HIDDEN_DIM,
            "num_classes": 3,
            "dropout_p": DROPOUT_P,
        },
        MODEL_PATH,
    )
    print(f"[train_model] Checkpoint saved → {MODEL_PATH}")

    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# Loading (trains if no checkpoint found)
# ---------------------------------------------------------------------------
def load_model() -> tuple[ToolRouterMLP, torch.device]:
    """Return a ready-to-use (model, device) pair.

    Trains from scratch when models/tool_router.pt does not exist.
    """
    if not os.path.exists(MODEL_PATH):
        print("[train_model] No checkpoint found — training from scratch …")
        return train_and_save()

    print(f"[train_model] Loading checkpoint from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = ToolRouterMLP(
        checkpoint["input_dim"],
        checkpoint["hidden_dim"],
        checkpoint["num_classes"],
        checkpoint["dropout_p"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# CLI entry point — run directly to force a fresh retrain
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the ToolRouter MLP classifier.")
    parser.add_argument("--force", action="store_true", help="Retrain even if a checkpoint exists.")
    args = parser.parse_args()

    if args.force and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print("[train_model] Existing checkpoint removed — retraining …")

    load_model()
