from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from anomaly_pipeline import (
    AutoencoderConfig,
    ScoreConfig,
    print_evaluation,
    run_anomaly_pipeline,
)


def load_fake_images(fake_dir: Path, img_size: int) -> torch.Tensor:
    files = sorted(fake_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found in {fake_dir}")

    to_tensor = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    images = []
    for path in files:
        image = Image.open(path).convert("RGB")
        images.append(to_tensor(image))
    return torch.stack(images)


def split_real_train(real_train_all: torch.Tensor, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(real_train_all), generator=generator)
    n_val = int(len(real_train_all) * val_fraction)
    train_idx = indices[:-n_val]
    val_idx = indices[-n_val:]
    return real_train_all[train_idx], real_train_all[val_idx]


def split_fakes(fake_images: torch.Tensor, n_fake_val: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n_fake_val >= len(fake_images):
        raise ValueError("n_fake_val must be smaller than the number of fake images")
    return fake_images[:n_fake_val], fake_images[n_fake_val:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AE-based anomaly detection on the deepfake dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--fake-dir", type=Path, default=Path("deepfake_imgs"))
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--ae-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--n-fake-val", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = torch.load(args.dataset_dir / "fruits_train.pt", weights_only=False)
    test_data = torch.load(args.dataset_dir / "fruits_test.pt", weights_only=False)

    real_train_all = train_data["composites"].float()
    real_test_all = test_data["composites"].float()
    fake_images = load_fake_images(args.fake_dir, args.img_size)

    real_train, real_val = split_real_train(real_train_all, args.val_fraction, args.seed)
    fake_val, fake_test = split_fakes(fake_images, args.n_fake_val)

    outputs = run_anomaly_pipeline(
        real_train=real_train,
        real_val=real_val,
        real_test=real_test_all,
        fake_val=fake_val,
        fake_test=fake_test,
        ae_config=AutoencoderConfig(
            img_size=args.img_size,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            epochs=args.ae_epochs,
        ),
        score_config=ScoreConfig(),
    )

    threshold = outputs["threshold"]
    print(
        f"Threshold strategy: {threshold.strategy} | "
        f"threshold={threshold.threshold:.4f}"
    )
    if threshold.balanced_accuracy is not None:
        print(f"Val balanced accuracy: {threshold.balanced_accuracy:.4f}")
    if threshold.best_f1 is not None:
        print(f"Val F1: {threshold.best_f1:.4f}")

    if "test_metrics" in outputs:
        print_evaluation("AE Anomaly Detection", outputs["test_metrics"])


if __name__ == "__main__":
    main()
