"""
Autoencoder-based anomaly detection for the deepfake project.

This module replaces the "frozen AE + classifier probe" stage with a proper
one-class pipeline:

1. Train the autoencoder on real images only.
2. Build anomaly features from reconstruction residuals and latent codes.
3. Calibrate a threshold on a validation split.
4. Score test images directly as normal/anomalous.

The small fake validation split is only used for threshold selection, never for
representation learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class AutoencoderConfig:
    img_size: int = 128
    latent_dim: int = 128
    batch_size: int = 256
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class ScoreConfig:
    error_grid: int = 8
    weight_grid_mean: float = 0.45
    weight_grid_max: float = 0.35
    weight_pixel_mean: float = 0.10
    weight_latent: float = 0.10


@dataclass
class ThresholdResult:
    threshold: float
    strategy: str
    balanced_accuracy: Optional[float] = None
    best_f1: Optional[float] = None
    target_fpr: Optional[float] = None


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
        )
        self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(z)
        x = x.view(-1, 256, 4, 4)
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def _make_loader(images: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=shuffle)


def train_autoencoder(
    model: ConvAutoencoder,
    real_train: torch.Tensor,
    real_val: torch.Tensor,
    config: AutoencoderConfig,
    device: torch.device,
) -> Dict[str, list]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_loader = _make_loader(real_train, config.batch_size, shuffle=True)
    val_loader = _make_loader(real_val, config.batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    model.to(device)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for (images,) in train_loader:
            images = images.to(device)
            recon = model(images)
            loss = F.mse_loss(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (images,) in val_loader:
                images = images.to(device)
                val_loss += F.mse_loss(model(images), images).item()
        val_loss /= max(len(val_loader), 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"AE epoch {epoch + 1:3d}/{config.epochs} "
                f"train_mse={train_loss:.6f} val_mse={val_loss:.6f}"
            )

    return history


@torch.no_grad()
def extract_anomaly_features(
    images: torch.Tensor,
    model: ConvAutoencoder,
    device: torch.device,
    img_size: int = 128,
    error_grid: int = 8,
) -> Dict[str, torch.Tensor]:
    features = {
        "grid_mean": [],
        "grid_max": [],
        "pixel_mean": [],
        "latent": [],
    }
    pool_kernel = img_size // error_grid

    for start in range(0, len(images), 64):
        batch = images[start : start + 64].to(device)
        latent = model.encode(batch)
        recon = model.decode(latent)

        error_map = (batch - recon).pow(2).mean(dim=1, keepdim=True)
        pooled = F.avg_pool2d(error_map, kernel_size=pool_kernel).flatten(1)

        features["grid_mean"].append(pooled.mean(dim=1).cpu())
        features["grid_max"].append(pooled.max(dim=1).values.cpu())
        features["pixel_mean"].append(error_map.flatten(1).mean(dim=1).cpu())
        features["latent"].append(latent.cpu())

    return {key: torch.cat(value, dim=0) for key, value in features.items()}


def fit_real_statistics(real_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    latent = real_features["latent"]
    return {
        "grid_mean_mu": real_features["grid_mean"].mean(),
        "grid_mean_sd": real_features["grid_mean"].std(unbiased=False) + 1e-6,
        "grid_max_mu": real_features["grid_max"].mean(),
        "grid_max_sd": real_features["grid_max"].std(unbiased=False) + 1e-6,
        "pixel_mean_mu": real_features["pixel_mean"].mean(),
        "pixel_mean_sd": real_features["pixel_mean"].std(unbiased=False) + 1e-6,
        "latent_mu": latent.mean(dim=0, keepdim=True),
        "latent_sd": latent.std(dim=0, unbiased=False, keepdim=True) + 1e-6,
    }


def score_from_features(
    features: Dict[str, torch.Tensor],
    stats: Dict[str, torch.Tensor],
    config: ScoreConfig,
) -> torch.Tensor:
    latent_z = (features["latent"] - stats["latent_mu"]) / stats["latent_sd"]
    latent_score = latent_z.pow(2).mean(dim=1)

    grid_mean_score = (features["grid_mean"] - stats["grid_mean_mu"]) / stats["grid_mean_sd"]
    grid_max_score = (features["grid_max"] - stats["grid_max_mu"]) / stats["grid_max_sd"]
    pixel_mean_score = (features["pixel_mean"] - stats["pixel_mean_mu"]) / stats["pixel_mean_sd"]

    score = (
        config.weight_grid_mean * grid_mean_score
        + config.weight_grid_max * grid_max_score
        + config.weight_pixel_mean * pixel_mean_score
        + config.weight_latent * latent_score
    )
    return score.float()


def calibrate_threshold(
    real_val_scores: torch.Tensor,
    fake_val_scores: Optional[torch.Tensor] = None,
    target_fpr: float = 0.05,
) -> ThresholdResult:
    real_np = real_val_scores.detach().cpu().numpy()

    if fake_val_scores is None or len(fake_val_scores) == 0:
        threshold = float(np.quantile(real_np, 1.0 - target_fpr))
        return ThresholdResult(
            threshold=threshold,
            strategy="real-quantile",
            target_fpr=target_fpr,
        )

    fake_np = fake_val_scores.detach().cpu().numpy()
    labels = np.concatenate([np.zeros_like(real_np), np.ones_like(fake_np)]).astype(int)
    scores = np.concatenate([real_np, fake_np])
    candidates = np.unique(scores)

    best_threshold = float(candidates[0])
    best_bal_acc = -1.0
    best_f1 = -1.0

    for threshold in candidates:
        preds = (scores >= threshold).astype(int)
        bal_acc = balanced_accuracy_score(labels, preds)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        if bal_acc > best_bal_acc or (np.isclose(bal_acc, best_bal_acc) and f1 > best_f1):
            best_threshold = float(threshold)
            best_bal_acc = float(bal_acc)
            best_f1 = float(f1)

    return ThresholdResult(
        threshold=best_threshold,
        strategy="balanced-accuracy",
        balanced_accuracy=best_bal_acc,
        best_f1=best_f1,
    )


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    preds = (scores >= threshold).astype(int)
    report = classification_report(
        labels,
        preds,
        target_names=["Real", "Fake"],
        digits=3,
        zero_division=0,
        output_dict=False,
    )
    metrics = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds),
        "classification_report": report,
        "predictions": preds,
        "scores": scores,
    }
    return metrics


def print_evaluation(title: str, metrics: Dict[str, object]) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(metrics["classification_report"])
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


def run_anomaly_pipeline(
    real_train: torch.Tensor,
    real_val: torch.Tensor,
    real_test: torch.Tensor,
    fake_val: Optional[torch.Tensor] = None,
    fake_test: Optional[torch.Tensor] = None,
    ae_config: Optional[AutoencoderConfig] = None,
    score_config: Optional[ScoreConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    ae_config = ae_config or AutoencoderConfig()
    score_config = score_config or ScoreConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvAutoencoder(latent_dim=ae_config.latent_dim).to(device)
    history = train_autoencoder(model, real_train, real_val, ae_config, device)

    real_train_features = extract_anomaly_features(
        real_train, model, device, img_size=ae_config.img_size, error_grid=score_config.error_grid
    )
    stats = fit_real_statistics(real_train_features)

    real_val_scores = score_from_features(
        extract_anomaly_features(real_val, model, device, ae_config.img_size, score_config.error_grid),
        stats,
        score_config,
    )
    fake_val_scores = None
    if fake_val is not None and len(fake_val) > 0:
        fake_val_scores = score_from_features(
            extract_anomaly_features(fake_val, model, device, ae_config.img_size, score_config.error_grid),
            stats,
            score_config,
        )

    threshold_result = calibrate_threshold(real_val_scores, fake_val_scores)

    real_test_scores = score_from_features(
        extract_anomaly_features(real_test, model, device, ae_config.img_size, score_config.error_grid),
        stats,
        score_config,
    )

    outputs: Dict[str, object] = {
        "model": model,
        "history": history,
        "stats": stats,
        "threshold": threshold_result,
        "real_val_scores": real_val_scores.cpu(),
        "real_test_scores": real_test_scores.cpu(),
    }

    if fake_test is not None and len(fake_test) > 0:
        fake_test_scores = score_from_features(
            extract_anomaly_features(fake_test, model, device, ae_config.img_size, score_config.error_grid),
            stats,
            score_config,
        )
        labels = np.concatenate(
            [
                np.zeros(len(real_test_scores), dtype=int),
                np.ones(len(fake_test_scores), dtype=int),
            ]
        )
        scores = np.concatenate([real_test_scores.cpu().numpy(), fake_test_scores.cpu().numpy()])
        metrics = evaluate_scores(labels, scores, threshold_result.threshold)
        outputs["fake_test_scores"] = fake_test_scores.cpu()
        outputs["test_metrics"] = metrics

    return outputs


def optimal_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    best_f1 = float(f1[best_idx])
    return best_threshold, best_f1
