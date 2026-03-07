import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import label


def dice_loss(pred, target, eps=1e-6):
    """
    Dice loss for binary segmentation.
    pred: logits (B, 1, H, W)
    target: binary mask (B, 1, H, W)
    """
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice


def poisson_loss(lam, y):
    """
    Poisson loss for count regression.
    lam: predicted rate (B,)
    y: ground-truth counts (B,)
    """
    return (lam - y * torch.log(lam + 1e-8)).mean()


def count_components(mask):
    """
    Counts connected components in a binary segmentation mask.
    mask: Tensor (1, H, W) or (B, 1, H, W)
    """
    mask = mask.squeeze().cpu().numpy()
    _, num = label(mask)
    return num


# ── Notebook utility functions ────────────────────────────────────────────────

def evaluate_models(unet, classifier_fnn, classifier_cnn, ds_full, ds_test, ground_truth, model_label="MODELS"):
    """
    Evaluate all three methods (threshold, FNN, CNN) on the test set.

    Args:
        unet:            Trained U-Net (segmentation backbone)
        classifier_fnn:  Trained FNN classifier
        classifier_cnn:  Trained CNN classifier
        ds_full:         Full labeled dataset (used for indexing)
        ds_test:         Test split (provides ds_test.indices)
        ground_truth:    DataFrame with 'coverage_pct' and 'status' columns
        model_label:     Label shown in the printed summary

    Returns:
        df (DataFrame): per-sample predictions for all methods
        accuracies (list[float]): [threshold_acc, fnn_acc, cnn_acc] in %
    """
    unet_device = next(unet.parameters()).device
    unet.eval()
    classifier_fnn.eval()
    classifier_cnn.eval()

    rows = []
    for idx in ds_test.indices:
        img, mask, label_val = ds_full[idx]
        img_t = img.unsqueeze(0).to(unet_device)

        with torch.no_grad():
            logits = unet(img_t)
            seg_mask = torch.sigmoid(logits)
            coverage = seg_mask.mean().item()
            threshold_pred = "APPROVE" if coverage > 0.4 else "REJECT"

            z = unet.get_latent(img_t)
            fnn_prob = classifier_fnn(z).item()
            fnn_pred = "APPROVE" if fnn_prob > 0.5 else "REJECT"

            z_map = unet.get_latent_map(img_t)
            cnn_prob = classifier_cnn(z_map).item()
            cnn_pred = "APPROVE" if cnn_prob > 0.5 else "REJECT"

        actual = "APPROVE" if label_val == 1 else "REJECT"
        actual_coverage = ground_truth.iloc[idx]["coverage_pct"]

        rows.append({
            "Image":     f"mosaic_{idx+1:04d}",
            "Coverage":  f"{actual_coverage*100:.1f}%",
            "Actual":    actual,
            "Threshold": threshold_pred,
            "FNN pred":  fnn_pred,
            "CNN pred":  cnn_pred,
        })

    df = pd.DataFrame(rows)

    threshold_correct = sum(df["Actual"] == df["Threshold"])
    fnn_correct       = sum(df["Actual"] == df["FNN pred"])
    cnn_correct       = sum(df["Actual"] == df["CNN pred"])

    accuracies = [
        threshold_correct / len(df) * 100,
        fnn_correct       / len(df) * 100,
        cnn_correct       / len(df) * 100,
    ]

    print(f"\n{model_label} — Evaluating on {len(df)} test samples (unseen during training)")
    print(f"{'='*60}")
    print(f"  Threshold (40% coverage): {accuracies[0]:.1f}% ({threshold_correct}/{len(df)} correct)")
    print(f"  FNN classifier:           {accuracies[1]:.1f}% ({fnn_correct}/{len(df)} correct)")
    print(f"  CNN classifier:           {accuracies[2]:.1f}% ({cnn_correct}/{len(df)} correct)")
    print(f"{'='*60}")

    return df, accuracies


def plot_accuracy_comparison(accuracies, labels=None, title="Accuracy Comparison"):
    """
    Bar chart comparing Threshold, FNN, and CNN accuracy on the test set.

    Args:
        accuracies: list of three floats [threshold_acc, fnn_acc, cnn_acc] in %
        labels:     optional list of three x-axis labels
        title:      chart title
    """
    if labels is None:
        labels = ["Threshold\n(40% coverage)", "FNN\nClassifier", "CNN\nClassifier"]
    colors = ["#3498db", "#e67e22", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accuracies, color=colors, edgecolor="white", linewidth=1.5)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_segmentation_results(unet, ds_full, ds_test, ground_truth,
                               title="U-Net segmentation results on TEST SET (unseen data)"):
    """
    3-row grid: original images | ground-truth masks | U-Net predictions
    for the first 8 samples of the test set.

    Args:
        unet:         Trained U-Net
        ds_full:      Full labeled dataset
        ds_test:      Test split (provides ds_test.indices)
        ground_truth: DataFrame with 'coverage_pct' column
        title:        Suptitle for the figure
    """
    unet.eval()
    unet_device = next(unet.parameters()).device
    test_sample_indices = list(ds_test.indices[:8])

    fig, axes = plt.subplots(3, 8, figsize=(18, 9))
    for col, idx in enumerate(test_sample_indices):
        img, mask, label_val = ds_full[idx]
        img_t = img.unsqueeze(0).to(unet_device)

        with torch.no_grad():
            pred = torch.sigmoid(unet(img_t)).squeeze().cpu()
        pred_bin = (pred > 0.5).float()

        status   = "APPROVE" if label_val == 1 else "REJECT"
        coverage = ground_truth.iloc[idx]["coverage_pct"] * 100

        axes[0, col].imshow(img.squeeze(), cmap="gray")
        axes[0, col].set_title(f"mosaic_{idx+1:04d}\n{status}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(mask.squeeze(), cmap="gray")
        axes[1, col].set_title(f"GT ({coverage:.1f}%)", fontsize=8)
        axes[1, col].axis("off")

        axes[2, col].imshow(pred_bin, cmap="gray")
        axes[2, col].set_title("U-Net pred", fontsize=8)
        axes[2, col].axis("off")

    for row, lbl in enumerate(["Original", "GT mask", "U-Net pred"]):
        axes[row, 0].set_ylabel(lbl, fontsize=11)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()


def plot_inference_result(img, gt_mask, seg_bin, IDX, coverage, actual_coverage, title_suffix=""):
    """
    3-panel plot: original image | ground-truth mask | U-Net prediction.
    Used in the full-pipeline inference sections.

    Args:
        img:             (1, H, W) or (H, W) image tensor
        gt_mask:         (1, H, W) or (H, W) ground-truth mask tensor
        seg_bin:         (1, H, W) or (H, W) binarised U-Net prediction tensor
        IDX:             dataset index (used in the title)
        coverage:        U-Net predicted intact fraction (float)
        actual_coverage: ground-truth intact fraction (float)
        title_suffix:    appended to suptitle, e.g. "BASIC MODEL"
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(img.squeeze(), cmap="gray")
    axes[0].set_title(f"Original: mosaic_{IDX+1:04d}", fontsize=11)
    axes[1].imshow(gt_mask.squeeze(), cmap="gray")
    axes[1].set_title(f"Ground truth mask\n{actual_coverage*100:.1f}% intact", fontsize=11)
    axes[2].imshow(seg_bin.squeeze().cpu(), cmap="gray")
    axes[2].set_title(f"U-Net prediction\n{coverage*100:.1f}% intact", fontsize=11)
    for ax in axes:
        ax.axis("off")

    suptitle = f"mosaic_{IDX+1:04d} — Full pipeline (TEST SET sample near 40%)"
    if title_suffix:
        suptitle += f" — {title_suffix}"
    plt.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_comparison_basic_vs_improved(acc_basic, acc_improved):
    """
    Side-by-side grouped bar chart comparing basic vs improved model accuracy.

    Args:
        acc_basic:    list [threshold_acc, fnn_acc, cnn_acc] for basic models
        acc_improved: list [threshold_acc, fnn_acc, cnn_acc] for improved models
    """
    methods = ["Threshold", "FNN", "CNN"]
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, acc_basic,    width, label="Basic Models",
                   color=["#D4A843", "#D47070", "#70A870"], alpha=0.6, edgecolor="gray")
    bars2 = ax.bar(x + width / 2, acc_improved, width, label="Improved Models",
                   color=["goldenrod", "tomato", "seagreen"], edgecolor="gray")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Basic vs Improved Models — Test Set Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim([0, 105])
    ax.axhline(y=95, color="red", linestyle="--", alpha=0.4, label="Target (95%)")
    ax.legend()

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, color="gray")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.show()

    print(f"\n{'='*65}")
    print(f"  IMPROVEMENT SUMMARY")
    print(f"{'='*65}")
    for m, b, i in zip(methods, acc_basic, acc_improved):
        delta = i - b
        print(f"  {m:12s}: {b:5.1f}% -> {i:5.1f}%  ({'+' if delta >= 0 else ''}{delta:.1f}% change)")
    print(f"{'='*65}")


def visualise_samples(approval_labels, ds_full, ds_test, ground_truth):
    """
    Print label distribution and visualise 8 samples from the test set.

    Args:
        approval_labels: list of int labels (1 = APPROVE, 0 = REJECT) for the full dataset
        ds_full:         Full labeled dataset
        ds_test:         Test split (provides ds_test.indices)
        ground_truth:    DataFrame with 'coverage_pct' column
    """
    print(f"Total samples: {len(approval_labels)}")
    print(f"Approved batches: {sum(approval_labels)} ({100*sum(approval_labels)/len(approval_labels):.1f}%)")
    print(f"Rejected batches: {len(approval_labels)-sum(approval_labels)} ({100*(1-sum(approval_labels)/len(approval_labels)):.1f}%)")

    fig, axes = plt.subplots(2, 8, figsize=(18, 5))
    test_indices = list(ds_test.indices[:8])

    for col, idx in enumerate(test_indices):
        img, mask, label = ds_full[idx]
        status = "APPROVE" if label == 1 else "REJECT"
        coverage = ground_truth.iloc[idx]["coverage_pct"] * 100

        axes[0, col].imshow(img.squeeze(), cmap="gray")
        axes[0, col].set_title(f"mosaic_{idx+1:04d}\n{status} ({coverage:.1f}%)", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(mask.squeeze(), cmap="gray")
        axes[1, col].set_title("GT mask", fontsize=8)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=11)
    axes[1, 0].set_ylabel("Mask", fontsize=11)
    plt.suptitle("Sample from test set (held-out data)", fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()


def plot_unet_history(history, title="U-Net Training"):
    """
    Plot train vs validation loss curves for the U-Net.

    Args:
        history: dict with 'train_loss' and 'val_loss' lists (one value per epoch)
        title:   chart title
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train Loss", color="steelblue")
    ax.plot(history["val_loss"],   label="Val Loss",   color="coral")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE + Dice)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_classifier_history(history, title="Classifier Training"):
    """
    Plot train/val loss and validation accuracy for a classifier.

    Args:
        history: dict with 'train_loss', 'val_loss', 'val_acc' lists
        title:   prefix used in subplot titles
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(history["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(history["val_loss"],   label="Val Loss",   color="coral")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (BCE)")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["val_acc"], label="Val Accuracy", color="seagreen")
    ax2.axhline(y=95, color="red", linestyle="--", alpha=0.5, label="Target (95%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} — Validation Accuracy")
    ax2.set_ylim([50, 102])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
