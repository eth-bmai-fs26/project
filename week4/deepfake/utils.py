"""
Shared utilities for the deepfake detection project.
Used by both the attention diffusion pipeline and the discriminator notebook.
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os, shutil, random

SEED = 42
IMG_SIZE = 64


def prepare_fruit_dataset(source_root, dest_root='./fruit_data', balance=True):
    """
    Extract apple and orange images from Fruits-360.
    - Excludes Pepper Orange, Pineapple, peeled oranges
    - Optionally downsamples the larger class to match the smaller one
    """
    if os.path.exists(dest_root):
        shutil.rmtree(dest_root)
    os.makedirs(dest_root)

    APPLE_KEYWORDS = ['apple']
    APPLE_EXCLUDE  = ['pineapple', 'custard']
    ORANGE_KEYWORDS = ['orange', 'clementine']
    ORANGE_EXCLUDE  = ['pepper', 'cherry', 'peeled']

    def classify_folder(name):
        name_lower = name.lower()
        if any(kw in name_lower for kw in APPLE_KEYWORDS):
            if not any(ex in name_lower for ex in APPLE_EXCLUDE):
                return 'apple'
        if any(kw in name_lower for kw in ORANGE_KEYWORDS):
            if not any(ex in name_lower for ex in ORANGE_EXCLUDE):
                return 'orange'
        return None

    for split_src, split_dst in [('Training', 'train'), ('Test', 'test')]:
        src_base = os.path.join(source_root, split_src)
        if not os.path.exists(src_base):
            for candidate in Path(source_root).rglob(split_src):
                if candidate.is_dir():
                    src_base = str(candidate)
                    break
        if not os.path.exists(src_base):
            print(f"Warning: Could not find {split_src} directory")
            continue

        for class_dir in sorted(os.listdir(src_base)):
            target_class = classify_folder(class_dir)
            if target_class is None:
                continue

            print(f"  ✅ {split_dst}: '{class_dir}' → {target_class}")

            src_dir = os.path.join(src_base, class_dir)
            dst_dir = os.path.join(dest_root, split_dst, target_class)
            os.makedirs(dst_dir, exist_ok=True)

            for img_file in os.listdir(src_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(src_dir, img_file)
                    dst_name = f"{class_dir}_{img_file}"
                    dst_path = os.path.join(dst_dir, dst_name)
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)

    if balance:
        print(f"\n⚖️  Balancing classes...")
        for split in ['train', 'test']:
            split_dir = os.path.join(dest_root, split)
            if not os.path.exists(split_dir):
                continue

            class_counts = {}
            class_files = {}
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, cls)
                files = [f for f in os.listdir(cls_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                class_counts[cls] = len(files)
                class_files[cls] = files

            min_count = min(class_counts.values())
            print(f"  {split}: min class has {min_count} images, downsampling others to match")

            for cls, files in class_files.items():
                if len(files) > min_count:
                    random.seed(SEED)
                    keep = set(random.sample(files, min_count))
                    removed = 0
                    for f in files:
                        if f not in keep:
                            os.remove(os.path.join(split_dir, cls, f))
                            removed += 1
                    print(f"    {cls}: removed {removed} images, kept {min_count}")

    print(f"\n📊 Final Dataset Summary:")
    print(f"{'─' * 40}")
    for split in ['train', 'test']:
        split_dir = os.path.join(dest_root, split)
        if os.path.exists(split_dir):
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, cls)
                n = len([f for f in os.listdir(cls_dir)
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {split:>5}/{cls:<10}: {n:>5} images")

    return dest_root


def load_fruit_data(dataset_path, batch_size=32, img_size=IMG_SIZE):
    """Download, prepare, and return datasets + loaders."""
    data_root = prepare_fruit_dataset(dataset_path, balance=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(os.path.join(data_root, 'train'), transform=transform)
    test_dataset  = ImageFolder(os.path.join(data_root, 'test'),  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    class_names = {v: k for k, v in train_dataset.class_to_idx.items()}

    print(f"Classes: {class_names}")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, test_dataset, train_loader, test_loader, class_names


def create_mask_and_crop_batch(images, threshold=0.93):
    """
    Batch masking for Fruits-360 white-background images.

    Args:
        images: (B, 3, H, W) tensor in [0, 1]
        threshold: pixels where all RGB channels > threshold = background

    Returns:
        masks:    (B, 1, H, W) — 1 where fruit is, 0 where background
        crops:    (B, 3, H, W) — fruit pixels only, background zeroed
        redacted: (B, 3, H, W) — background only, fruit area zeroed
    """
    is_background = (images > threshold).all(dim=1, keepdim=True)
    masks = (~is_background).float()
    crops = images * masks
    redacted = images * (1 - masks)
    return masks, crops, redacted


def visualize_masking(images, n=5):
    """Show masking results for a few samples."""
    masks, crops, redacted = create_mask_and_crop_batch(images[:n])

    fig, axes = plt.subplots(4, n, figsize=(3*n, 10))
    row_labels = ['Original', 'Mask', 'Crop (fruit)', 'Redacted (hole)']

    for i in range(n):
        axes[0, i].imshow(images[i].permute(1, 2, 0).numpy())
        axes[1, i].imshow(masks[i, 0].numpy(), cmap='gray')
        axes[2, i].imshow(crops[i].permute(1, 2, 0).numpy())
        axes[3, i].imshow(redacted[i].permute(1, 2, 0).numpy())
        for row in range(4):
            axes[row, i].axis('off')

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, rotation=0, labelpad=70, va='center')

    plt.suptitle('Masking Pipeline', fontsize=14)
    plt.tight_layout()
    plt.show()


def show_batch(loader, class_names, n=8):
    """Display a batch of images with their labels."""
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    fig.suptitle('Sample Training Images', fontsize=14)

    for row, class_idx in enumerate([0, 1]):
        mask = labels == class_idx
        class_imgs = images[mask][:n]
        for col in range(min(n, len(class_imgs))):
            axes[row, col].imshow(class_imgs[col].permute(1, 2, 0).numpy())
            axes[row, col].set_title(class_names[class_idx], fontsize=10)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def find_samples_per_class(loader, class_to_idx, n_per_class=3):
    """
    Search across ALL batches to reliably find n samples of each class.
    Returns dict: {class_idx: images_tensor}
    """
    collected = {idx: [] for idx in class_to_idx.values()}
    target_n = n_per_class

    for images, labels in loader:
        for cls_idx in collected:
            if len(collected[cls_idx]) >= target_n:
                continue
            mask = labels == cls_idx
            if mask.any():
                cls_imgs = images[mask]
                need = target_n - len(collected[cls_idx])
                collected[cls_idx].extend(cls_imgs[:need])

        if all(len(v) >= target_n for v in collected.values()):
            break

    result = {}
    for cls_idx, img_list in collected.items():
        if len(img_list) > 0:
            result[cls_idx] = torch.stack(img_list)

    return result


def collect_all_images(loader):
    """Collect all images from a loader into a single tensor."""
    all_imgs = []
    for images, _ in loader:
        all_imgs.append(images)
    return torch.cat(all_imgs, dim=0)
