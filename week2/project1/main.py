import torch
from torch.utils.data import DataLoader

# Dataset: loads (image, binary_mask) pairs from data/
from week2.project1.dataset import SegmentationDataset

# U-Net model and its training function
from week2.project1.unet import UNet, train_unet

# FNN and CNN counters with their training functions
from week2.project1.counter import BrokenCounter, BrokenCounterCNN, train_counter, train_counter_cnn

# Utility for counting connected components in a binary mask
from week2.project1.utils import count_components


# --------------------------------------------------
# 1. DATASET AND DATALOADER
# --------------------------------------------------

# Create the segmentation dataset
# Each sample is: image [1, H, W], mask [1, H, W]
ds = SegmentationDataset("data")

# DataLoader with batch_size=1 (dataset is very small)
# shuffle=False so that index i always corresponds to broken_labels[i]
dl = DataLoader(ds, batch_size=1, shuffle=False)


# --------------------------------------------------
# 2. TRAIN THE U-NET (SUPERVISED SEGMENTATION)
# --------------------------------------------------

# Initialize the U-Net
# in_ch = 1 because images are grayscale
# out_ch = 1 because segmentation is binary (object / background)
unet = UNet(in_ch=1, base_ch=32)

# Train the U-Net using the segmentation masks
# The model will overfit (expected with only 6 images)
train_unet(unet, dl)


# --------------------------------------------------
# 3. GROUND-TRUTH BROKEN-OBJECT COUNTS
# --------------------------------------------------

# Number of broken fragments per image (order matches sorted dataset)
# ceramics-1: 9, ceramics-2: 5, ceramics-3: 12,
# ceramics-4: 0, ceramics-5: 28, ceramics-6: 12
broken_labels = [9, 5, 12, 0, 28, 12]


# --------------------------------------------------
# 4. TRAIN THE FNN BROKEN-OBJECT COUNTER
# --------------------------------------------------

# FNN counter: operates on the POOLED latent vector (B, 64)
# produced by UNet.get_latent() (global average pooling applied)
counter_fnn = BrokenCounter(unet.latent_dim)

train_counter(unet, counter_fnn, dl, broken_labels)


# --------------------------------------------------
# 5. TRAIN THE CNN BROKEN-OBJECT COUNTER
# --------------------------------------------------

# CNN counter: operates on the SPATIAL latent map (B, 64, 64, 64)
# produced by UNet.get_latent_map() (no global average pooling)
counter_cnn = BrokenCounterCNN(in_channels=unet.latent_dim)

train_counter_cnn(unet, counter_cnn, dl, broken_labels)


# --------------------------------------------------
# 6. COMPARISON: FNN vs CNN COUNTER ON ALL IMAGES
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

unet.eval()
counter_fnn.eval()
counter_cnn.eval()

print("\nInference results — FNN vs CNN Counter")
print("-" * 58)
print(f"{'Img':<12} {'Actual':>8} {'FNN pred':>10} {'CNN pred':>10} {'Segments':>10}")
print("-" * 58)

for idx in range(len(ds)):
    img, _ = ds[idx]
    img_t = img.unsqueeze(0).to(device)

    with torch.no_grad():
        # Segmentation mask
        logits = unet(img_t)
        seg_mask = (torch.sigmoid(logits) > 0.5).float()
        total_objects = count_components(seg_mask)

        # FNN path: pooled latent vector
        z = unet.get_latent(img_t)
        fnn_pred = int(torch.round(counter_fnn(z)).item())

        # CNN path: spatial latent map
        z_map = unet.get_latent_map(img_t)
        cnn_pred = int(torch.round(counter_cnn(z_map)).item())

    actual = broken_labels[idx]
    print(f"ceramics-{idx+1:<3} {actual:>8} {fnn_pred:>10} {cnn_pred:>10} {total_objects:>10}")
