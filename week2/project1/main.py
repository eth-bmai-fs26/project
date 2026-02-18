
from torch.utils.data import DataLoader

# Dataset: loads (image, binary_mask) pairs from data/
from week2.project1.dataset import SegmentationDataset

# U-Net model and its training function
from week2.project1.unet import UNet, train_unet

# Feedforward network for counting broken objects
from week2.project1.counter import BrokenCounter, train_counter


# --------------------------------------------------
# 1. DATASET AND DATALOADER
# --------------------------------------------------

# Create the segmentation dataset
# Each sample is: image [1, H, W], mask [1, H, W]
ds = SegmentationDataset("data")

# DataLoader with batch_size=1 (dataset is very small)
# shuffle=True to avoid always seeing images in the same order
dl = DataLoader(ds, batch_size=1, shuffle=True)


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
# 3. TRAIN THE BROKEN-OBJECT COUNTER
# --------------------------------------------------

# Ground-truth number of broken objects for each image
# IMPORTANT: the order must match the dataset indexing
broken_labels = [1, 1, 1, 1, 2, 3]

# Initialize the feedforward network
# latent_dim is defined by the U-Net bottleneck
counter = BrokenCounter(unet.latent_dim)

# Train the counter using:
# - latent vectors extracted from the trained U-Net
# - Poisson loss for count regression
train_counter(unet, counter, dl, broken_labels)




# pick one image from the dataset (for example the first one)
idx = 4
img, _ = ds[idx]
img = img.unsqueeze(0)   # add batch dimension: [1, C, H, W]

device = "cuda" if torch.cuda.is_available() else "cpu"
img = img.to(device)

unet.eval()
counter.eval()

with torch.no_grad():
    # --- SEGMENTATION ---
    logits = unet(img)
    seg_mask = (torch.sigmoid(logits) > 0.5).float()

    # --- COUNT TOTAL OBJECTS ---
    total_objects = count_components(seg_mask)

    # --- LATENT VECTOR ---
    latent = unet.get_latent(img)

    # --- COUNT BROKEN OBJECTS ---
    broken_objects = counter(latent)
    broken_objects = int(torch.round(broken_objects).item())

actual_broken_objects = broken_labels[idx]
print("Inference results")
print("-----------------")
print(f"Total objects detected     : {total_objects}")
print(f"Broken objects estimated   : {broken_objects}")
print(f"Actual broken objects : {actual_broken_objects}")
