# UNet-Based Object Segmentation and Broken Component Counting

This project implements a small computer vision pipeline for:

1. Binary object segmentation using a U-Net  
2. Object counting via connected components  
3. Broken-object estimation using a latent-space feedforward network  

The system is designed for small synthetic datasets and focuses on architectural clarity and modularity.

---

## 📁 Project Structure

```
week2_app1_draft/
│
├── data/
│   ├── original/        # Input images
│   └── segmented/       # Corresponding binary masks
│
├── dataset.py           # Custom PyTorch Dataset
├── unet.py              # U-Net model + segmentation training
├── counter.py           # Feedforward network for broken-object counting
├── utils.py             # Loss functions and utility methods
├── main.py              # Training + inference pipeline
└── README.md
```

---

## Pipeline Overview

The system works in two stages.

### Stage 1 — Segmentation

A U-Net is trained in a supervised manner to perform binary segmentation:

- **Input:** image  
- **Output:** binary mask (object vs background)

Typical loss:
- Binary Cross-Entropy (BCE)
- Optionally Dice loss

---

### Stage 2 — Broken Object Counting

Instead of counting directly from segmentation masks, we:

1. Extract the **latent representation** from the U-Net bottleneck  
2. Apply Global Average Pooling  
3. Feed the resulting latent vector into a small feedforward network  
4. Use Poisson regression to predict the number of broken objects  

---

## U-Net Architecture

The U-Net consists of:

- Encoder (downsampling via stride-2 convolutions)
- Bottleneck
- Decoder (transpose convolutions + skip connections)
- Final 1×1 convolution for binary output

### Latent Representation

The bottleneck feature map has shape:

```
(base_ch * 2, H/4, W/4)
```

For example, with:

- Input size = 256×256  
- base_ch = 32  

The latent feature map is:

```
(64, 64, 64)
```

However, this is still a spatial tensor.

To obtain a fixed-size latent vector, we apply **Global Average Pooling**:

```
z_c = (1 / (H * W)) * sum_{i,j} F_{c,i,j}
```

Resulting latent vector size:

```
latent_dim = base_ch * 2
```

So if `base_ch = 32`, then:

```
latent_dim = 64
```

This vector is used as input to the counting network.

---

## 📦 Files Description

### `dataset.py`

- Loads aligned image–mask pairs  
- Ignores hidden folders (e.g. `.ipynb_checkpoints`)  
- Returns tensors:
  - image: `(1, H, W)`
  - mask: `(1, H, W)`

---

### `unet.py`

Contains:

- `ConvBlock`
- `UNet` model
- `get_latent()` method
- `train_unet()` function

`get_latent()`:

- Extracts bottleneck feature map  
- Applies global average pooling  
- Returns latent vector  

---

### `utils.py`

Contains:

- `dice_loss()`
- `poisson_loss()`
- `count_connected_components()`

Connected components are computed from the predicted binary segmentation to obtain total object count.

---

### `counter.py`

Contains:

- `BrokenCounter` (Feedforward network)
- `train_counter()` function

The counter:

- Takes latent vector as input  
- Outputs λ (Poisson rate)  
- Uses Softplus activation to ensure λ > 0  
- Trained with Poisson loss  

---

### `main.py`

Pipeline execution:

1. Load dataset  
2. Train U-Net  
3. Extract latent vectors  
4. Train broken-object counter  
5. Perform inference:
   - Segment image  
   - Count total objects  
   - Predict number of broken objects  

---

## 🔢 Loss Functions

### Dice Loss

```
Dice = (2 * sum(p * y)) / (sum(p) + sum(y))
Loss = 1 - Dice
```

Used to handle class imbalance in segmentation.

---

### Poisson Loss (for count regression)

```
L = λ - y * log(λ)
```

Where:

- `y` = true count  
- `λ` = predicted rate  


