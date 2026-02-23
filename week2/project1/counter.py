import torch
import torch.nn as nn
import torch.optim as optim
from week2.project1.utils import poisson_loss


class BrokenCounter(nn.Module):
    """
    FNN-based broken-object counter.
    Takes a latent VECTOR (B, latent_dim) produced by UNet.get_latent()
    (i.e. after global average pooling) and regresses a Poisson rate.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()   # ensures positive Poisson rate
        )

    def forward(self, z):
        return self.net(z).squeeze(1)


class BrokenCounterCNN(nn.Module):
    """
    CNN-based broken-object counter.
    Takes the spatial bottleneck feature MAP (B, in_channels, H', W') produced
    by UNet.get_latent_map() — no prior global average pooling — and regresses
    a Poisson rate. Two convolutional layers process spatial structure before
    the final global pooling step, giving the network access to the spatial
    distribution of activations (e.g. fragment density, cluster locations).
    """

    def __init__(self, in_channels: int = 64):
        """
        Args:
            in_channels: number of channels in the latent map.
                         Default 64 matches UNet with base_ch=32.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 64x64 -> 32x32
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                   # 32x32 -> 1x1 (GAP)
        )
        self.head = nn.Sequential(
            nn.Flatten(),                              # (B, 16, 1, 1) -> (B, 16)
            nn.Linear(16, 1),
            nn.Softplus()                              # ensures positive Poisson rate
        )

    def forward(self, z_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_map: latent feature map, shape (B, in_channels, H', W')
        Returns:
            Predicted Poisson rate (count), shape (B,)
        """
        x = self.features(z_map)
        return self.head(x).squeeze(1)


def train_counter(unet, counter, dataloader, broken_labels, epochs=200):
    """
    Train the FNN BrokenCounter using pooled latent vectors from UNet.
    UNet weights are frozen (eval mode, no_grad).

    Args:
        unet: trained UNet instance (frozen)
        counter: BrokenCounter instance to train
        dataloader: DataLoader yielding (img, mask) pairs, batch_size=1
        broken_labels: list of ground-truth counts, one per sample
        epochs: number of training epochs
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unet.eval()
    counter.to(device)

    optimizer = optim.Adam(counter.parameters(), lr=1e-3)

    for ep in range(epochs):
        total_loss = 0.0

        for i, (img, _) in enumerate(dataloader):
            img = img.to(device)
            y = torch.tensor([broken_labels[i]], dtype=torch.float, device=device)

            with torch.no_grad():
                z = unet.get_latent(img)   # (B, latent_dim) — pooled vector

            lam = counter(z)
            loss = poisson_loss(lam, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if ep % 20 == 0:
            print(f"[CounterFNN] Epoch {ep} | Loss: {total_loss:.4f}")


def train_counter_cnn(unet, counter_cnn, dataloader, broken_labels, epochs=200, lr=1e-3):
    """
    Train the CNN BrokenCounterCNN using the spatial bottleneck map from UNet.
    UNet weights are frozen (eval mode, no_grad).

    Args:
        unet: trained UNet instance (frozen)
        counter_cnn: BrokenCounterCNN instance to train
        dataloader: DataLoader yielding (img, mask) pairs, batch_size=1
        broken_labels: list of ground-truth counts, one per sample
        epochs: number of training epochs
        lr: learning rate for Adam
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unet.eval()
    counter_cnn.to(device)

    optimizer = optim.Adam(counter_cnn.parameters(), lr=lr)

    for ep in range(epochs):
        total_loss = 0.0

        for i, (img, _) in enumerate(dataloader):
            img = img.to(device)
            y = torch.tensor([broken_labels[i]], dtype=torch.float, device=device)

            with torch.no_grad():
                z_map = unet.get_latent_map(img)   # (B, latent_dim, H', W') — spatial map

            lam = counter_cnn(z_map)
            loss = poisson_loss(lam, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if ep % 20 == 0:
            print(f"[CounterCNN] Epoch {ep} | Loss: {total_loss:.4f}")
