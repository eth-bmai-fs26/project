import torch
import torch.nn as nn
import torch.optim as optim
from week2.utils import dice_loss


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, 2, 1)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, 2, 1)
        self.dec1 = ConvBlock(base_ch * 4, base_ch)

        self.up2 = nn.ConvTranspose2d(base_ch, base_ch, 4, 2, 1)
        self.dec2 = ConvBlock(base_ch * 2, base_ch)

        # Output (binary segmentation)
        self.out = nn.Conv2d(base_ch, 1, 1)

        # Latent vector size after Global Average Pooling
        self.latent_dim = base_ch * 2

    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        b = self.bottleneck(d2)

        u1 = self.up1(b)
        u1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(u2)

        return self.out(d2)

    def get_latent(self, x):
        """
        Returns the latent VECTOR obtained by
        averaging the bottleneck feature map
        over spatial dimensions.
        """
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        b = self.bottleneck(d2)          # (B, C, H/4, W/4)
        z = b.mean(dim=(2, 3))            # Global Average Pooling → (B, C)
        return z


def train_unet(model, dataloader, epochs=200, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0

        for img, mask in dataloader:
            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)
            loss = bce(pred, mask) + dice_loss(pred, mask).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if ep % 20 == 0:
            print(f"[UNet] Epoch {ep} | Loss: {epoch_loss:.4f}")
