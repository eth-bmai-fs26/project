import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    """
    A simple UNet with Class Conditioning.
    Supports grayscale (in_channels=1) and RGB (in_channels=3) images.

    args:
    - base_ch: number of channels in the first layer (doubles every downsample)
    - emb_dim: dimension of the time and class embedding
    - num_classes: number of classes for conditioning (default 10 for MNIST)
    - in_channels: number of input image channels (1=grayscale, 3=RGB)
    
    note: output channels are always the same as input channels, since we're predicting noise which has the same shape as the input image.
    """
    def __init__(self, base_ch=128, emb_dim=64, num_classes=10, in_channels=1):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = in_channels
        self.base_ch = base_ch  
        self.emb_dim = emb_dim    

        # This learns a unique vector of size 'emb_dim' for each class (0-9).
        self.label_emb = nn.Embedding(num_classes, emb_dim)

        self.cond_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        # Encoder
        self.enc1 = ConvBlock(self.in_ch, base_ch, emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)

        self.enc2 = ConvBlock(base_ch, base_ch * 2, emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)

        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, emb_dim)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)

        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, emb_dim)
        self.down4 = nn.Conv2d(base_ch * 8, base_ch * 8, 4, stride=2, padding=1)

        # Bottleneck
        self.bot = ConvBlock(base_ch * 8, base_ch * 8, emb_dim)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 4, stride=2, padding=1)
        self.dec4 = ConvBlock(base_ch * 8 + base_ch * 8, base_ch * 4, emb_dim)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)
        self.dec3 = ConvBlock(base_ch * 4 + base_ch * 4, base_ch * 2, emb_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 2 + base_ch * 2, base_ch, emb_dim)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch + base_ch, base_ch, emb_dim)

        self.out = nn.Conv2d(base_ch, self.out_ch, 1)    

    def forward(self, x, t, labels):
        """
        x: (B, C, H, W)
        t: (B,)
        labels: (B,) IntTensor containing class indices (0-9)
        """
        
        # create time and label embeddings and combined them into a single conditioning embedding
        time_emb = sinusoidal_time_embedding(t, self.emb_dim)
        label_emb = self.label_emb(labels)
        cond_emb = self.cond_mlp(time_emb + label_emb)
        
        # encoder
        e1 = self.enc1(x, cond_emb)            # (B, base_ch)
        d1 = self.down1(e1)                    # (B, base_ch)
        e2 = self.enc2(d1, cond_emb)           # (B, base_ch * 2)
        d2 = self.down2(e2)                    # (B, base_ch * 2)
        e3 = self.enc3(d2, cond_emb)           # (B, base_ch * 4)
        d3 = self.down3(e3)                    # (B, base_ch * 4)
        e4 = self.enc4(d3, cond_emb)           # (B, base_ch * 8)
        d4 = self.down4(e4)                    # (B, base_ch * 8)

        # bottleneck
        b = self.bot(d4, cond_emb)             # (B, base_ch * 8)

        # decoder
        u4 = self.up4(b)                                               # (B, base_ch * 4,  ~8,  ~8)
        u4 = F.interpolate(u4, size=e4.shape[2:], mode='nearest')     # align to e3's spatial size
        u4 = torch.cat([u4, e4], dim=1)                               # (B, base_ch * 8,   8,   8)
        u4 = self.dec4(u4, cond_emb)
        
        u3 = self.up3(u4)                                             # (B, base_ch * 2,  16,  16)
        u3 = torch.cat([u3, e3], dim=1)                               # (B, base_ch * 4,  16,  16)
        u3 = self.dec3(u3, cond_emb)                                  # (B, base_ch * 2,   8,   8)

        u2 = self.up2(u3)                                             # (B, base_ch * 2,  16,  16)
        u2 = torch.cat([u2, e2], dim=1)                               # (B, base_ch * 4,  16,  16)
        u2 = self.dec2(u2, cond_emb)                                  # (B, base_ch,      16,  16)

        u1 = self.up1(u2)                                             # (B, base_ch,      32,  32)
        u1 = torch.cat([u1, e1], dim=1)                               # (B, base_ch * 2,  32,  32)
        u1 = self.dec1(u1, cond_emb)                                  # (B, base_ch,      32,  32)

        return self.out(u1)                                           # (B,   C,          32,  32)

class ConvBlock(nn.Module):
    """
    Helper class which handles the convolution operation for the UNet.
    """
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
        )
        # FiLM-style conditioning from embedding -> scale & shift
        self.to_scale = nn.Linear(emb_dim, out_ch)
        self.to_shift = nn.Linear(emb_dim, out_ch)

    def forward(self, x, cond_emb):
        h = self.conv(x)
        # reshape FiLM params to (B, C, 1, 1) --> for each image in batch (noising is done differently per image), for each channel, scale and shift each entire feature map!
        scale = self.to_scale(cond_emb).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(cond_emb).unsqueeze(-1).unsqueeze(-1)
        return h * (1 + scale) + shift


# --- Sinusoidal timestep embedding (classic DDPM-style) ---
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Helper function which creates transformer like positional encodings.

    timesteps: (B,) long
    returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0,1))
    return emb  # (B, dim)
