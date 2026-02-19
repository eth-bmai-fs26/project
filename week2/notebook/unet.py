import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    """
    A simple UNet for 28x28 MNIST with Class Conditioning
    """
    def __init__(self, base_ch=64, emb_dim=64, num_classes=10): # <--- CHANGE 1: Add num_classes arg
        super().__init__()
        self.in_ch = 1      
        self.out_ch = 1     
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

        # Bottleneck
        self.bot = ConvBlock(base_ch * 2, base_ch * 2, emb_dim)    

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)     
        self.dec1 = ConvBlock(base_ch * 2 + base_ch * 2, base_ch, emb_dim)     
        self.up2 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)         
        self.dec2 = ConvBlock(base_ch + base_ch, base_ch, emb_dim)     

        self.out = nn.Conv2d(base_ch, self.out_ch, 1)    

    def forward(self, x, t, labels=None): # <--- CHANGE 3: Add labels argument
        """
        x: (B, 1, 28, 28)
        t: (B,)
        labels: (B,) IntTensor containing values 0-9
        """
        # 1. Create time embedding 
        time_emb = sinusoidal_time_embedding(t, self.emb_dim)
        
        # 2. Add Class conditioning
        if labels is not None:
            label_emb = self.label_emb(labels) # Lookup embedding: (B, emb_dim)
            time_emb = time_emb + label_emb    # Simple fusion: (B, emb_dim)
            
        # 3. Pass combined embedding through MLP
        cond_emb = self.cond_mlp(time_emb)

        # The rest of the network remains EXACTLY the same!
        # The 'cond_emb' now carries both time and class info to every block.
        
        # encoder
        e1 = self.enc1(x, cond_emb)                
        d1 = self.down1(e1)                     
        e2 = self.enc2(d1, cond_emb)               
        d2 = self.down2(e2)                     

        # bottleneck
        b = self.bot(d2, cond_emb)

        # decoder
        u1 = self.up1(b)                        
        u1 = torch.cat([u1, e2], dim=1)         
        u1 = self.dec1(u1, cond_emb)               
        u2 = self.up2(u1)                       
        u2 = torch.cat([u2, e1], dim=1)         
        u2 = self.dec2(u2, cond_emb)               

        return self.out(u2)

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
