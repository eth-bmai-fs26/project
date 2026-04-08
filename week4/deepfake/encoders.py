"""
Variational Autoencoder (VAE) and CLIP Encoder — PyTorch Implementation
========================================================================
This module provides clean, well-documented implementations of:
  1. A convolutional Variational Autoencoder (VAE)
  2. A CLIP-style dual-encoder (image + text) with contrastive learning

Both models are fully functional and can be trained on real data.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PART 1 — Variational Autoencoder (VAE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Encoder(nn.Module):
    """
    Convolutional encoder that maps an input image to the parameters
    (mean μ, log-variance log σ²) of a diagonal Gaussian in latent space.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        layers = []
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = h_dim

        self.conv_layers = nn.Sequential(*layers)

        # Projection to latent distribution parameters
        # For a 64×64 input the spatial size after 5 stride-2 convs is 2×2
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_layers(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Transposed-convolutional decoder that maps a latent vector z back to
    image space.
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]

        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 4)
        self.initial_channels = hidden_dims[0]

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        # Final layer → pixel values in [0, 1]
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Sigmoid(),
            )
        )

        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.initial_channels, 2, 2)
        return self.deconv_layers(h)


class VAE(nn.Module):
    """
    Full Variational Autoencoder.

    Architecture
    ────────────
    Encoder  →  μ, log σ²  →  reparameterize  →  z  →  Decoder  →  x̂

    Loss = Reconstruction (BCE) + β · KL divergence

    Args:
        in_channels:  Number of image channels (1 for grayscale, 3 for RGB).
        latent_dim:   Dimensionality of the latent space.
        hidden_dims:  Channel sizes for each conv block.
        beta:         Weight on the KL term (β-VAE when β > 1).
        img_size:     Expected spatial resolution (default 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        beta: float = 1.0,
        img_size: int = 64,
    ):
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.encoder = Encoder(in_channels, latent_dim, hidden_dims)
        decoder_hidden = list(reversed(hidden_dims)) if hidden_dims else None
        self.decoder = Decoder(in_channels, latent_dim, decoder_hidden)

    # ── reparameterization trick ──────────────────────────────────────────
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + σ · ε,  where ε ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    # ── forward pass ──────────────────────────────────────────────────────
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    # ── loss ──────────────────────────────────────────────────────────────
    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, reconstruction_loss, kl_divergence).
        """
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total = recon_loss + self.beta * kl
        return total, recon_loss, kl

    # ── utilities ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, n: int = 16, device: str = "cpu") -> torch.Tensor:
        """Generate n images by sampling z ~ N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode — useful for evaluation."""
        x_recon, _, _ = self.forward(x)
        return x_recon


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PART 2 — CLIP Encoder (Contrastive Language-Image Pretraining)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PatchEmbedding(nn.Module):
    """Split an image into fixed-size patches and linearly embed each one."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with pre-norm compatible interface."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN → MHSA → residual → LN → FFN → residual."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) used as the CLIP image encoder.

    Prepends a learnable [CLS] token and adds positional embeddings.
    The [CLS] token's final representation serves as the image embedding.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.ln_post = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        # Return [CLS] token representation
        return self.ln_post(x[:, 0])


class TextTransformer(nn.Module):
    """
    Transformer-based text encoder for CLIP.

    Uses causal (auto-regressive) attention masking.  The representation
    at the [EOS] position is taken as the text embedding.
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        max_seq_len: int = 77,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.ln_final = nn.LayerNorm(embed_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) integer token ids.
        Returns:
            (B, embed_dim) text embedding taken from the last non-padding position.
        """
        B, L = input_ids.shape
        x = self.token_embed(input_ids) + self.pos_embed[:, :L]
        mask = self._causal_mask(L, x.device)

        for blk in self.blocks:
            x = blk(x, mask)

        x = self.ln_final(x)

        # Gather the embedding at the [EOS] position (last token per sample)
        # In practice, [EOS] is placed right after the real tokens.
        eos_indices = input_ids.argmax(dim=-1)  # highest token id ≈ EOS
        x = x[torch.arange(B, device=x.device), eos_indices]
        return x


class CLIPModel(nn.Module):
    """
    CLIP — Contrastive Language-Image Pretraining.

    Architecture
    ────────────
    Image  →  ViT  →  projection  →  l2-normalise  ─┐
                                                      ├─ cosine similarity matrix
    Text   →  TextTransformer  →  projection  →  l2  ─┘

    Trained with a symmetric cross-entropy loss (InfoNCE) over the
    image–text similarity matrix.

    Args:
        embed_dim:       Joint embedding dimensionality.
        vision_cfg:      Kwargs for VisionTransformer.
        text_cfg:        Kwargs for TextTransformer.
        temperature_init: Initial value for the learnable temperature τ.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        vision_cfg: Optional[dict] = None,
        text_cfg: Optional[dict] = None,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        vision_cfg = vision_cfg or {}
        text_cfg = text_cfg or {}

        v_dim = vision_cfg.get("embed_dim", 768)
        t_dim = text_cfg.get("embed_dim", 512)

        self.visual = VisionTransformer(**vision_cfg)
        self.text = TextTransformer(**text_cfg)

        self.visual_proj = nn.Linear(v_dim, embed_dim, bias=False)
        self.text_proj = nn.Linear(t_dim, embed_dim, bias=False)

        # Learnable temperature (log scale for numerical stability)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature_init)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to l2-normalised embeddings."""
        x = self.visual(images)
        x = self.visual_proj(x)
        return F.normalize(x, dim=-1)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode token sequences to l2-normalised embeddings."""
        x = self.text(input_ids)
        x = self.text_proj(x)
        return F.normalize(x, dim=-1)

    def forward(
        self, images: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits_per_image: (B, B) cosine similarities scaled by τ.
            logits_per_text:  (B, B) transpose of the above.
        """
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def contrastive_loss(
        self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor
    ) -> torch.Tensor:
        """
        Symmetric InfoNCE loss.

        The ground-truth pairing is the diagonal of the similarity matrix:
        image_i ↔ text_i.
        """
        B = logits_per_image.size(0)
        labels = torch.arange(B, device=logits_per_image.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PART 3 — Training helpers & demo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def train_vae(
    model: VAE,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    """Train the VAE and return per-epoch losses."""
    model.to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            x_recon, mu, logvar = model(images)
            loss, recon, kl = model.loss_function(images, x_recon, mu, logvar)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        history.append(avg)
        print(f"[VAE] Epoch {epoch:>3}/{epochs}  loss={avg:.4f}")

    return history


def train_clip(
    model: CLIPModel,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 3e-4,
    device: str = "cpu",
) -> List[float]:
    """Train CLIP and return per-epoch losses."""
    model.to(device).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for images, tokens in dataloader:
            images, tokens = images.to(device), tokens.to(device)
            logits_img, logits_txt = model(images, tokens)
            loss = model.contrastive_loss(logits_img, logits_txt)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        history.append(avg)
        print(f"[CLIP] Epoch {epoch:>3}/{epochs}  loss={avg:.4f}")

    return history


# ── quick demo on random data ────────────────────────────────────────────────

def demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # ── VAE demo ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  VAE Demo — random 64×64 RGB images")
    print("=" * 60)

    vae = VAE(in_channels=3, latent_dim=64, hidden_dims=[32, 64, 128, 256, 512], beta=1.0)
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}\n")

    fake_images = torch.rand(128, 3, 64, 64)
    vae_loader = DataLoader(TensorDataset(fake_images), batch_size=32, shuffle=True)

    train_vae(vae, vae_loader, epochs=5, device=device)

    samples = vae.sample(4, device=device)
    print(f"\nGenerated {samples.shape[0]} samples of shape {samples.shape[1:]}")

    # ── CLIP demo ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CLIP Demo — random 224×224 images + token ids")
    print("=" * 60)

    clip = CLIPModel(
        embed_dim=256,
        vision_cfg=dict(img_size=224, patch_size=16, embed_dim=384, depth=6, num_heads=6),
        text_cfg=dict(vocab_size=10000, max_seq_len=32, embed_dim=256, depth=4, num_heads=4),
    )
    print(f"CLIP parameters: {sum(p.numel() for p in clip.parameters()):,}\n")

    fake_imgs = torch.rand(64, 3, 224, 224)
    fake_tokens = torch.randint(0, 10000, (64, 32))
    clip_loader = DataLoader(TensorDataset(fake_imgs, fake_tokens), batch_size=16, shuffle=True)

    train_clip(clip, clip_loader, epochs=3, device=device)

    # Encode a single image / text pair
    with torch.no_grad():
        img_emb = clip.encode_image(fake_imgs[:1].to(device))
        txt_emb = clip.encode_text(fake_tokens[:1].to(device))
        sim = (img_emb @ txt_emb.t()).item()
        print(f"\nCosine similarity (random pair): {sim:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    demo()

