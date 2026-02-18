import torch
import torch.nn as nn
import torch.optim as optim
from week2.project1.utils import poisson_loss


class BrokenCounter(nn.Module):
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


def train_counter(unet, counter, dataloader, broken_labels, epochs=200):
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
                z = unet.get_latent(img)   # (B, latent_dim)

            lam = counter(z)
            loss = poisson_loss(lam, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if ep % 20 == 0:
            print(f"[Counter] Epoch {ep} | Loss: {total_loss:.4f}")
