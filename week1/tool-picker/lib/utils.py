"""General-purpose helpers for the Tool Router notebook."""

import random
import numpy as np
import torch


def set_seed(seed):
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
