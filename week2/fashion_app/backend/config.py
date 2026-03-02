"""
config.py
─────────
All constants, paths, and device configuration.
Import from here rather than defining values inline in other modules.
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
APP_DIR         = os.path.dirname(BASE_DIR)  # fashion_app/
OUTPUT_DIR      = os.path.join(APP_DIR, "output")
HISTORY_DIR     = os.path.join(APP_DIR, "history")
COMBINED_CSV    = os.path.join(APP_DIR, "data", "fashion_combined.csv")
EMBEDDINGS_PATH = os.path.join(APP_DIR, "data", "fashion_embeddings.npy")

# ── API ───────────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://litellm.sph-prod.ethz.ch/v1"

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please create a .env file with your key.")

# ── Model names ───────────────────────────────────────────────────────────────
FASHION_CLIP_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL         = "anthropic/claude-sonnet-4-5"
IMAGE_MODEL       = "azure/dall-e-3"

# ── CLIP quality threshold ────────────────────────────────────────────────────
CLIP_SCORE_THRESHOLD = 0.28   # cosine-sim rescaled to [0, 1]

# ── Device ────────────────────────────────────────────────────────────────────
# FIX: Force CPU — fashion-CLIP's projection head produces NaN pooler_output
# on Apple MPS due to float16 precision issues. CPU is fast enough at inference.
# NOTE: Do NOT change to "mps" even if torch.backends.mps.is_available().
device = "cuda" if torch.cuda.is_available() else "cpu"