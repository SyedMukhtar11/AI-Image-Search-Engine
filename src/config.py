from pathlib import Path
import torch


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent


# Dataset
IMAGE_DIR = BASE_DIR / "data" / "images"

# Stored Week 4 embeddings
EMBEDDING_FILE = BASE_DIR / "data" / "image_embeddings.npy"
FILENAME_FILE = BASE_DIR / "data" / "image_filenames.npy"


# CLIP
MODEL_NAME = "openai/clip-vit-base-patch32"


# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Default number of search results
DEFAULT_TOP_K = 5