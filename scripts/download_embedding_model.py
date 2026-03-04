"""
Download the embedding model into project model_cache so Render (and local) load from disk.
Run during build: pip install -r requirements.txt && python scripts/download_embedding_model.py
Then at runtime the engine finds the model in model_cache (no download, faster startup).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = PROJECT_ROOT / "model_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_CACHE)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

print("Downloading embedding model into", MODEL_CACHE, "...")

from sentence_transformers import SentenceTransformer

SentenceTransformer("all-mpnet-base-v2")
print("Done. Model cached at", MODEL_CACHE)
