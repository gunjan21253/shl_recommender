"""
Vector Store Builder
====================
Builds two complementary search indexes:
1. FAISS dense vector index (semantic similarity via sentence-transformers)
2. BM25 sparse index (keyword matching via rank_bm25)

The hybrid of both gives significantly better Recall@10 than either alone.

Usage:
    python -m recommender.build_index
"""

import json
import pickle
import logging
import re
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_FILE = DATA_DIR / "shl_catalog.json"
INDEX_FILE = DATA_DIR / "shl_index.faiss"
ASSESSMENTS_FILE = DATA_DIR / "assessments.pkl"
BM25_FILE = DATA_DIR / "bm25_index.pkl"
TEXTS_FILE = DATA_DIR / "texts.pkl"

EMBEDDING_MODEL = "all-mpnet-base-v2"

TEST_TYPE_MAP = {
    "A": "Ability and Aptitude cognitive reasoning",
    "B": "Biodata and Situational Judgement behavioral",
    "C": "Competencies leadership management",
    "D": "Development and 360 feedback",
    "E": "Assessment Exercises simulation practical",
    "K": "Knowledge and Skills technical proficiency",
    "P": "Personality and Behavior traits values culture",
    "S": "Simulations realistic job preview",
}

ROLE_KEYWORDS = {
    "developer": "software engineer programmer coding",
    "analyst": "data analysis reporting insights",
    "sales": "customer relationship revenue persuasion",
    "manager": "leadership team management strategy",
    "executive": "C-suite director senior leadership vision",
    "customer service": "support empathy communication resolution",
    "finance": "accounting numbers budgeting risk",
    "hr": "human resources recruitment talent people",
}


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_embedding_text(assessment: dict) -> str:
    """
    Create a rich, information-dense text for embedding.
    Quality of this text directly impacts retrieval quality.
    """
    name = assessment.get("name", "")
    description = assessment.get("description", "")
    test_types = assessment.get("test_types", [])
    duration = assessment.get("duration", 0)
    remote = assessment.get("remote_support", "No")
    adaptive = assessment.get("adaptive_support", "No")
    languages = assessment.get("languages", [])

    type_descriptions = [TEST_TYPE_MAP.get(t, t) for t in test_types]

    name_lower = name.lower()
    inferred_roles = []
    for keyword, expansion in ROLE_KEYWORDS.items():
        if keyword in name_lower:
            inferred_roles.append(expansion)

    text_parts = [
        f"Assessment Name: {name}",
        f"Test Categories: {', '.join(type_descriptions)}" if type_descriptions else "",
        f"Description: {description}" if description else "",
        f"Duration: {duration} minutes" if duration else "",
        f"Remote Testing: {remote}",
        f"Adaptive Testing: {adaptive}",
        f"Languages: {', '.join(languages)}" if languages else "",
        f"Related Skills: {', '.join(inferred_roles)}" if inferred_roles else "",
    ]

    return clean_text(" | ".join([p for p in text_parts if p]))


def create_bm25_tokens(text: str) -> list:
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return [t for t in tokens if len(t) > 1]


def build_indexes(assessments: list):
    """Build FAISS dense + BM25 sparse indexes."""
    log.info(f"Building indexes for {len(assessments)} assessments...")

    texts = [create_embedding_text(a) for a in assessments]

    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    log.info("Generating embeddings (this takes ~2-5 minutes)...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    log.info(f"FAISS index: {faiss_index.ntotal} vectors of dimension {dimension}")

    log.info("Building BM25 index...")
    tokenized_texts = [create_bm25_tokens(t) for t in texts]
    bm25_index = BM25Okapi(tokenized_texts)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(INDEX_FILE))
    log.info(f"Saved FAISS index -> {INDEX_FILE}")

    with open(ASSESSMENTS_FILE, "wb") as f:
        pickle.dump(assessments, f)
    log.info(f"Saved assessments -> {ASSESSMENTS_FILE}")

    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25_index, f)
    log.info(f"Saved BM25 index -> {BM25_FILE}")

    with open(TEXTS_FILE, "wb") as f:
        pickle.dump(texts, f)
    log.info(f"Saved texts -> {TEXTS_FILE}")

    return faiss_index, bm25_index, assessments


def main():
    if not CATALOG_FILE.exists():
        log.error(f"Catalog file not found: {CATALOG_FILE}")
        log.error("Run: python -m scraper.catalog_scraper")
        return

    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    log.info(f"Loaded {len(assessments)} assessments from catalog")

    if len(assessments) < 100:
        log.warning("Very few assessments - scraping may have failed")

    build_indexes(assessments)
    log.info("Index build complete!")


if __name__ == "__main__":
    main()
