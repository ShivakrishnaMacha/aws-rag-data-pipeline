#!/usr/bin/env python3
"""Embedding stage: batched sentence-transformer embeddings with a disk cache.

Model: sentence-transformers/all-MiniLM-L6-v2 (384 dims, Apache-2.0).
The cache (processed/embeddings_cache.jsonl) makes re-runs idempotent:
unchanged chunk_ids are never re-embedded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "s3_mock" / "processed-zone"
CACHE_PATH = PROCESSED / "embeddings_cache.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_cache() -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    if CACHE_PATH.exists():
        with CACHE_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cache[rec["chunk_id"]] = rec["embedding"]
    return cache


def save_cache(cache: dict[str, list[float]]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as fh:
        for chunk_id, emb in cache.items():
            fh.write(json.dumps({"chunk_id": chunk_id, "embedding": emb}) + "\n")


def embed_chunks(chunks: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings, chunk_ids) for all chunks, using the cache."""
    model = get_model()
    cache = load_cache()
    missing = [c for c in chunks if c["chunk_id"] not in cache]
    t0 = time.time()
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vecs = model.encode(texts, batch_size=len(texts), show_progress_bar=False,
                            normalize_embeddings=True)
        for c, v in zip(batch, vecs):
            cache[c["chunk_id"]] = [float(x) for x in v]
        print(f"embedded {min(i + BATCH_SIZE, len(missing))}/{len(missing)} chunks "
              f"({time.time() - t0:.1f}s)", flush=True)
    save_cache(cache)
    ids = [c["chunk_id"] for c in chunks]
    mat = np.array([cache[cid] for cid in ids], dtype=np.float32)
    return mat, ids


if __name__ == "__main__":
    from pipeline.chunk import chunk_all, load_ingested_docs

    chunks = chunk_all(load_ingested_docs())
    mat, ids = embed_chunks(chunks)
    print(f"embeddings: {mat.shape}, cache size: {len(load_cache())}")
