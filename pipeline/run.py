#!/usr/bin/env python3
"""Orchestrator: ingest -> chunk -> embed -> index. Idempotent.

Unchanged documents are skipped at the ingest step (content-hash manifest)
and unchanged chunks are never re-embedded (embedding cache).
"""
from __future__ import annotations

from pipeline.chunk import chunk_all, load_ingested_docs
from pipeline.embed import embed_chunks
from pipeline.index import build_index
from pipeline.ingest import ingest


def main() -> None:
    summary = ingest()
    print(f"ingest: {summary['new']} new, {summary['unchanged']} unchanged, "
          f"{summary['invalid']} invalid, {summary['total_docs']} total docs")
    docs = load_ingested_docs()
    chunks = chunk_all(docs)
    print(f"chunked {len(docs)} docs -> {len(chunks)} chunks")
    embeddings, chunk_ids = embed_chunks(chunks)
    print(f"embeddings: {embeddings.shape}")
    build_index(embeddings, chunk_ids, chunks)
    print("pipeline complete: raw zone -> processed zone index + metadata")


if __name__ == "__main__":
    main()
