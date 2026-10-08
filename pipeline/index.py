#!/usr/bin/env python3
"""Index stage: FAISS dense index + DuckDB chunk metadata.

- FAISS ``IndexFlatIP`` over L2-normalized embeddings (= cosine similarity).
- Chunk metadata (chunk_id, doc_id, title, topic, section, offsets, text)
  goes to DuckDB ``metadata.duckdb``. Production swap: DynamoDB table with
  ``chunk_id`` as the partition key — same access pattern, pay-per-request.
- Prints index stats: n_chunks, dim, build time, index size on disk.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "s3_mock" / "processed-zone"
INDEX_PATH = PROCESSED / "faiss.index"
IDS_PATH = PROCESSED / "chunk_ids.json"
DB_PATH = PROCESSED / "metadata.duckdb"


def build_index(embeddings: np.ndarray, chunk_ids: list[str], chunks: list[dict]) -> dict:
    t0 = time.time()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(INDEX_PATH))
    IDS_PATH.write_text(json.dumps(chunk_ids), encoding="utf-8")

    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS chunks")
    con.execute(
        """CREATE TABLE chunks (
               chunk_id VARCHAR PRIMARY KEY, doc_id VARCHAR, title VARCHAR,
               topic VARCHAR, section VARCHAR, char_start INTEGER,
               char_end INTEGER, text VARCHAR)"""
    )
    rows = [
        (c["chunk_id"], c["doc_id"], c["title"], c["topic"], c["section"],
         c["char_start"], c["char_end"], c["text"])
        for c in chunks
    ]
    con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.close()

    build_s = time.time() - t0
    stats = {
        "n_chunks": len(chunk_ids),
        "dim": dim,
        "build_time_s": round(build_s, 2),
        "index_bytes": INDEX_PATH.stat().st_size,
        "db_bytes": DB_PATH.stat().st_size,
    }
    print(json.dumps(stats, indent=2))
    return stats


def load_index() -> tuple[faiss.IndexFlatIP, list[str]]:
    index = faiss.read_index(str(INDEX_PATH))
    chunk_ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    return index, chunk_ids


if __name__ == "__main__":
    from pipeline.chunk import chunk_all, load_ingested_docs
    from pipeline.embed import embed_chunks

    chunks = chunk_all(load_ingested_docs())
    mat, ids = embed_chunks(chunks)
    build_index(mat, ids, chunks)
