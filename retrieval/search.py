"""Hybrid retrieval: BM25 (lexical) + dense cosine (semantic), fused and ranked.

Score fusion: min-max normalize each channel to [0, 1], then
fused = ALPHA * dense + (1 - ALPHA) * bm25. Top-k returned with citations.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "s3_mock" / "processed-zone"

ALPHA = 0.6  # weight on dense channel


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def scatter_scores(dist: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    """Scatter FAISS (sorted distance, label) pairs back to per-row order.

    FAISS ``index.search`` returns distances sorted best-first along with the
    row labels; fusing with per-chunk BM25 requires scores aligned to chunk
    order, hence this scatter. Getting this wrong silently pairs each chunk
    with another chunk's dense score.
    """
    scores = np.zeros(n, dtype=np.float32)
    scores[labels] = dist
    return scores


class HybridRetriever:
    def __init__(self, alpha: float = ALPHA):
        from pipeline.embed import get_model
        from pipeline.index import IDS_PATH, load_index

        self.alpha = alpha
        self.index, self.chunk_ids = load_index()
        self.model = get_model()
        con = duckdb.connect(str(PROCESSED / "metadata.duckdb"), read_only=True)
        rows = con.execute(
            "SELECT chunk_id, doc_id, title, topic, section, text FROM chunks"
        ).fetchall()
        con.close()
        self.meta = {
            r[0]: {"chunk_id": r[0], "doc_id": r[1], "title": r[2],
                   "topic": r[3], "section": r[4], "text": r[5]}
            for r in rows
        }
        # keep only chunks present in the FAISS id list, in that order
        self.ordered = [self.meta[cid] for cid in self.chunk_ids if cid in self.meta]
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.ordered])

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def fuse(self, dense: np.ndarray, bm25: np.ndarray) -> np.ndarray:
        return self.alpha * self._minmax(dense) + (1 - self.alpha) * self._minmax(bm25)

    def search(self, query: str, k: int = 5, chunks_per_doc: int = 2) -> dict:
        """Hybrid search with document-level score aggregation.

        Chunks are scored by fused BM25+dense, then scores are *summed* per
        document: a doc whose chunks together cover all query terms outranks
        docs that match only part of the query. This is the standard
        production pattern for chunked corpora (retrieve docs, then passages).
        """
        t0 = time.time()
        qvec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        dist, labels = self.index.search(qvec, len(self.ordered))
        dense_scores = scatter_scores(dist[0], labels[0], len(self.ordered))
        bm25_scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float32)
        fused = self.fuse(dense_scores, bm25_scores)

        # Aggregate chunk scores per document (sum), keep best chunks per doc.
        doc_scores: dict[str, float] = {}
        doc_chunks: dict[str, list[tuple[int, float]]] = {}
        for i, c in enumerate(self.ordered):
            doc_scores[c["doc_id"]] = doc_scores.get(c["doc_id"], 0.0) + float(fused[i])
            doc_chunks.setdefault(c["doc_id"], []).append((i, float(fused[i])))
        ranked_docs = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]

        results = []
        for rank, (doc_id, doc_score) in enumerate(ranked_docs, start=1):
            top_chunks = sorted(doc_chunks[doc_id], key=lambda t: t[1], reverse=True)[:chunks_per_doc]
            first = self.ordered[top_chunks[0][0]]
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "title": first["title"],
                    "topic": first["topic"],
                    "score": round(doc_score, 4),
                    "chunks": [
                        {
                            "chunk_id": self.ordered[i]["chunk_id"],
                            "section": self.ordered[i]["section"],
                            "text": self.ordered[i]["text"],
                            "score": round(s, 4),
                        }
                        for i, s in top_chunks
                    ],
                }
            )
        return {
            "query": query,
            "results": results,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
