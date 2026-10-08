"""Unit tests for the RAG pipeline: chunking, dedupe, fusion scoring."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.chunk import chunk_text  # noqa: E402
from pipeline.ingest import validate_doc  # noqa: E402


def test_chunk_overlap_math():
    text = " ".join(f"sentence number {i} with some filler words." for i in range(60))
    chunks = chunk_text(text, chunk_size=200, overlap=60)
    assert len(chunks) > 1
    # consecutive chunks must overlap in content
    for (t1, s1, e1), (t2, s2, e2) in zip(chunks, chunks[1:]):
        words1 = set(t1.split())
        words2 = set(t2.split())
        assert words1 & words2, "consecutive chunks must share overlapping words"
        assert s2 < e1, "offsets must overlap"
    # offsets stay within the text
    for _, s, e in chunks:
        assert 0 <= s < e <= len(text)


def test_chunk_empty_and_short():
    assert chunk_text("") == []
    short = chunk_text("hello world", chunk_size=600, overlap=120)
    assert len(short) == 1 and short[0][0] == "hello world"


def test_validate_doc_rejects_bad_sections():
    bad = {"doc_id": "D1", "title": "T", "topic": "x",
           "body": "# T\n\nno required sections here\n"}
    good = {"doc_id": "D1", "title": "T", "topic": "x",
            "body": "# T\n\n## Symptoms\ny\n\n## Likely root cause\nz\n\n## Remediation\nw\n"}
    errors = validate_doc(bad)
    assert any("Symptoms" in e for e in errors)
    assert validate_doc(good) == []


def test_dedupe_by_content_hash(tmp_path):
    from pipeline.ingest import sha256_of

    f1 = tmp_path / "a.md"
    f2 = tmp_path / "b.md"
    f1.write_text("same content")
    f2.write_text("same content")
    assert sha256_of(f1) == sha256_of(f2)
    f2.write_text("different content")
    assert sha256_of(f1) != sha256_of(f2)


def test_fusion_scoring_deterministic():
    # fusion must be deterministic and bounded in [0, 1]; exercise the real
    # HybridRetriever.fuse / _minmax implementations (no model needed).
    from retrieval.search import HybridRetriever, scatter_scores

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.alpha = 0.6
    dense = np.array([0.9, 0.1, 0.5], dtype=np.float32)
    bm25 = np.array([2.0, 8.0, 5.0], dtype=np.float32)
    fused = retriever.fuse(dense, bm25)
    assert fused.shape == dense.shape
    assert np.all(fused >= 0) and np.all(fused <= 1)
    # best-on-both-channels doc must win
    assert int(np.argmax(fused)) == 0
    assert np.allclose(fused, retriever.fuse(dense, bm25))  # deterministic


def test_scatter_scores_restores_chunk_order():
    # Regression test: FAISS returns (distances, labels) sorted best-first;
    # scatter_scores must restore per-chunk order before BM25 fusion.
    from retrieval.search import scatter_scores

    # chunk rows 0..4, FAISS returns them sorted by score with labels
    dist = np.array([0.9, 0.7, 0.5, 0.3, 0.1], dtype=np.float32)
    labels = np.array([3, 0, 4, 1, 2])
    out = scatter_scores(dist, labels, 5)
    assert np.allclose(out, [0.7, 0.3, 0.1, 0.9, 0.5])
    # naive use of the sorted array would misattribute every chunk's score
    assert not np.allclose(out, dist)
