"""Streamlit chat demo: query the RAG pipeline, see grounded answers + citations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from retrieval.answer import compose_answer  # noqa: E402
from retrieval.search import HybridRetriever  # noqa: E402

st.set_page_config(page_title="RAG Data Pipeline — Ops Runbook Search", layout="wide")
st.title("🔎 RAG Data Pipeline — Ops Runbook Search")
st.caption("Hybrid BM25 + dense retrieval over synthetic data-engineering runbooks. "
           "Answers are extractive and citation-grounded.")


def _ensure_index() -> None:
    """Build the corpus + index on first launch (e.g. Streamlit Community Cloud).

    Locally the pipeline is run via ``python pipeline/run.py``; on a fresh
    cloud instance the processed zone doesn't exist yet, so bootstrap it here.
    Heavy deps (torch, sentence-transformers) come from requirements.txt and
    the model downloads from the Hugging Face Hub on first run (~3-5 min).
    """
    import subprocess

    index_path = ROOT / "s3_mock" / "processed-zone" / "faiss.index"
    if index_path.exists():
        return
    with st.spinner("First launch: generating corpus and building the vector index "
                    "(one-time, ~3-5 min)…"):
        subprocess.run([sys.executable, "data/generate_corpus.py"],
                       cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "pipeline.run"],
                       cwd=ROOT, check=True)


@st.cache_resource
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


@st.cache_data
def corpus_stats() -> dict:
    manifest = json.loads((ROOT / "pipeline" / "manifest.json").read_text())
    ids = json.loads((ROOT / "s3_mock" / "processed-zone" / "chunk_ids.json").read_text())
    return {"docs": len(manifest["docs"]), "chunks": len(ids)}


_ensure_index()
retriever = get_retriever()
stats = corpus_stats()

with st.sidebar:
    st.header("Corpus stats")
    st.metric("Documents", stats["docs"])
    st.metric("Chunks indexed", stats["chunks"])
    st.metric("Embedding dim", 384)
    st.markdown("**Stack**: S3 zones (mocked) → chunk → MiniLM embeddings → "
                "FAISS + DuckDB → hybrid BM25+dense → Streamlit")
    st.markdown("Production path (Bedrock, OpenSearch, DynamoDB) is documented in the README.")

query = st.text_input("Ask about a pipeline incident",
                      placeholder="e.g. How do I troubleshoot consumer group lag climbing past 1M messages?")
k = st.slider("Top-k documents", 1, 10, 5)

if query:
    with st.spinner("Retrieving…"):
        sr = retriever.search(query, k=k)
        ans = compose_answer(sr)
    st.subheader("Answer")
    st.write(ans["answer"])
    st.caption(f"⏱ retrieval latency: {ans['latency_ms']} ms · "
               f"{ans['retrieved_docs']} docs considered")
    st.subheader("Citations")
    for c in ans["citations"]:
        st.markdown(f"- **[{c['doc_id']}]** {c['title']} — *{c['section']}* (score {c['score']})")
    with st.expander("Retrieved documents (raw chunks)"):
        for r in sr["results"]:
            st.markdown(f"**[{r['doc_id']}] {r['title']}** (doc score {r['score']})")
            for ch in r["chunks"]:
                st.markdown(f"*{ch['section']}* (chunk score {ch['score']})")
                st.write(ch["text"][:800])
            st.divider()
else:
    st.info("Try one of the eval questions, e.g. "
            "“How do I troubleshoot stage retries exhausted after shuffle read timeouts?”")
