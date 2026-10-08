#!/usr/bin/env python3
"""Chunking stage: recursive character chunking with overlap.

Chunk ID scheme: ``<doc_id>#<section>#<n>`` where n is the ordinal chunk index
within the section. Metadata recorded per chunk: doc_id, title, topic,
section, char offsets, chunk length.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120


def split_into_sections(body: str) -> list[tuple[str, str]]:
    """Split a markdown runbook into (section_heading, section_text) pairs."""
    sections: list[tuple[str, str]] = []
    current_heading = "preamble"
    current_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines or not sections:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return sections


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[str, int, int]]:
    """Chunk text into (chunk_text, start_offset, end_offset) with overlap.

    Splits greedily on sentence boundaries first, then on whitespace, so chunks
    end at natural breaks whenever possible.
    """
    if not text.strip():
        return []
    # Build sentence-ish pieces.
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks: list[tuple[str, int, int]] = []
    current: list[str] = []
    current_len = 0

    def flush(pieces: list[str]) -> None:
        joined = " ".join(pieces)
        start = text.find(joined)
        if start < 0:
            start = 0
        chunks.append((joined, start, start + len(joined)))

    for sent in sentences:
        if current and current_len + 1 + len(sent) > chunk_size:
            flush(current)
            # overlap: keep trailing pieces that fit within `overlap` chars
            kept: list[str] = []
            kept_len = 0
            for piece in reversed(current):
                if kept_len + len(piece) + 1 > overlap:
                    break
                kept.insert(0, piece)
                kept_len += len(piece) + 1
            current = kept
            current_len = kept_len
        current.append(sent)
        current_len += len(sent) + 1
    if current:
        flush(current)
    # Recompute exact offsets sequentially to keep them consistent
    fixed: list[tuple[str, int, int]] = []
    cursor = 0
    for ctext, _, _ in chunks:
        idx = text.find(ctext, cursor)
        if idx < 0:
            idx = cursor
        fixed.append((ctext, idx, idx + len(ctext)))
        cursor = idx + max(1, len(ctext) - overlap)
    return fixed


def chunk_doc(doc: dict) -> list[dict]:
    out: list[dict] = []
    for section, section_text in split_into_sections(doc["body"]):
        for n, (ctext, start, end) in enumerate(chunk_text(section_text)):
            out.append(
                {
                    "chunk_id": f"{doc['doc_id']}#{section}#{n}",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "topic": doc["topic"],
                    "section": section,
                    "text": ctext,
                    "char_start": start,
                    "char_end": end,
                }
            )
    return out


def chunk_all(docs: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for doc in docs:
        chunks.extend(chunk_doc(doc))
    return chunks


def load_ingested_docs() -> list[dict]:
    from pipeline.ingest import RAW_ZONE, load_manifest

    manifest = load_manifest()
    docs = []
    for doc_id, entry in manifest["docs"].items():
        md_file = RAW_ZONE / entry["file"]
        docs.append(
            {
                "doc_id": doc_id,
                "title": entry["title"],
                "topic": entry["topic"],
                "body": md_file.read_text(encoding="utf-8"),
            }
        )
    return docs


if __name__ == "__main__":
    docs = load_ingested_docs()
    chunks = chunk_all(docs)
    print(json.dumps({"docs": len(docs), "chunks": len(chunks)}, indent=2))
