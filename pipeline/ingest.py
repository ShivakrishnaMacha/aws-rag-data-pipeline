#!/usr/bin/env python3
"""Ingest stage: raw zone -> validated, deduped documents + manifest.

Production mapping: the local ``s3_mock/raw-zone`` and ``s3_mock/processed-zone``
directories stand in for S3 buckets (e.g. ``s3://<bucket>/raw/`` and
``s3://<bucket>/processed/``). The code paths are identical to an S3-backed
implementation: read object -> validate -> dedupe by content hash -> manifest.
Swap ``local_*`` helpers for boto3 ``get_object``/``put_object`` to go live.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_ZONE = ROOT / "s3_mock" / "raw-zone"          # stand-in for s3://<bucket>/raw/
PROCESSED_ZONE = ROOT / "s3_mock" / "processed-zone"  # stand-in for s3://<bucket>/processed/
MANIFEST_PATH = ROOT / "pipeline" / "manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_doc(doc: dict) -> list[str]:
    """Return a list of validation errors (empty == valid)."""
    errors: list[str] = []
    for field in ("doc_id", "title", "topic", "body"):
        if not doc.get(field):
            errors.append(f"missing field: {field}")
    body = doc.get("body", "")
    for section in ("## Symptoms", "## Likely root cause", "## Remediation"):
        if section not in body:
            errors.append(f"missing section: {section}")
    return errors


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"docs": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def sync_raw_zone() -> int:
    """Copy the generated corpus into the raw zone (simulates S3 landing)."""
    corpus_dir = ROOT / "data" / "corpus"
    meta_path = ROOT / "data" / "corpus_metadata.json"
    RAW_ZONE.mkdir(parents=True, exist_ok=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    by_file = {m["file"]: m for m in meta}
    count = 0
    for md_file in sorted(corpus_dir.glob("*.md")):
        dest = RAW_ZONE / md_file.name
        if not dest.exists() or sha256_of(md_file) != sha256_of(dest):
            shutil.copy2(md_file, dest)
        m = by_file[md_file.name]
        sidecar = dest.with_suffix(".json")
        sidecar.write_text(json.dumps(m, indent=2), encoding="utf-8")
        count += 1
    return count


def ingest() -> dict:
    """Ingest raw-zone objects; returns summary stats. Idempotent."""
    synced = sync_raw_zone()
    manifest = load_manifest()
    known = manifest.get("docs", {})

    stats = {"scanned": 0, "new": 0, "unchanged": 0, "invalid": 0, "errors": []}
    for md_file in sorted(RAW_ZONE.glob("*.md")):
        stats["scanned"] += 1
        content_hash = sha256_of(md_file)
        sidecar = md_file.with_suffix(".json")
        if not sidecar.exists():
            stats["invalid"] += 1
            stats["errors"].append(f"{md_file.name}: missing metadata sidecar")
            continue
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        doc = {**meta, "body": md_file.read_text(encoding="utf-8")}
        errors = validate_doc(doc)
        if errors:
            stats["invalid"] += 1
            stats["errors"].extend(f"{md_file.name}: {e}" for e in errors)
            continue
        prev = known.get(doc["doc_id"])
        if prev and prev.get("content_hash") == content_hash:
            stats["unchanged"] += 1
            continue
        known[doc["doc_id"]] = {
            "content_hash": content_hash,
            "title": doc["title"],
            "topic": doc["topic"],
            "file": md_file.name,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        stats["new"] += 1

    manifest["docs"] = known
    manifest["last_ingest_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    stats["synced_raw_objects"] = synced
    stats["total_docs"] = len(known)
    return stats


if __name__ == "__main__":
    summary = ingest()
    print(json.dumps(summary, indent=2))
