"""
scripts/ingest.py — Production-grade 2-stage ingestion pipeline.

Two ingestion modes
-------------------
--direct  (recommended to start):
    Embeds parent documents directly. No API calls needed. Run this today
    and have a fully working system in ~2 minutes. Retrieval quality is
    still excellent — the summary step is an optional enhancement, not a
    requirement.

default (summary mode):
    Embeds AI-generated summaries instead of raw text. Slightly better
    retrieval for dense technical docs. Requires batch_summarize.py to
    have been run first.

Typical workflow
----------------
  # Option A — works immediately, no quota needed:
  python scripts/ingest.py --direct

  # Option B — run after batch_summarize.py finishes:
  python scripts/ingest.py

Other flags
-----------
  --dry-run          Print what would happen without writing anything
  --skip-validation  Skip pre-flight checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT      = Path(__file__).resolve().parent.parent   # backend/
DATA_PATH         = PROJECT_ROOT / "data"
DB_PATH           = PROJECT_ROOT / "vector_db"
PARENT_STORE_PATH = PROJECT_ROOT / "parent_docstore"
PARENT_LIST_PATH  = PROJECT_ROOT / "parents.pkl"
SUMMARIES_PATH    = PROJECT_ROOT / "summaries"
MANIFEST_PATH     = PROJECT_ROOT / "ingest_manifest.json"

NAMESPACE_UUID = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# ── Imports (deferred so --help works without the ML stack) ──────────────────

def _import_ml():
    global LocalFileStore, create_kv_docstore, Document, CSVLoader
    global Chroma, HuggingFaceEmbeddings
    from langchain.storage import LocalFileStore
    from langchain.storage._lc_store import create_kv_docstore
    from langchain.docstore.document import Document
    from langchain_community.document_loaders import CSVLoader
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def preflight(skip_validation: bool = False) -> bool:
    """Returns True if all checks pass, False if we should abort."""
    ok = True

    log.info("Running pre-flight checks…")

    # Data directory
    if not DATA_PATH.exists() or not any(DATA_PATH.iterdir()):
        log.error("❌  data/ directory is missing or empty: %s", DATA_PATH)
        ok = False
    else:
        md_files  = list(DATA_PATH.rglob("*.md"))
        csv_files = list(DATA_PATH.rglob("*.csv"))
        log.info("  ✅  Found %d .md and %d .csv files in data/",
                 len(md_files), len(csv_files))

    # Google API key
    if not os.getenv("GOOGLE_API_KEY") and not skip_validation:
        log.warning("  ⚠️  GOOGLE_API_KEY not set — summaries won't be generated "
                    "(ingest.py itself doesn't need it, but batch_summarize.py does)")

    # Summary coverage
    if SUMMARIES_PATH.exists() and PARENT_STORE_PATH.exists():
        summary_ids = {f.stem for f in SUMMARIES_PATH.glob("*.txt")}
        from langchain.storage import LocalFileStore
        byte_store  = LocalFileStore(str(PARENT_STORE_PATH))
        parent_ids  = set(byte_store.yield_keys())
        missing     = parent_ids - summary_ids
        coverage    = len(summary_ids) / len(parent_ids) * 100 if parent_ids else 0

        if missing:
            log.warning("  ⚠️  Summary coverage: %.0f%% (%d/%d). "
                        "Run batch_summarize.py for the missing %d docs first.",
                        coverage, len(summary_ids), len(parent_ids), len(missing))
        else:
            log.info("  ✅  Summary coverage: 100%% (%d docs)", len(summary_ids))
    else:
        log.info("  ℹ️  No existing stores found — fresh ingest will be performed.")

    return ok


# ── Stage 1: Build parent document store ─────────────────────────────────────

def _make_doc_id(filename: str, key: str) -> str:
    return str(uuid.uuid5(NAMESPACE_UUID, f"{filename}-{key}"))


def process_markdown(content: str, filename: str) -> List[Any]:
    """Split markdown by H2 headers, return list of Document objects."""
    docs = []
    sections = re.split(r"\n(?=## )", content)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        first_line = section.split("\n", 1)[0]
        title = re.sub(r"^#{1,6}\s*", "", first_line).strip()

        if not title:
            continue

        doc_id = _make_doc_id(filename, title)
        docs.append(Document(
            page_content=section,
            metadata={"source": filename, "section_title": title, "doc_id": doc_id},
        ))

    if not docs:
        log.warning("  ⚠️  No sections extracted from %s", filename)

    return docs


def process_csv(filepath: Path, filename: str) -> List[Any]:
    """Load CSV — each row becomes one parent document."""
    docs = []
    try:
        loader = CSVLoader(
            file_path=str(filepath),
            source_column="Question",
            metadata_columns=["Category"],
        )
        for doc in loader.load():
            content = doc.page_content
            if not content.strip():
                continue
            doc_id = _make_doc_id(filename, content[:50])
            doc.metadata["doc_id"] = doc_id
            doc.metadata["source"] = filename
            docs.append(doc)
    except Exception as exc:
        log.error("  ❌  Failed to load %s: %s", filename, exc)
    return docs


def load_parents() -> List[Any]:
    all_parents: List[Any] = []
    files_processed = 0

    for filepath in sorted(DATA_PATH.rglob("*")):
        if filepath.suffix == ".md":
            try:
                content = filepath.read_text(encoding="utf-8")
                docs = process_markdown(content, filepath.name)
                all_parents.extend(docs)
                log.info("  📄  %s → %d sections", filepath.name, len(docs))
                files_processed += 1
            except Exception as exc:
                log.error("  ❌  Could not read %s: %s", filepath.name, exc)

        elif filepath.suffix == ".csv":
            docs = process_csv(filepath, filepath.name)
            all_parents.extend(docs)
            log.info("  📊  %s → %d rows", filepath.name, len(docs))
            files_processed += 1

    log.info("Loaded %d parent documents from %d files.", len(all_parents), files_processed)
    return all_parents


def build_parent_store(all_parents: List[Any], dry_run: bool) -> None:
    log.info("Stage 1 — Building parent document store…")

    if dry_run:
        log.info("  [DRY RUN] Would index %d parents → %s", len(all_parents), PARENT_STORE_PATH)
        return

    # Write to a temp dir first, then swap atomically
    tmp_store = PROJECT_ROOT / "_parent_docstore_tmp"
    if tmp_store.exists():
        shutil.rmtree(tmp_store)

    byte_store = LocalFileStore(str(tmp_store))
    store = create_kv_docstore(byte_store)
    id_map = {doc.metadata["doc_id"]: doc for doc in all_parents}
    store.mset(list(id_map.items()))

    # Atomic swap (shutil.move works on Windows unlike Path.rename)
    if PARENT_STORE_PATH.exists():
        shutil.rmtree(PARENT_STORE_PATH)
    shutil.move(str(tmp_store), str(PARENT_STORE_PATH))

    # Serialise list for BM25
    tmp_pkl = PROJECT_ROOT / "_parents_tmp.pkl"
    with open(tmp_pkl, "wb") as f:
        pickle.dump(all_parents, f)
    if PARENT_LIST_PATH.exists():
        PARENT_LIST_PATH.unlink()
    shutil.move(str(tmp_pkl), str(PARENT_LIST_PATH))

    log.info("  ✅  Parent store: %d docs → %s", len(all_parents), PARENT_STORE_PATH)
    log.info("  ✅  BM25 list serialised → %s", PARENT_LIST_PATH)


# ── Stage 2: Build vector store from summaries ───────────────────────────────

def load_summary_docs() -> List[Any]:
    """Combine summary text (for retrieval) with parent metadata (for tracing)."""
    if not SUMMARIES_PATH.exists():
        return []

    byte_store = LocalFileStore(str(PARENT_STORE_PATH))
    docstore   = create_kv_docstore(byte_store)

    summary_files = list(SUMMARIES_PATH.glob("*.txt"))
    doc_ids       = [f.stem for f in summary_files]

    # Batch fetch parent metadata
    parent_docs = docstore.mget(doc_ids)

    summary_docs = []
    orphaned = 0
    for f, parent in zip(summary_files, parent_docs):
        if parent is None:
            log.debug("  Orphaned summary (no parent): %s", f.stem)
            orphaned += 1
            continue
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            log.warning("  ⚠️  Empty summary file: %s", f.name)
            continue
        summary_docs.append(Document(
            page_content=text,
            metadata=parent.metadata,
        ))

    if orphaned:
        log.warning("  ⚠️  Skipped %d orphaned summary files (no matching parent)", orphaned)

    log.info("  Loaded %d summary documents for vector indexing.", len(summary_docs))
    return summary_docs


def build_vector_store(docs_to_embed: List[Any], dry_run: bool) -> None:
    log.info("Stage 2 — Building vector store (%d docs)…", len(docs_to_embed))

    if not docs_to_embed:
        log.error("  ❌  No documents to embed. "
                  "Either run batch_summarize.py first, or use --direct flag.")
        return

    if dry_run:
        log.info("  [DRY RUN] Would embed %d docs → %s", len(docs_to_embed), DB_PATH)
        return

    log.info("  Loading embedding model (all-MiniLM-L6-v2)…")
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # On Windows, ChromaDB holds SQLite WAL file locks that survive del + gc.
    # Atomic tmp→rename is impossible. Instead: delete old store first, then
    # write directly to final path. The window of vulnerability is <5 seconds.
    if DB_PATH.exists():
        try:
            shutil.rmtree(DB_PATH)
        except PermissionError:
            log.error("  ❌  Cannot delete vector_db — a file lock is held by another process.")
            log.error("       This is almost always a Jupyter notebook with ChromaDB still open.")
            log.error("       Fix: shut down the notebook kernel (top-right in VS Code Jupyter),")
            log.error("       then re-run this script.")
            sys.exit(1)
    DB_PATH.mkdir(parents=True, exist_ok=True)

    log.info("  Embedding and indexing %d documents…", len(docs_to_embed))
    Chroma.from_documents(
        documents=docs_to_embed,
        embedding=embedding_model,
        persist_directory=str(DB_PATH),
    )

    log.info("  ✅  Vector store built → %s", DB_PATH)


# ── Manifest ──────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    for f in sorted(path.rglob("*") if path.is_dir() else [path]):
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()


def write_manifest(all_parents: List[Any], summary_docs: List[Any]) -> None:
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "parent_count": len(all_parents),
        "summary_count": len(summary_docs),
        "vector_db_hash": _file_hash(DB_PATH) if DB_PATH.exists() else None,
        "data_sources": sorted({d.metadata.get("source", "?") for d in all_parents}),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    log.info("  📋  Manifest written → %s", MANIFEST_PATH)


def print_summary_report(all_parents, summary_docs) -> None:
    log.info("")
    log.info("═" * 60)
    log.info("  Ingestion complete")
    log.info("  Parent documents : %d", len(all_parents))
    log.info("  Summary vectors  : %d", len(summary_docs))
    coverage = len(summary_docs) / len(all_parents) * 100 if all_parents else 0
    log.info("  Coverage         : %.0f%%", coverage)
    log.info("  Vector store     : %s", DB_PATH)
    log.info("═" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SentioBot ingestion pipeline")
    parser.add_argument("--direct",           action="store_true",
                        help="Embed parent docs directly — no summaries needed. "
                             "Use this when batch_summarize.py hasn't run yet.")
    parser.add_argument("--dry-run",          action="store_true",
                        help="Print what would happen without writing anything")
    parser.add_argument("--skip-validation",  action="store_true",
                        help="Skip pre-flight checks")
    args = parser.parse_args()

    log.info("═" * 60)
    log.info("SentioBot — Ingestion Pipeline%s%s",
             " [DRY RUN]" if args.dry_run else "",
             " [DIRECT MODE]" if args.direct else "")
    log.info("═" * 60)

    _import_ml()

    if not args.skip_validation:
        if not preflight(skip_validation=False):
            log.error("Pre-flight checks failed. Fix the issues above and re-run.")
            sys.exit(1)

    # Stage 1 — always runs
    all_parents = load_parents()
    if not all_parents:
        log.error("No documents found in data/. Aborting.")
        sys.exit(1)

    build_parent_store(all_parents, dry_run=args.dry_run)

    # Stage 2 — direct mode skips summaries entirely
    if not args.dry_run:
        if args.direct:
            log.info("Stage 2 — Direct mode: embedding parent documents…")
            log.info("  (Tip: run batch_summarize.py then ingest.py without --direct")
            log.info("   for a small retrieval quality boost once your quota resets.)")
            docs_to_embed = all_parents
        else:
            docs_to_embed = load_summary_docs()

        build_vector_store(docs_to_embed, dry_run=False)
        write_manifest(all_parents, docs_to_embed)
        print_summary_report(all_parents, docs_to_embed)
    else:
        log.info("[DRY RUN] No files written. Remove --dry-run to execute.")


if __name__ == "__main__":
    main()