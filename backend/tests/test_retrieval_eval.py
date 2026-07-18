"""Deterministic retrieval eval as a CI gate. Zero tokens: runs the frozen
base-ensemble hit@k eval against the baked index and asserts the 0.913 baseline
still holds. A retrieval regression (index or ranking change) fails the build.

I8-2: the eval opens Chroma read-write and writes a results JSON, so the test
runs it against a TMP COPY of the index with the output redirected to tmp - it
never dirties the committed results/ or vector_db/.

Slower than the other tests because it loads the MiniLM embedding model (a free,
public HuggingFace download) and the Chroma index.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "backend/vector_db/chroma.sqlite3"


@pytest.mark.skipif(not INDEX.exists(), reason="baked Chroma index not present")
def test_retrieval_hit_at_5_reproduces_baseline(tmp_path):
    # Copy the index + parent list to tmp so the eval's Chroma open (which
    # rewrites internal metadata) and its results JSON never touch tracked files.
    vdb = tmp_path / "vector_db"
    shutil.copytree(REPO / "backend/vector_db", vdb)
    parents = tmp_path / "parents.pkl"
    shutil.copy(REPO / "backend/parents.pkl", parents)

    env = {
        **os.environ,
        "VECTOR_DB_PATH": str(vdb),
        "PARENT_LIST_PATH": str(parents),
        "SENTIOBOT_RESULTS_DIR": str(tmp_path),
        "DEBUG": "true",
        "JWT_SECRET": os.environ.get("JWT_SECRET", "ci-test-secret-0123456789abcdef"),
    }
    subprocess.run(
        [sys.executable, "-m", "backend.scripts.quality_baseline_retrieval"],
        cwd=REPO, check=True, capture_output=True, timeout=900, env=env,
    )
    d = json.load(open(tmp_path / "quality_retrieval_run_1.json", encoding="utf-8"))
    assert d["hit_rate_at_5"] == 0.913, f"retrieval hit@5 regressed: {d['hit_rate_at_5']}"
    assert d["misses_at_5"] == ["doc-01", "doc-13"], f"miss set changed: {d['misses_at_5']}"
