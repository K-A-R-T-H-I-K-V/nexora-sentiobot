"""Deterministic retrieval eval as a CI gate. Zero tokens: runs the frozen
base-ensemble hit@k eval against the baked index and asserts the 0.913 baseline
still holds. A retrieval regression (index or ranking change) fails the build.

Slower than the other tests because it loads the MiniLM embedding model (a free,
public HuggingFace download) and the baked Chroma index.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "backend/vector_db/chroma.sqlite3"


@pytest.mark.skipif(not INDEX.exists(), reason="baked Chroma index not present")
def test_retrieval_hit_at_5_reproduces_baseline():
    subprocess.run(
        [sys.executable, "-m", "backend.scripts.quality_baseline_retrieval"],
        cwd=REPO, check=True, capture_output=True, timeout=900,
    )
    d = json.load(open(REPO / "results/quality_retrieval_run_1.json", encoding="utf-8"))
    assert d["hit_rate_at_5"] == 0.913, f"retrieval hit@5 regressed: {d['hit_rate_at_5']}"
    assert d["misses_at_5"] == ["doc-01", "doc-13"], f"miss set changed: {d['misses_at_5']}"
