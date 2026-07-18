"""Pytest setup: make the suite runnable in CI with NO secrets, NO network to
Supabase, and NO paid LLM calls.

Every test here is free and deterministic. We set a dummy JWT secret + DEBUG so
importing modules that read settings does not refuse to boot, and put the repo
root on sys.path so `backend.*` imports resolve when pytest runs from anywhere.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JWT_SECRET", "ci-test-secret-not-for-production-0123456789abcdef")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("GROQ_API_KEY", "")  # tests never call the LLM
os.environ.setdefault("SUPABASE_URL", "http://localhost")  # tests never call Supabase
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "ci")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
