"""
config.py — Centralised settings using Pydantic BaseSettings.

All values come from environment variables or the backend/.env file. The
.env file and the vector / parent store paths are resolved against the
backend package directory (BASE_DIR), so the app boots identically from any
working directory and inside the container, not just from backend/.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

# backend/ package directory (this file lives in backend/core/config.py).
BASE_DIR = Path(__file__).resolve().parents[1]

_PLACEHOLDER_JWT = "change-me-in-production-use-openssl-rand"


class Settings(BaseSettings):
    # --- Core ---
    app_name: str = "SentioBot API"
    debug: bool = False
    # Raw comma-separated origins from the ALLOWED_ORIGINS env var. Kept as a
    # plain str (not list[str]) so pydantic-settings does not try to JSON
    # decode it; exposed as a list via the allowed_origins property below.
    cors_origins: str = Field(
        default="http://localhost:3000", validation_alias="ALLOWED_ORIGINS"
    )

    # --- Auth ---
    jwt_secret: str = _PLACEHOLDER_JWT
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 1 week

    # --- LLM ---
    # Provider is config-selectable so a future swap is config, not a migration.
    llm_provider: str = "groq"  # "groq" (ratified default) or "gemini"
    llm_temperature: float = 0.0  # deterministic for eval reproducibility
    llm_max_tokens: int = 2048
    # Groq (ratified provider)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    # Gemini (kept selectable via LLM_PROVIDER=gemini)
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    # Max tool ROUNDS before the graph force-finalizes (answers without tools).
    # This, together with tool-call dedupe, makes the tool path converge instead
    # of looping on repeated calls and burning the Groq free-tier daily quota.
    agent_max_tool_rounds: int = 4

    # --- Supabase ---
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    # --- Retrieval ---
    # Increment 4: base ensemble (BM25 + vector) is the default. The multi-query
    # retriever (extra LLM call per query) is kept behind this flag, reversible
    # and A/B-able. Baselines showed it added no hit-rate here.
    use_multiquery: bool = False

    # --- Vector / Embedding (absolute paths resolved against backend/) ---
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db_path: str = str(BASE_DIR / "vector_db")
    parent_store_path: str = str(BASE_DIR / "parent_docstore")
    parent_list_path: str = str(BASE_DIR / "parents.pkl")
    summaries_path: str = str(BASE_DIR / "summaries")

    # --- Cache ---
    # Set REDIS_URL to redis://... for the L3 tier, else in-process L1/L2 only.
    redis_url: str = ""
    cache_ttl_seconds: int = 3600  # 1 hour for identical queries
    semantic_cache_threshold: float = 0.92

    # --- Rate limiting (field present; limiter not yet wired, see FORWARD TASKS) ---
    rate_limit_per_minute: int = 20

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def allowed_origins(self) -> list[str]:
        """CORS origins as a list, parsed from the comma-separated env value."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _require_real_jwt_secret(self):
        """Refuse to boot with the placeholder/empty JWT secret (warn in debug)."""
        if self.jwt_secret in ("", _PLACEHOLDER_JWT):
            msg = (
                "JWT_SECRET is unset or still the placeholder. Set a strong "
                "secret in backend/.env (generate with: openssl rand -hex 32)."
            )
            if self.debug:
                log.warning("%s Continuing because DEBUG is on.", msg)
            else:
                raise ValueError(msg)
        return self


@lru_cache()
def get_settings() -> Settings:
    return Settings()
