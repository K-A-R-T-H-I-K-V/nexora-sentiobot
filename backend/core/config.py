"""
config.py — Centralised settings using Pydantic BaseSettings.
All values come from environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- Core ---
    app_name: str = "SentioBot API"
    debug: bool = False
    allowed_origins: list[str] = ["http://localhost:3000", "https://your-vercel-app.vercel.app"]

    # --- Auth ---
    jwt_secret: str = "change-me-in-production-use-openssl-rand"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 1 week

    # --- LLM ---
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # --- Supabase ---
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    # --- Vector / Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db_path: str = "vector_db"
    parent_store_path: str = "parent_docstore"
    parent_list_path: str = "parents.pkl"
    summaries_path: str = "summaries"

    # --- Cache ---
    # Set to redis://... for Redis, or leave empty to use in-process LRU cache
    redis_url: str = ""
    cache_ttl_seconds: int = 3600  # 1 hour for identical queries

    # --- Rate limiting ---
    rate_limit_per_minute: int = 20

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
