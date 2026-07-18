# SentioBot backend - multi-stage image.
#
# Stage 1 (builder) installs the Python deps (incl. build-essential for any
# native wheels) into an isolated venv. Stage 2 (runtime) copies just that venv
# and the app, so build-essential is NOT in the final image. Torch is the +cpu
# build (see requirements.txt) which keeps the image ~3.2GB instead of ~13GB.
#
# The RAG index (vector_db / parent_docstore / parents.pkl) is BAKED in by the
# COPY below: unlike .env / venv / caches, it is NOT listed in .dockerignore, so
# it is included in the build context and copied into the image. That makes the
# container self-contained (no persistent volume). A corpus change means
# re-running ingest and rebuilding the image.

# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /app/requirements.txt
# The PyTorch CPU extra-index-url is declared inside requirements.txt.
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Stage 2: runtime ----
FROM python:3.11-slim AS runtime
WORKDIR /app

# curl only, for the compose/container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Bring in the prebuilt venv (no build tools in the final image).
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy the backend source + baked index into the package dir, so that
# `backend.api.main:app` and all `backend.*` imports resolve as they do locally.
# .dockerignore keeps .env, venv, and caches out.
COPY . /app/backend

EXPOSE 8000
# Listen on $PORT so the same image runs on any host: local/compose/CI default to
# 8000, Hugging Face Spaces routes to app_port (set it to 8000), and Cloud Run
# injects its own $PORT. Workers default to 1 (one copy of the embedding model in
# memory - safe on free tiers); raise via WEB_CONCURRENCY where RAM allows.
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
