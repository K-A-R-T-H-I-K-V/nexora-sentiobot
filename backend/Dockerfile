# SentioBot backend - multi-stage image.
#
# Stage 1 (builder) installs the Python deps (incl. build-essential for any
# native wheels) into an isolated venv. Stage 2 (runtime) copies just that venv
# and the app, so build-essential is NOT in the final image. Embeddings run on
# ONNX Runtime (fastembed), NOT PyTorch, so the image is ~1.5GB (was ~2.8GB with
# torch) and its RAM drops to ~280MB (measured under a 512MB cap), fitting free
# scale-to-zero hosts.
#
# Two things are BAKED in so the container is self-contained (no volume, no
# cold-start download): the RAG index (vector_db / parent_docstore / parents.pkl,
# NOT excluded by .dockerignore) and the ONNX embedding model (downloaded during
# build into FASTEMBED_CACHE_DIR). A corpus change means re-running ingest and
# rebuilding the image.

# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /app/requirements.txt
# --timeout/--retries so a slow or flaky network does not fail the build.
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r /app/requirements.txt

# Bake the ONNX embedding model into the image so there is no first-request
# download on a cold start (it lands in FASTEMBED_CACHE_DIR, copied to runtime).
ENV FASTEMBED_CACHE_DIR=/opt/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir='/opt/fastembed')"

# ---- Stage 2: runtime ----
FROM python:3.11-slim AS runtime
WORKDIR /app

# curl only, for the compose/container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Bring in the prebuilt venv (no build tools in the final image) and the baked
# ONNX embedding model.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/fastembed /opt/fastembed
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV FASTEMBED_CACHE_DIR=/opt/fastembed
ENV TOKENIZERS_PARALLELISM=false

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
