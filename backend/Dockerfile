FROM python:3.11-slim

# Run the app as the `backend` package from /app so `backend.api.main:app`
# and all `backend.*` imports resolve identically to a local run from the
# repo root. Build context is ./backend (see docker-compose.yml / railway).
WORKDIR /app

# Install system deps (needed for some ML packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the backend source into the package dir. .dockerignore excludes venv,
# .env, and the vector/parent stores (those are mounted or rebuilt at runtime).
COPY . /app/backend

EXPOSE 8000
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
