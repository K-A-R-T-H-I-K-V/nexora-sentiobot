"""
onnx_embeddings.py - the all-MiniLM-L6-v2 embedding model on ONNX Runtime.

Increment 9: we previously ran MiniLM through PyTorch (langchain-huggingface +
sentence-transformers + torch), which alone made the backend image ~2.8GB and
its resident memory ~1GB - too heavy for free scale-to-zero hosts. This runs the
SAME model on ONNX Runtime via fastembed: no torch, ~1/10th the footprint, image
~1.5GB and its RAM to ~280MB. Verified bit-identical retrieval (query cosine 1.0 vs the PyTorch model,
hit@5 0.913 with the same misses against the existing index), so the frozen
baseline holds and the index does not need rebuilding.

The model files are baked into the image at build time (see backend/Dockerfile,
FASTEMBED_CACHE_DIR) so there is no first-request download on a cold start.
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

# fastembed uses the fully-qualified hub name; the app config uses the short alias.
_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"


def _resolve(model_name: str) -> str:
    return model_name if "/" in model_name else f"sentence-transformers/{model_name}"


class OnnxMiniLMEmbeddings(Embeddings):
    """LangChain Embeddings backed by fastembed (ONNX Runtime). Drop-in for the
    old HuggingFaceEmbeddings, minus PyTorch."""

    def __init__(self, model_name: str = _DEFAULT):
        self._model = TextEmbedding(
            model_name=_resolve(model_name),
            cache_dir=os.environ.get("FASTEMBED_CACHE_DIR") or None,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, next(iter(self._model.embed([text])))))


@lru_cache(maxsize=2)
def get_embeddings(model_name: str = _DEFAULT) -> OnnxMiniLMEmbeddings:
    """Cached singleton so the model loads once per process."""
    return OnnxMiniLMEmbeddings(model_name)
