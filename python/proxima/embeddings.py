"""Text embedding via an off-the-shelf sentence-transformers model.

IMPORTANT framing: the embedding model is **not** part of Proxima. It is the
input layer that turns text into vectors; the project is the search engine that
operates on those vectors. We use a small, fast, widely-used model so the demo
is reproducible and the focus stays on the index.
"""

from __future__ import annotations

import numpy as np

# 384-dim, ~80MB, strong quality/speed tradeoff, L2-normalized outputs -> inner
# product equals cosine similarity (the engine's InnerProduct metric).
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEmbedder:
    """Lazily-loaded wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 normalize: bool = True):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            # Apple Silicon GPU (MPS) when present, else CPU. This is the same
            # device that will later embed live demo queries.
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self.dim: int = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 256,
               show_progress: bool = False) -> np.ndarray:
        """Encode a list of strings into a contiguous float32 (n, dim) array.

        Suitable for small inputs (e.g. a single query). For large corpora use
        `encode_chunked`, which is robust against MPS stalls on huge inputs."""
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string into a (dim,) float32 vector."""
        return self.encode([text])[0]

    def _empty_cache(self) -> None:
        # Release accumulated GPU buffers between chunks. A single giant encode()
        # on MPS can wedge the Metal command queue; chunk + empty_cache avoids it.
        try:
            import torch

            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    def encode_chunked(self, texts: list[str], chunk_size: int = 20_000,
                       batch_size: int = 256, progress=None) -> np.ndarray:
        """Encode a large corpus in bounded chunks, clearing the device cache
        between them. `progress(done, total, elapsed_s)` is called after each
        chunk so callers can print *flushed* progress (tqdm to a pipe buffers).
        """
        import time

        n = len(texts)
        out = np.empty((n, self.dim), dtype=np.float32)
        t0 = time.perf_counter()
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            emb = self.model.encode(
                texts[start:end],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            out[start:end] = np.ascontiguousarray(emb, dtype=np.float32)
            self._empty_cache()
            if progress is not None:
                progress(end, n, time.perf_counter() - t0)
        return out
