from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 normalize: bool = True):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self.dim: int = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 256,
               show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _empty_cache(self) -> None:
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
