"""High-level semantic search: ties the embedder, the native index, and the
document metadata into a single `query(text) -> ranked docs` interface.

This is what the demo and the CLI drive. The heavy lifting (the actual nearest-
neighbor search) happens in the Rust core; this class only embeds the query and
maps result ids back to documents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._core import FlatIndex, HnswIndex, Metric
from .embeddings import DEFAULT_MODEL, TextEmbedder
from .store import load_corpus, load_metadata


@dataclass
class SearchResult:
    rank: int
    score: float
    title: str
    text: str
    url: str


class SemanticSearch:
    def __init__(self, index, docs: list[dict], embedder: TextEmbedder,
                 manifest: dict | None = None, index_kind: str = "FlatIndex (exact)"):
        self.index = index
        self.docs = docs
        self.embedder = embedder
        self.manifest = manifest or {}
        self.index_kind = index_kind

    @classmethod
    def from_corpus(cls, data_dir: str | Path, embedder: TextEmbedder | None = None,
                    prefer_hnsw: bool = True):
        """Load a built corpus into a searchable index.

        If a persisted HNSW index (`<data_dir>/hnsw.sfidx`, e.g. from
        `scripts/build_index.py`) is present and `prefer_hnsw` is set, it is
        loaded directly — instant startup and sub-ms search even at million
        scale. Otherwise an exact flat index is built from the vectors.
        """
        data_dir = Path(data_dir)
        hnsw_path = data_dir / "hnsw.sfidx"

        if prefer_hnsw and hnsw_path.exists():
            # The persisted graph already holds the vectors, so skip the
            # (potentially multi-GB) vectors.npy load entirely.
            docs, manifest = load_metadata(data_dir)
            index = HnswIndex.load(str(hnsw_path))
            index_kind = f"HNSW (approximate, ef_search={index.ef_search})"
        else:
            vectors, docs, manifest = load_corpus(data_dir)
            metric = getattr(Metric, manifest.get("metric", "InnerProduct"))
            index = FlatIndex(dim=vectors.shape[1], metric=metric)
            index.add(vectors)
            index_kind = "FlatIndex (exact)"

        if embedder is None:
            embedder = TextEmbedder(manifest.get("model", DEFAULT_MODEL))
        return cls(index, docs, embedder, manifest, index_kind)

    def query(self, text: str, k: int = 10) -> tuple[list[SearchResult], float]:
        """Return (results, latency_ms). Latency covers only the index search,
        not query embedding, so it is comparable across index types."""
        qv = self.embedder.encode_one(text)
        t0 = time.perf_counter()
        ids, scores = self.index.search(qv, k=k)
        latency_ms = (time.perf_counter() - t0) * 1e3

        results: list[SearchResult] = []
        for rank, (i, s) in enumerate(zip(ids.tolist(), scores.tolist())):
            if i < 0:
                continue
            d = self.docs[i]
            results.append(SearchResult(rank=rank + 1, score=float(s),
                                        title=d["title"], text=d["text"],
                                        url=d.get("url", "")))
        return results, latency_ms
