from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._core import FlatIndex, HnswIndex, Metric
from .corpus import trim_snippet
from .embeddings import DEFAULT_MODEL, TextEmbedder
from .store import load_corpus, load_metadata


@dataclass
class SearchResult:
    rank: int
    score: float
    title: str
    text: str
    url: str


@dataclass
class TraceStep:
    layer: int
    title: str
    score: float


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
        data_dir = Path(data_dir)
        hnsw_path = data_dir / "hnsw.sfidx"

        if prefer_hnsw and hnsw_path.exists():
            docs, manifest = load_metadata(data_dir)
            index = HnswIndex.load(str(hnsw_path))
            if index.size != len(docs):
                raise ValueError(
                    f"index/docs count mismatch in {data_dir}: hnsw.sfidx has "
                    f"{index.size} vectors vs meta.jsonl has {len(docs)} docs"
                )
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
                                        title=d["title"],
                                        text=trim_snippet(d["text"],
                                                          self.manifest.get("snippet_chars", 500),
                                                          already_cut=not self.manifest.get("clean_snippets", False)),
                                        url=d.get("url", "")))
        return results, latency_ms

    def query_traced(self, text: str, k: int = 10, max_trace: int = 40
                     ) -> tuple[list[SearchResult], float, list[TraceStep], int]:
        if not hasattr(self.index, "search_traced"):
            results, latency_ms = self.query(text, k=k)
            return results, latency_ms, [], 0

        qv = self.embedder.encode_one(text)
        t0 = time.perf_counter()
        id_scores, trace, total_visited = self.index.search_traced(qv, k=k, max_trace=max_trace)
        latency_ms = (time.perf_counter() - t0) * 1e3

        results: list[SearchResult] = []
        for rank, (i, s) in enumerate(id_scores):
            if i < 0:
                continue
            d = self.docs[i]
            results.append(SearchResult(rank=rank + 1, score=float(s), title=d["title"],
                                        text=trim_snippet(d["text"],
                                                          self.manifest.get("snippet_chars", 500),
                                                          already_cut=not self.manifest.get("clean_snippets", False)),
                                        url=d.get("url", "")))

        trace_steps = [TraceStep(layer=int(layer), title=self.docs[i]["title"], score=float(s))
                      for layer, i, s in trace if i >= 0]
        return results, latency_ms, trace_steps, total_visited
