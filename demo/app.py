"""Proxima demo web app.

A minimal FastAPI server that loads a built corpus into the engine and serves:
  GET /                  the search UI (static HTML)
  GET /api/search?q=&k=  JSON results + index-search latency
  GET /api/info          corpus / engine metadata

Run:
  PROXIMA_CORPUS=data/wiki_simple \
    .venv/bin/uvicorn demo.app:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from proxima.search import SemanticSearch

STATIC = Path(__file__).resolve().parent / "static"
# Prefer the largest available corpus; each falls back to the next if missing.
CORPUS_CANDIDATES = ["data/wiki_1m", "data/wiki_simple", "data/wiki_smoke"]

state: dict = {}


def _pick_corpus() -> str:
    env = os.environ.get("PROXIMA_CORPUS")
    candidates = [env, *CORPUS_CANDIDATES] if env else CORPUS_CANDIDATES
    for c in candidates:
        if c and Path(c, "manifest.json").exists():
            return c
    raise RuntimeError("no corpus found; run scripts/build_corpus.py first")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the corpus + embedding model once at startup.
    corpus = _pick_corpus()
    print(f"[proxima] loading corpus: {corpus}", flush=True)
    ss = SemanticSearch.from_corpus(corpus)
    # Warm the index: a freshly-deserialized graph is cold in CPU cache, so the
    # first few searches are slow. Pre-touch it so the first real query is fast.
    try:
        qv = ss.embedder.encode_one("warmup query about history science and art")
        for _ in range(8):
            ss.index.search(qv, k=10)
    except Exception:
        pass
    state["search"] = ss
    state["corpus"] = corpus
    print(f"[proxima] ready: {ss.index.size:,} docs ({ss.index_kind})", flush=True)
    yield
    state.clear()


app = FastAPI(title="Proxima", lifespan=lifespan)


@app.get("/api/info")
def info() -> JSONResponse:
    ss: SemanticSearch = state["search"]
    return JSONResponse({
        "corpus": state["corpus"],
        "count": ss.index.size,
        "dim": ss.index.dim,
        "model": ss.manifest.get("model"),
        "metric": ss.manifest.get("metric"),
        "index": ss.index_kind,
        "memory_mb": round(ss.index.memory_bytes / 1e6, 1),
    })


@app.get("/api/search")
def search(q: str, k: int = 10) -> JSONResponse:
    ss: SemanticSearch = state["search"]
    results, latency_ms = ss.query(q, k=k)
    return JSONResponse({
        "query": q,
        "latency_ms": round(latency_ms, 3),
        "count": ss.index.size,
        "results": [r.__dict__ for r in results],
    })


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC / "index.html")
