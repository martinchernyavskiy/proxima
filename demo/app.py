from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, Response

from proxima.search import SemanticSearch

STATIC = Path(__file__).resolve().parent / "static"
CORPUS_CANDIDATES = ["data/wiki_1m", "data/wiki_simple", "data/wiki_smoke"]

state: dict = {}


def _pick_corpus() -> str:
    env = os.environ.get("PROXIMA_CORPUS")
    if env:
        if Path(env, "manifest.json").exists():
            return env
        raise RuntimeError(
            f"PROXIMA_CORPUS={env!r} has no manifest.json; "
            "check the path (typo?) or build it first"
        )
    for c in CORPUS_CANDIDATES:
        if Path(c, "manifest.json").exists():
            return c
    raise RuntimeError("no corpus found; run scripts/build_corpus.py first")


@asynccontextmanager
async def lifespan(app: FastAPI):
    corpus = _pick_corpus()
    print(f"[proxima] loading corpus: {corpus}", flush=True)
    ss = SemanticSearch.from_corpus(corpus)
    try:
        warmup_texts = [
            "history and science",
            "art and culture",
            "geography and nature",
            "sports and entertainment",
        ]
        for text in warmup_texts:
            qv = ss.embedder.encode_one(text)
            for _ in range(6):
                if hasattr(ss.index, "search_traced"):
                    ss.index.search_traced(qv, k=10, max_trace=40)
                else:
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
def search(q: str, k: int = Query(10, ge=0, le=100)) -> JSONResponse:
    ss: SemanticSearch = state["search"]
    results, latency_ms, trace, total_visited = ss.query_traced(q, k=k)
    return JSONResponse({
        "query": q,
        "latency_ms": round(latency_ms, 3),
        "count": ss.index.size,
        "results": [r.__dict__ for r in results],
        "trace": [t.__dict__ for t in trace],
        "trace_total_visited": total_visited,
    })


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)
