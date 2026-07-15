#!/usr/bin/env python3
"""Build and persist an ANN index from a corpus's embedding matrix.

    python scripts/build_index.py --corpus data/wiki_1m --type hnsw

Writes `<corpus>/<type>.sfidx`. The demo auto-loads `hnsw.sfidx` if present, so
a million-vector corpus serves sub-millisecond queries without rebuilding the
graph (which takes minutes) on every startup.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from proxima import FlatIndex, HnswIndex, Metric, PqIndex


def _add_chunked(idx, vectors, chunk: int = 50_000) -> None:
    """Insert a (possibly memory-mapped) matrix in chunks. This keeps peak RAM
    low: only a `chunk`-sized contiguous copy is resident at a time instead of a
    second full copy of the whole matrix alongside the growing index."""
    n = len(vectors)
    t0 = time.perf_counter()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        idx.add(np.ascontiguousarray(vectors[start:end], dtype=np.float32))
        done = end
        rate = done / (time.perf_counter() - t0 + 1e-9)
        print(f"  added {done:,}/{n:,} ({rate:,.0f}/s)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="built corpus dir")
    p.add_argument("--type", choices=["hnsw", "flat", "pq"], default="hnsw")
    p.add_argument("--m", type=int, default=16, help="HNSW M")
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=64)
    p.add_argument("--pq-m", type=int, default=16, help="PQ subspaces")
    p.add_argument("--chunk", type=int, default=50_000)
    args = p.parse_args()

    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "manifest.json").read_text())
    metric = getattr(Metric, manifest.get("metric", "InnerProduct"))
    # Memory-map the matrix so the full 1.5 GB+ is never all resident at once.
    vectors = np.load(corpus / "vectors.npy", mmap_mode="r")
    n, dim = vectors.shape
    out = corpus / f"{args.type}.sfidx"
    print(f"corpus={args.corpus} vectors={vectors.shape} metric={manifest.get('metric')} "
          f"-> {args.type}", flush=True)

    t0 = time.perf_counter()
    if args.type == "hnsw":
        idx = HnswIndex(dim=dim, metric=metric, m=args.m,
                        ef_construction=args.ef_construction, ef_search=args.ef_search)
        _add_chunked(idx, vectors, args.chunk)
    elif args.type == "flat":
        idx = FlatIndex(dim=dim, metric=metric)
        _add_chunked(idx, vectors, args.chunk)
    else:  # pq
        idx = PqIndex(dim=dim, metric=metric, m=args.pq_m)
        rng = np.random.default_rng(0)
        sample_idx = np.sort(rng.choice(n, size=min(100_000, n), replace=False))
        idx.train(np.ascontiguousarray(vectors[sample_idx], dtype=np.float32))
        print("  trained PQ codebooks", flush=True)
        _add_chunked(idx, vectors, args.chunk)

    build_s = time.perf_counter() - t0
    idx.save(str(out))
    size_mb = out.stat().st_size / 1e6
    print(f"built {args.type} on {idx.size:,} vectors in {build_s:.1f}s "
          f"({idx.size / build_s:,.0f}/s)", flush=True)
    print(f"saved {out} ({size_mb:.1f} MB on disk)", flush=True)


if __name__ == "__main__":
    main()
