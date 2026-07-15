#!/usr/bin/env python3
"""Run the Proxima benchmark suite and print a results table.

    # benchmark over a built corpus
    python bench/run_bench.py --corpus data/wiki_simple --nq 1000 --k 10

    # or over synthetic normalized vectors
    python bench/run_bench.py --n 200000 --dim 384 --nq 1000

At M0 only the exact FlatIndex exists, so recall is trivially 1.0 — the value is
establishing the latency / QPS / memory baseline and the recall methodology that
the HNSW index (M2) is measured against.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # local `harness` module

from harness import benchmark_index, exact_ground_truth, format_table  # noqa: E402

from proxima import FlatIndex, HnswIndex, Metric  # noqa: E402
from proxima.store import load_corpus  # noqa: E402


def load_base(args) -> tuple[np.ndarray, str]:
    if args.corpus:
        vectors, _docs, manifest = load_corpus(args.corpus)
        return vectors, manifest.get("metric", "InnerProduct")
    rng = np.random.default_rng(args.seed)
    v = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)  # unit vectors -> IP == cosine
    return np.ascontiguousarray(v), "InnerProduct"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=str, default=None, help="built corpus dir")
    p.add_argument("--n", type=int, default=100_000, help="synthetic base size")
    p.add_argument("--dim", type=int, default=384, help="synthetic dim")
    p.add_argument("--nq", type=int, default=1000, help="number of queries")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m", type=int, default=16, help="HNSW M (neighbors/layer)")
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef", type=str, default="16,32,64,128",
                   help="comma-separated ef_search sweep")
    p.add_argument("--no-hnsw", action="store_true", help="exact baseline only")
    p.add_argument("--json", type=str, default=None, help="also write results JSON")
    args = p.parse_args()

    base, metric_name = load_base(args)
    metric = getattr(Metric, metric_name)
    rng = np.random.default_rng(args.seed + 1)
    qi = rng.choice(len(base), size=min(args.nq, len(base)), replace=False)
    queries = np.ascontiguousarray(base[qi])

    print(f"base={len(base):,}  dim={base.shape[1]}  queries={len(queries):,}  "
          f"k={args.k}  metric={metric_name}")
    print("computing exact ground truth ...", flush=True)
    gt = exact_ground_truth(base, queries, args.k, metric)

    results = []
    flat = FlatIndex(dim=base.shape[1], metric=metric)
    flat.add(base)
    results.append(benchmark_index("FlatIndex (exact)", flat, queries, args.k, gt))

    if not args.no_hnsw:
        print(f"building HNSW (M={args.m}, ef_construction={args.ef_construction}) ...",
              flush=True)
        hnsw = HnswIndex(dim=base.shape[1], metric=metric, m=args.m,
                         ef_construction=args.ef_construction)
        t0 = time.perf_counter()
        hnsw.add(base)
        build_s = time.perf_counter() - t0
        print(f"      built {hnsw.size:,} nodes in {build_s:.1f}s "
              f"({hnsw.size / build_s:,.0f}/s), index {hnsw.memory_bytes / 1e6:.0f} MB")
        # Sweep ef_search: each setting is a point on the recall/latency curve.
        for ef in (int(x) for x in args.ef.split(",")):
            hnsw.ef_search = ef
            results.append(benchmark_index(f"HNSW(ef={ef})", hnsw, queries, args.k, gt))

    print()
    print(format_table(results))

    if args.json:
        Path(args.json).write_text(json.dumps([r.as_dict() for r in results], indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
