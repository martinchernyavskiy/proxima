#!/usr/bin/env python3
"""SIFT1M benchmark — Proxima vs FAISS on identical data, queries, and
ground truth.

    python bench/bench_sift.py                  # full 1M, all indexes + FAISS
    python bench/bench_sift.py --n 200000 --no-faiss   # quick subset check

Reports recall@k (against the dataset's held-out ground truth), single-query
latency p50/p99, throughput, and index memory for the exact, HNSW, and PQ
indexes from both engines. "Within X× of FAISS" is an honest, credible flex.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import load_sift  # noqa: E402
from harness import benchmark_index, exact_ground_truth, format_table  # noqa: E402

from proxima import FlatIndex, HnswIndex, Metric, PqIndex  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/sift/sift")
    p.add_argument("--n", type=int, default=0, help="limit base size (0 = all 1M)")
    p.add_argument("--nq", type=int, default=1000, help="number of queries")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--m", type=int, default=16, help="HNSW M")
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef", type=str, default="32,64,128")
    p.add_argument("--pq-m", type=int, default=16, help="PQ subspaces")
    p.add_argument("--no-faiss", action="store_true")
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    ds = load_sift(args.root)
    subset = bool(args.n and args.n < len(ds.base))
    base = np.ascontiguousarray(ds.base[: args.n] if args.n else ds.base, dtype=np.float32)
    queries = np.ascontiguousarray(ds.queries[: args.nq], dtype=np.float32)
    train = ds.train if ds.train is not None else base
    metric = getattr(Metric, ds.metric)
    dim = base.shape[1]
    efs = [int(x) for x in args.ef.split(",")]

    print(f"{ds.name}: base={len(base):,} dim={dim} queries={len(queries):,} "
          f"k={args.k} metric={ds.metric}")

    # The shipped ground truth is for the full 1M base; if we subset the base
    # the ids no longer apply, so recompute exact GT on the subset.
    if subset:
        print("  subset: recomputing exact ground truth ...", flush=True)
        gt = exact_ground_truth(base, queries, args.k, metric)
    else:
        gt = ds.ground_truth[: args.nq]
    results = []

    def build(label, fn):
        t0 = time.perf_counter()
        idx = fn()
        print(f"  built {label} in {time.perf_counter() - t0:.1f}s", flush=True)
        return idx

    # ---- Proxima ----
    px_flat = build("PX Flat", lambda: _add(FlatIndex(dim=dim, metric=metric), base))
    results.append(benchmark_index("PX Flat (exact)", px_flat, queries, args.k, gt))

    px_hnsw = build(f"PX HNSW (M={args.m})",
                    lambda: _add(HnswIndex(dim=dim, metric=metric, m=args.m,
                                           ef_construction=args.ef_construction), base))
    for ef in efs:
        px_hnsw.ef_search = ef
        results.append(benchmark_index(f"PX HNSW(ef={ef})", px_hnsw, queries, args.k, gt))

    def make_pq():
        idx = PqIndex(dim=dim, metric=metric, m=args.pq_m)
        idx.train(np.ascontiguousarray(train, dtype=np.float32))
        idx.add(base)
        return idx
    px_pq = build(f"PX PQ (m={args.pq_m})", make_pq)
    results.append(benchmark_index(f"PX PQ(m={args.pq_m})", px_pq, queries, args.k, gt))
    print(f"  PX PQ compression: {px_pq.compression_ratio:.1f}x "
          f"({px_pq.raw_bytes / 1e6:.0f}MB -> {px_pq.memory_bytes / 1e6:.1f}MB)")

    # ---- FAISS ----
    if not args.no_faiss:
        import faiss_compare as fc

        f_flat = build("FAISS Flat", lambda: fc.build_flat(base, ds.metric))
        results.append(benchmark_index("FAISS Flat", f_flat, queries, args.k, gt))

        f_hnsw = build(f"FAISS HNSW (M={args.m})",
                       lambda: fc.build_hnsw(base, ds.metric, m=args.m,
                                             ef_construction=args.ef_construction))
        for ef in efs:
            f_hnsw.ef_search = ef
            results.append(benchmark_index(f"FAISS HNSW(ef={ef})", f_hnsw, queries, args.k, gt))

        f_pq = build(f"FAISS PQ (m={args.pq_m})",
                     lambda: fc.build_pq(base, ds.metric, m=args.pq_m, train=train))
        results.append(benchmark_index(f"FAISS PQ(m={args.pq_m})", f_pq, queries, args.k, gt))

    print()
    print(format_table(results))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps([r.as_dict() for r in results], indent=2))
        print(f"\nwrote {args.json}")


def _add(index, vectors):
    index.add(vectors)
    return index


if __name__ == "__main__":
    main()
