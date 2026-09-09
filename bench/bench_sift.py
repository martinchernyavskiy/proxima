from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import load_sift
from harness import (LATENCY_SAMPLES, benchmark_index, exact_ground_truth,
                     format_table, write_results)

from proxima import FlatIndex, HnswIndex, Metric, PqIndex


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
    p.add_argument("--latency-samples", type=int, default=LATENCY_SAMPLES,
                   help="single-query timings sampled per index")
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()
    if args.n < 0:
        p.error("--n must be >= 0")
    if args.nq < 0:
        p.error("--nq must be >= 0")
    if args.latency_samples < 0:
        p.error("--latency-samples must be >= 0")

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

    if subset:
        print("  subset: recomputing exact ground truth ...", flush=True)
        gt = exact_ground_truth(base, queries, args.k, metric)
    else:
        gt = ds.ground_truth[: args.nq]
    results = []

    def build(label, fn):
        t0 = time.perf_counter()
        idx = fn()
        build_s = time.perf_counter() - t0
        print(f"  built {label} in {build_s:.1f}s", flush=True)
        return idx, build_s

    def record(label, index, build_s):
        results.append(benchmark_index(label, index, queries, args.k, gt,
                                       latency_samples=args.latency_samples,
                                       build_seconds=build_s))

    px_flat, t_flat = build("PX Flat",
                            lambda: _add(FlatIndex(dim=dim, metric=metric), base))
    record("PX Flat (exact)", px_flat, t_flat)

    px_hnsw, t_hnsw = build(f"PX HNSW (M={args.m})",
                            lambda: _add(HnswIndex(dim=dim, metric=metric, m=args.m,
                                                   ef_construction=args.ef_construction), base))
    for ef in efs:
        px_hnsw.ef_search = ef
        record(f"PX HNSW(ef={ef})", px_hnsw, t_hnsw)

    def make_pq():
        idx = PqIndex(dim=dim, metric=metric, m=args.pq_m)
        idx.train(np.ascontiguousarray(train, dtype=np.float32))
        idx.add(base)
        return idx
    px_pq, t_pq = build(f"PX PQ (m={args.pq_m})", make_pq)
    record(f"PX PQ(m={args.pq_m})", px_pq, t_pq)
    print(f"  PX PQ compression: {px_pq.compression_ratio:.1f}x "
          f"({px_pq.raw_bytes / 1e6:.0f}MB -> {px_pq.memory_bytes / 1e6:.1f}MB)")

    del px_flat, px_hnsw, px_pq
    gc.collect()

    if not args.no_faiss:
        import faiss_compare as fc

        f_flat, t_f_flat = build("FAISS Flat", lambda: fc.build_flat(base, ds.metric))
        record("FAISS Flat", f_flat, t_f_flat)

        f_hnsw, t_f_hnsw = build(f"FAISS HNSW (M={args.m})",
                                 lambda: fc.build_hnsw(base, ds.metric, m=args.m,
                                                       ef_construction=args.ef_construction))
        for ef in efs:
            f_hnsw.ef_search = ef
            record(f"FAISS HNSW(ef={ef})", f_hnsw, t_f_hnsw)

        f_pq, t_f_pq = build(f"FAISS PQ (m={args.pq_m})",
                             lambda: fc.build_pq(base, ds.metric, m=args.pq_m, train=train))
        record(f"FAISS PQ(m={args.pq_m})", f_pq, t_f_pq)

    print()
    print(format_table(results))
    if args.json:
        print(f"\nwrote {write_results(args.json, results)}")


def _add(index, vectors):
    index.add(vectors)
    return index


if __name__ == "__main__":
    main()
