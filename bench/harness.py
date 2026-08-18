from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from proxima import FlatIndex, Metric


@dataclass
class BenchResult:
    name: str
    n_base: int
    dim: int
    k: int
    recall_at_k: float | None
    p50_ms: float
    p99_ms: float
    mean_ms: float
    qps_1t: float
    qps_mt: float
    memory_mb: float

    def as_dict(self) -> dict:
        return asdict(self)


def exact_ground_truth(base: np.ndarray, queries: np.ndarray, k: int,
                       metric: Metric = Metric.InnerProduct) -> np.ndarray:
    idx = FlatIndex(dim=base.shape[1], metric=metric)
    idx.add(np.ascontiguousarray(base, dtype=np.float32))
    ids, _ = idx.search_batch(np.ascontiguousarray(queries, dtype=np.float32),
                              k=k, num_threads=0)
    return ids


def recall_at_k(approx_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    nq = len(approx_ids)
    total = 0.0
    for i in range(nq):
        approx = {int(x) for x in approx_ids[i][:k] if x >= 0}
        truth = {int(x) for x in gt_ids[i][:k] if x >= 0}
        if truth:
            total += len(approx & truth) / len(truth)
    return total / nq if nq else 0.0


def measure_latency(index, queries: np.ndarray, k: int, repeats: int = 3
                    ) -> tuple[float, float, float]:
    if len(queries) == 0:
        return 0.0, 0.0, 0.0
    times: list[float] = []
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            index.search(q, k=k)
            times.append((time.perf_counter() - t0) * 1e3)
    arr = np.asarray(times)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 99)), float(arr.mean())


def measure_throughput(index, queries: np.ndarray, k: int, num_threads: int) -> float:
    index.search_batch(queries[: min(64, len(queries))], k=k, num_threads=num_threads)
    t0 = time.perf_counter()
    index.search_batch(queries, k=k, num_threads=num_threads)
    dt = time.perf_counter() - t0
    return len(queries) / dt if dt > 0 else float("inf")


def benchmark_index(name: str, index, queries: np.ndarray, k: int,
                    gt_ids: np.ndarray | None, latency_n: int = 300) -> BenchResult:
    q = np.ascontiguousarray(queries, dtype=np.float32)
    approx_ids, _ = index.search_batch(q, k=k, num_threads=0)
    recall = recall_at_k(approx_ids, gt_ids, k) if gt_ids is not None else None
    p50, p99, mean = measure_latency(index, q[:latency_n], k)
    qps_1t = measure_throughput(index, q, k, num_threads=1)
    qps_mt = measure_throughput(index, q, k, num_threads=0)
    return BenchResult(
        name=name, n_base=index.size, dim=index.dim, k=k,
        recall_at_k=recall, p50_ms=p50, p99_ms=p99, mean_ms=mean,
        qps_1t=qps_1t, qps_mt=qps_mt, memory_mb=index.memory_bytes / 1e6,
    )


def format_table(results: list[BenchResult]) -> str:
    cols = ["name", "n_base", "dim", "k", "recall_at_k", "p50_ms", "p99_ms",
            "qps_1t", "qps_mt", "memory_mb"]
    headers = ["index", "N", "dim", "k", "recall@k", "p50(ms)", "p99(ms)",
               "QPS(1t)", "QPS(mt)", "mem(MB)"]

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}" if v < 100 else f"{v:,.0f}"
        return f"{v:,}" if isinstance(v, int) else str(v)

    rows = [[fmt(getattr(r, c)) for c in cols] for r in results]
    widths = [max([len(headers[i])] + [len(row[i]) for row in rows]) for i in range(len(cols))]
    line = lambda parts: "  ".join(p.rjust(widths[i]) for i, p in enumerate(parts))
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])
