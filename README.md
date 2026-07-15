# Proxima

**A from-scratch, GPU-accelerable vector search engine — written in Rust, demonstrated through semantic search.**

> Search by *meaning*, not keywords. Type "a quiet beach town in southern Europe" and get the most semantically similar items from a corpus of millions, in milliseconds — the same retrieval technology that powers vector databases (Pinecone, Weaviate, Milvus) and the RAG layer of modern AI systems.

Proxima is an **approximate nearest-neighbor (ANN) vector search engine** built from the ground up: a hand-written **HNSW** graph index, an exact brute-force baseline, **product quantization** for memory compression, and an honest benchmark harness measuring recall / latency / memory against the exact ground truth and against FAISS.

📖 **[Design writeup](docs/WRITEUP.md)** — HNSW internals, the parallel-build regression story, and the full FAISS comparison. · 🚀 **[Deploy the demo](docs/DEPLOY.md)** to a free hosted URL.

> **Scope, honestly:** this project is the *search infrastructure* — the index, the search algorithms, the systems engineering and the benchmarking. Embeddings come from an off-the-shelf model (sentence-transformers); training embedding models is explicitly **not** part of the project.

---

## Architecture

![Proxima architecture: Python orchestration → PyO3 bindings → Rust core (FlatIndex / HnswIndex / PqIndex / distance kernels), with a CUDA satellite accelerating exact search and FAISS as an external comparison baseline](docs/assets/architecture.svg)

- **`crates/core`** — the engine, pure Rust, no Python dependency. Directly `cargo test`-able and benchmarkable.
- **`crates/py`** — thin [PyO3](https://pyo3.rs) bindings exposing the engine to Python as `proxima._core` (built with [maturin](https://www.maturin.rs)). Releases the GIL around native search.
- **`python/proxima`** — the Python package: corpus loading, embedding pipeline, benchmarks, demo.

## Status

Built milestone-by-milestone so it is resume-ready early and never an unfinished repo.

| Milestone | What | State |
|-----------|------|-------|
| **M0** | Exact (flat) brute-force search + embeddings + minimal demo, end-to-end | ✅ done |
| **M1** | GPU exact search (CUDA via FFI) + speedup number | ✅ done — **18.9× over multicore CPU** (35× single-thread), exact |
| **M2** | From-scratch **HNSW** index — recall@10 vs exact, latency | ✅ done (the centerpiece) |
| **M3** | Scale to millions + **product quantization** + FAISS comparison | ✅ done |
| **M4** | Polished demo (1M-scale), README diagram, results tables | ✅ done |

## Results

Apple M3 Pro · 100k Wikipedia (Simple English) articles · 384-dim embeddings · k=10 · 1,000 queries. Recall is measured against the exact flat index (ground truth). p50/p99 are single-query latency; QPS is throughput single-threaded (`1t`) and across all cores (`mt`).

| index | recall@10 | p50 (ms) | p99 (ms) | QPS (1t) | QPS (mt) | mem |
|-------|----------:|---------:|---------:|---------:|---------:|----:|
| Flat (exact) | 1.000 | 4.44 | 5.76 | 231 | 532 | 154 MB |
| HNSW `ef=16` | 0.974 | **0.22** | 0.39 | 4,499 | **31,392** | 174 MB |
| HNSW `ef=32` | 0.991 | 0.38 | 0.59 | 2,681 | 17,590 | 174 MB |
| HNSW `ef=64` | 0.996 | 0.70 | 1.13 | 1,551 | 10,379 | 174 MB |
| HNSW `ef=128` | 0.998 | 1.22 | 2.22 | 807 | 5,890 | 174 MB |
| HNSW `ef=256` | 0.999 | 2.32 | 3.56 | 448 | 3,047 | 174 MB |

**HNSW reaches 99.6% recall@10 at 0.70 ms p50 — ~6× faster than exact single-threaded — and sustains 30k+ QPS at 97% recall.** `ef_search` is the recall/latency dial. (Reproduce: `python bench/run_bench.py --corpus data/wiki_simple`.)

![HNSW's recall/latency dial vs. exact search on 100k Wikipedia articles](docs/assets/recall_latency_wiki.svg)

### SIFT1M — head-to-head vs FAISS

The canonical 1M-vector ANN benchmark (128-dim, **held-out** queries + exact ground truth), run against [FAISS](https://github.com/facebookresearch/faiss) on identical data, queries, and ground truth. Apple M3 Pro, k=10, 1,000 queries.

| index | recall@10 | p50 (ms) | QPS (1t) | QPS (mt) | mem |
|-------|----------:|---------:|---------:|---------:|----:|
| **PX** Flat (exact) | 0.999 | 9.99 | 101 | 223 | 512 MB |
| **PX** HNSW `ef=32` | 0.905 | 0.114 | 8,966 | 58,646 | 684 MB |
| **PX** HNSW `ef=64` | 0.967 | 0.202 | 5,146 | 34,449 | 684 MB |
| **PX** HNSW `ef=128` | 0.991 | 0.366 | 2,857 | 19,170 | 684 MB |
| **PX** PQ `m=16` | 0.540 | 5.66 | 176 | 1,190 | **16 MB** |
| FAISS Flat | 0.999 | 8.31 | 1,630 | 2,534 | 512 MB |
| FAISS HNSW `ef=64` | 0.964 | 0.125 | 7,454 | 49,445 | 656 MB |
| FAISS HNSW `ef=128` | 0.989 | 0.228 | 4,639 | 26,096 | 656 MB |
| FAISS PQ `m=16` | 0.533 | 3.15 | 316 | 1,859 | 16 MB |

**Takeaways (honest):**
- **Recall matches FAISS** at every operating point — Proxima's HNSW graph and PQ codebooks are correct (recall is even marginally higher, e.g. 0.967 vs 0.964 at ef=64).
- **HNSW query latency/throughput is within ~1.4–1.6× of FAISS** — e.g. 0.20 ms vs 0.125 ms p50; 34k vs 49k QPS multi-thread. Strong for a hand-written engine.
- **PQ compresses 512 MB → 16 MB (32×)** with the same recall tradeoff as FAISS PQ.
- **HNSW build is parallelized** across cores (rayon, per-node locking; the query path stays lock-free): ~**12× faster** than single-threaded, bringing the SIFT1M build to the same ballpark as FAISS (~tens of seconds).
- **Where FAISS still wins, and why:** exact-flat throughput — FAISS uses a BLAS GEMM, Proxima a straightforward SIMD scan (~16×). That's honest, well-understood headroom (a blocked/BLAS matmul would close it), not a correctness gap.

![SIFT1M recall vs. latency: Proxima HNSW vs. FAISS HNSW](docs/assets/recall_latency_sift1m.svg)

(Reproduce: `python bench/bench_sift.py`; regenerate these charts with `python bench/plot_results.py`.)

### GPU exact search (CUDA)

A custom CUDA k-NN kernel (one block per query, base stored transposed for
coalesced reads, block-level top-k reduction) accelerates the exact brute-force
baseline. RTX 4070 Ti vs. Ryzen 7 7800X3D (8 cores), 1M × 128, 2,000 queries, k=10:

| | time | throughput | speedup |
|---|-----:|-----------:|--------:|
| CPU flat, 1 thread | 20.2 s | 99 q/s | 1× |
| CPU flat, all cores | 10.8 s | 185 q/s | 1.9× |
| **GPU exact (CUDA)** | **0.57 s** | **3,496 q/s** | **18.9× / 35.4×** |

The GPU's top-k is cross-checked against the CPU index on every run — **exact
agreement (1.0000)**, so it's accelerated, not approximated. (Reproduce on an
NVIDIA machine: `cargo run --release --example gpu_knn --features cuda` — see
[docs/SETUP-GPU.md](docs/SETUP-GPU.md).)

## Build & develop

Requires a Rust toolchain and Python 3.9+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release          # builds proxima._core into the venv

cargo test -p proxima-core         # pure-Rust engine tests
```

### Quick start

```python
import numpy as np
from proxima import FlatIndex, HnswIndex, PqIndex, Metric

# Exact baseline
flat = FlatIndex(dim=384, metric=Metric.InnerProduct)
flat.add(np.random.rand(100_000, 384).astype("float32"))
ids, scores = flat.search(np.random.rand(384).astype("float32"), k=10)

# Approximate HNSW — same interface, ~ms latency at scale
hnsw = HnswIndex(dim=384, metric=Metric.InnerProduct, m=16, ef_search=64)
hnsw.add(vectors)
hnsw.save("index.sfidx"); hnsw = HnswIndex.load("index.sfidx")

# Metadata-filtered search — "nearest neighbors WHERE category = X"
mask = category_ids == target_category   # bool array, one entry per stored vector
ids, scores = hnsw.search_filtered(query, mask, k=10)
```

### The semantic-search demo

```bash
python scripts/build_corpus.py --limit 1000000 --config 20231101.en --out data/wiki_1m
python scripts/build_index.py  --corpus data/wiki_1m --type hnsw   # persist the graph
make demo                                                          # http://127.0.0.1:8000
```

Type a natural-language query → semantically ranked Wikipedia results + the live index-search latency. Serves **sub-millisecond search over 1M articles** (the graph is warmed at startup). With no corpus built, `make corpus` builds a quick 100k Simple-English set.

### Reproduce the benchmarks

```bash
python bench/run_bench.py  --corpus data/wiki_simple   # HNSW vs exact (recall/latency sweep)
python bench/bench_sift.py                             # Proxima vs FAISS on SIFT1M
```

## Project layout

```
crates/core/src/   distance.rs · flat.rs · hnsw.rs · pq.rs · metric.rs   (the engine)
crates/py/src/     lib.rs                                                (PyO3 bindings)
python/proxima/    embeddings · corpus · store · search                (orchestration)
bench/             harness · run_bench · bench_sift · faiss_compare · datasets
demo/              app.py + static/index.html                            (FastAPI + UI)
scripts/           build_corpus · build_index · search_cli
tests/             Python binding tests (Rust unit tests live in crates/core)
```

## The engineering, in brief

- **Hand-written HNSW** — multi-layer navigable small-world graph: construction, geometric layer assignment, the neighbor-selection diversity heuristic, greedy descent + best-first beam search; tunable `M` / `ef_construction` / `ef_search`. **Parallel construction** (rayon + per-node `RwLock`, deadlock-free) with a **lock-free query path** (one generic traversal serves both).
- **SIMD distance kernels** — branch-free, lane-parallel accumulation that auto-vectorizes to NEON / AVX on stable Rust (no `-ffast-math`), plus a custom hasher for the graph's visited set.
- **Product quantization** — k-means codebooks + asymmetric distance computation; 32× compression with a measured recall tradeoff.
- **Metadata-filtered search** — `search_filtered` on all three indexes ("nearest neighbors *where category = X*"). HNSW routes through non-matching nodes to preserve graph connectivity but only admits matches into the result set, so a selective filter costs search time, not silently-wrong recall.
- **Cache-aware layout** — contiguous row-major vectors for traversal locality; rayon-parallel batch search with the GIL released.
- **Persistence** — every index serializes to disk, so a million-vector graph loads in seconds instead of rebuilding.
- **Rigorous evaluation** — recall@k vs exact ground truth, p50/p99 latency, QPS, memory, and a head-to-head FAISS comparison on SIFT1M.

## License

MIT
