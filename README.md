# Proxima

**A from-scratch, GPU-accelerable vector search engine, written in Rust and demonstrated through semantic search.**

> Search by *meaning*, not keywords. Type "a quiet beach town in southern Europe" and get the most semantically similar items from a corpus of millions, in milliseconds. It's the same kind of retrieval that sits underneath vector databases and the RAG layer of most modern AI systems.

Proxima is an **approximate nearest-neighbor (ANN) vector search engine**, built from the ground up. It has a hand-written **HNSW** graph index, an exact brute-force baseline, **product quantization** for memory compression, and a benchmark harness that checks recall, latency, and memory against exact ground truth and head-to-head against [FAISS](https://github.com/facebookresearch/faiss) on SIFT1M, matching its recall at every operating point and running modestly ahead on multi-threaded HNSW throughput, while FAISS keeps the edge single-threaded and leads clearly on batched exact-flat.

🚀 **[Deploy the demo](docs/DEPLOY.md)** to a free hosted URL.

> **Scope:** this project is the *search infrastructure*: the index, the search algorithms, the systems engineering, and the benchmarking. Embeddings come from an off-the-shelf model (sentence-transformers); training embedding models is **not** part of the project.

---

## Architecture

![Proxima architecture: Python orchestration → PyO3 bindings → Rust core (FlatIndex / HnswIndex / PqIndex / distance kernels), with a CUDA satellite accelerating exact search and FAISS as an external comparison baseline](docs/assets/architecture.svg)

- **`crates/core`**: the engine, pure Rust, no Python dependency. Directly `cargo test`-able and benchmarkable.
- **`crates/py`**: thin [PyO3](https://pyo3.rs) bindings exposing the engine to Python as `proxima._core` (built with [maturin](https://www.maturin.rs)). Releases the GIL around native search.
- **`python/proxima`**: the Python package for corpus loading, the embedding pipeline, benchmarks, and the demo.

## Status

Built in stages, each one a complete, working index before moving to the next.

| Milestone | What | State |
|-----------|------|-------|
| **M0** | Exact (flat) brute-force search + embeddings + minimal demo, end-to-end | ✅ done |
| **M1** | GPU exact search (CUDA via FFI) + speedup number | ✅ done. **27× over multicore CPU** (48× single-thread) on an RTX 5070, exact |
| **M2** | From-scratch **HNSW** index: recall@10 vs exact, latency | ✅ done (the centerpiece) |
| **M3** | Scale to millions + **product quantization** + FAISS comparison | ✅ done |
| **M4** | Polished demo (1M-scale), README diagram, results tables | ✅ done |

## Results

AMD Ryzen 7 7800X3D (8 cores / 16 threads) · 99k Wikipedia (Simple English) articles · 384-dim embeddings · k=10 · 1,000 held-out queries (excluded from the indexed set, not just from the training set). Recall is measured against the exact flat index (ground truth). p50/p99 are single-query latency; QPS is throughput single-threaded (`1t`) and across all cores (`mt`).

| index | recall@10 | p50 (ms) | p99 (ms) | QPS (1t) | QPS (mt) | mem |
|-------|----------:|---------:|---------:|---------:|---------:|----:|
| Flat (exact) | 1.000 | 4.03 | 4.59 | 250 | 1,977 | 152 MB |
| HNSW `ef=16` | 0.877 | **0.065** | 0.115 | 14,374 | **112,718** | 173 MB |
| HNSW `ef=32` | 0.941 | 0.110 | 0.179 | 8,765 | 67,855 | 173 MB |
| HNSW `ef=64` | 0.979 | 0.209 | 0.302 | 4,766 | 38,696 | 173 MB |
| HNSW `ef=128` | 0.993 | 0.365 | 0.514 | 2,645 | 22,023 | 173 MB |
| HNSW `ef=256` | 0.996 | 0.665 | 0.933 | 1,542 | 12,437 | 173 MB |

**HNSW reaches 97.9% recall@10 at 0.21 ms p50, about 19× faster than exact single-threaded, and sustains 39k QPS at that recall level** (or 113k QPS at 88% recall for the fastest setting). `ef_search` is the recall/latency dial. (Reproduce: `python bench/run_bench.py --corpus data/wiki_simple --ef 16,32,64,128,256 --json bench/results/wiki_simple_100k.json`.)

![HNSW's recall/latency dial vs. exact search on 100k Wikipedia articles](docs/assets/recall_latency_wiki.svg)

### SIFT1M: head-to-head vs FAISS

The canonical 1M-vector ANN benchmark (128-dim, **held-out** queries + exact ground truth), run against [FAISS](https://github.com/facebookresearch/faiss) on identical data, queries, and ground truth. AMD Ryzen 7 7800X3D (8 cores / 16 threads), k=10, 1,000 queries.

| index | recall@10 | p50 (ms) | QPS (1t) | QPS (mt) | mem |
|-------|----------:|---------:|---------:|---------:|----:|
| **PX** Flat (exact) | 0.999 | 9.60 | 104 | 156 | 512 MB |
| **PX** HNSW `ef=32` | 0.904 | 0.071 | 14,142 | 90,896 | 685 MB |
| **PX** HNSW `ef=64` | 0.965 | 0.127 | 8,065 | 50,579 | 685 MB |
| **PX** HNSW `ef=128` | 0.990 | 0.239 | 4,341 | 27,369 | 685 MB |
| **PX** PQ `m=16` | 0.538 | 7.51 | 133 | 1,014 | **16 MB** |
| FAISS Flat | 0.999 | 13.12 | 600 | 773 | 512 MB |
| FAISS HNSW `ef=32` | 0.898 | 0.072 | 14,588 | 79,759 | 656 MB |
| FAISS HNSW `ef=64` | 0.962 | 0.128 | 8,273 | 46,254 | 656 MB |
| FAISS HNSW `ef=128` | 0.990 | 0.240 | 4,380 | 25,459 | 656 MB |
| FAISS PQ `m=16` | 0.532 | 7.71 | 128 | 1,000 | 16 MB |

**Takeaways:**
- **Recall matches FAISS** at every operating point: Proxima's HNSW graph and PQ codebooks are correct (recall is even marginally higher at every operating point in all four invocations — e.g. 0.965 vs 0.962 at ef=64).
- **Multi-threaded throughput runs modestly ahead; single-threaded runs modestly behind.** Across four full invocations Proxima led FAISS on every one of twelve multi-threaded measurements, by a median of **1.09×** (range 1.01–1.26). Single-threaded it trails, median 0.93×. The spread matters as much as the number: run-to-run variance is 15–22% even with every figure taken as the median of seven timed passes, so read this as a modest repeatable lead rather than a precise multiple.
- **PQ compresses 512 MB down to 16 MB (32×)** with the same recall tradeoff as FAISS PQ.
- **HNSW build is parallelized** across cores (rayon, per-node locking; the query path stays lock-free), finishing the full 1M-vector SIFT index in **40 s** against FAISS's 54 s — faster in all four invocations.
- **Where FAISS still wins, and why:** batched exact-flat throughput. FAISS uses a BLAS GEMM; Proxima uses a scalar scan over 8-wide accumulators that LLVM auto-vectorizes under `-C target-cpu=native`, about 5× slower on that path (773 vs 156 QPS multi-threaded). Single-query flat latency actually favors Proxima (9.6 ms vs 13.1 ms) — the GEMM only pays off once the batch is large. A blocked matmul would close it; it is not a correctness problem.

![SIFT1M recall vs. latency: Proxima HNSW vs. FAISS HNSW](docs/assets/recall_latency_sift1m.svg)

(Reproduce: `python bench/bench_sift.py --json bench/results/sift1m.json`; regenerate these charts with `python bench/plot_results.py`.)

### GPU exact search (CUDA)

A custom CUDA k-NN kernel (one block per query, base stored transposed for
coalesced reads, block-level top-k reduction) accelerates the exact brute-force
baseline. RTX 5070 vs. Ryzen 7 7800X3D (8 cores / 16 threads), 1M × 128, 2,000 queries,
k=10, on an otherwise idle machine:

| | time | throughput | speedup |
|---|-----:|-----------:|--------:|
| CPU flat, 1 thread | 22.9 s | 87 q/s | 1× |
| CPU flat, all cores | 12.9 s | 155 q/s | 1.8× |
| **GPU exact (CUDA)** | **0.48 s** | **4,142 q/s** | **26.7× / 47.5×** |

The GPU's top-k is cross-checked against the CPU index on every run, with
**exact agreement (1.0000)** since both are exact algorithms, just at different
speeds. (Reproduce on an
NVIDIA machine, CUDA 12.8+: `cargo run --release --example gpu_knn --features cuda -- 1000000 128 2000 10 0 5`; see
[docs/SETUP-GPU.md](docs/SETUP-GPU.md).)

Every figure is the median of 5 timed runs taken inside one invocation, so
`bench/results/gpu.json` holds exactly the numbers printed above along with the device,
driver, CPU and commit that produced them.

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

vectors = np.random.rand(100_000, 384).astype("float32")
query = np.random.rand(384).astype("float32")

# Exact baseline
flat = FlatIndex(dim=384, metric=Metric.InnerProduct)
flat.add(vectors)
ids, scores = flat.search(query, k=10)

# Approximate HNSW, same interface, ~ms latency at scale
hnsw = HnswIndex(dim=384, metric=Metric.InnerProduct, m=16, ef_search=64)
hnsw.add(vectors)
hnsw.save("index.sfidx"); hnsw = HnswIndex.load("index.sfidx")

# Metadata-filtered search: "nearest neighbors WHERE category = X"
category_ids = np.random.randint(0, 8, size=len(vectors))
mask = category_ids == 3   # bool array, one entry per stored vector
ids, scores = hnsw.search_filtered(query, mask, k=10)
```

### The semantic-search demo

```bash
python scripts/build_corpus.py --limit 1000000 --config 20231101.en --out data/wiki_1m
python scripts/build_index.py  --corpus data/wiki_1m --type hnsw   # persist the graph
make demo                                                          # http://127.0.0.1:8000
```

Type a natural-language query → semantically ranked Wikipedia results + the live index-search latency. Index-search latency stays well under a millisecond on the corpora measured above (the graph is warmed at startup); embedding the query itself takes longer than the search. With no corpus built, `make corpus` builds a quick 100k Simple-English set.

### Reproduce the benchmarks

```bash
python bench/run_bench.py  --corpus data/wiki_simple   # HNSW vs exact (recall/latency sweep)
python bench/bench_sift.py                             # Proxima vs FAISS on SIFT1M
```

`bench_sift.py` needs the SIFT1M corpus, which isn't bundled (168 MB compressed). Fetch it once:

```bash
mkdir -p data/sift
curl -o data/sift/sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf data/sift/sift.tar.gz -C data/sift   # -> data/sift/sift/*.fvecs
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

## License

MIT
