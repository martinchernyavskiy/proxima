VENV ?= .venv
PY   := $(VENV)/bin/python

.PHONY: help build test test-rust test-py bench demo corpus clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

build:  ## build the Rust engine into the venv (maturin develop --release)
	$(VENV)/bin/maturin develop --release

test: test-rust test-py  ## run all tests (Rust core + Python bindings)

test-rust:  ## pure-Rust core unit tests
	cargo test -p proxima-core --release

test-py:  ## Python binding tests
	$(PY) -m pytest tests/ -q

bench:  ## benchmark on synthetic vectors (recall / latency / QPS / memory)
	$(PY) bench/run_bench.py --n 100000 --dim 384 --nq 1000 --k 10

corpus:  ## build the 100k Wikipedia corpus into data/wiki_simple
	$(PY) scripts/build_corpus.py --limit 100000 --out data/wiki_simple

index:  ## build + persist an HNSW index for a corpus (CORPUS=data/wiki_simple)
	$(PY) scripts/build_index.py --corpus $(or $(CORPUS),data/wiki_simple) --type hnsw

demo:  ## run the demo web app at http://127.0.0.1:8000 (auto-picks largest corpus)
	$(VENV)/bin/uvicorn demo.app:app --port 8000

clean:  ## remove build artifacts and downloaded data
	cargo clean
	rm -rf data **/__pycache__ .pytest_cache
