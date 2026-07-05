# Container image for the SearchForge semantic-search demo.
#
# Builds the Rust engine, bakes a small Wikipedia corpus + HNSW index at image
# build time (so startup is fast), and serves the FastAPI app. Tuned for
# Hugging Face Spaces (Docker SDK, port 7860) but portable to any container host
# (Render, Fly.io, Railway, a VPS).
FROM python:3.11-slim

# --- system toolchain + Rust ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# Build for a portable x86-64 baseline, NOT target-cpu=native: the build host
# and run host may differ, and native codegen could emit unsupported opcodes.
ENV RUSTFLAGS="-C target-cpu=x86-64-v2"

# Cache HF model/datasets inside the image so runtime startup needs no network.
ENV HF_HOME=/app/.hf

WORKDIR /app

# --- Python deps (CPU-only torch keeps the image lean) ---
RUN pip install --no-cache-dir maturin \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir sentence-transformers datasets fastapi "uvicorn[standard]"

# --- build the Rust extension into the environment ---
COPY Cargo.toml Cargo.lock pyproject.toml README.md ./
COPY .cargo ./.cargo
COPY crates ./crates
COPY python ./python
RUN maturin build --release --out /tmp/wheels \
 && pip install --no-cache-dir /tmp/wheels/*.whl

COPY demo ./demo
COPY scripts ./scripts

# --- bake a small, recruiter-legible corpus + HNSW index ---
RUN python scripts/build_corpus.py --limit 30000 --config 20231101.simple --out data/wiki_demo \
 && python scripts/build_index.py --corpus data/wiki_demo --type hnsw

ENV SEARCHFORGE_CORPUS=data/wiki_demo
EXPOSE 7860
CMD ["uvicorn", "demo.app:app", "--host", "0.0.0.0", "--port", "7860"]
