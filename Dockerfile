FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

ENV RUSTFLAGS="-C target-cpu=x86-64-v2"

ENV HF_HOME=/app/.hf

WORKDIR /app

RUN pip install --no-cache-dir maturin==1.15.0 \
 && pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir sentence-transformers==6.0.1 datasets==5.0.1 fastapi==0.141.1 "uvicorn[standard]==0.52.4"

COPY Cargo.toml Cargo.lock pyproject.toml README.md ./
COPY .cargo ./.cargo
COPY crates ./crates
COPY python ./python
RUN maturin build --release --out /tmp/wheels \
 && pip install --no-cache-dir /tmp/wheels/*.whl

COPY demo ./demo
COPY scripts ./scripts

RUN python scripts/build_corpus.py --limit 30000 --config 20231101.simple --out data/wiki_demo \
 && python scripts/build_index.py --corpus data/wiki_demo --type hnsw

ENV PROXIMA_CORPUS=data/wiki_demo
EXPOSE 7860
CMD ["uvicorn", "demo.app:app", "--host", "0.0.0.0", "--port", "7860"]
