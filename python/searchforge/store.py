"""Persistence for a built corpus: the embedding matrix + document metadata.

Layout under `data_dir`:
  vectors.npy   float32 (n, dim) embedding matrix
  meta.jsonl    one JSON document per line, aligned by row to vectors.npy
  manifest.json small header (counts, dim, model, metric)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .corpus import Doc


def save_corpus(data_dir: str | Path, vectors: np.ndarray, docs: list[Doc],
                model_name: str, metric: str = "InnerProduct") -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    assert vectors.shape[0] == len(docs), "vectors and docs must align"
    np.save(data_dir / "vectors.npy", np.ascontiguousarray(vectors, dtype=np.float32))

    with open(data_dir / "meta.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    manifest = {
        "count": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "model": model_name,
        "metric": metric,
    }
    with open(data_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_metadata(data_dir: str | Path) -> tuple[list[dict], dict]:
    """Load just the documents + manifest (no embedding matrix). Used when a
    prebuilt index already holds the vectors, to avoid a redundant multi-GB
    load of vectors.npy."""
    data_dir = Path(data_dir)
    with open(data_dir / "meta.jsonl") as f:
        docs = [json.loads(line) for line in f]
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)
    return docs, manifest


def load_corpus(data_dir: str | Path) -> tuple[np.ndarray, list[dict], dict]:
    data_dir = Path(data_dir)
    vectors = np.ascontiguousarray(np.load(data_dir / "vectors.npy"), dtype=np.float32)
    docs, manifest = load_metadata(data_dir)
    return vectors, docs, manifest
