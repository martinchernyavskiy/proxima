from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._core import Metric
from .corpus import Doc

_KNOWN_METRICS = ("L2", "InnerProduct")


def resolve_metric(name: str) -> Metric:
    if name not in _KNOWN_METRICS:
        raise ValueError(f"unknown metric {name!r}; expected one of {_KNOWN_METRICS}")
    return getattr(Metric, name)


def save_corpus(data_dir: str | Path, vectors: np.ndarray, docs: list[Doc],
                model_name: str, metric: str = "InnerProduct",
                snippet_chars: int = 500) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D (n, dim), got shape {vectors.shape}")
    if vectors.shape[0] != len(docs):
        raise ValueError(
            f"vectors/docs count mismatch: {vectors.shape[0]} vectors vs {len(docs)} docs"
        )
    np.save(data_dir / "vectors.npy", np.ascontiguousarray(vectors, dtype=np.float32))

    with open(data_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    manifest = {
        "count": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "model": model_name,
        "metric": metric,
        "snippet_chars": int(snippet_chars),
        "clean_snippets": True,
    }
    with open(data_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_metadata(data_dir: str | Path) -> tuple[list[dict], dict]:
    data_dir = Path(data_dir)
    with open(data_dir / "meta.jsonl", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)
    expected = manifest.get("count")
    if expected is not None and len(docs) != expected:
        raise ValueError(
            f"metadata count mismatch in {data_dir}: meta.jsonl has {len(docs)} docs "
            f"but manifest.json says {expected}"
        )
    return docs, manifest


def load_corpus(data_dir: str | Path) -> tuple[np.ndarray, list[dict], dict]:
    data_dir = Path(data_dir)
    vectors = np.ascontiguousarray(np.load(data_dir / "vectors.npy"), dtype=np.float32)
    docs, manifest = load_metadata(data_dir)
    if vectors.shape[0] != len(docs):
        raise ValueError(
            f"vectors/docs count mismatch in {data_dir}: "
            f"{vectors.shape[0]} vectors vs {len(docs)} docs"
        )
    return vectors, docs, manifest
