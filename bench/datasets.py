from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def read_fvecs(path: str | Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    return np.ascontiguousarray(rows[:, 1:].view(np.float32))


def read_ivecs(path: str | Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return np.ascontiguousarray(raw.reshape(-1, dim + 1)[:, 1:])


@dataclass
class AnnDataset:
    base: np.ndarray
    queries: np.ndarray
    ground_truth: np.ndarray
    metric: str
    train: np.ndarray | None = None
    name: str = "dataset"


def load_sift(root: str | Path = "data/sift/sift") -> AnnDataset:
    root = Path(root)
    base = read_fvecs(root / "sift_base.fvecs")
    queries = read_fvecs(root / "sift_query.fvecs")
    gt = read_ivecs(root / "sift_groundtruth.ivecs")
    train_path = root / "sift_learn.fvecs"
    train = read_fvecs(train_path) if train_path.exists() else None
    return AnnDataset(base=base, queries=queries, ground_truth=gt,
                      metric="L2", train=train, name="SIFT1M")
