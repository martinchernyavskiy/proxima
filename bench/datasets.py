"""Standard ANN benchmark datasets.

SIFT1M (the canonical 1M-vector benchmark) ships in the TEXMEX `.fvecs`/`.ivecs`
format: each record is an int32 dimension followed by that many float32 (fvecs)
or int32 (ivecs) values. It comes with 10k held-out query vectors and exact
ground-truth neighbor ids — so recall is measured against a real, independent
answer key rather than queries sampled from the base set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def read_fvecs(path: str | Path) -> np.ndarray:
    """Read an `.fvecs` file into a contiguous float32 (n, dim) array."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    return np.ascontiguousarray(rows[:, 1:].view(np.float32))


def read_ivecs(path: str | Path) -> np.ndarray:
    """Read an `.ivecs` file into a contiguous int32 (n, dim) array."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return np.ascontiguousarray(raw.reshape(-1, dim + 1)[:, 1:])


@dataclass
class AnnDataset:
    base: np.ndarray       # (n, dim) float32
    queries: np.ndarray    # (nq, dim) float32
    ground_truth: np.ndarray  # (nq, gt_k) int32 — true neighbor ids per query
    metric: str            # "L2" or "InnerProduct"
    train: np.ndarray | None = None  # optional learn set (PQ codebook training)
    name: str = "dataset"


def load_sift(root: str | Path = "data/sift/sift") -> AnnDataset:
    """Load SIFT1M from an extracted TEXMEX directory."""
    root = Path(root)
    base = read_fvecs(root / "sift_base.fvecs")
    queries = read_fvecs(root / "sift_query.fvecs")
    gt = read_ivecs(root / "sift_groundtruth.ivecs")
    train_path = root / "sift_learn.fvecs"
    train = read_fvecs(train_path) if train_path.exists() else None
    # SIFT vectors are raw descriptors compared by Euclidean distance.
    return AnnDataset(base=base, queries=queries, ground_truth=gt,
                      metric="L2", train=train, name="SIFT1M")
