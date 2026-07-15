"""Tests for the HNSW approximate index: recall against the exact baseline,
the ef_search recall/latency dial, and interface parity with FlatIndex.
"""

import numpy as np
import pytest

from proxima import FlatIndex, HnswIndex, Metric


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return _normalize(rng.standard_normal((5000, 48)).astype(np.float32))


@pytest.fixture
def queries():
    rng = np.random.default_rng(7)
    return _normalize(rng.standard_normal((200, 48)).astype(np.float32))


def _recall(approx_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    hits = 0
    for a, g in zip(approx_ids, gt_ids):
        hits += len(set(int(x) for x in a if x >= 0) & set(int(x) for x in g))
    return hits / (len(approx_ids) * k)


def test_hnsw_high_recall_vs_exact(data, queries):
    flat = FlatIndex(dim=48, metric=Metric.InnerProduct)
    flat.add(data)
    hnsw = HnswIndex(dim=48, metric=Metric.InnerProduct, ef_search=128)
    hnsw.add(data)
    assert hnsw.size == len(data)

    gt, _ = flat.search_batch(queries, k=10)
    approx, _ = hnsw.search_batch(queries, k=10)
    assert _recall(approx, gt, 10) > 0.95


def test_ef_search_improves_recall(data, queries):
    flat = FlatIndex(dim=48, metric=Metric.InnerProduct)
    flat.add(data)
    gt, _ = flat.search_batch(queries, k=10)

    hnsw = HnswIndex(dim=48, metric=Metric.InnerProduct)
    hnsw.add(data)

    hnsw.ef_search = 16
    low, _ = hnsw.search_batch(queries, k=10)
    hnsw.ef_search = 200
    high, _ = hnsw.search_batch(queries, k=10)
    assert _recall(high, gt, 10) >= _recall(low, gt, 10)
    assert _recall(high, gt, 10) > 0.95


def test_l2_metric_recall(data, queries):
    flat = FlatIndex(dim=48, metric=Metric.L2)
    flat.add(data)
    hnsw = HnswIndex(dim=48, metric=Metric.L2, ef_search=128)
    hnsw.add(data)
    gt, _ = flat.search_batch(queries, k=10)
    approx, _ = hnsw.search_batch(queries, k=10)
    assert _recall(approx, gt, 10) > 0.95


def test_interface_parity_and_metadata(data):
    hnsw = HnswIndex(dim=48, metric=Metric.L2, m=16, ef_construction=100, ef_search=50)
    hnsw.add(data)
    assert hnsw.size == len(data) == len(hnsw)
    assert hnsw.dim == 48
    assert hnsw.metric == Metric.L2
    assert hnsw.ef_search == 50
    hnsw.ef_search = 80
    assert hnsw.ef_search == 80
    assert hnsw.memory_bytes > 0


def test_validation():
    with pytest.raises(Exception):
        HnswIndex(dim=0)
    with pytest.raises(Exception):
        HnswIndex(dim=8, m=1)  # m must be >= 2
