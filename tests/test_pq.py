import numpy as np
import pytest

from proxima import FlatIndex, Metric, PqIndex


def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((4000, 64)).astype(np.float32)


@pytest.fixture
def queries():
    rng = np.random.default_rng(9)
    return rng.standard_normal((200, 64)).astype(np.float32)


def _recall(approx, gt, k):
    hits = sum(len(set(int(x) for x in a if x >= 0) & set(int(x) for x in g))
               for a, g in zip(approx, gt))
    return hits / (len(approx) * k)


def test_lifecycle_and_recall(data, queries):
    flat = FlatIndex(dim=64, metric=Metric.L2)
    flat.add(data)
    gt, _ = flat.search_batch(queries, k=10)

    pq = PqIndex(dim=64, metric=Metric.L2, m=8)
    assert not pq.is_trained
    pq.train(data)
    assert pq.is_trained
    pq.add(data)
    assert pq.size == len(data) == len(pq)

    approx, _ = pq.search_batch(queries, k=10)
    assert _recall(approx, gt, 10) > 0.25


def test_recall_at_100_high(data, queries):
    flat = FlatIndex(dim=64, metric=Metric.L2)
    flat.add(data)
    gt, _ = flat.search_batch(queries, k=100)
    pq = PqIndex(dim=64, metric=Metric.L2, m=16)
    pq.train(data)
    pq.add(data)
    approx, _ = pq.search_batch(queries, k=100)
    assert _recall(approx, gt, 100) > 0.55


def test_compression_ratio():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((40000, 64)).astype(np.float32)
    pq = PqIndex(dim=64, metric=Metric.L2, m=8)
    pq.train(base)
    pq.add(base)
    assert pq.raw_bytes == len(base) * 64 * 4
    assert pq.compression_ratio > 20


def test_must_train_before_use(data):
    pq = PqIndex(dim=64, metric=Metric.L2, m=8)
    with pytest.raises(Exception):
        pq.add(data)
    with pytest.raises(Exception):
        pq.search(data[0], k=5)


def test_dim_divisibility():
    with pytest.raises(Exception):
        PqIndex(dim=10, m=3)


def test_invalid_m_rejected():
    with pytest.raises(Exception):
        PqIndex(dim=8, m=0)


def test_small_train_sample_not_degenerate(data, queries):
    flat = FlatIndex(dim=64, metric=Metric.L2)
    flat.add(data)
    gt, _ = flat.search_batch(queries, k=10)

    pq = PqIndex(dim=64, metric=Metric.L2, m=8, train_sample=50)
    pq.train(data)
    pq.add(data)
    assert pq.is_trained and pq.size == len(data)

    approx, _ = pq.search_batch(queries, k=10)
    assert _recall(approx, gt, 10) > 0.15


def test_inner_product_metric(queries):
    rng = np.random.default_rng(1)
    data = _normalize(rng.standard_normal((4000, 64)).astype(np.float32))
    q = _normalize(queries)
    flat = FlatIndex(dim=64, metric=Metric.InnerProduct)
    flat.add(data)
    gt, _ = flat.search_batch(q, k=10)
    pq = PqIndex(dim=64, metric=Metric.InnerProduct, m=16)
    pq.train(data)
    pq.add(data)
    approx, _ = pq.search_batch(q, k=10)
    assert _recall(approx, gt, 10) > 0.3
