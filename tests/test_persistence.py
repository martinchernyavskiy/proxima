import numpy as np
import pytest

from proxima import FlatIndex, HnswIndex, Metric, PqIndex


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3000, 64)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_flat_roundtrip(tmp_path, data):
    idx = FlatIndex(dim=64, metric=Metric.L2)
    idx.add(data)
    before, _ = idx.search(data[5].copy(), k=10)
    path = str(tmp_path / "flat.sfidx")
    idx.save(path)
    loaded = FlatIndex.load(path)
    after, _ = loaded.search(data[5].copy(), k=10)
    assert loaded.size == idx.size
    assert loaded.metric == Metric.L2
    assert before.tolist() == after.tolist()


def test_hnsw_roundtrip(tmp_path, data):
    idx = HnswIndex(dim=64, metric=Metric.InnerProduct, ef_search=123)
    idx.add(data)
    before, _ = idx.search(data[5].copy(), k=10)
    path = str(tmp_path / "hnsw.sfidx")
    idx.save(path)
    loaded = HnswIndex.load(path)
    after, _ = loaded.search(data[5].copy(), k=10)
    assert loaded.size == idx.size and loaded.ef_search == 123
    assert loaded.metric == Metric.InnerProduct
    assert before.tolist() == after.tolist()


def test_pq_roundtrip(tmp_path, data):
    idx = PqIndex(dim=64, metric=Metric.InnerProduct, m=4)
    idx.train(data)
    idx.add(data)
    before, _ = idx.search(data[5].copy(), k=10)
    path = str(tmp_path / "pq.sfidx")
    idx.save(path)
    loaded = PqIndex.load(path)
    after, _ = loaded.search(data[5].copy(), k=10)
    assert loaded.is_trained and loaded.size == idx.size and loaded.m == 4
    assert loaded.metric == Metric.InnerProduct
    assert before.tolist() == after.tolist()


def test_save_untrained_pq_raises(tmp_path):
    idx = PqIndex(dim=64, metric=Metric.L2, m=8)
    with pytest.raises(Exception):
        idx.save(str(tmp_path / "x.sfidx"))
