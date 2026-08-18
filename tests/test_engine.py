import numpy as np
import pytest

from proxima import FlatIndex, HnswIndex, Metric


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return _normalize(rng.standard_normal((3000, 64)).astype(np.float32))


def test_inner_product_matches_numpy(data):
    idx = FlatIndex(dim=64, metric=Metric.InnerProduct)
    idx.add(data)
    rng = np.random.default_rng(1)
    q = _normalize(rng.standard_normal(64).astype(np.float32))

    ids, sims = idx.search(q, k=10)
    ref = np.argsort(-(data @ q))[:10]
    assert set(ids.tolist()) == set(ref.tolist())
    for i, s in zip(ids.tolist(), sims.tolist()):
        assert abs(s - float(data[i] @ q)) < 1e-4


def test_l2_matches_numpy(data):
    idx = FlatIndex(dim=64, metric=Metric.L2)
    idx.add(data)
    rng = np.random.default_rng(2)
    q = _normalize(rng.standard_normal(64).astype(np.float32))

    ids, dists = idx.search(q, k=10)
    d_all = np.linalg.norm(data - q, axis=1)
    ref = np.argsort(d_all)[:10]
    assert set(ids.tolist()) == set(ref.tolist())
    assert np.all(np.diff(dists) >= -1e-4)
    assert abs(dists[0] - d_all[ref[0]]) < 1e-3


def test_self_is_nearest(data):
    idx = FlatIndex(dim=64, metric=Metric.InnerProduct)
    idx.add(data)
    for i in (0, 100, 2999):
        ids, sims = idx.search(data[i].copy(), k=1)
        assert ids[0] == i
        assert sims[0] == pytest.approx(1.0, abs=1e-3)


def test_batch_equals_single(data):
    idx = FlatIndex(dim=64, metric=Metric.InnerProduct)
    idx.add(data)
    rng = np.random.default_rng(3)
    queries = _normalize(rng.standard_normal((50, 64)).astype(np.float32))

    bids, _ = idx.search_batch(queries, k=8, num_threads=4)
    for qi in range(len(queries)):
        sids, _ = idx.search(queries[qi], k=8)
        assert bids[qi].tolist() == sids.tolist()


def test_padding_when_fewer_than_k():
    idx = FlatIndex(dim=4, metric=Metric.L2)
    idx.add(np.eye(3, 4, dtype=np.float32))
    ids, _ = idx.search(np.zeros(4, dtype=np.float32), k=5)
    assert (ids[3:] == -1).all()
    assert sorted(ids[:3].tolist()) == [0, 1, 2]


def test_incremental_add(data):
    idx = FlatIndex(dim=64, metric=Metric.InnerProduct)
    idx.add(data[:1000])
    idx.add(data[1000:])
    assert idx.size == len(data) == len(idx)


def test_dim_mismatch_raises():
    idx = FlatIndex(dim=8, metric=Metric.L2)
    with pytest.raises(Exception):
        idx.add(np.zeros((10, 9), dtype=np.float32))
    idx.add(np.zeros((10, 8), dtype=np.float32))
    with pytest.raises(Exception):
        idx.search(np.zeros(7, dtype=np.float32), k=3)


def test_metric_and_metadata():
    idx = FlatIndex(dim=16, metric=Metric.L2)
    assert idx.metric == Metric.L2
    assert idx.dim == 16
    idx.add(np.zeros((4, 16), dtype=np.float32))
    assert idx.memory_bytes == 4 * 16 * 4


def test_non_contiguous_input_not_corrupted(data):
    for metric in (Metric.InnerProduct, Metric.L2):
        for cls in (FlatIndex, HnswIndex):
            idx = cls(dim=64, metric=metric)
            idx.add(np.asfortranarray(data))
            for i in (0, 123, 2999):
                q = np.ascontiguousarray(data[i])
                ids, _ = idx.search(q, k=1)
                assert ids[0] == i, (cls.__name__, metric, i)
    idx = FlatIndex(dim=64, metric=Metric.L2)
    idx.add(data)
    strided = np.zeros(128, dtype=np.float32)[::2]
    strided[:] = data[42]
    ids, _ = idx.search(strided, k=1)
    assert ids[0] == 42
