import numpy as np
import pytest

from proxima import FlatIndex, HnswIndex, Metric, PqIndex


def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return _normalize(rng.standard_normal((3000, 48)).astype(np.float32))


@pytest.fixture
def mask(data):
    rng = np.random.default_rng(1)
    return rng.random(len(data)) < 0.2


class TestFlatIndexFiltered:
    def test_results_all_satisfy_mask(self, data, mask):
        idx = FlatIndex(dim=48, metric=Metric.InnerProduct)
        idx.add(data)
        q = data[7].copy()
        ids, _ = idx.search_filtered(q, mask, k=10)
        for i in ids:
            if i >= 0:
                assert mask[i], f"result {i} should satisfy the filter"

    def test_matches_bruteforce_reference(self, data, mask):
        idx = FlatIndex(dim=48, metric=Metric.InnerProduct)
        idx.add(data)
        q = data[123].copy()
        ids, scores = idx.search_filtered(q, mask, k=10)

        allowed = np.nonzero(mask)[0]
        sims = data[allowed] @ q
        ref_order = allowed[np.argsort(-sims)][:10]
        assert set(ids.tolist()) == set(ref_order.tolist())

    def test_batch_filtered_matches_single(self, data, mask):
        idx = FlatIndex(dim=48, metric=Metric.L2)
        idx.add(data)
        queries = data[:20].copy()
        bids, _ = idx.search_batch_filtered(queries, mask, k=5)
        for i, q in enumerate(queries):
            sids, _ = idx.search_filtered(q, mask, k=5)
            assert bids[i].tolist() == sids.tolist()

    def test_wrong_mask_length_raises(self, data, mask):
        idx = FlatIndex(dim=48, metric=Metric.L2)
        idx.add(data)
        with pytest.raises(Exception):
            idx.search_filtered(data[0], mask[:-1], k=5)

    def test_zero_matches_returns_padding(self, data):
        idx = FlatIndex(dim=48, metric=Metric.L2)
        idx.add(data)
        empty_mask = np.zeros(len(data), dtype=bool)
        ids, _ = idx.search_filtered(data[0].copy(), empty_mask, k=5)
        assert (ids == -1).all()


class TestHnswIndexFiltered:
    def test_results_all_satisfy_mask(self, data, mask):
        idx = HnswIndex(dim=48, metric=Metric.InnerProduct, ef_search=128)
        idx.add(data)
        for q in data[:20]:
            ids, _ = idx.search_filtered(q.copy(), mask, k=10)
            for i in ids:
                if i >= 0:
                    assert mask[i], f"result {i} should satisfy the filter"

    def test_recall_vs_exact_filtered_ground_truth(self, data, mask):
        flat = FlatIndex(dim=48, metric=Metric.InnerProduct)
        flat.add(data)
        idx = HnswIndex(dim=48, metric=Metric.InnerProduct, ef_search=200)
        idx.add(data)

        queries = data[:100].copy()
        gt, _ = flat.search_batch_filtered(queries, mask, k=10)
        approx, _ = idx.search_batch_filtered(queries, mask, k=10)

        hits = sum(len(set(int(x) for x in a if x >= 0) & set(int(x) for x in g))
                   for a, g in zip(approx, gt))
        recall = hits / (len(queries) * 10)
        assert recall > 0.85, f"filtered recall@10 was {recall:.3f}"

    def test_batch_filtered_matches_single(self, data, mask):
        idx = HnswIndex(dim=48, metric=Metric.L2, ef_search=64)
        idx.add(data)
        queries = data[:15].copy()
        bids, _ = idx.search_batch_filtered(queries, mask, k=5)
        for i, q in enumerate(queries):
            sids, _ = idx.search_filtered(q, mask, k=5)
            assert bids[i].tolist() == sids.tolist()

    def test_wrong_mask_length_raises(self, data, mask):
        idx = HnswIndex(dim=48, metric=Metric.L2)
        idx.add(data)
        with pytest.raises(Exception):
            idx.search_filtered(data[0], mask[:-1], k=5)


class TestPqIndexFiltered:
    def test_results_all_satisfy_mask_and_recall_reasonable(self, data, mask):
        flat = FlatIndex(dim=48, metric=Metric.L2)
        flat.add(data)
        pq = PqIndex(dim=48, metric=Metric.L2, m=16)
        pq.train(data)
        pq.add(data)

        queries = data[:50].copy()
        gt, _ = flat.search_batch_filtered(queries, mask, k=10)
        approx, _ = pq.search_batch_filtered(queries, mask, k=10)
        for row in approx:
            for i in row:
                if i >= 0:
                    assert mask[i]
        hits = sum(len(set(int(x) for x in a if x >= 0) & set(int(x) for x in g))
                   for a, g in zip(approx, gt))
        recall = hits / (len(queries) * 10)
        assert recall > 0.2, f"filtered PQ recall@10 was {recall:.3f}"

    def test_wrong_mask_length_raises(self, data, mask):
        pq = PqIndex(dim=48, metric=Metric.L2, m=16)
        pq.train(data)
        pq.add(data)
        with pytest.raises(Exception):
            pq.search_filtered(data[0], mask[:-1], k=5)

    def test_untrained_raises(self, data, mask):
        pq = PqIndex(dim=48, metric=Metric.L2, m=16)
        with pytest.raises(Exception):
            pq.search_filtered(data[0], mask, k=5)
