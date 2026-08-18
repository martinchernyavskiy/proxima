from __future__ import annotations

import os

import faiss
import numpy as np

_METRIC = {"L2": faiss.METRIC_L2, "InnerProduct": faiss.METRIC_INNER_PRODUCT}


def _index_bytes(index) -> int:
    return int(faiss.serialize_index(index).nbytes)


class FaissAdapter:
    def __init__(self, index, dim: int, n: int, name: str, set_ef=None):
        self.index = index
        self._dim = dim
        self._n = n
        self.name = name
        self._mem = _index_bytes(index)
        self._set_ef = set_ef

    @property
    def ef_search(self):
        return self.index.hnsw.efSearch

    @ef_search.setter
    def ef_search(self, ef: int):
        if self._set_ef is not None:
            self._set_ef(ef)

    def search(self, q: np.ndarray, k: int = 10):
        qq = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
        d, i = self.index.search(qq, k)
        return i[0], d[0]

    def search_batch(self, queries: np.ndarray, k: int = 10, num_threads: int = 0):
        faiss.omp_set_num_threads(num_threads if num_threads > 0 else (os.cpu_count() or 1))
        qq = np.ascontiguousarray(queries, dtype=np.float32)
        d, i = self.index.search(qq, k)
        return i, d

    @property
    def size(self):
        return self._n
    @property
    def dim(self):
        return self._dim
    @property
    def memory_bytes(self):
        return self._mem


def build_flat(base: np.ndarray, metric: str) -> FaissAdapter:
    d = base.shape[1]
    index = faiss.IndexFlatL2(d) if metric == "L2" else faiss.IndexFlatIP(d)
    index.add(base)
    return FaissAdapter(index, d, base.shape[0], "FAISS Flat")


def build_hnsw(base: np.ndarray, metric: str, m: int = 16,
               ef_construction: int = 200, ef_search: int = 64) -> FaissAdapter:
    d = base.shape[1]
    index = faiss.IndexHNSWFlat(d, m, _METRIC[metric])
    index.hnsw.efConstruction = ef_construction
    index.add(base)
    index.hnsw.efSearch = ef_search
    return FaissAdapter(index, d, base.shape[0], "FAISS HNSW",
                        set_ef=lambda ef: setattr(index.hnsw, "efSearch", ef))


def build_pq(base: np.ndarray, metric: str, m: int = 8, nbits: int = 8,
             train: np.ndarray | None = None) -> FaissAdapter:
    d = base.shape[1]
    index = faiss.IndexPQ(d, m, nbits, _METRIC[metric])
    index.train(np.ascontiguousarray(base if train is None else train, dtype=np.float32))
    index.add(base)
    return FaissAdapter(index, d, base.shape[0], "FAISS PQ")
