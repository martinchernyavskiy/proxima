//! Exact ("flat") brute-force nearest-neighbor index.
//!
//! This is Proxima's ground truth: it scans every stored vector for each
//! query, so its results are exact by construction. The approximate HNSW index
//! is measured for recall against it, and the future GPU path accelerates
//! exactly this computation. Vectors are stored contiguously row-major for
//! cache-friendly streaming, and batch search is parallelized across queries
//! with rayon.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::distance::{inner_product, l2_sqr};
use crate::metric::Metric;

/// One entry in the bounded top-k heap. `key` is normalized so that *smaller is
/// more relevant* for both metrics (L2 → squared distance, InnerProduct →
/// negated dot product), which lets a single max-heap serve both: its root is
/// always the worst of the current best-k and is the entry we evict.
#[derive(Clone, Copy)]
struct Cand {
    key: f32,
    id: i64,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    // total_cmp gives a deterministic total order even with NaN, so the heap is
    // always well-formed.
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.total_cmp(&other.key)
    }
}

/// Exact brute-force vector index.
#[derive(Serialize, Deserialize)]
pub struct FlatIndex {
    dim: usize,
    metric: Metric,
    n: usize,
    data: Vec<f32>, // n * dim, row-major
}

impl FlatIndex {
    /// Create an empty index for `dim`-dimensional vectors.
    pub fn new(dim: usize, metric: Metric) -> Self {
        assert!(dim > 0, "dim must be > 0");
        FlatIndex { dim, metric, n: 0, data: Vec::new() }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    /// Bytes held by the raw vector store (excludes container overhead).
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Persist the index to `path` (bincode).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        crate::save_to(self, path)
    }

    /// Load an index previously written by [`FlatIndex::save`].
    pub fn load(path: &Path) -> std::io::Result<Self> {
        crate::load_from(path)
    }

    /// Append row-major vectors. `vectors.len()` must be a multiple of `dim`.
    /// IDs are assigned sequentially in insertion order.
    pub fn add(&mut self, vectors: &[f32]) {
        assert!(
            vectors.len().is_multiple_of(self.dim),
            "vectors length {} is not a multiple of dim {}",
            vectors.len(),
            self.dim
        );
        self.data.extend_from_slice(vectors);
        self.n += vectors.len() / self.dim;
    }

    /// Top-`k` for a single query. `out_ids` and `out_dists` must each hold `k`
    /// slots; results are written best-first. If the index holds fewer than `k`
    /// vectors, trailing slots are padded with id `-1`.
    pub fn search(&self, query: &[f32], k: usize, out_ids: &mut [i64], out_dists: &mut [f32]) {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        match self.metric {
            Metric::L2 => self.scan::<true>(query, k, None, out_ids, out_dists),
            Metric::InnerProduct => self.scan::<false>(query, k, None, out_ids, out_dists),
        }
    }

    /// Top-`k` restricted to vectors where `filter[id]` is `true`. `filter` must
    /// have one entry per stored vector (`filter.len() == self.len()`). Exact:
    /// every vector is still scanned, so this is a real filtered nearest-
    /// neighbor search, not a post-hoc reranking of an unfiltered top-k.
    pub fn search_filtered(&self, query: &[f32], k: usize, filter: &[bool], out_ids: &mut [i64],
                           out_dists: &mut [f32]) {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        assert_eq!(filter.len(), self.n, "filter length {} must equal index size {}",
                   filter.len(), self.n);
        match self.metric {
            Metric::L2 => self.scan::<true>(query, k, Some(filter), out_ids, out_dists),
            Metric::InnerProduct => self.scan::<false>(query, k, Some(filter), out_ids, out_dists),
        }
    }

    // Monomorphized per metric (the const bool is resolved at compile time, so
    // the inner branch and distance call inline with no runtime dispatch).
    fn scan<const L2: bool>(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&[bool]>,
        out_ids: &mut [i64],
        out_dists: &mut [f32],
    ) {
        if k == 0 {
            return;
        }
        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
        for (i, v) in self.data.chunks_exact(self.dim).enumerate() {
            if let Some(f) = filter {
                if !f[i] {
                    continue;
                }
            }
            let key = if L2 { l2_sqr(query, v) } else { -inner_product(query, v) };
            if heap.len() < k {
                heap.push(Cand { key, id: i as i64 });
            } else if key < heap.peek().unwrap().key {
                // Better than the current worst-of-best: evict and insert.
                heap.pop();
                heap.push(Cand { key, id: i as i64 });
            }
        }

        // into_sorted_vec yields ascending order by `key`, i.e. best-first.
        let sorted = heap.into_sorted_vec();
        for (j, c) in sorted.iter().enumerate() {
            out_ids[j] = c.id;
            out_dists[j] = if L2 { c.key.sqrt() } else { -c.key };
        }
        // Pad with finite sentinels (id = -1) when fewer than k results exist.
        for j in sorted.len()..k {
            out_ids[j] = -1;
            out_dists[j] = if L2 { f32::MAX } else { f32::MIN };
        }
    }

    /// Top-`k` for a batch of `nq` queries (row-major, `nq * dim`). Outputs are
    /// `nq * k`, row-major. Queries are independent and uniform-cost, so this
    /// scales near-linearly across cores. `num_threads == 0` uses rayon's global
    /// pool; a positive value runs on a local pool of that size (useful for
    /// measuring single- vs multi-threaded speedup).
    pub fn search_batch(
        &self,
        queries: &[f32],
        k: usize,
        out_ids: &mut [i64],
        out_dists: &mut [f32],
        num_threads: usize,
    ) {
        if k == 0 {
            return;
        }
        let dim = self.dim;
        let mut run = || {
            out_ids
                .par_chunks_mut(k)
                .zip(out_dists.par_chunks_mut(k))
                .enumerate()
                .for_each(|(qi, (ids, dists))| {
                    self.search(&queries[qi * dim..(qi + 1) * dim], k, ids, dists);
                });
        };
        if num_threads == 0 {
            run();
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to build rayon pool");
            pool.install(run);
        }
    }

    /// Batch version of [`FlatIndex::search_filtered`]: the same `filter` is
    /// applied to every query in the batch (the common case — e.g. "search
    /// within category X" for many queries at once).
    pub fn search_batch_filtered(&self, queries: &[f32], k: usize, filter: &[bool],
                                 out_ids: &mut [i64], out_dists: &mut [f32], num_threads: usize) {
        assert_eq!(filter.len(), self.n, "filter length {} must equal index size {}",
                   filter.len(), self.n);
        if k == 0 {
            return;
        }
        let dim = self.dim;
        let mut run = || {
            out_ids
                .par_chunks_mut(k)
                .zip(out_dists.par_chunks_mut(k))
                .enumerate()
                .for_each(|(qi, (ids, dists))| {
                    self.search_filtered(&queries[qi * dim..(qi + 1) * dim], k, filter, ids, dists);
                });
        };
        if num_threads == 0 {
            run();
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to build rayon pool");
            pool.install(run);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::l2_sqr;

    // Deterministic pseudo-random data via a splitmix64-style generator, so
    // tests are reproducible without an RNG dependency.
    fn gen(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n * dim)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                // map to roughly [-1, 1)
                (z as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn self_is_nearest_l2() {
        let (n, dim) = (200usize, 16usize);
        let data = gen(n, dim, 42);
        let mut idx = FlatIndex::new(dim, Metric::L2);
        idx.add(&data);
        assert_eq!(idx.len(), n);

        let mut ids = [0i64; 1];
        let mut dists = [0f32; 1];
        for i in 0..n {
            idx.search(&data[i * dim..(i + 1) * dim], 1, &mut ids, &mut dists);
            assert_eq!(ids[0], i as i64, "vector {i} should be its own NN");
            assert!(dists[0].abs() < 1e-3, "self distance ~0");
        }
    }

    #[test]
    fn topk_sorted_and_matches_reference_l2() {
        let (n, dim, k) = (1000usize, 24usize, 10usize);
        let data = gen(n, dim, 7);
        let q = gen(1, dim, 999);
        let mut idx = FlatIndex::new(dim, Metric::L2);
        idx.add(&data);

        let mut ids = vec![0i64; k];
        let mut dists = vec![0f32; k];
        idx.search(&q, k, &mut ids, &mut dists);

        for j in 1..k {
            assert!(dists[j] >= dists[j - 1] - 1e-4, "ascending L2");
        }
        // Independent reference for the nearest distance.
        let mut best = f32::MAX;
        for i in 0..n {
            let d = l2_sqr(&q, &data[i * dim..(i + 1) * dim]).sqrt();
            if d < best {
                best = d;
            }
        }
        assert!((dists[0] - best).abs() < 1e-3, "top-1 distance matches reference");
    }

    #[test]
    fn batch_matches_single_ip() {
        let (n, dim, k, nq) = (500usize, 12usize, 5usize, 33usize);
        let data = gen(n, dim, 3);
        let queries = gen(nq, dim, 88);
        let mut idx = FlatIndex::new(dim, Metric::InnerProduct);
        idx.add(&data);

        let mut bids = vec![0i64; nq * k];
        let mut bd = vec![0f32; nq * k];
        idx.search_batch(&queries, k, &mut bids, &mut bd, 4);

        for qi in 0..nq {
            let mut sids = vec![0i64; k];
            let mut sd = vec![0f32; k];
            idx.search(&queries[qi * dim..(qi + 1) * dim], k, &mut sids, &mut sd);
            assert_eq!(&bids[qi * k..(qi + 1) * k], &sids[..], "batch == single (ids)");
            for j in 1..k {
                assert!(sd[j] <= sd[j - 1] + 1e-4, "descending IP similarity");
            }
        }
    }

    #[test]
    fn padding_when_fewer_than_k() {
        let dim = 4;
        let mut idx = FlatIndex::new(dim, Metric::L2);
        idx.add(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // 2 vectors
        let mut ids = vec![0i64; 5];
        let mut dists = vec![0f32; 5];
        idx.search(&[0.0, 0.0, 0.0, 0.0], 5, &mut ids, &mut dists);
        assert_eq!(ids[0], 0);
        assert_eq!(&ids[2..], &[-1, -1, -1], "trailing ids padded with -1");
    }

    #[test]
    fn filtered_search_only_returns_matching_ids_and_matches_reference() {
        let (n, dim, k) = (2000usize, 24usize, 10usize);
        let data = gen(n, dim, 11);
        let q = gen(1, dim, 456);
        let mut idx = FlatIndex::new(dim, Metric::InnerProduct);
        idx.add(&data);

        // Filter: keep only even ids.
        let filter: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let mut ids = vec![0i64; k];
        let mut dists = vec![0f32; k];
        idx.search_filtered(&q, k, &filter, &mut ids, &mut dists);

        for &id in &ids {
            assert!(id >= 0 && id % 2 == 0, "result {id} must satisfy the filter");
        }
        for j in 1..k {
            assert!(dists[j] <= dists[j - 1] + 1e-4, "descending IP similarity");
        }

        // Independent reference: brute-force over only the even-id subset.
        let mut scored: Vec<(i64, f32)> = (0..n)
            .filter(|i| i % 2 == 0)
            .map(|i| (i as i64, inner_product(&q, &data[i * dim..(i + 1) * dim])))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let expect: std::collections::HashSet<i64> = scored[..k].iter().map(|(id, _)| *id).collect();
        let got: std::collections::HashSet<i64> = ids.into_iter().collect();
        assert_eq!(got, expect, "filtered top-k must match a brute-force scan of only the filtered subset");
    }

    #[test]
    fn filtered_search_pads_when_fewer_matches_than_k() {
        let dim = 4;
        let mut idx = FlatIndex::new(dim, Metric::L2);
        idx.add(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]); // 3 vectors
        let filter = [true, false, false]; // only id 0 passes
        let mut ids = vec![0i64; 5];
        let mut dists = vec![0f32; 5];
        idx.search_filtered(&[0.0, 0.0, 0.0, 0.0], 5, &filter, &mut ids, &mut dists);
        assert_eq!(ids[0], 0);
        assert_eq!(&ids[1..], &[-1, -1, -1, -1], "only one match exists; rest padded");
    }

    #[test]
    #[should_panic(expected = "filter length")]
    fn filtered_search_rejects_wrong_length_filter() {
        let dim = 4;
        let mut idx = FlatIndex::new(dim, Metric::L2);
        idx.add(&[0.0, 0.0, 0.0, 0.0]);
        let bad_filter = [true, true]; // wrong length (index has 1 vector)
        let mut ids = vec![0i64; 1];
        let mut dists = vec![0f32; 1];
        idx.search_filtered(&[0.0, 0.0, 0.0, 0.0], 1, &bad_filter, &mut ids, &mut dists);
    }
}
