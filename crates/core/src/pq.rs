use std::collections::BinaryHeap;
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::distance::{inner_product, l2_sqr};
use crate::metric::Metric;

#[derive(Clone, Copy, Debug)]
pub struct PqParams {
    pub m: usize,
    pub nbits: usize,
    pub train_iters: usize,
    pub train_sample: usize,
    pub seed: u64,
}

impl Default for PqParams {
    fn default() -> Self {
        PqParams { m: 8, nbits: 8, train_iters: 25, train_sample: 50_000, seed: 0x5EED }
    }
}

struct SplitMix(u64);
impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[derive(Serialize, Deserialize)]
pub struct ProductQuantizer {
    dim: usize,
    m: usize,
    dsub: usize,
    k: usize,
    codebooks: Vec<f32>,
}

impl ProductQuantizer {
    fn centroid(&self, sub: usize, c: usize) -> &[f32] {
        let base = (sub * self.k + c) * self.dsub;
        &self.codebooks[base..base + self.dsub]
    }

    pub fn train(training: &[f32], dim: usize, params: PqParams) -> Self {
        assert!(params.m >= 1, "m must be >= 1");
        assert!(dim.is_multiple_of(params.m), "dim must be divisible by m");
        assert!(
            (1..=8).contains(&params.nbits),
            "nbits must be in 1..=8 (got {}): codes are stored as u8, so larger \
             centroid indices would silently truncate",
            params.nbits
        );
        let dsub = dim / params.m;
        let k = 1usize << params.nbits;
        assert!(training.len().is_multiple_of(dim), "training length not a multiple of dim");
        let n_all = training.len() / dim;
        assert!(n_all >= k, "need at least k={k} training vectors, got {n_all}");
        assert!(
            training.iter().all(|x| x.is_finite()),
            "training data must not contain NaN or infinite values"
        );

        let mut rng = SplitMix(params.seed);
        let n = if params.train_sample > 0 {
            params.train_sample.clamp(k, n_all)
        } else {
            n_all
        };
        let sample: Vec<usize> = if n == n_all {
            (0..n_all).collect()
        } else {
            (0..n).map(|_| rng.below(n_all)).collect()
        };

        let m = params.m;
        let books: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|sub| {
                let mut sub_data = vec![0.0f32; n * dsub];
                for (i, &row) in sample.iter().enumerate() {
                    let src = &training[row * dim + sub * dsub..row * dim + sub * dsub + dsub];
                    sub_data[i * dsub..(i + 1) * dsub].copy_from_slice(src);
                }
                kmeans(&sub_data, n, dsub, k, params.train_iters,
                       params.seed ^ (sub as u64).wrapping_mul(0x9E3779B97F4A7C15))
            })
            .collect();

        let mut codebooks = vec![0.0f32; m * k * dsub];
        for (sub, book) in books.into_iter().enumerate() {
            let base = sub * k * dsub;
            codebooks[base..base + k * dsub].copy_from_slice(&book);
        }
        ProductQuantizer { dim, m, dsub, k, codebooks }
    }

    pub fn encode_into(&self, v: &[f32], out: &mut [u8]) {
        for sub in 0..self.m {
            let qs = &v[sub * self.dsub..(sub + 1) * self.dsub];
            let mut best = 0u32;
            let mut best_d = f32::MAX;
            for c in 0..self.k {
                let d = l2_sqr(qs, self.centroid(sub, c));
                if d < best_d {
                    best_d = d;
                    best = c as u32;
                }
            }
            out[sub] = best as u8;
        }
    }

    fn adc_table(&self, query: &[f32], metric: Metric) -> Vec<f32> {
        let mut table = vec![0.0f32; self.m * self.k];
        for sub in 0..self.m {
            let qs = &query[sub * self.dsub..(sub + 1) * self.dsub];
            for c in 0..self.k {
                let cen = self.centroid(sub, c);
                table[sub * self.k + c] = match metric {
                    Metric::L2 => l2_sqr(qs, cen),
                    Metric::InnerProduct => -inner_product(qs, cen),
                };
            }
        }
        table
    }
}

fn kmeans(data: &[f32], n: usize, d: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix(seed);
    let mut centroids = vec![0.0f32; k * d];
    let mut order: Vec<usize> = (0..n).collect();
    for c in 0..k {
        let j = c + rng.below(n - c);
        order.swap(c, j);
        let idx = order[c];
        centroids[c * d..(c + 1) * d].copy_from_slice(&data[idx * d..(idx + 1) * d]);
    }

    let mut assign = vec![0u32; n];
    let mut sums = vec![0.0f32; k * d];
    let mut counts = vec![0u32; k];

    for _ in 0..iters {
        assign.par_iter_mut().enumerate().for_each(|(i, a)| {
            let v = &data[i * d..(i + 1) * d];
            let mut best = 0u32;
            let mut best_d = f32::MAX;
            for c in 0..k {
                let dist = l2_sqr(v, &centroids[c * d..(c + 1) * d]);
                if dist < best_d {
                    best_d = dist;
                    best = c as u32;
                }
            }
            *a = best;
        });

        sums.iter_mut().for_each(|x| *x = 0.0);
        counts.iter_mut().for_each(|x| *x = 0);
        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            let v = &data[i * d..(i + 1) * d];
            let s = &mut sums[c * d..(c + 1) * d];
            for j in 0..d {
                s[j] += v[j];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                let idx = rng.below(n);
                centroids[c * d..(c + 1) * d].copy_from_slice(&data[idx * d..(idx + 1) * d]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..d {
                    centroids[c * d + j] = sums[c * d + j] * inv;
                }
            }
        }
    }
    centroids
}

#[derive(Clone, Copy)]
struct Cand {
    key: f32,
    id: i64,
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.key.total_cmp(&o.key)
    }
}

#[derive(Serialize, Deserialize)]
pub struct PqIndex {
    metric: Metric,
    pq: ProductQuantizer,
    codes: Vec<u8>,
    n: usize,
}

impl PqIndex {
    pub fn train(training: &[f32], dim: usize, metric: Metric, params: PqParams) -> Self {
        let pq = ProductQuantizer::train(training, dim, params);
        PqIndex { metric, pq, codes: Vec::new(), n: 0 }
    }

    pub fn dim(&self) -> usize {
        self.pq.dim
    }
    pub fn m(&self) -> usize {
        self.pq.m
    }
    pub fn nbits(&self) -> usize {
        self.pq.k.trailing_zeros() as usize
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

    pub fn memory_bytes(&self) -> usize {
        self.codes.len() + self.pq.codebooks.len() * std::mem::size_of::<f32>()
    }

    pub fn raw_bytes(&self) -> usize {
        self.n * self.pq.dim * std::mem::size_of::<f32>()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        crate::save_to(self, path)
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        crate::load_from(path)
    }

    pub fn add(&mut self, vectors: &[f32]) {
        let dim = self.pq.dim;
        assert!(vectors.len().is_multiple_of(dim), "vectors length not a multiple of dim");
        let count = vectors.len() / dim;
        let m = self.pq.m;
        let mut new_codes = vec![0u8; count * m];
        new_codes
            .par_chunks_mut(m)
            .enumerate()
            .for_each(|(i, code)| {
                self.pq.encode_into(&vectors[i * dim..(i + 1) * dim], code);
            });
        self.codes.extend_from_slice(&new_codes);
        self.n += count;
    }

    pub fn search(&self, query: &[f32], k: usize, out_ids: &mut [i64], out_dists: &mut [f32]) {
        assert_eq!(query.len(), self.pq.dim, "query dim mismatch");
        self.scan(query, k, None, out_ids, out_dists);
    }

    pub fn search_filtered(&self, query: &[f32], k: usize, filter: &[bool], out_ids: &mut [i64],
                           out_dists: &mut [f32]) {
        assert_eq!(query.len(), self.pq.dim, "query dim mismatch");
        assert_eq!(filter.len(), self.n, "filter length {} must equal index size {}",
                   filter.len(), self.n);
        self.scan(query, k, Some(filter), out_ids, out_dists);
    }

    fn scan(&self, query: &[f32], k: usize, filter: Option<&[bool]>, out_ids: &mut [i64],
           out_dists: &mut [f32]) {
        if k == 0 {
            return;
        }
        assert_eq!(out_ids.len(), k, "out_ids length {} must equal k ({k})", out_ids.len());
        assert_eq!(out_dists.len(), k, "out_dists length {} must equal k ({k})", out_dists.len());
        assert!(
            query.iter().all(|x| x.is_finite()),
            "query must not contain NaN or infinite values"
        );
        let stride = self.pq.k;
        let table = self.pq.adc_table(query, self.metric);

        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
        for (i, code) in self.codes.chunks_exact(self.pq.m).enumerate() {
            if let Some(f) = filter {
                if !f[i] {
                    continue;
                }
            }
            let mut key = 0.0f32;
            for sub in 0..self.pq.m {
                key += table[sub * stride + code[sub] as usize];
            }
            if heap.len() < k {
                heap.push(Cand { key, id: i as i64 });
            } else if key.total_cmp(&heap.peek().unwrap().key).is_lt() {
                heap.pop();
                heap.push(Cand { key, id: i as i64 });
            }
        }

        let sorted = heap.into_sorted_vec();
        for (j, c) in sorted.iter().enumerate() {
            out_ids[j] = c.id;
            out_dists[j] = match self.metric {
                Metric::L2 => c.key.sqrt(),
                Metric::InnerProduct => -c.key,
            };
        }
        for j in sorted.len()..k {
            out_ids[j] = -1;
            out_dists[j] = match self.metric {
                Metric::L2 => f32::MAX,
                Metric::InnerProduct => f32::MIN,
            };
        }
    }

    pub fn search_batch(&self, queries: &[f32], k: usize, out_ids: &mut [i64],
                        out_dists: &mut [f32], num_threads: usize) {
        self.search_batch_impl(queries, k, None, out_ids, out_dists, num_threads);
    }

    pub fn search_batch_filtered(&self, queries: &[f32], k: usize, filter: &[bool],
                                 out_ids: &mut [i64], out_dists: &mut [f32], num_threads: usize) {
        assert_eq!(filter.len(), self.n, "filter length {} must equal index size {}",
                   filter.len(), self.n);
        self.search_batch_impl(queries, k, Some(filter), out_ids, out_dists, num_threads);
    }

    fn search_batch_impl(&self, queries: &[f32], k: usize, filter: Option<&[bool]>,
                        out_ids: &mut [i64], out_dists: &mut [f32], num_threads: usize) {
        if k == 0 {
            return;
        }
        let dim = self.pq.dim;
        assert!(
            queries.len().is_multiple_of(dim),
            "queries length {} is not a multiple of dim {}",
            queries.len(),
            dim
        );
        let nq = queries.len() / dim;
        assert_eq!(out_ids.len(), nq * k,
                   "out_ids length {} must equal nq * k ({nq} * {k})", out_ids.len());
        assert_eq!(out_dists.len(), nq * k,
                   "out_dists length {} must equal nq * k ({nq} * {k})", out_dists.len());
        let mut run = || {
            out_ids
                .par_chunks_mut(k)
                .zip(out_dists.par_chunks_mut(k))
                .enumerate()
                .for_each(|(qi, (ids, dists))| {
                    self.scan(&queries[qi * dim..(qi + 1) * dim], k, filter, ids, dists);
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
    use crate::flat::FlatIndex;

    fn gen(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = SplitMix(seed);
        (0..n * dim).map(|_| (s.next_u64() as f32 / u64::MAX as f32) * 2.0 - 1.0).collect()
    }

    fn recall(m: usize, k: usize) -> f64 {
        let (n, dim) = (3000usize, 32usize);
        let data = gen(n, dim, 1);
        let params = PqParams { m, train_iters: 20, train_sample: 0, ..Default::default() };
        let mut pq = PqIndex::train(&data, dim, Metric::L2, params);
        pq.add(&data);

        let mut flat = FlatIndex::new(dim, Metric::L2);
        flat.add(&data);

        let queries = gen(150, dim, 77);
        let nq = 150;
        let mut hit = 0usize;
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let (mut pids, mut pd) = (vec![0i64; k], vec![0f32; k]);
            pq.search(q, k, &mut pids, &mut pd);
            let (mut fids, mut fd) = (vec![0i64; k], vec![0f32; k]);
            flat.search(q, k, &mut fids, &mut fd);
            let truth: std::collections::HashSet<i64> = fids.into_iter().collect();
            for id in pids {
                if id >= 0 && truth.contains(&id) {
                    hit += 1;
                }
            }
        }
        hit as f64 / (nq * k) as f64
    }

    #[test]
    fn pq_beats_random_and_improves_with_m() {
        let r4 = recall(4, 10);
        let r8 = recall(8, 10);
        assert!(r8 > 0.2, "recall@10 with m=8 was {r8:.3}");
        assert!(r8 > r4, "more subspaces should improve recall: m4={r4:.3} m8={r8:.3}");
    }

    #[test]
    fn pq_recall_at_100_is_high() {
        let r = recall(8, 100);
        assert!(r > 0.6, "recall@100 with m=8 was {r:.3}");
    }

    #[test]
    fn compression_ratio() {
        let (n, dim) = (2000usize, 64usize);
        let data = gen(n, dim, 5);
        let mut pq = PqIndex::train(&data, dim, Metric::L2,
                                    PqParams { m: 8, ..Default::default() });
        pq.add(&data);
        assert_eq!(pq.codes.len(), n * 8);
        assert!(pq.raw_bytes() as f64 / (n * 8) as f64 > 30.0);
    }

    #[test]
    fn filtered_search_only_returns_matching_ids_and_recall_holds() {
        let (n, dim, k) = (3000usize, 32usize, 10usize);
        let data = gen(n, dim, 1);
        let params = PqParams { m: 8, train_iters: 20, train_sample: 0, ..Default::default() };
        let mut pq = PqIndex::train(&data, dim, Metric::L2, params);
        pq.add(&data);
        let mut flat = FlatIndex::new(dim, Metric::L2);
        flat.add(&data);

        let filter: Vec<bool> = (0..n).map(|i| i % 4 == 0).collect();
        let queries = gen(100, dim, 77);
        let mut hit = 0usize;
        for qi in 0..100 {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let (mut pids, mut pd) = (vec![0i64; k], vec![0f32; k]);
            pq.search_filtered(q, k, &filter, &mut pids, &mut pd);
            for &id in &pids {
                assert!(id < 0 || filter[id as usize], "result {id} must satisfy the filter");
            }
            let (mut fids, mut fd) = (vec![0i64; k], vec![0f32; k]);
            flat.search_filtered(q, k, &filter, &mut fids, &mut fd);
            let truth: std::collections::HashSet<i64> = fids.into_iter().collect();
            for id in pids {
                if id >= 0 && truth.contains(&id) {
                    hit += 1;
                }
            }
        }
        let r = hit as f64 / (100 * k) as f64;
        assert!(r > 0.2, "filtered recall@10 was {r:.3}");
    }

    #[test]
    #[should_panic(expected = "filter length")]
    fn filtered_search_rejects_wrong_length_filter() {
        let (n, dim) = (500usize, 16usize);
        let data = gen(n, dim, 9);
        let mut pq = PqIndex::train(&data, dim, Metric::L2, PqParams { m: 4, ..Default::default() });
        pq.add(&data);
        let bad_filter = vec![true; n - 1];
        let (mut ids, mut d) = (vec![0i64; 5], vec![0f32; 5]);
        pq.search_filtered(&gen(1, dim, 2), 5, &bad_filter, &mut ids, &mut d);
    }

    #[test]
    #[should_panic(expected = "must not contain NaN or infinite values")]
    fn train_rejects_non_finite_training_data() {
        let (n, dim) = (300usize, 16usize);
        let mut data = gen(n, dim, 1);
        data[5] = f32::NAN;
        PqIndex::train(&data, dim, Metric::L2, PqParams { m: 4, ..Default::default() });
    }

    #[test]
    #[should_panic(expected = "must not contain NaN or infinite values")]
    fn search_rejects_non_finite_query() {
        let (n, dim) = (300usize, 16usize);
        let data = gen(n, dim, 1);
        let mut pq = PqIndex::train(&data, dim, Metric::L2, PqParams { m: 4, ..Default::default() });
        pq.add(&data);
        let mut query = gen(1, dim, 2);
        query[0] = f32::INFINITY;
        let (mut ids, mut d) = (vec![0i64; 5], vec![0f32; 5]);
        pq.search(&query, 5, &mut ids, &mut d);
    }

    #[test]
    #[should_panic(expected = "must equal k")]
    fn search_rejects_mismatched_output_length() {
        let (n, dim) = (300usize, 16usize);
        let data = gen(n, dim, 1);
        let mut pq = PqIndex::train(&data, dim, Metric::L2, PqParams { m: 4, ..Default::default() });
        pq.add(&data);
        let (mut ids, mut d) = (vec![0i64; 3], vec![0f32; 5]);
        pq.search(&gen(1, dim, 2), 5, &mut ids, &mut d);
    }
}
