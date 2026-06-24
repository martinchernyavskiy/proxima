//! Hierarchical Navigable Small World (HNSW) approximate nearest-neighbor index.
//!
//! HNSW (Malkov & Yashunin, 2016) is a multi-layer proximity graph. The bottom
//! layer connects every vector; each higher layer is an exponentially sparser
//! "express lane". A search greedily descends the sparse upper layers to land
//! near the query, then runs a best-first beam search (width `ef`) on the dense
//! bottom layer. This turns the O(N) brute-force scan into roughly O(log N)
//! distance evaluations, at the cost of *approximate* recall — the tradeoff this
//! whole project is built to measure.
//!
//! Lineage note: HNSW's layered, skip-on-the-express-lane structure is the
//! direct descendant of the skip list — a probabilistic tower of linked lists.
//!
//! Tunables:
//!   * `m`               — neighbors per node per layer (degree). Bottom layer
//!                         gets `2*m`. Higher `m` → better recall, more memory.
//!   * `ef_construction` — beam width while building. Higher → better graph.
//!   * `ef_search`       — beam width while querying. The recall/latency dial.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::distance::{inner_product, l2_sqr};
use crate::metric::Metric;

/// A trivial hasher for the per-search `visited` set. Keys are node ids (`u32`),
/// so the default SipHash is pure overhead — it dominates construction time. A
/// single multiply by a 64-bit odd constant (Fibonacci hashing) spreads the bits
/// enough for the small, short-lived sets used here.
#[derive(Default)]
struct U32Hasher(u64);
impl Hasher for U32Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write_u32(&mut self, x: u32) {
        self.0 = (x as u64).wrapping_mul(0x9E3779B97F4A7C15);
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("visited set only holds u32 ids")
    }
}
type VisitedSet = HashSet<u32, BuildHasherDefault<U32Hasher>>;

/// Construction / search parameters.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams { m: 16, ef_construction: 200, ef_search: 64, seed: 0x5EED }
    }
}

/// A (distance, id) pair. `dist` is in the engine's "smaller is closer" key
/// space (L2 → squared distance, InnerProduct → negated dot), so the same
/// comparisons drive both metrics. `Ord` is by distance via `total_cmp`.
#[derive(Clone, Copy, PartialEq)]
struct Neighbor {
    dist: f32,
    id: u32,
}
impl Eq for Neighbor {}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

/// HNSW graph index.
#[derive(Serialize, Deserialize)]
pub struct Hnsw {
    dim: usize,
    metric: Metric,
    params: HnswParams,
    m_max0: usize, // max degree on layer 0
    m_max: usize,  // max degree on layers > 0
    ml: f64,       // level-generation normalization, 1 / ln(m)

    data: Vec<f32>,             // n * dim, row-major (same layout as FlatIndex)
    links: Vec<Vec<Vec<u32>>>,  // links[node][layer] -> neighbor ids
    levels: Vec<usize>,         // top layer of each node
    entry_point: Option<u32>,
    max_level: usize,
    rng_state: u64,
    n: usize,
}

impl Hnsw {
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert!(params.m >= 2, "m must be >= 2");
        Hnsw {
            dim,
            metric,
            params,
            m_max0: params.m * 2,
            m_max: params.m,
            ml: 1.0 / (params.m as f64).ln(),
            data: Vec::new(),
            links: Vec::new(),
            levels: Vec::new(),
            entry_point: None,
            max_level: 0,
            rng_state: params.seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1),
            n: 0,
        }
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
    pub fn params(&self) -> HnswParams {
        self.params
    }
    pub fn set_ef_search(&mut self, ef: usize) {
        self.params.ef_search = ef;
    }

    /// Persist the full graph + vectors to `path` (bincode). Loading is then
    /// near-instant versus rebuilding the graph from scratch.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        crate::save_to(self, path)
    }

    /// Load an index previously written by [`Hnsw::save`].
    pub fn load(path: &Path) -> std::io::Result<Self> {
        crate::load_from(path)
    }

    /// Resident memory of the index: vectors + adjacency lists.
    pub fn memory_bytes(&self) -> usize {
        let vecs = self.data.len() * std::mem::size_of::<f32>();
        let mut edges = 0usize;
        for node in &self.links {
            for layer in node {
                edges += layer.capacity() * std::mem::size_of::<u32>();
            }
        }
        vecs + edges
    }

    fn vec_at(&self, id: u32) -> &[f32] {
        let i = id as usize * self.dim;
        &self.data[i..i + self.dim]
    }

    // Distance in "smaller is closer" key space.
    #[inline]
    fn key(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            Metric::L2 => l2_sqr(a, b),
            Metric::InnerProduct => -inner_product(a, b),
        }
    }

    // Random level ~ floor(-ln(U) * mL), the geometric distribution that makes
    // upper layers exponentially sparse.
    fn random_level(&mut self) -> usize {
        // splitmix64 step for a deterministic uniform in (0, 1].
        self.rng_state = self.rng_state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        let u = ((z >> 11) as f64 / (1u64 << 53) as f64).max(f64::MIN_POSITIVE);
        (-u.ln() * self.ml) as usize
    }

    /// Append row-major vectors, inserting each into the graph.
    pub fn add(&mut self, vectors: &[f32]) {
        assert!(vectors.len() % self.dim == 0, "vectors length not a multiple of dim");
        let count = vectors.len() / self.dim;
        self.data.reserve(vectors.len());
        for c in 0..count {
            let v = &vectors[c * self.dim..(c + 1) * self.dim];
            self.insert(v);
        }
    }

    fn insert(&mut self, v: &[f32]) {
        let id = self.n as u32;
        self.data.extend_from_slice(v);
        self.n += 1;
        let level = self.random_level();
        self.levels.push(level);
        self.links.push((0..=level).map(|_| Vec::new()).collect());

        // First node becomes the entry point.
        let ep = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                self.max_level = level;
                return;
            }
            Some(ep) => ep,
        };

        let mut cur = ep;
        // Phase 1: greedily descend the layers above this node's top level.
        if self.max_level > level {
            for lc in (level + 1..=self.max_level).rev() {
                cur = self.greedy_descend(v, cur, lc);
            }
        }

        // Phase 2: from this node's top level down to 0, find ef_construction
        // candidates, select neighbors via the heuristic, and link bidirectionally.
        let mut entry_points = vec![cur];
        let start = level.min(self.max_level);
        for lc in (0..=start).rev() {
            let candidates =
                self.search_layer(v, &entry_points, self.params.ef_construction, lc);
            let m = if lc == 0 { self.m_max0 } else { self.m_max };
            let selected = self.select_neighbors(&candidates, m);

            // Link id <-> each selected neighbor on this layer.
            for &nb in &selected {
                self.links[id as usize][lc].push(nb);
                self.links[nb as usize][lc].push(id);
                // Keep the neighbor's degree bounded.
                let cap = if lc == 0 { self.m_max0 } else { self.m_max };
                if self.links[nb as usize][lc].len() > cap {
                    self.prune(nb, lc, cap);
                }
            }

            // The full candidate set seeds the next (lower) layer's search.
            entry_points = candidates.iter().map(|c| c.id).collect();
            if entry_points.is_empty() {
                entry_points = vec![cur];
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(id);
        }
    }

    // Greedy single-best descent on an upper layer (equivalent to search_layer
    // with ef = 1, but cheaper).
    fn greedy_descend(&self, q: &[f32], entry: u32, layer: usize) -> u32 {
        let mut best = entry;
        let mut best_d = self.key(q, self.vec_at(entry));
        loop {
            let mut improved = false;
            for &nb in &self.links[best as usize][layer] {
                let d = self.key(q, self.vec_at(nb));
                if d < best_d {
                    best_d = d;
                    best = nb;
                    improved = true;
                }
            }
            if !improved {
                return best;
            }
        }
    }

    // Best-first beam search on a single layer. Returns up to `ef` nearest nodes
    // (unsorted Vec; the caller sorts/selects). This is the core graph traversal.
    fn search_layer(&self, q: &[f32], entry: &[u32], ef: usize, layer: usize) -> Vec<Neighbor> {
        let mut visited: VisitedSet =
            HashSet::with_capacity_and_hasher(ef * 4, BuildHasherDefault::default());
        // Candidates: min-heap (closest expanded first) via Reverse.
        let mut candidates: BinaryHeap<std::cmp::Reverse<Neighbor>> = BinaryHeap::new();
        // Results W: max-heap (farthest on top) so we can drop it when full.
        let mut w: BinaryHeap<Neighbor> = BinaryHeap::new();

        for &e in entry {
            if visited.insert(e) {
                let d = self.key(q, self.vec_at(e));
                candidates.push(std::cmp::Reverse(Neighbor { dist: d, id: e }));
                w.push(Neighbor { dist: d, id: e });
            }
        }
        while w.len() > ef {
            w.pop();
        }

        while let Some(std::cmp::Reverse(c)) = candidates.pop() {
            // If the closest remaining candidate is farther than the worst result,
            // no unvisited node can improve W — stop.
            let worst = w.peek().map(|n| n.dist).unwrap_or(f32::MAX);
            if c.dist > worst && w.len() >= ef {
                break;
            }
            for &nb in &self.links[c.id as usize][layer] {
                if visited.insert(nb) {
                    let d = self.key(q, self.vec_at(nb));
                    let worst = w.peek().map(|n| n.dist).unwrap_or(f32::MAX);
                    if d < worst || w.len() < ef {
                        candidates.push(std::cmp::Reverse(Neighbor { dist: d, id: nb }));
                        w.push(Neighbor { dist: d, id: nb });
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }
            }
        }

        w.into_vec()
    }

    // HNSW neighbor-selection heuristic (paper Algorithm 4, simple form): prefer
    // a *diverse* neighbor set over the strict m-nearest. Keep candidate `c`
    // only if it is closer to the base point than to every already-selected
    // neighbor — this avoids clustering all edges in one direction and keeps the
    // graph navigable.
    fn select_neighbors(&self, candidates: &[Neighbor], m: usize) -> Vec<u32> {
        let mut sorted = candidates.to_vec();
        sorted.sort_unstable(); // ascending by distance to base
        let mut result: Vec<Neighbor> = Vec::with_capacity(m);
        for c in sorted {
            if result.len() >= m {
                break;
            }
            let c_vec = self.vec_at(c.id);
            let diverse = result
                .iter()
                .all(|r| self.key(c_vec, self.vec_at(r.id)) >= c.dist);
            if diverse {
                result.push(c);
            }
        }
        result.into_iter().map(|n| n.id).collect()
    }

    // Re-select a node's neighbors on `layer` down to `cap`, using the heuristic.
    fn prune(&mut self, node: u32, layer: usize, cap: usize) {
        let nvec_start = node as usize * self.dim;
        // Take the current neighbor list out, score each by distance to `node`,
        // then keep the heuristic-selected `cap` best.
        let current = std::mem::take(&mut self.links[node as usize][layer]);
        let nvec = &self.data[nvec_start..nvec_start + self.dim];
        let cands: Vec<Neighbor> = current
            .iter()
            .map(|&nb| Neighbor { dist: self.key(nvec, self.vec_at(nb)), id: nb })
            .collect();
        let kept = self.select_neighbors(&cands, cap);
        self.links[node as usize][layer] = kept;
    }

    /// Top-`k` for a single query using beam width `ef` (clamped to at least `k`).
    /// `out_ids` / `out_dists` hold `k` slots; unfilled slots get id `-1`.
    pub fn search(&self, query: &[f32], k: usize, ef: usize, out_ids: &mut [i64],
                  out_dists: &mut [f32]) {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        if k == 0 {
            return;
        }
        let fill_empty = |out_ids: &mut [i64], out_dists: &mut [f32]| {
            for j in 0..k {
                out_ids[j] = -1;
                out_dists[j] = match self.metric {
                    Metric::L2 => f32::MAX,
                    Metric::InnerProduct => f32::MIN,
                };
            }
        };
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return fill_empty(out_ids, out_dists),
        };

        let mut cur = ep;
        for lc in (1..=self.max_level).rev() {
            cur = self.greedy_descend(query, cur, lc);
        }
        let ef = ef.max(k);
        let mut w = self.search_layer(query, &[cur], ef, 0);
        w.sort_unstable(); // ascending by key (best first)

        let take = k.min(w.len());
        for j in 0..take {
            out_ids[j] = w[j].id as i64;
            out_dists[j] = match self.metric {
                Metric::L2 => w[j].dist.sqrt(),
                Metric::InnerProduct => -w[j].dist,
            };
        }
        for j in take..k {
            out_ids[j] = -1;
            out_dists[j] = match self.metric {
                Metric::L2 => f32::MAX,
                Metric::InnerProduct => f32::MIN,
            };
        }
    }

    /// Batch search, parallelized across queries with rayon. Mirrors
    /// `FlatIndex::search_batch` so the benchmark harness is index-agnostic.
    pub fn search_batch(&self, queries: &[f32], k: usize, ef: usize, out_ids: &mut [i64],
                        out_dists: &mut [f32], num_threads: usize) {
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
                    self.search(&queries[qi * dim..(qi + 1) * dim], k, ef, ids, dists);
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
        let mut s = seed;
        (0..n * dim)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                (z as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    // Recall of HNSW top-k against the exact flat index over many queries.
    fn recall(metric: Metric, n: usize, dim: usize, k: usize, ef: usize) -> f64 {
        let data = gen(n, dim, 1);
        let mut hnsw = Hnsw::new(dim, metric, HnswParams { ef_search: ef, ..Default::default() });
        hnsw.add(&data);
        assert_eq!(hnsw.len(), n);

        let mut flat = FlatIndex::new(dim, metric);
        flat.add(&data);

        let queries = gen(200, dim, 99);
        let nq = 200;
        let mut hit = 0usize;
        let mut total = 0usize;
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let (mut hids, mut hd) = (vec![0i64; k], vec![0f32; k]);
            hnsw.search(q, k, ef, &mut hids, &mut hd);
            let (mut fids, mut fd) = (vec![0i64; k], vec![0f32; k]);
            flat.search(q, k, &mut fids, &mut fd);
            let truth: std::collections::HashSet<i64> = fids.into_iter().collect();
            for id in hids {
                if id >= 0 && truth.contains(&id) {
                    hit += 1;
                }
            }
            total += k;
        }
        hit as f64 / total as f64
    }

    #[test]
    fn high_recall_inner_product() {
        let r = recall(Metric::InnerProduct, 5000, 32, 10, 100);
        assert!(r > 0.95, "recall@10 was {r:.3}, expected > 0.95");
    }

    #[test]
    fn high_recall_l2() {
        let r = recall(Metric::L2, 5000, 32, 10, 100);
        assert!(r > 0.95, "recall@10 was {r:.3}, expected > 0.95");
    }

    #[test]
    fn higher_ef_improves_recall() {
        let low = recall(Metric::L2, 4000, 48, 10, 16);
        let high = recall(Metric::L2, 4000, 48, 10, 200);
        assert!(high >= low, "ef=200 recall {high:.3} should be >= ef=16 recall {low:.3}");
        assert!(high > 0.95);
    }

    #[test]
    fn handles_small_and_empty() {
        let mut h = Hnsw::new(4, Metric::L2, HnswParams::default());
        let (mut ids, mut d) = (vec![0i64; 5], vec![0f32; 5]);
        h.search(&[0.0, 0.0, 0.0, 0.0], 5, 16, &mut ids, &mut d);
        assert!(ids.iter().all(|&x| x == -1)); // empty index
        h.add(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        h.search(&[1.0, 0.0, 0.0, 0.0], 5, 16, &mut ids, &mut d);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[2], -1); // only 2 nodes -> padded
    }
}
