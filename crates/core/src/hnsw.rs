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
//! **Concurrency.** Construction is parallelized across vectors with rayon. The
//! adjacency lists are *moved* into a temporary `Vec<RwLock<..>>` mirror for the
//! build; independent inserts proceed concurrently and serialize only when they
//! touch the same node, and a thread never holds two node locks at once
//! (deadlock-free). After the build the inner `Vec`s are moved back into the
//! plain `links`, so the **query path is lock-free** — `search` reads the plain
//! adjacency lists with zero locking or copying. One generic traversal (the
//! [`Graph`] trait) serves both: `Plain` for queries, `Locked` for the build.
//!
//! Tunables: `m` (degree; bottom layer gets `2*m`), `ef_construction` (build beam
//! width), `ef_search` (query beam width — the recall/latency dial).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;
use std::sync::{Mutex, RwLock};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::distance::{inner_product, l2_sqr};
use crate::metric::Metric;

/// A trivial hasher for the per-search `visited` set. Keys are node ids (`u32`),
/// so the default SipHash is pure overhead. A single multiply by a 64-bit odd
/// constant (Fibonacci hashing) spreads the bits enough for these small sets.
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

/// A (distance, id) pair in "smaller is closer" key space (L2 → squared
/// distance, InnerProduct → negated dot). `Ord` is by distance via `total_cmp`.
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

// Abstracts neighbor iteration so one traversal implementation serves both the
// lock-free query path and the locked, concurrent build path.
trait Graph {
    fn for_each<F: FnMut(u32)>(&self, node: u32, layer: usize, f: F);
}

/// Plain adjacency lists — used by queries (no concurrent writers → no locks).
struct Plain<'a>(&'a [Vec<Vec<u32>>]);
impl Graph for Plain<'_> {
    #[inline]
    fn for_each<F: FnMut(u32)>(&self, node: u32, layer: usize, mut f: F) {
        if let Some(nbrs) = self.0[node as usize].get(layer) {
            for &x in nbrs {
                f(x);
            }
        }
    }
}

/// Per-node-locked adjacency lists — used only during parallel construction.
struct Locked<'a>(&'a [RwLock<Vec<Vec<u32>>>]);
impl Graph for Locked<'_> {
    #[inline]
    fn for_each<F: FnMut(u32)>(&self, node: u32, layer: usize, mut f: F) {
        let g = self.0[node as usize].read().unwrap();
        if let Some(nbrs) = g.get(layer) {
            for &x in nbrs {
                f(x);
            }
        }
    }
}

// Entry-point state shared across build threads (a tiny critical section).
struct Entry {
    point: Option<u32>,
    max_level: usize,
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

    data: Vec<f32>,            // n * dim, row-major
    links: Vec<Vec<Vec<u32>>>, // links[node][layer] -> neighbor ids (plain: lock-free query)
    levels: Vec<usize>,        // top layer of each node
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

    /// Persist the full graph + vectors to `path` (bincode).
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

    #[inline]
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

    // Random level ~ floor(-ln(U) * mL): the geometric distribution that makes
    // upper layers exponentially sparse. Single-threaded (called in `add`).
    fn random_level(&mut self) -> usize {
        self.rng_state = self.rng_state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        let u = ((z >> 11) as f64 / (1u64 << 53) as f64).max(f64::MIN_POSITIVE);
        (-u.ln() * self.ml) as usize
    }

    /// Append row-major vectors and link them into the graph in parallel.
    pub fn add(&mut self, vectors: &[f32]) {
        assert!(vectors.len().is_multiple_of(self.dim), "vectors length not a multiple of dim");
        let count = vectors.len() / self.dim;
        if count == 0 {
            return;
        }
        let start = self.n;

        // Sequential prep: data, levels (RNG is single-threaded).
        self.data.extend_from_slice(vectors);
        let mut new_levels = Vec::with_capacity(count);
        for _ in 0..count {
            new_levels.push(self.random_level());
        }

        // Move existing adjacency lists into a locked mirror and append empty
        // slots for the new nodes. Moves are pointer-cheap (no list copying).
        let mut locked: Vec<RwLock<Vec<Vec<u32>>>> =
            std::mem::take(&mut self.links).into_iter().map(RwLock::new).collect();
        for &lvl in &new_levels {
            locked.push(RwLock::new((0..=lvl).map(|_| Vec::new()).collect()));
        }
        self.levels.extend_from_slice(&new_levels);
        self.n += count;

        let entry = Mutex::new(Entry { point: self.entry_point, max_level: self.max_level });
        let mut first = start;
        {
            let mut e = entry.lock().unwrap();
            if e.point.is_none() {
                e.point = Some(start as u32);
                e.max_level = self.levels[start];
                first = start + 1;
            }
        }

        // Parallel linking. `this` is a shared reborrow; the closure only reads
        // self and mutates through the locks in `locked` / `entry`.
        {
            let this: &Hnsw = self;
            (first..start + count)
                .into_par_iter()
                .for_each(|id| this.link_node(id as u32, &locked, &entry));
        }

        // Move adjacency lists back into the plain, lock-free representation.
        self.links = locked.into_iter().map(|l| l.into_inner().unwrap()).collect();
        let e = entry.into_inner().unwrap();
        self.entry_point = e.point;
        self.max_level = e.max_level;
    }

    // Insert node `id` (slots pre-allocated) into the locked graph.
    fn link_node(&self, id: u32, links: &[RwLock<Vec<Vec<u32>>>], entry: &Mutex<Entry>) {
        let v = self.vec_at(id);
        let level = self.levels[id as usize];
        let graph = Locked(links);
        let (mut cur, max_level) = {
            let e = entry.lock().unwrap();
            (e.point.expect("entry set before parallel phase"), e.max_level)
        };

        for lc in (level + 1..=max_level).rev() {
            cur = self.greedy_descend(&graph, v, cur, lc);
        }

        let mut entry_points = vec![cur];
        let top = level.min(max_level);
        for lc in (0..=top).rev() {
            let candidates =
                self.search_layer(&graph, v, &entry_points, self.params.ef_construction, lc, None);
            let cap = if lc == 0 { self.m_max0 } else { self.m_max };
            let selected = self.select_neighbors(&candidates, cap);
            for &nb in &selected {
                self.connect(links, id, nb, lc, cap);
                self.connect(links, nb, id, lc, cap);
            }
            entry_points = candidates.iter().map(|c| c.id).collect();
            if entry_points.is_empty() {
                entry_points = vec![cur];
            }
        }

        if level > max_level {
            let mut e = entry.lock().unwrap();
            if level > e.max_level {
                e.max_level = level;
                e.point = Some(id);
            }
        }
    }

    // Add `neighbor` to `node`'s list on `layer`, pruning back to `cap` via the
    // heuristic if it overflows. Holds only `node`'s write lock (no nested
    // locks), so concurrent inserts can never deadlock.
    fn connect(&self, links: &[RwLock<Vec<Vec<u32>>>], node: u32, neighbor: u32, layer: usize,
               cap: usize) {
        let mut g = links[node as usize].write().unwrap();
        g[layer].push(neighbor);
        if g[layer].len() > cap {
            let nvec = self.vec_at(node);
            let cands: Vec<Neighbor> = g[layer]
                .iter()
                .map(|&nb| Neighbor { dist: self.key(nvec, self.vec_at(nb)), id: nb })
                .collect();
            g[layer] = self.select_neighbors(&cands, cap);
        }
    }

    // Greedy single-best descent on an upper layer (search_layer with ef = 1).
    fn greedy_descend<G: Graph>(&self, g: &G, q: &[f32], entry: u32, layer: usize) -> u32 {
        let mut best = entry;
        let mut best_d = self.key(q, self.vec_at(entry));
        loop {
            let mut improved = false;
            g.for_each(best, layer, |nb| {
                let d = self.key(q, self.vec_at(nb));
                if d < best_d {
                    best_d = d;
                    best = nb;
                    improved = true;
                }
            });
            if !improved {
                return best;
            }
        }
    }

    // Best-first beam search on a single layer; returns up to `ef` nearest nodes
    // that pass `filter` (or all nodes, if `filter` is `None`).
    //
    // A node's *expansion* (whether we visit its neighbors) is decided purely by
    // distance, exactly as in the unfiltered search; its *admission* into the
    // result set `w` additionally requires the filter. This lets the traversal
    // route through non-matching nodes to reach matching ones beyond them,
    // rather than treating excluded nodes as absent from the graph — the
    // standard technique for combining ANN search with attribute filtering.
    // With `filter = None` this is byte-for-byte the unfiltered algorithm (the
    // `passes` check is always true), so unfiltered search is unaffected.
    fn search_layer<G: Graph>(&self, g: &G, q: &[f32], entry: &[u32], ef: usize, layer: usize,
                              filter: Option<&[bool]>) -> Vec<Neighbor> {
        let mut visited: VisitedSet =
            HashSet::with_capacity_and_hasher(ef * 4, BuildHasherDefault::default());
        let mut candidates: BinaryHeap<std::cmp::Reverse<Neighbor>> = BinaryHeap::new();
        let mut w: BinaryHeap<Neighbor> = BinaryHeap::new();
        let passes = |id: u32| match filter {
            Some(f) => f[id as usize],
            None => true,
        };

        for &e in entry {
            if visited.insert(e) {
                let d = self.key(q, self.vec_at(e));
                candidates.push(std::cmp::Reverse(Neighbor { dist: d, id: e }));
                if passes(e) {
                    w.push(Neighbor { dist: d, id: e });
                }
            }
        }
        while w.len() > ef {
            w.pop();
        }

        while let Some(std::cmp::Reverse(c)) = candidates.pop() {
            let worst = w.peek().map(|n| n.dist).unwrap_or(f32::MAX);
            if c.dist > worst && w.len() >= ef {
                break;
            }
            g.for_each(c.id, layer, |nb| {
                if visited.insert(nb) {
                    let d = self.key(q, self.vec_at(nb));
                    let worst = w.peek().map(|n| n.dist).unwrap_or(f32::MAX);
                    if d < worst || w.len() < ef {
                        candidates.push(std::cmp::Reverse(Neighbor { dist: d, id: nb }));
                        if passes(nb) {
                            w.push(Neighbor { dist: d, id: nb });
                            if w.len() > ef {
                                w.pop();
                            }
                        }
                    }
                }
            });
        }
        w.into_vec()
    }

    // HNSW neighbor-selection heuristic (paper Algorithm 4, simple form): keep
    // candidate `c` only if it is closer to the base point than to every
    // already-selected neighbor — favors a diverse, navigable neighbor set.
    fn select_neighbors(&self, candidates: &[Neighbor], m: usize) -> Vec<u32> {
        let mut sorted = candidates.to_vec();
        sorted.sort_unstable();
        let mut result: Vec<Neighbor> = Vec::with_capacity(m);
        for c in sorted {
            if result.len() >= m {
                break;
            }
            let c_vec = self.vec_at(c.id);
            let diverse = result.iter().all(|r| self.key(c_vec, self.vec_at(r.id)) >= c.dist);
            if diverse {
                result.push(c);
            }
        }
        result.into_iter().map(|n| n.id).collect()
    }

    /// Top-`k` for a single query using beam width `ef` (clamped to at least `k`).
    /// Lock-free: reads the plain adjacency lists directly.
    pub fn search(&self, query: &[f32], k: usize, ef: usize, out_ids: &mut [i64],
                  out_dists: &mut [f32]) {
        self.search_impl(query, k, ef, None, out_ids, out_dists);
    }

    /// Top-`k` restricted to vectors where `filter[id]` is `true`. `filter`
    /// must have one entry per stored vector (`filter.len() == self.len()`).
    ///
    /// The graph traversal still expands through non-matching nodes — only
    /// matching nodes are admitted into the result set — so a selective filter
    /// costs more distance evaluations (more of the graph gets explored)
    /// rather than silently starving the result set. `ef` is the same
    /// recall/latency dial as unfiltered search: raise it if a selective
    /// filter returns fewer than `k` results.
    pub fn search_filtered(&self, query: &[f32], k: usize, ef: usize, filter: &[bool],
                          out_ids: &mut [i64], out_dists: &mut [f32]) {
        assert_eq!(filter.len(), self.n, "filter length {} must equal index size {}",
                   filter.len(), self.n);
        self.search_impl(query, k, ef, Some(filter), out_ids, out_dists);
    }

    fn search_impl(&self, query: &[f32], k: usize, ef: usize, filter: Option<&[bool]>,
                  out_ids: &mut [i64], out_dists: &mut [f32]) {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        if k == 0 {
            return;
        }
        let pad = |out_ids: &mut [i64], out_dists: &mut [f32], from: usize| {
            for j in from..k {
                out_ids[j] = -1;
                out_dists[j] = match self.metric {
                    Metric::L2 => f32::MAX,
                    Metric::InnerProduct => f32::MIN,
                };
            }
        };
        let ep = match self.entry_point {
            Some(p) => p,
            None => return pad(out_ids, out_dists, 0),
        };

        let graph = Plain(&self.links);
        let mut cur = ep;
        for lc in (1..=self.max_level).rev() {
            cur = self.greedy_descend(&graph, query, cur, lc);
        }
        let ef = ef.max(k);
        let mut w = self.search_layer(&graph, query, &[cur], ef, 0, filter);
        w.sort_unstable();

        let take = k.min(w.len());
        for j in 0..take {
            out_ids[j] = w[j].id as i64;
            out_dists[j] = match self.metric {
                Metric::L2 => w[j].dist.sqrt(),
                Metric::InnerProduct => -w[j].dist,
            };
        }
        pad(out_ids, out_dists, take);
    }

    /// Batch search, parallelized across queries with rayon.
    pub fn search_batch(&self, queries: &[f32], k: usize, ef: usize, out_ids: &mut [i64],
                        out_dists: &mut [f32], num_threads: usize) {
        self.search_batch_impl(queries, k, ef, None, out_ids, out_dists, num_threads);
    }

    /// Batch version of [`Hnsw::search_filtered`]: the same `filter` is applied
    /// to every query in the batch.
    // Adding `ef` and `filter` to the unfiltered signature pushes this past
    // clippy's default arity threshold; a named-options struct would be
    // over-engineering for an internal, already fully-documented method.
    #[allow(clippy::too_many_arguments)]
    pub fn search_batch_filtered(&self, queries: &[f32], k: usize, ef: usize, filter: &[bool],
                                out_ids: &mut [i64], out_dists: &mut [f32], num_threads: usize) {
        assert_eq!(filter.len(), self.n, "filter length {} must equal index size {}",
                   filter.len(), self.n);
        self.search_batch_impl(queries, k, ef, Some(filter), out_ids, out_dists, num_threads);
    }

    #[allow(clippy::too_many_arguments)]
    fn search_batch_impl(&self, queries: &[f32], k: usize, ef: usize, filter: Option<&[bool]>,
                        out_ids: &mut [i64], out_dists: &mut [f32], num_threads: usize) {
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
                    self.search_impl(&queries[qi * dim..(qi + 1) * dim], k, ef, filter, ids, dists);
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
    fn incremental_add_matches_total() {
        let (n, dim) = (3000usize, 32usize);
        let data = gen(n, dim, 5);
        let mut hnsw = Hnsw::new(dim, Metric::L2, HnswParams::default());
        hnsw.add(&data[..1500 * dim]);
        hnsw.add(&data[1500 * dim..]);
        assert_eq!(hnsw.len(), n);
        let mut ids = vec![0i64; 5];
        let mut d = vec![0f32; 5];
        hnsw.search(&data[7 * dim..8 * dim], 5, 64, &mut ids, &mut d);
        assert_eq!(ids[0], 7, "a stored vector is its own nearest neighbor");
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

    #[test]
    fn filtered_search_only_returns_matching_ids() {
        let (n, dim, k, ef) = (5000usize, 32usize, 10usize, 100usize);
        let data = gen(n, dim, 1);
        let mut hnsw = Hnsw::new(dim, Metric::InnerProduct, HnswParams::default());
        hnsw.add(&data);

        // ~10% of ids pass — a genuinely selective filter, not a near-no-op.
        let filter: Vec<bool> = (0..n).map(|i| i % 10 == 0).collect();
        let queries = gen(50, dim, 42);
        for qi in 0..50 {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let (mut ids, mut d) = (vec![0i64; k], vec![0f32; k]);
            hnsw.search_filtered(q, k, ef, &filter, &mut ids, &mut d);
            for &id in &ids {
                if id >= 0 {
                    assert!(filter[id as usize], "result id {id} must satisfy the filter");
                }
            }
        }
    }

    #[test]
    fn filtered_search_recall_vs_exact_filtered_ground_truth() {
        let (n, dim, k, ef) = (5000usize, 32usize, 10usize, 200usize);
        let data = gen(n, dim, 3);
        let mut hnsw = Hnsw::new(dim, Metric::L2, HnswParams { ef_search: ef, ..Default::default() });
        hnsw.add(&data);
        let mut flat = FlatIndex::new(dim, Metric::L2);
        flat.add(&data);

        let filter: Vec<bool> = (0..n).map(|i| i % 5 == 0).collect(); // 20% pass
        let queries = gen(100, dim, 7);
        let mut hit = 0usize;
        let mut total = 0usize;
        for qi in 0..100 {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let (mut hids, mut hd) = (vec![0i64; k], vec![0f32; k]);
            hnsw.search_filtered(q, k, ef, &filter, &mut hids, &mut hd);
            let (mut fids, mut fd) = (vec![0i64; k], vec![0f32; k]);
            flat.search_filtered(q, k, &filter, &mut fids, &mut fd);
            let truth: std::collections::HashSet<i64> = fids.into_iter().collect();
            for id in hids {
                if id >= 0 && truth.contains(&id) {
                    hit += 1;
                }
            }
            total += k;
        }
        let r = hit as f64 / total as f64;
        assert!(r > 0.9, "filtered recall@10 was {r:.3}, expected > 0.9");
    }

    #[test]
    #[should_panic(expected = "filter length")]
    fn filtered_search_rejects_wrong_length_filter() {
        let (n, dim) = (100usize, 8usize);
        let mut h = Hnsw::new(dim, Metric::L2, HnswParams::default());
        h.add(&gen(n, dim, 1));
        let bad_filter = vec![true; n - 1]; // wrong length
        let (mut ids, mut d) = (vec![0i64; 5], vec![0f32; 5]);
        h.search_filtered(&gen(1, dim, 2), 5, 64, &bad_filter, &mut ids, &mut d);
    }
}
