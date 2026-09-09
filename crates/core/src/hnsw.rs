use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;
use std::sync::{Mutex, RwLock};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::distance::{inner_product, l2_sqr};
use crate::metric::Metric;

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

type TracedSearchResult = (Vec<(u32, f32)>, Vec<(u32, u32, f32)>, usize);

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

trait Graph {
    fn for_each<F: FnMut(u32)>(&self, node: u32, layer: usize, f: F);
}

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

struct Locked<'a>(&'a [RwLock<Vec<Vec<u32>>>]);
impl Graph for Locked<'_> {
    #[inline]
    fn for_each<F: FnMut(u32)>(&self, node: u32, layer: usize, mut f: F) {
        let g = self.0[node as usize].read().unwrap_or_else(|e| e.into_inner());
        if let Some(nbrs) = g.get(layer) {
            for &x in nbrs {
                f(x);
            }
        }
    }
}

struct Entry {
    point: Option<u32>,
    max_level: usize,
}

#[derive(Serialize, Deserialize)]
pub struct Hnsw {
    dim: usize,
    metric: Metric,
    params: HnswParams,
    m_max0: usize,
    m_max: usize,
    ml: f64,

    data: Vec<f32>,
    links: Vec<Vec<Vec<u32>>>,
    levels: Vec<usize>,
    entry_point: Option<u32>,
    max_level: usize,
    rng_state: u64,
    n: usize,
}

impl Hnsw {
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert!(params.m >= 2, "m must be >= 2");
        assert!(params.ef_construction >= 1, "ef_construction must be >= 1");
        assert!(params.ef_search >= 1, "ef_search must be >= 1");
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

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        crate::save_to(self, path)
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        crate::load_from(path)
    }

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

    #[inline]
    fn key(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            Metric::L2 => l2_sqr(a, b),
            Metric::InnerProduct => -inner_product(a, b),
        }
    }

    #[inline]
    fn to_natural(&self, d: f32) -> f32 {
        match self.metric {
            Metric::L2 => d.sqrt(),
            Metric::InnerProduct => -d,
        }
    }

    fn random_level(&mut self) -> usize {
        self.rng_state = self.rng_state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        let u = ((z >> 11) as f64 / (1u64 << 53) as f64).max(f64::MIN_POSITIVE);
        (-u.ln() * self.ml) as usize
    }

    pub fn add(&mut self, vectors: &[f32]) {
        assert!(vectors.len().is_multiple_of(self.dim), "vectors length not a multiple of dim");
        assert!(
            vectors.iter().all(|x| x.is_finite()),
            "vectors must not contain NaN or infinite values"
        );
        let count = vectors.len() / self.dim;
        if count == 0 {
            return;
        }
        let start = self.n;

        self.data.extend_from_slice(vectors);
        let mut new_levels = Vec::with_capacity(count);
        for _ in 0..count {
            new_levels.push(self.random_level());
        }

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
            let mut e = entry.lock().unwrap_or_else(|e| e.into_inner());
            if e.point.is_none() {
                e.point = Some(start as u32);
                e.max_level = self.levels[start];
                first = start + 1;
            }
        }

        let outcome = {
            let this: &Hnsw = self;
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (first..start + count)
                    .into_par_iter()
                    .for_each(|id| this.link_node(id as u32, &locked, &entry));
            }))
        };

        self.links = locked
            .into_iter()
            .map(|l| l.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect();
        let e = entry.into_inner().unwrap_or_else(|e| e.into_inner());
        self.entry_point = e.point;
        self.max_level = e.max_level;
        if let Err(payload) = outcome {
            std::panic::resume_unwind(payload);
        }
    }

    fn link_node(&self, id: u32, links: &[RwLock<Vec<Vec<u32>>>], entry: &Mutex<Entry>) {
        let v = self.vec_at(id);
        let level = self.levels[id as usize];
        let graph = Locked(links);
        let (mut cur, max_level) = {
            let e = entry.lock().unwrap_or_else(|e| e.into_inner());
            (e.point.expect("entry set before parallel phase"), e.max_level)
        };

        for lc in (level + 1..=max_level).rev() {
            cur = self.greedy_descend(&graph, v, cur, lc, Some(id));
        }

        let mut entry_points = vec![cur];
        let top = level.min(max_level);
        for lc in (0..=top).rev() {
            let candidates = self.search_layer(&graph, v, &entry_points, self.params.ef_construction,
                                               lc, None, Some(id));
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
            let mut e = entry.lock().unwrap_or_else(|e| e.into_inner());
            if level > e.max_level {
                e.max_level = level;
                e.point = Some(id);
            }
        }
    }

    fn connect(&self, links: &[RwLock<Vec<Vec<u32>>>], node: u32, neighbor: u32, layer: usize,
               cap: usize) {
        let mut g = links[node as usize].write().unwrap_or_else(|e| e.into_inner());
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

    fn greedy_descend<G: Graph>(&self, g: &G, q: &[f32], entry: u32, layer: usize,
                               exclude: Option<u32>) -> u32 {
        let mut best = entry;
        let mut best_d = self.key(q, self.vec_at(entry));
        loop {
            let mut improved = false;
            g.for_each(best, layer, |nb| {
                if Some(nb) == exclude {
                    return;
                }
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

    #[allow(clippy::too_many_arguments)]
    fn search_layer<G: Graph>(&self, g: &G, q: &[f32], entry: &[u32], ef: usize, layer: usize,
                              filter: Option<&[bool]>, exclude: Option<u32>) -> Vec<Neighbor> {
        let mut visited: VisitedSet =
            HashSet::with_capacity_and_hasher(ef * 4, BuildHasherDefault::default());
        let mut candidates: BinaryHeap<std::cmp::Reverse<Neighbor>> = BinaryHeap::new();
        let mut w: BinaryHeap<Neighbor> = BinaryHeap::new();
        let passes = |id: u32| match filter {
            Some(f) => f[id as usize],
            None => true,
        };
        let admissible = |id: u32| Some(id) != exclude;

        for &e in entry {
            if admissible(e) && visited.insert(e) {
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
                if admissible(nb) && visited.insert(nb) {
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

    fn search_layer_traced<G: Graph>(&self, g: &G, q: &[f32], entry: &[u32], ef: usize,
                                     layer: usize) -> (Vec<Neighbor>, Vec<(u32, f32)>) {
        let mut visited: VisitedSet =
            HashSet::with_capacity_and_hasher(ef * 4, BuildHasherDefault::default());
        let mut candidates: BinaryHeap<std::cmp::Reverse<Neighbor>> = BinaryHeap::new();
        let mut w: BinaryHeap<Neighbor> = BinaryHeap::new();
        let mut order: Vec<(u32, f32)> = Vec::new();

        for &e in entry {
            if visited.insert(e) {
                let d = self.key(q, self.vec_at(e));
                order.push((e, d));
                candidates.push(std::cmp::Reverse(Neighbor { dist: d, id: e }));
                w.push(Neighbor { dist: d, id: e });
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
                    order.push((nb, d));
                    let worst = w.peek().map(|n| n.dist).unwrap_or(f32::MAX);
                    if d < worst || w.len() < ef {
                        candidates.push(std::cmp::Reverse(Neighbor { dist: d, id: nb }));
                        w.push(Neighbor { dist: d, id: nb });
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }
            });
        }
        (w.into_vec(), order)
    }

    fn even_sample<T: Copy>(items: &[T], cap: usize) -> Vec<T> {
        if items.is_empty() || cap == 0 {
            return Vec::new();
        }
        if items.len() <= cap {
            return items.to_vec();
        }
        (0..cap)
            .map(|i| items[i * (items.len() - 1) / (cap - 1).max(1)])
            .collect()
    }

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

    pub fn search(&self, query: &[f32], k: usize, ef: usize, out_ids: &mut [i64],
                  out_dists: &mut [f32]) {
        self.search_impl(query, k, ef, None, out_ids, out_dists);
    }

    pub fn search_filtered(&self, query: &[f32], k: usize, ef: usize, filter: &[bool],
                          out_ids: &mut [i64], out_dists: &mut [f32]) {
        assert_eq!(filter.len(), self.n, "filter length {} must equal index size {}",
                   filter.len(), self.n);
        self.search_impl(query, k, ef, Some(filter), out_ids, out_dists);
    }

    fn search_impl(&self, query: &[f32], k: usize, ef: usize, filter: Option<&[bool]>,
                  out_ids: &mut [i64], out_dists: &mut [f32]) {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        assert!(
            query.iter().all(|x| x.is_finite()),
            "query must not contain NaN or infinite values"
        );
        if k == 0 {
            return;
        }
        assert_eq!(out_ids.len(), k, "out_ids length {} must equal k ({k})", out_ids.len());
        assert_eq!(out_dists.len(), k, "out_dists length {} must equal k ({k})", out_dists.len());
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
            cur = self.greedy_descend(&graph, query, cur, lc, None);
        }
        let ef = ef.max(k);
        let mut w = self.search_layer(&graph, query, &[cur], ef, 0, filter, None);
        w.sort_unstable();

        let take = k.min(w.len());
        for j in 0..take {
            out_ids[j] = w[j].id as i64;
            out_dists[j] = self.to_natural(w[j].dist);
        }
        pad(out_ids, out_dists, take);
    }

    pub fn search_traced(&self, query: &[f32], k: usize, ef: usize, max_trace: usize)
                        -> TracedSearchResult {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        assert!(
            query.iter().all(|x| x.is_finite()),
            "query must not contain NaN or infinite values"
        );
        if k == 0 {
            return (Vec::new(), Vec::new(), 0);
        }
        let ep = match self.entry_point {
            Some(p) => p,
            None => return (Vec::new(), Vec::new(), 0),
        };

        let graph = Plain(&self.links);
        let mut trace: Vec<(u32, u32, f32)> = Vec::new();
        let mut cur = ep;
        for lc in (1..=self.max_level).rev() {
            cur = self.greedy_descend(&graph, query, cur, lc, None);
            let d = self.key(query, self.vec_at(cur));
            trace.push((lc as u32, cur, self.to_natural(d)));
        }
        let ef = ef.max(k);
        let (mut w, order) = self.search_layer_traced(&graph, query, &[cur], ef, 0);
        w.sort_unstable();
        let total_visited = order.len();
        trace.extend(
            Self::even_sample(&order, max_trace)
                .into_iter()
                .map(|(id, d)| (0u32, id, self.to_natural(d))),
        );

        let take = k.min(w.len());
        let results = w[..take].iter().map(|n| (n.id, self.to_natural(n.dist))).collect();

        (results, trace, total_visited)
    }

    pub fn search_batch(&self, queries: &[f32], k: usize, ef: usize, out_ids: &mut [i64],
                        out_dists: &mut [f32], num_threads: usize) {
        self.search_batch_impl(queries, k, ef, None, out_ids, out_dists, num_threads);
    }

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
        assert!(ids.iter().all(|&x| x == -1));
        h.add(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        h.search(&[1.0, 0.0, 0.0, 0.0], 5, 16, &mut ids, &mut d);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[2], -1);
    }

    #[test]
    fn filtered_search_only_returns_matching_ids() {
        let (n, dim, k, ef) = (5000usize, 32usize, 10usize, 100usize);
        let data = gen(n, dim, 1);
        let mut hnsw = Hnsw::new(dim, Metric::InnerProduct, HnswParams::default());
        hnsw.add(&data);

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

        let filter: Vec<bool> = (0..n).map(|i| i % 5 == 0).collect();
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
        let bad_filter = vec![true; n - 1];
        let (mut ids, mut d) = (vec![0i64; 5], vec![0f32; 5]);
        h.search_filtered(&gen(1, dim, 2), 5, 64, &bad_filter, &mut ids, &mut d);
    }

    #[test]
    fn traced_search_matches_untraced_search() {
        let (n, dim, k, ef) = (2000usize, 32usize, 10usize, 64usize);
        let mut hnsw = Hnsw::new(dim, Metric::InnerProduct, HnswParams { ef_search: ef, ..Default::default() });
        hnsw.add(&gen(n, dim, 11));

        let query = gen(1, dim, 77);
        let (mut ids, mut dists) = (vec![0i64; k], vec![0f32; k]);
        hnsw.search(&query, k, ef, &mut ids, &mut dists);

        let (traced_results, trace, total_visited) = hnsw.search_traced(&query, k, ef, 1000);
        let traced_ids: Vec<i64> = traced_results.iter().map(|&(id, _)| id as i64).collect();
        let traced_dists: Vec<f32> = traced_results.iter().map(|&(_, d)| d).collect();
        assert_eq!(ids, traced_ids, "search_traced must find the same ids as search");
        for (a, b) in dists.iter().zip(traced_dists.iter()) {
            assert!((a - b).abs() < 1e-4, "scores must match: {a} vs {b}");
        }
        assert!(!trace.is_empty(), "trace should record at least the entry point");
        assert!(total_visited > 0);
        for &(_, id, _) in &trace {
            assert!((id as usize) < n, "trace must only contain real node ids");
        }
    }

    #[test]
    fn traced_search_trace_is_capped() {
        let (n, dim, ef) = (5000usize, 32usize, 200usize);
        let mut hnsw = Hnsw::new(dim, Metric::L2, HnswParams { ef_search: ef, ..Default::default() });
        hnsw.add(&gen(n, dim, 3));

        let query = gen(1, dim, 5);
        let (_, trace, total_visited) = hnsw.search_traced(&query, 10, ef, 5);
        let layer0_steps = trace.iter().filter(|&&(layer, _, _)| layer == 0).count();
        assert!(layer0_steps <= 5, "layer-0 trace must respect max_trace, got {layer0_steps}");
        assert!(total_visited >= layer0_steps, "total_visited must be at least the sampled layer-0 steps");
    }

    #[test]
    fn traced_search_on_empty_index() {
        let h = Hnsw::new(8, Metric::L2, HnswParams::default());
        let (results, trace, total_visited) = h.search_traced(&[0.0; 8], 5, 16, 40);
        assert!(results.is_empty());
        assert!(trace.is_empty());
        assert_eq!(total_visited, 0);
    }

    #[test]
    #[should_panic(expected = "ef_search must be >= 1")]
    fn new_rejects_zero_ef_search() {
        Hnsw::new(4, Metric::L2, HnswParams { ef_search: 0, ..Default::default() });
    }

    #[test]
    #[should_panic(expected = "must not contain NaN or infinite values")]
    fn search_rejects_non_finite_query() {
        let mut h = Hnsw::new(4, Metric::L2, HnswParams::default());
        h.add(&gen(50, 4, 1));
        let (mut ids, mut d) = (vec![0i64; 5], vec![0f32; 5]);
        h.search(&[0.0, f32::NAN, 0.0, 0.0], 5, 16, &mut ids, &mut d);
    }

    #[test]
    #[should_panic(expected = "must not contain NaN or infinite values")]
    fn search_traced_rejects_non_finite_query() {
        let mut h = Hnsw::new(4, Metric::L2, HnswParams::default());
        h.add(&gen(50, 4, 1));
        h.search_traced(&[f32::INFINITY, 0.0, 0.0, 0.0], 5, 16, 10);
    }

    #[test]
    #[should_panic(expected = "must equal nq * k")]
    fn search_batch_rejects_mismatched_output_length() {
        let (n, dim, k) = (100usize, 8usize, 5usize);
        let mut h = Hnsw::new(dim, Metric::L2, HnswParams::default());
        h.add(&gen(n, dim, 1));
        let queries = gen(3, dim, 2);
        let mut ids = vec![0i64; k];
        let mut d = vec![0f32; k];
        h.search_batch(&queries, k, 16, &mut ids, &mut d, 0);
    }

    #[test]
    #[should_panic(expected = "must equal k")]
    fn search_rejects_mismatched_output_length() {
        let (n, dim) = (100usize, 8usize);
        let mut h = Hnsw::new(dim, Metric::L2, HnswParams::default());
        h.add(&gen(n, dim, 1));
        let mut ids = vec![0i64; 3];
        let mut d = vec![0f32; 5];
        h.search(&gen(1, dim, 2), 5, 16, &mut ids, &mut d);
    }

    #[test]
    #[should_panic(expected = "must not contain NaN or infinite values")]
    fn add_rejects_non_finite_vectors() {
        let mut h = Hnsw::new(4, Metric::L2, HnswParams::default());
        h.add(&[0.0, f32::NAN, 0.0, 0.0]);
    }

    #[test]
    fn no_self_loops_after_parallel_build() {
        for seed in 1..=5u64 {
            let (n, dim) = (20000usize, 16usize);
            let data = gen(n, dim, seed);
            let mut h = Hnsw::new(dim, Metric::L2, HnswParams { m: 8, ef_construction: 100, ..Default::default() });
            h.add(&data);
            for (id, layers) in h.links.iter().enumerate() {
                for (layer, nbrs) in layers.iter().enumerate() {
                    assert!(!nbrs.contains(&(id as u32)),
                        "node {id} contains itself as a neighbor on layer {layer} (seed {seed})");
                }
            }
        }
    }

    #[test]
    fn no_links_above_node_level_after_parallel_build() {
        for seed in 1..=4u64 {
            let (n, dim) = (10000usize, 16usize);
            let data = gen(n, dim, seed);
            let mut h = Hnsw::new(dim, Metric::L2,
                HnswParams { m: 4, ef_construction: 200, seed, ..Default::default() });
            h.add(&data);
            assert!(h.max_level >= 2,
                "seed {seed} only reached level {}, too flat to exercise entry-point growth",
                h.max_level);
            assert_eq!(h.links.len(), h.levels.len(), "seed {seed}");
            for (id, layers) in h.links.iter().enumerate() {
                assert!(layers.len() <= h.levels[id] + 1,
                    "node {id} holds {} adjacency layers but sits at level {} (seed {seed})",
                    layers.len(), h.levels[id]);
            }
            let ep = h.entry_point.expect("a populated index must have an entry point") as usize;
            assert!(h.max_level <= h.levels[ep],
                "entry point {ep} sits at level {} but max_level is {} (seed {seed})",
                h.levels[ep], h.max_level);
        }
    }
}
