//! GPU vs CPU exact-search benchmark (milestone M1). Run on a CUDA machine:
//!
//!   cargo run --release --example gpu_knn --features cuda
//!   cargo run --release --example gpu_knn --features cuda -- 1000000 128 2000 10
//!
//! Args (all optional): N base vectors, dim, nq queries, k. It builds the same
//! data for both engines, checks that the GPU's exact top-k agrees with the CPU
//! flat index (both are exact, so agreement should be ~1.0), and reports the
//! speedup over both single- and multi-threaded CPU search.

use std::time::Instant;

use searchforge_core::gpu::CudaKnn;
use searchforge_core::{FlatIndex, Metric};

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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, def: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(def);
    let (n, dim, nq, k) = (arg(1, 1_000_000), arg(2, 128), arg(3, 2000), arg(4, 10));
    let metric = Metric::L2;
    println!("exact k-NN: N={n} dim={dim} queries={nq} k={k} metric=L2");

    let base = gen(n, dim, 1);
    let queries = gen(nq, dim, 2);

    // CPU baselines (single- and multi-threaded).
    let mut flat = FlatIndex::new(dim, metric);
    flat.add(&base);
    let (mut cids, mut cd) = (vec![0i64; nq * k], vec![0f32; nq * k]);

    let t = Instant::now();
    flat.search_batch(&queries, k, &mut cids, &mut cd, 1);
    let cpu1 = t.elapsed().as_secs_f64();

    let t = Instant::now();
    flat.search_batch(&queries, k, &mut cids, &mut cd, 0);
    let cpu_mt = t.elapsed().as_secs_f64();

    // GPU: upload once, warm once, then time.
    let gpu = CudaKnn::new(&base, dim, metric).expect("GPU init");
    let (mut gids, mut gd) = (vec![0i64; nq * k], vec![0f32; nq * k]);
    gpu.search(&queries, k, &mut gids, &mut gd).expect("GPU warmup");
    let t = Instant::now();
    gpu.search(&queries, k, &mut gids, &mut gd).expect("GPU search");
    let gpu_t = t.elapsed().as_secs_f64();

    // Correctness: GPU and CPU are both exact, so their top-k sets should match.
    let mut agree = 0usize;
    for q in 0..nq {
        let cpu_set: std::collections::HashSet<i64> =
            cids[q * k..(q + 1) * k].iter().copied().collect();
        agree += gids[q * k..(q + 1) * k].iter().filter(|id| cpu_set.contains(id)).count();
    }
    let agreement = agree as f64 / (nq * k) as f64;

    println!("\n  CPU flat (1 thread) : {cpu1:8.3} s   ({:>8.0} q/s)", nq as f64 / cpu1);
    println!("  CPU flat (all cores): {cpu_mt:8.3} s   ({:>8.0} q/s)", nq as f64 / cpu_mt);
    println!("  GPU exact           : {gpu_t:8.3} s   ({:>8.0} q/s)", nq as f64 / gpu_t);
    println!("\n  speedup vs CPU 1-thread : {:.1}x", cpu1 / gpu_t);
    println!("  speedup vs CPU all-core : {:.1}x", cpu_mt / gpu_t);
    println!("  GPU/CPU top-{k} agreement : {agreement:.4}  (expect ~1.0 — both exact)");
    assert!(agreement > 0.99, "GPU results disagree with exact CPU — kernel bug");
    println!("\n  OK: GPU exact search matches the CPU ground truth.");
}
