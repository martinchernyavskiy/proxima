const LANES: usize = 8;

#[inline]
pub fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2_sqr: mismatched vector lengths");
    let mut acc = [0.0f32; LANES];
    let mut ai = a.chunks_exact(LANES);
    let mut bi = b.chunks_exact(LANES);
    for (ac, bc) in ai.by_ref().zip(bi.by_ref()) {
        for l in 0..LANES {
            let d = ac[l] - bc[l];
            acc[l] += d * d;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ai.remainder().iter().zip(bi.remainder().iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "inner_product: mismatched vector lengths");
    let mut acc = [0.0f32; LANES];
    let mut ai = a.chunks_exact(LANES);
    let mut bi = b.chunks_exact(LANES);
    for (ac, bc) in ai.by_ref().zip(bi.by_ref()) {
        for l in 0..LANES {
            acc[l] += ac[l] * bc[l];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ai.remainder().iter().zip(bi.remainder().iter()) {
        sum += x * y;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_naive(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }
    fn ip_naive(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn kernels_match_naive_across_lengths() {
        for len in [1usize, 3, 7, 8, 9, 16, 17, 31, 64, 100] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.91 + 1.0).cos()).collect();
            assert!((l2_sqr(&a, &b) - l2_naive(&a, &b)).abs() < 1e-3, "l2 len {len}");
            assert!((inner_product(&a, &b) - ip_naive(&a, &b)).abs() < 1e-3, "ip len {len}");
        }
    }

    #[test]
    fn l2_zero_for_identical() {
        let a = [1.0f32, -2.0, 3.5, 4.0, 0.0, 9.9, 7.0, 8.0, 1.0];
        assert!(l2_sqr(&a, &a).abs() < 1e-5);
    }

    #[test]
    #[should_panic]
    fn l2_sqr_mismatched_lengths_panics() {
        let a = [0.0f32; 3];
        let b = [0.0f32; 16];
        l2_sqr(&a, &b);
    }

    #[test]
    #[should_panic]
    fn inner_product_mismatched_lengths_panics() {
        let a = [0.0f32; 3];
        let b = [0.0f32; 16];
        inner_product(&a, &b);
    }
}
