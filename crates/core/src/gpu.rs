//! CUDA GPU exact k-NN (feature `cuda`) — milestone M1.
//!
//! A thin, safe wrapper over the CUDA host functions in `cuda/knn.cu`. The base
//! matrix is uploaded to the device once (`new`) and reused across searches, so
//! the transfer cost is amortized like a real index. Results use the same
//! "smaller key = closer" convention as the CPU [`FlatIndex`](crate::FlatIndex)
//! and are converted back to public distances, so GPU and CPU are drop-in
//! comparable — which is exactly how the benchmark validates the GPU path.

use std::os::raw::{c_float, c_int, c_longlong};

use crate::metric::Metric;

#[repr(C)]
struct SfGpuIndex {
    _private: [u8; 0],
}

extern "C" {
    fn knn_gpu_create(base: *const c_float, n: c_int, dim: c_int) -> *mut SfGpuIndex;
    fn knn_gpu_search(idx: *const SfGpuIndex, queries: *const c_float, nq: c_int, k: c_int,
                      metric: c_int, out_ids: *mut c_longlong, out_keys: *mut c_float) -> c_int;
    fn knn_gpu_free(idx: *mut SfGpuIndex);
}

/// A GPU-resident exact brute-force k-NN index.
pub struct CudaKnn {
    handle: *mut SfGpuIndex,
    dim: usize,
    n: usize,
    metric: Metric,
}

// The handle is a device pointer owned exclusively by this struct.
unsafe impl Send for CudaKnn {}

impl CudaKnn {
    /// Upload a row-major `(n, dim)` base matrix to GPU memory.
    pub fn new(base: &[f32], dim: usize, metric: Metric) -> Result<Self, String> {
        assert!(dim > 0 && base.len().is_multiple_of(dim), "base length not a multiple of dim");
        let n = base.len() / dim;
        let handle = unsafe { knn_gpu_create(base.as_ptr(), n as c_int, dim as c_int) };
        if handle.is_null() {
            return Err("knn_gpu_create failed (CUDA malloc/copy — out of memory?)".into());
        }
        Ok(CudaKnn { handle, dim, n, metric })
    }

    pub fn len(&self) -> usize {
        self.n
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Exact top-`k` for a batch of queries (row-major `nq * dim`). Fills
    /// `out_ids` / `out_dists` (`nq * k`), converting the internal key to the
    /// public distance (L2 → Euclidean; InnerProduct → similarity).
    pub fn search(&self, queries: &[f32], k: usize, out_ids: &mut [i64],
                  out_dists: &mut [f32]) -> Result<(), String> {
        assert!(queries.len().is_multiple_of(self.dim), "query length not a multiple of dim");
        let nq = queries.len() / self.dim;
        assert!(out_ids.len() >= nq * k && out_dists.len() >= nq * k, "output buffers too small");
        let metric = match self.metric {
            Metric::L2 => 0,
            Metric::InnerProduct => 1,
        };
        // The kernel writes keys into out_dists; convert them in place afterward.
        let rc = unsafe {
            knn_gpu_search(self.handle, queries.as_ptr(), nq as c_int, k as c_int, metric,
                           out_ids.as_mut_ptr(), out_dists.as_mut_ptr())
        };
        if rc != 0 {
            return Err(format!("knn_gpu_search failed (CUDA error {rc})"));
        }
        for d in out_dists.iter_mut().take(nq * k) {
            let key = *d;
            *d = match self.metric {
                Metric::L2 => key.max(0.0).sqrt(),
                Metric::InnerProduct => -key,
            };
        }
        Ok(())
    }
}

impl Drop for CudaKnn {
    fn drop(&mut self) {
        unsafe { knn_gpu_free(self.handle) }
    }
}
