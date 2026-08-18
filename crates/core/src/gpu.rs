use std::os::raw::{c_float, c_int, c_longlong};

use crate::metric::Metric;

#[repr(C)]
struct PxGpuIndex {
    _private: [u8; 0],
}

extern "C" {
    fn knn_gpu_create(base: *const c_float, n: c_int, dim: c_int) -> *mut PxGpuIndex;
    fn knn_gpu_search(idx: *const PxGpuIndex, queries: *const c_float, nq: c_int, k: c_int,
                      metric: c_int, out_ids: *mut c_longlong, out_keys: *mut c_float) -> c_int;
    fn knn_gpu_free(idx: *mut PxGpuIndex);
}

pub struct CudaKnn {
    handle: *mut PxGpuIndex,
    dim: usize,
    n: usize,
    metric: Metric,
}

unsafe impl Send for CudaKnn {}

impl CudaKnn {
    pub fn new(base: &[f32], dim: usize, metric: Metric) -> Result<Self, String> {
        assert!(dim > 0 && base.len().is_multiple_of(dim), "base length not a multiple of dim");
        let n = base.len() / dim;
        let handle = unsafe { knn_gpu_create(base.as_ptr(), n as c_int, dim as c_int) };
        if handle.is_null() {
            return Err("knn_gpu_create failed (CUDA malloc/copy, out of memory?)".into());
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

    pub fn search(&self, queries: &[f32], k: usize, out_ids: &mut [i64],
                  out_dists: &mut [f32]) -> Result<(), String> {
        assert!(queries.len().is_multiple_of(self.dim), "query length not a multiple of dim");
        let nq = queries.len() / self.dim;
        assert!(out_ids.len() >= nq * k && out_dists.len() >= nq * k, "output buffers too small");
        if k == 0 {
            return Ok(());
        }
        assert!(
            queries.iter().all(|x| x.is_finite()),
            "query must not contain NaN or infinite values"
        );
        let metric = match self.metric {
            Metric::L2 => 0,
            Metric::InnerProduct => 1,
        };
        let rc = unsafe {
            knn_gpu_search(self.handle, queries.as_ptr(), nq as c_int, k as c_int, metric,
                           out_ids.as_mut_ptr(), out_dists.as_mut_ptr())
        };
        if rc != 0 {
            return Err(match rc {
                -1 => format!("k={k} is invalid for the GPU kernel (must be between 1 and its max-k limit)"),
                -2 => format!(
                    "k={k} exceeds the GPU kernel's shared-memory budget for this vector \
                     dimensionality; try a smaller k"
                ),
                _ => format!("knn_gpu_search failed (CUDA error {rc})"),
            });
        }
        for (d, &id) in out_dists.iter_mut().zip(out_ids.iter()).take(nq * k) {
            let key = *d;
            *d = if id < 0 {
                match self.metric {
                    Metric::L2 => f32::MAX,
                    Metric::InnerProduct => f32::MIN,
                }
            } else {
                match self.metric {
                    Metric::L2 => key.sqrt(),
                    Metric::InnerProduct => -key,
                }
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
