//! Python bindings (PyO3 + rust-numpy) for the SearchForge engine.
//!
//! This layer is deliberately thin: it validates shapes, hands numpy buffers to
//! the pure-Rust core as `&[f32]`, and releases the GIL around the native
//! search so Python threads can run concurrently. All search logic lives in
//! `searchforge-core`.

use std::path::PathBuf;

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use searchforge_core::{
    FlatIndex as CoreFlat, Hnsw as CoreHnsw, HnswParams, Metric as CoreMetric,
    PqIndex as CorePq, PqParams,
};

/// Distance metric, mirrored from the core enum so Python never sees Rust types.
#[pyclass(name = "Metric", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
enum Metric {
    L2,
    InnerProduct,
}

impl From<Metric> for CoreMetric {
    fn from(m: Metric) -> Self {
        match m {
            Metric::L2 => CoreMetric::L2,
            Metric::InnerProduct => CoreMetric::InnerProduct,
        }
    }
}

/// Exact brute-force vector index (see `searchforge_core::FlatIndex`).
#[pyclass(name = "FlatIndex")]
struct PyFlatIndex {
    inner: CoreFlat,
}

#[pymethods]
impl PyFlatIndex {
    #[new]
    #[pyo3(signature = (dim, metric = Metric::InnerProduct))]
    fn new(dim: usize, metric: Metric) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be > 0"));
        }
        Ok(Self { inner: CoreFlat::new(dim, metric.into()) })
    }

    /// Append an `(n, dim)` float32 array of vectors.
    fn add(&mut self, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        let shape = vectors.shape();
        if shape[1] != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "vector dim {} does not match index dim {}",
                shape[1],
                self.inner.dim()
            )));
        }
        let slice = vectors
            .as_slice()
            .map_err(|_| PyValueError::new_err("vectors must be C-contiguous float32"))?;
        self.inner.add(slice);
        Ok(())
    }

    /// Top-`k` for a single `(dim,)` query. Returns `(ids, distances)` as two
    /// 1-D numpy arrays of length `k`.
    #[pyo3(signature = (query, k = 10))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
        if query.shape()[0] != self.inner.dim() {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        let q = query
            .as_slice()
            .map_err(|_| PyValueError::new_err("query must be C-contiguous float32"))?;
        let mut ids = vec![0i64; k];
        let mut dists = vec![0f32; k];
        // Release the GIL: the native scan touches no Python state.
        py.allow_threads(|| self.inner.search(q, k, &mut ids, &mut dists));
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    /// Top-`k` for an `(nq, dim)` batch of queries. Returns `(ids, distances)`
    /// as two `(nq, k)` numpy arrays. Parallelized across queries with rayon;
    /// `num_threads = 0` uses all cores.
    #[pyo3(signature = (queries, k = 10, num_threads = 0))]
    fn search_batch<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        num_threads: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
        let shape = queries.shape();
        let (nq, dim) = (shape[0], shape[1]);
        if dim != self.inner.dim() {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        let q = queries
            .as_slice()
            .map_err(|_| PyValueError::new_err("queries must be C-contiguous float32"))?;
        let mut ids = vec![0i64; nq * k];
        let mut dists = vec![0f32; nq * k];
        py.allow_threads(|| {
            self.inner.search_batch(q, k, &mut ids, &mut dists, num_threads)
        });
        let ids = Array2::from_shape_vec((nq, k), ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dists = Array2::from_shape_vec((nq, k), dists)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.len()
    }
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    #[getter]
    fn metric(&self) -> Metric {
        match self.inner.metric() {
            CoreMetric::L2 => Metric::L2,
            CoreMetric::InnerProduct => Metric::InnerProduct,
        }
    }
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(&path).map_err(|e| PyIOError::new_err(e.to_string()))
    }
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = CoreFlat::load(&path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Approximate HNSW graph index (see `searchforge_core::Hnsw`).
#[pyclass(name = "HnswIndex")]
struct PyHnsw {
    inner: CoreHnsw,
    ef_search: usize,
}

#[pymethods]
impl PyHnsw {
    #[new]
    #[pyo3(signature = (dim, metric = Metric::InnerProduct, m = 16,
                        ef_construction = 200, ef_search = 64, seed = 24301))]
    fn new(dim: usize, metric: Metric, m: usize, ef_construction: usize,
           ef_search: usize, seed: u64) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be > 0"));
        }
        if m < 2 {
            return Err(PyValueError::new_err("m must be >= 2"));
        }
        let params = HnswParams { m, ef_construction, ef_search, seed };
        Ok(Self { inner: CoreHnsw::new(dim, metric.into(), params), ef_search })
    }

    /// Append an `(n, dim)` float32 array, inserting each vector into the graph.
    fn add(&mut self, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        if vectors.shape()[1] != self.inner.dim() {
            return Err(PyValueError::new_err("vector dim mismatch"));
        }
        let slice = vectors
            .as_slice()
            .map_err(|_| PyValueError::new_err("vectors must be C-contiguous float32"))?;
        self.inner.add(slice);
        Ok(())
    }

    #[pyo3(signature = (query, k = 10))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
        if query.shape()[0] != self.inner.dim() {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        let q = query
            .as_slice()
            .map_err(|_| PyValueError::new_err("query must be C-contiguous float32"))?;
        let ef = self.ef_search;
        let mut ids = vec![0i64; k];
        let mut dists = vec![0f32; k];
        py.allow_threads(|| self.inner.search(q, k, ef, &mut ids, &mut dists));
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[pyo3(signature = (queries, k = 10, num_threads = 0))]
    fn search_batch<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        num_threads: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
        let shape = queries.shape();
        let (nq, dim) = (shape[0], shape[1]);
        if dim != self.inner.dim() {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        let q = queries
            .as_slice()
            .map_err(|_| PyValueError::new_err("queries must be C-contiguous float32"))?;
        let ef = self.ef_search;
        let mut ids = vec![0i64; nq * k];
        let mut dists = vec![0f32; nq * k];
        py.allow_threads(|| self.inner.search_batch(q, k, ef, &mut ids, &mut dists, num_threads));
        let ids = Array2::from_shape_vec((nq, k), ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dists = Array2::from_shape_vec((nq, k), dists)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[getter]
    fn ef_search(&self) -> usize {
        self.ef_search
    }
    #[setter]
    fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef;
    }
    #[getter]
    fn size(&self) -> usize {
        self.inner.len()
    }
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    #[getter]
    fn metric(&self) -> Metric {
        match self.inner.metric() {
            CoreMetric::L2 => Metric::L2,
            CoreMetric::InnerProduct => Metric::InnerProduct,
        }
    }
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(&path).map_err(|e| PyIOError::new_err(e.to_string()))
    }
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = CoreHnsw::load(&path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let ef_search = inner.params().ef_search;
        Ok(Self { inner, ef_search })
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Product-quantized compressed index (see `searchforge_core::PqIndex`).
///
/// Lifecycle: construct → `train(training)` → `add(vectors)` → `search(...)`.
#[pyclass(name = "PqIndex")]
struct PyPq {
    inner: Option<CorePq>,
    dim: usize,
    metric: CoreMetric,
    params: PqParams,
}

#[pymethods]
impl PyPq {
    #[new]
    #[pyo3(signature = (dim, metric = Metric::InnerProduct, m = 8, nbits = 8,
                        train_iters = 25, train_sample = 50000, seed = 24301))]
    fn new(dim: usize, metric: Metric, m: usize, nbits: usize, train_iters: usize,
           train_sample: usize, seed: u64) -> PyResult<Self> {
        if dim == 0 || dim % m != 0 {
            return Err(PyValueError::new_err("dim must be > 0 and divisible by m"));
        }
        if !(1..=8).contains(&nbits) {
            return Err(PyValueError::new_err("nbits must be in 1..=8"));
        }
        Ok(Self {
            inner: None,
            dim,
            metric: metric.into(),
            params: PqParams { m, nbits, train_iters, train_sample, seed },
        })
    }

    /// Learn the codebooks from an `(n, dim)` training set.
    fn train(&mut self, py: Python<'_>, training: PyReadonlyArray2<f32>) -> PyResult<()> {
        if training.shape()[1] != self.dim {
            return Err(PyValueError::new_err("training dim mismatch"));
        }
        let slice = training
            .as_slice()
            .map_err(|_| PyValueError::new_err("training must be C-contiguous float32"))?;
        let (dim, metric, params) = (self.dim, self.metric, self.params);
        let inner = py.allow_threads(|| CorePq::train(slice, dim, metric, params));
        self.inner = Some(inner);
        Ok(())
    }

    /// Encode and append an `(n, dim)` array. Requires `train` first.
    fn add(&mut self, py: Python<'_>, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        if vectors.shape()[1] != self.dim {
            return Err(PyValueError::new_err("vector dim mismatch"));
        }
        let slice = vectors
            .as_slice()
            .map_err(|_| PyValueError::new_err("vectors must be C-contiguous float32"))?;
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("index is not trained; call train() first"))?;
        py.allow_threads(|| inner.add(slice));
        Ok(())
    }

    #[pyo3(signature = (query, k = 10))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("index is not trained"))?;
        if query.shape()[0] != self.dim {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        let q = query
            .as_slice()
            .map_err(|_| PyValueError::new_err("query must be C-contiguous float32"))?;
        let mut ids = vec![0i64; k];
        let mut dists = vec![0f32; k];
        py.allow_threads(|| inner.search(q, k, &mut ids, &mut dists));
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[pyo3(signature = (queries, k = 10, num_threads = 0))]
    fn search_batch<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        num_threads: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("index is not trained"))?;
        let shape = queries.shape();
        let (nq, dim) = (shape[0], shape[1]);
        if dim != self.dim {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        let q = queries
            .as_slice()
            .map_err(|_| PyValueError::new_err("queries must be C-contiguous float32"))?;
        let mut ids = vec![0i64; nq * k];
        let mut dists = vec![0f32; nq * k];
        py.allow_threads(|| inner.search_batch(q, k, &mut ids, &mut dists, num_threads));
        let ids = Array2::from_shape_vec((nq, k), ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dists = Array2::from_shape_vec((nq, k), dists)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[getter]
    fn is_trained(&self) -> bool {
        self.inner.is_some()
    }
    #[getter]
    fn size(&self) -> usize {
        self.inner.as_ref().map_or(0, |x| x.len())
    }
    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }
    #[getter]
    fn m(&self) -> usize {
        self.params.m
    }
    #[getter]
    fn metric(&self) -> Metric {
        match self.metric {
            CoreMetric::L2 => Metric::L2,
            CoreMetric::InnerProduct => Metric::InnerProduct,
        }
    }
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.as_ref().map_or(0, |x| x.memory_bytes())
    }
    #[getter]
    fn raw_bytes(&self) -> usize {
        self.inner.as_ref().map_or(0, |x| x.raw_bytes())
    }
    /// Raw float32 bytes / compressed bytes.
    #[getter]
    fn compression_ratio(&self) -> f64 {
        match &self.inner {
            Some(x) if x.memory_bytes() > 0 => x.raw_bytes() as f64 / x.memory_bytes() as f64,
            _ => 0.0,
        }
    }
    fn save(&self, path: PathBuf) -> PyResult<()> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("index is not trained"))?;
        inner.save(&path).map_err(|e| PyIOError::new_err(e.to_string()))
    }
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = CorePq::load(&path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let (dim, metric, m) = (inner.dim(), inner.metric(), inner.m());
        Ok(Self {
            inner: Some(inner),
            dim,
            metric,
            params: PqParams { m, nbits: 8, train_iters: 25, train_sample: 50_000, seed: 24301 },
        })
    }
    fn __len__(&self) -> usize {
        self.size()
    }
}

/// The native module, imported as `searchforge._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Metric>()?;
    m.add_class::<PyFlatIndex>()?;
    m.add_class::<PyHnsw>()?;
    m.add_class::<PyPq>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
