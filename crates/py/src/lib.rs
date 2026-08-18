#![allow(clippy::useless_conversion, clippy::type_complexity)]

use std::borrow::Cow;
use std::path::PathBuf;

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use proxima_core::{
    FlatIndex as CoreFlat, Hnsw as CoreHnsw, HnswParams, Metric as CoreMetric,
    PqIndex as CorePq, PqParams,
};

fn rows_c<'a>(a: &'a PyReadonlyArray2<'_, f32>) -> Cow<'a, [f32]> {
    if a.is_c_contiguous() {
        Cow::Borrowed(a.as_slice().expect("c-contiguous"))
    } else {
        Cow::Owned(a.as_array().iter().copied().collect())
    }
}
fn vec_c<'a>(a: &'a PyReadonlyArray1<'_, f32>) -> Cow<'a, [f32]> {
    if a.is_c_contiguous() {
        Cow::Borrowed(a.as_slice().expect("c-contiguous"))
    } else {
        Cow::Owned(a.as_array().iter().copied().collect())
    }
}

fn mask_c<'a>(a: &'a PyReadonlyArray1<'_, bool>) -> Cow<'a, [bool]> {
    if a.is_c_contiguous() {
        Cow::Borrowed(a.as_slice().expect("c-contiguous"))
    } else {
        Cow::Owned(a.as_array().iter().copied().collect())
    }
}

fn check_finite(v: &[f32]) -> PyResult<()> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(PyValueError::new_err("query must not contain NaN or infinite values"))
    }
}

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

    fn add(&mut self, py: Python<'_>, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        let shape = vectors.shape();
        if shape[1] != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "vector dim {} does not match index dim {}",
                shape[1],
                self.inner.dim()
            )));
        }
        let data = rows_c(&vectors);
        py.allow_threads(|| self.inner.add(&data));
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
        let q = vec_c(&query);
        check_finite(&q)?;
        let mut ids = vec![0i64; k];
        let mut dists = vec![0f32; k];
        py.allow_threads(|| self.inner.search(&q, k, &mut ids, &mut dists));
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
        let q = rows_c(&queries);
        check_finite(&q)?;
        let mut ids = vec![0i64; nq * k];
        let mut dists = vec![0f32; nq * k];
        py.allow_threads(|| {
            self.inner.search_batch(&q, k, &mut ids, &mut dists, num_threads)
        });
        let ids = Array2::from_shape_vec((nq, k), ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dists = Array2::from_shape_vec((nq, k), dists)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[pyo3(signature = (query, mask, k = 10))]
    fn search_filtered<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        mask: PyReadonlyArray1<bool>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
        if query.shape()[0] != self.inner.dim() {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        if mask.shape()[0] != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "mask length {} must equal index size {}", mask.shape()[0], self.inner.len()
            )));
        }
        let q = vec_c(&query);
        check_finite(&q)?;
        let m = mask_c(&mask);
        let mut ids = vec![0i64; k];
        let mut dists = vec![0f32; k];
        py.allow_threads(|| self.inner.search_filtered(&q, k, &m, &mut ids, &mut dists));
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[pyo3(signature = (queries, mask, k = 10, num_threads = 0))]
    fn search_batch_filtered<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        mask: PyReadonlyArray1<bool>,
        k: usize,
        num_threads: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
        let shape = queries.shape();
        let (nq, dim) = (shape[0], shape[1]);
        if dim != self.inner.dim() {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        if mask.shape()[0] != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "mask length {} must equal index size {}", mask.shape()[0], self.inner.len()
            )));
        }
        let q = rows_c(&queries);
        check_finite(&q)?;
        let m = mask_c(&mask);
        let mut ids = vec![0i64; nq * k];
        let mut dists = vec![0f32; nq * k];
        py.allow_threads(|| {
            self.inner.search_batch_filtered(&q, k, &m, &mut ids, &mut dists, num_threads)
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
        if ef_construction < 1 {
            return Err(PyValueError::new_err("ef_construction must be >= 1"));
        }
        if ef_search < 1 {
            return Err(PyValueError::new_err("ef_search must be >= 1"));
        }
        let params = HnswParams { m, ef_construction, ef_search, seed };
        Ok(Self { inner: CoreHnsw::new(dim, metric.into(), params), ef_search })
    }

    fn add(&mut self, py: Python<'_>, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        if vectors.shape()[1] != self.inner.dim() {
            return Err(PyValueError::new_err("vector dim mismatch"));
        }
        let data = rows_c(&vectors);
        py.allow_threads(|| self.inner.add(&data));
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
        let q = vec_c(&query);
        check_finite(&q)?;
        let ef = self.ef_search;
        let mut ids = vec![0i64; k];
        let mut dists = vec![0f32; k];
        py.allow_threads(|| self.inner.search(&q, k, ef, &mut ids, &mut dists));
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[pyo3(signature = (query, k = 10, max_trace = 40))]
    fn search_traced(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        max_trace: usize,
    ) -> PyResult<(Vec<(i64, f32)>, Vec<(u32, i64, f32)>, usize)> {
        if query.shape()[0] != self.inner.dim() {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        let q = vec_c(&query);
        check_finite(&q)?;
        let ef = self.ef_search;
        let (results, trace, total_visited) =
            py.allow_threads(|| self.inner.search_traced(&q, k, ef, max_trace));
        let results = results.into_iter().map(|(id, s)| (id as i64, s)).collect();
        let trace = trace.into_iter().map(|(layer, id, s)| (layer, id as i64, s)).collect();
        Ok((results, trace, total_visited))
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
        let q = rows_c(&queries);
        check_finite(&q)?;
        let ef = self.ef_search;
        let mut ids = vec![0i64; nq * k];
        let mut dists = vec![0f32; nq * k];
        py.allow_threads(|| self.inner.search_batch(&q, k, ef, &mut ids, &mut dists, num_threads));
        let ids = Array2::from_shape_vec((nq, k), ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dists = Array2::from_shape_vec((nq, k), dists)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[pyo3(signature = (query, mask, k = 10))]
    fn search_filtered<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        mask: PyReadonlyArray1<bool>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
        if query.shape()[0] != self.inner.dim() {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        if mask.shape()[0] != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "mask length {} must equal index size {}", mask.shape()[0], self.inner.len()
            )));
        }
        let q = vec_c(&query);
        check_finite(&q)?;
        let m = mask_c(&mask);
        let ef = self.ef_search;
        let mut ids = vec![0i64; k];
        let mut dists = vec![0f32; k];
        py.allow_threads(|| self.inner.search_filtered(&q, k, ef, &m, &mut ids, &mut dists));
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[pyo3(signature = (queries, mask, k = 10, num_threads = 0))]
    fn search_batch_filtered<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        mask: PyReadonlyArray1<bool>,
        k: usize,
        num_threads: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
        let shape = queries.shape();
        let (nq, dim) = (shape[0], shape[1]);
        if dim != self.inner.dim() {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        if mask.shape()[0] != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "mask length {} must equal index size {}", mask.shape()[0], self.inner.len()
            )));
        }
        let q = rows_c(&queries);
        check_finite(&q)?;
        let m = mask_c(&mask);
        let ef = self.ef_search;
        let mut ids = vec![0i64; nq * k];
        let mut dists = vec![0f32; nq * k];
        py.allow_threads(|| {
            self.inner.search_batch_filtered(&q, k, ef, &m, &mut ids, &mut dists, num_threads)
        });
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
        self.inner.set_ef_search(ef);
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
        if m == 0 {
            return Err(PyValueError::new_err("m must be >= 1"));
        }
        if dim == 0 || !dim.is_multiple_of(m) {
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

    fn train(&mut self, py: Python<'_>, training: PyReadonlyArray2<f32>) -> PyResult<()> {
        if training.shape()[1] != self.dim {
            return Err(PyValueError::new_err("training dim mismatch"));
        }
        let k = 1usize << self.params.nbits;
        if training.shape()[0] < k {
            return Err(PyValueError::new_err(format!(
                "training set too small: need at least {k} rows for nbits={}, got {}",
                self.params.nbits,
                training.shape()[0]
            )));
        }
        let training_data = rows_c(&training);
        let (dim, metric, params) = (self.dim, self.metric, self.params);
        let inner = py.allow_threads(|| CorePq::train(&training_data, dim, metric, params));
        self.inner = Some(inner);
        Ok(())
    }

    fn add(&mut self, py: Python<'_>, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        if vectors.shape()[1] != self.dim {
            return Err(PyValueError::new_err("vector dim mismatch"));
        }
        let data = rows_c(&vectors);
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("index is not trained; call train() first"))?;
        py.allow_threads(|| inner.add(&data));
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
        let q = vec_c(&query);
        check_finite(&q)?;
        let mut ids = vec![0i64; k];
        let mut dists = vec![0f32; k];
        py.allow_threads(|| inner.search(&q, k, &mut ids, &mut dists));
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
        let q = rows_c(&queries);
        check_finite(&q)?;
        let mut ids = vec![0i64; nq * k];
        let mut dists = vec![0f32; nq * k];
        py.allow_threads(|| inner.search_batch(&q, k, &mut ids, &mut dists, num_threads));
        let ids = Array2::from_shape_vec((nq, k), ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dists = Array2::from_shape_vec((nq, k), dists)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[pyo3(signature = (query, mask, k = 10))]
    fn search_filtered<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        mask: PyReadonlyArray1<bool>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("index is not trained"))?;
        if query.shape()[0] != self.dim {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        if mask.shape()[0] != inner.len() {
            return Err(PyValueError::new_err(format!(
                "mask length {} must equal index size {}", mask.shape()[0], inner.len()
            )));
        }
        let q = vec_c(&query);
        check_finite(&q)?;
        let m = mask_c(&mask);
        let mut ids = vec![0i64; k];
        let mut dists = vec![0f32; k];
        py.allow_threads(|| inner.search_filtered(&q, k, &m, &mut ids, &mut dists));
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    #[pyo3(signature = (queries, mask, k = 10, num_threads = 0))]
    fn search_batch_filtered<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        mask: PyReadonlyArray1<bool>,
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
        if mask.shape()[0] != inner.len() {
            return Err(PyValueError::new_err(format!(
                "mask length {} must equal index size {}", mask.shape()[0], inner.len()
            )));
        }
        let q = rows_c(&queries);
        check_finite(&q)?;
        let m = mask_c(&mask);
        let mut ids = vec![0i64; nq * k];
        let mut dists = vec![0f32; nq * k];
        py.allow_threads(|| {
            inner.search_batch_filtered(&q, k, &m, &mut ids, &mut dists, num_threads)
        });
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
        let (dim, metric, m, nbits) = (inner.dim(), inner.metric(), inner.m(), inner.nbits());
        Ok(Self {
            inner: Some(inner),
            dim,
            metric,
            params: PqParams { m, nbits, train_iters: 25, train_sample: 50_000, seed: 24301 },
        })
    }
    fn __len__(&self) -> usize {
        self.size()
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Metric>()?;
    m.add_class::<PyFlatIndex>()?;
    m.add_class::<PyHnsw>()?;
    m.add_class::<PyPq>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
