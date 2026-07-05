//! SearchForge core — a from-scratch vector search engine (pure Rust).
//!
//! This crate has no Python dependency, so it can be unit-tested and
//! benchmarked directly with `cargo test` / `cargo bench`. The Python bindings
//! live in a separate crate (`searchforge-py`) and are a thin layer on top.
//!
//! Today it exposes the exact [`FlatIndex`] baseline; the approximate HNSW
//! index and product quantization land here as the project progresses, all
//! sharing the [`distance`] kernels and [`Metric`] selector.

pub mod distance;
pub mod flat;
pub mod hnsw;
pub mod metric;
pub mod pq;

pub use flat::FlatIndex;
pub use hnsw::{Hnsw, HnswParams};
pub use metric::Metric;
pub use pq::{PqIndex, PqParams};

use std::io;
use std::path::Path;

/// Serialize any index to a file (bincode). Used by each index's `save`.
pub(crate) fn save_to<T: serde::Serialize>(value: &T, path: &Path) -> io::Result<()> {
    let writer = io::BufWriter::new(std::fs::File::create(path)?);
    bincode::serialize_into(writer, value)
        .map_err(|e| io::Error::other(e.to_string()))
}

/// Deserialize an index from a file written by `save_to`.
pub(crate) fn load_from<T: serde::de::DeserializeOwned>(path: &Path) -> io::Result<T> {
    let reader = io::BufReader::new(std::fs::File::open(path)?);
    bincode::deserialize_from(reader)
        .map_err(|e| io::Error::other(e.to_string()))
}
