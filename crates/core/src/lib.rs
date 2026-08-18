pub mod distance;
pub mod flat;
pub mod hnsw;
pub mod metric;
pub mod pq;

#[cfg(feature = "cuda")]
pub mod gpu;

pub use flat::FlatIndex;
pub use hnsw::{Hnsw, HnswParams};
pub use metric::Metric;
pub use pq::{PqIndex, PqParams};

use std::io;
use std::io::Write as _;
use std::path::Path;

pub(crate) fn save_to<T: serde::Serialize>(value: &T, path: &Path) -> io::Result<()> {
    let mut writer = io::BufWriter::new(std::fs::File::create(path)?);
    bincode::serialize_into(&mut writer, value)
        .map_err(|e| io::Error::other(e.to_string()))?;
    writer.flush()
}

pub(crate) fn load_from<T: serde::de::DeserializeOwned>(path: &Path) -> io::Result<T> {
    let reader = io::BufReader::new(std::fs::File::open(path)?);
    bincode::deserialize_from(reader)
        .map_err(|e| io::Error::other(e.to_string()))
}
