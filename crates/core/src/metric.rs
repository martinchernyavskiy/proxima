//! Distance / similarity metric selector.

use serde::{Deserialize, Serialize};

/// The metric an index ranks by.
///
/// * [`Metric::L2`] — squared Euclidean is used internally for ranking (the
///   square root is monotonic, so it is skipped on the hot path); the public
///   API reports the true Euclidean distance. Smaller is more similar.
/// * [`Metric::InnerProduct`] — dot product, which equals cosine similarity for
///   unit-normalized vectors. Larger is more similar.
///
/// Sentence-Transformer / CLIP embeddings are typically L2-normalized, so
/// `InnerProduct` is the natural default for semantic search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    L2,
    InnerProduct,
}
