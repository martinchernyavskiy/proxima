"""Proxima: a from-scratch GPU-accelerable vector search engine.

The heavy lifting lives in the native extension ``proxima._core`` (Rust, built
via PyO3/maturin). This package re-exports the engine types and adds thin,
ergonomic Python helpers (corpus loading, embeddings, benchmarking) on top.
"""

from ._core import FlatIndex, HnswIndex, Metric, PqIndex

__all__ = ["FlatIndex", "HnswIndex", "PqIndex", "Metric"]
__version__ = "0.1.0"
