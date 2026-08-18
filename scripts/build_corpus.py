from __future__ import annotations

import argparse
import time

from proxima.corpus import load_wikipedia
from proxima.embeddings import DEFAULT_MODEL, TextEmbedder
from proxima.store import save_corpus


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=100_000, help="max documents")
    p.add_argument("--out", type=str, default="data/wiki_simple", help="output dir")
    p.add_argument("--config", type=str, default="20231101.simple",
                   help="wikimedia/wikipedia config (e.g. 20231101.en for full)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=20_000,
                   help="docs per encode chunk (device cache cleared between)")
    args = p.parse_args()

    print(f"[1/3] Loading up to {args.limit:,} Wikipedia docs ({args.config}) ...",
          flush=True)
    t0 = time.perf_counter()
    docs = load_wikipedia(limit=args.limit, config=args.config)
    print(f"      loaded {len(docs):,} docs in {time.perf_counter() - t0:.1f}s")

    print(f"[2/3] Embedding with {args.model} ...", flush=True)
    embedder = TextEmbedder(args.model)
    print(f"      device={embedder.device} dim={embedder.dim}", flush=True)

    def progress(done, total, elapsed):
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(f"      embedded {done:,}/{total:,} ({rate:,.0f} docs/s, "
              f"ETA {eta / 60:.1f} min)", flush=True)

    t0 = time.perf_counter()
    vectors = embedder.encode_chunked([d.embed_text for d in docs],
                                      chunk_size=args.chunk_size,
                                      batch_size=args.batch_size, progress=progress)
    dt = time.perf_counter() - t0
    print(f"      embedded {len(docs):,} docs in {dt:.1f}s "
          f"({len(docs) / dt:,.0f} docs/s), shape={vectors.shape}", flush=True)

    print(f"[3/3] Saving to {args.out} ...", flush=True)
    save_corpus(args.out, vectors, docs, model_name=args.model)
    mb = vectors.nbytes / 1e6
    print(f"      done. vectors={mb:.1f}MB  ({args.out}/vectors.npy)")


if __name__ == "__main__":
    main()
