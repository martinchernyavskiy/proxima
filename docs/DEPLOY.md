# Deploying the demo

The demo (`demo/app.py`) is a FastAPI server that embeds each query and searches
an HNSW index over a Wikipedia corpus. The included [`Dockerfile`](../Dockerfile)
builds the Rust engine, bakes the same 99k-article Simple-English Wikipedia
corpus the benchmarks above use, plus its index, at image-build time, and serves on port `7860`.

Because the query has to be embedded server-side, the image includes PyTorch +
sentence-transformers, so the running container needs roughly 2 GB of RAM. That
rules out most free tiers, which cap at 512 MB.

**Hugging Face Spaces** is the recommended host. Creating a Space that runs on
compute now requires a PRO subscription ($9/month as of September 2026); the
CPU Basic hardware it then gives you (2 vCPU, 16 GB RAM) costs nothing per hour.
Static Spaces remain free but cannot run a server.

## Hugging Face Spaces (recommended)

1. Create an account at <https://huggingface.co/join> and subscribe to PRO.
2. **New → Space.** Choose **Docker → Blank**, hardware **CPU Basic**, name it e.g. `proxima`.
3. Put this repo's files in the Space's git repo. Easiest is to add the Space as a remote and push:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/proxima
   git push space main
   ```
4. Ensure the Space's `README.md` begins with this frontmatter so HF builds the Dockerfile on port 7860 (the Space-creation UI adds it; if you overwrote the README by pushing, re-add it or set the port under **Settings**):
   ```yaml
   ---
   title: Proxima
   emoji: 🔎
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
5. HF builds the image (first build ~35–45 min: it compiles Rust, installs Torch, and embeds 99k articles) and serves at `https://huggingface.co/spaces/<your-username>/proxima`.

**Notes**
- The corpus is baked at build time, so restarts are fast. To change its size or source, edit the `build_corpus.py` line in the `Dockerfile` (e.g. `--limit 100000`, or `--config 20231101.en` for full English Wikipedia, which needs a larger image and more RAM).
- Query latency on free CPU is dominated by embedding the query (~20 ms); the index search itself stays ~1 ms and is what the UI reports.
- To iterate on the build without pushing each time, build locally: `docker build -t proxima . && docker run -p 7860:7860 proxima`, then open <http://localhost:7860>.

## Other hosts

The same `Dockerfile` works on any container platform. Point it at the Dockerfile
and expose port `7860` (or set `--port $PORT` for platforms that inject a `PORT`
env var).

Be aware of the memory floor. As of September 2026 the free tiers on Render and
Koyeb are both 512 MB, which is not enough to hold PyTorch plus the index, so the
container will not start. Fly.io no longer offers a free tier and Railway is a
trial credit rather than an ongoing one. Anything with 2 GB or more works —
Google Cloud Run and Oracle Cloud's always-free instances both qualify, though
both want a card on file.

If you want this to fit a 512 MB tier, the change worth making is replacing
PyTorch and sentence-transformers with ONNX Runtime and an ONNX export of the
same MiniLM model. That drops the image by roughly an order of magnitude and the
resident set to a few hundred megabytes, at no cost to search quality — the
engine itself never touches the embedding model.
