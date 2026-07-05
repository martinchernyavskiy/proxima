# Deploying the demo

The demo (`demo/app.py`) is a FastAPI server that embeds each query and searches
an HNSW index over a Wikipedia corpus. The included [`Dockerfile`](../Dockerfile)
builds the Rust engine, bakes a small (30k-article Simple-English Wikipedia)
corpus + index at image-build time, and serves on port `7860`.

Because the query has to be embedded server-side, the image includes PyTorch +
sentence-transformers (~2 GB). **Hugging Face Spaces** is the recommended host:
free, 16 GB RAM, no credit card, and purpose-built for this.

## Hugging Face Spaces (recommended, free)

1. Create a free account at <https://huggingface.co/join>.
2. **New → Space.** Choose **Docker → Blank**, hardware **CPU basic (free)**, name it e.g. `searchforge`.
3. Put this repo's files in the Space's git repo. Easiest is to add the Space as a remote and push:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/searchforge
   git push space main
   ```
4. Ensure the Space's `README.md` begins with this frontmatter so HF builds the Dockerfile on port 7860 (the Space-creation UI adds it; if you overwrote the README by pushing, re-add it or set the port under **Settings**):
   ```yaml
   ---
   title: SearchForge
   emoji: 🔎
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
5. HF builds the image (first build ~10–20 min: it compiles Rust, installs Torch, and embeds 30k articles) and serves at `https://huggingface.co/spaces/<your-username>/searchforge`. That URL is what you put on your résumé.

**Notes**
- The corpus is baked at build time, so restarts are fast. To change its size or source, edit the `build_corpus.py` line in the `Dockerfile` (e.g. `--limit 100000`, or `--config 20231101.en` for full English Wikipedia — larger images/RAM).
- Query latency on free CPU is dominated by embedding the query (~20 ms); the index search itself stays ~1 ms and is what the UI reports.
- To iterate on the build without pushing each time, build locally: `docker build -t searchforge . && docker run -p 7860:7860 searchforge`, then open <http://localhost:7860>.

## Other hosts

The same `Dockerfile` works on any container platform — **Render**, **Fly.io**,
**Railway**, or a VPS. Point the platform at the Dockerfile and expose port
`7860` (or set `--port $PORT` in the start command for platforms that inject a
`PORT` env var). These generally need a credit card on file even for free tiers,
which is why Spaces is the recommended default.
