# Running the CUDA GPU benchmark (milestone M1)

The GPU exact-search path (`crates/core/cuda/knn.cu` + `src/gpu.rs`) is
feature-gated behind `--features cuda`, so it's inert on machines without an
NVIDIA toolchain. To get a real, validated GPU-vs-CPU speedup number, build and
run it on the desktop with the **RTX 4070 Ti** (Ada, compute capability 8.9,
with plenty of VRAM for million-vector exact search).

The benchmark (`examples/gpu_knn.rs`) generates its own synthetic data, so it
needs **only the source code**, no corpus or dataset download.

---

## Step 0: get the code onto the desktop

Cleanest (and you want this for the résumé anyway): push to GitHub, then clone on the desktop.

```bash
# on the Mac, from the repo:
gh repo create proxima --public --source=. --remote=origin --push
#   ...or: git remote add origin https://github.com/<you>/proxima.git && git push -u origin main

# on the desktop:
git clone https://github.com/<you>/proxima.git && cd proxima
```

(A plain folder copy works too. The `data/` and `target/` dirs are gitignored and not needed.)

## Step 1: prerequisites

Both OSes need: the **NVIDIA driver** (already installed for gaming), the
**CUDA Toolkit 12.x** (provides `nvcc`), and **Rust** (<https://rustup.rs>).
Verify with `nvcc --version` and `nvidia-smi`.

### Windows
- Install the **CUDA Toolkit** from NVIDIA (sets the `CUDA_PATH` env var automatically).
- Install **Visual Studio 2022 Build Tools** with the *"Desktop development with C++"* workload: `nvcc` needs the MSVC host compiler (`cl.exe`).
- **Build from the "x64 Native Tools Command Prompt for VS 2022"** (Start menu) so both `cl.exe` and `nvcc` are on `PATH`. This is the single most common source of build errors.

### Linux
- Install the CUDA Toolkit (distro package or NVIDIA's apt/dnf repo) and `gcc`.
- Ensure `/usr/local/cuda/bin` is on `PATH` and `nvcc --version` works.

## Step 2: build and run

```bash
cargo run --release --example gpu_knn --features cuda
# custom sizes: N base, dim, num-queries, k
cargo run --release --example gpu_knn --features cuda -- 1000000 128 2000 10
```

Expected output (numbers will vary):

```
exact k-NN: N=1000000 dim=128 queries=2000 k=10 metric=L2
  CPU flat (1 thread) :   17.900 s   (     112 q/s)
  CPU flat (all cores):    2.700 s   (     741 q/s)
  GPU exact           :    0.090 s   (   22000 q/s)
  speedup vs CPU 1-thread : 199.0x
  speedup vs CPU all-core :  30.0x
  GPU/CPU top-10 agreement : 1.0000  (expect ~1.0, both exact)
  OK: GPU exact search matches the CPU ground truth.
```

The **agreement check is the important part**: the GPU and the CPU flat index are
both exact, so their top-k must match (~1.0). That's what proves the kernel is
correct, not just fast. Drop the numbers straight into the README's results
section and the résumé bullet.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvcc: command not found` | CUDA Toolkit not installed or not on `PATH` (Linux: add `/usr/local/cuda/bin`; Windows: reopen the VS Native Tools prompt). |
| `cl.exe not found` / host compiler error (Windows) | Build from the **x64 Native Tools Command Prompt**, or run `vcvars64.bat` first. |
| Linker can't find `cudart` | Set `CUDA_PATH` to the toolkit root (the build script adds `$CUDA_PATH/lib/x64` on Windows, `/lib64` on Linux). |
| `no kernel image is available` at runtime | Your GPU's compute capability differs from `sm_89`; edit the `-gencode` arch in `crates/core/build.rs` (find it via `nvidia-smi --query-gpu=compute_cap --format=csv`). |
| CUDA out of memory | Lower `N` (the first CLI arg); the base matrix must fit in 12 GB VRAM. |

Most build errors here come down to a toolchain or `PATH` issue rather than
the kernel itself; the table above covers the common ones.
