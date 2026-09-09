# Running the CUDA GPU benchmark (milestone M1)

The GPU exact-search path (`crates/core/cuda/knn.cu` + `src/gpu.rs`) is
feature-gated behind `--features cuda`, so it's inert on machines without an
NVIDIA toolchain. To get a real, validated GPU-vs-CPU speedup number, build and
run it on a machine with a CUDA-capable card and enough VRAM to hold the base
matrix.

The benchmark (`examples/gpu_knn.rs`) generates its own synthetic data, so it
needs **only the source code**, no corpus or dataset download.

---

## Step 0: prerequisites

You need the **NVIDIA driver**, the **CUDA Toolkit**, and **Rust**
(<https://rustup.rs>). Verify with `nvcc --version` and `nvidia-smi`.

Toolkit version matters. Blackwell cards (RTX 50-series, compute capability
12.0) need **CUDA 12.8 or newer** — `compute_120` simply doesn't exist in
earlier toolkits, and `nvcc` rejects it outright rather than falling back to
something that runs. Older architectures are fine on 12.x.

### Windows
- Install the **CUDA Toolkit** from NVIDIA (sets the `CUDA_PATH` env var automatically).
- Install **Visual Studio 2022 Build Tools** with the *"Desktop development with C++"* workload: `nvcc` needs the MSVC host compiler (`cl.exe`).
- **Build from the "x64 Native Tools Command Prompt for VS 2022"** (Start menu) so both `cl.exe` and `nvcc` are on `PATH`. This is the single most common source of build errors.

### Linux
- Install the CUDA Toolkit (distro package or NVIDIA's apt/dnf repo) and `gcc`.
- Ensure `/usr/local/cuda/bin` is on `PATH` and `nvcc --version` works.

## Step 1: check what you're building for

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
```

`build.rs` targets compute capability **12.0** by default, and emits PTX
alongside the SASS so a newer card can JIT it. If `compute_cap` reports
something else, set `PX_CUDA_ARCH` to it without the dot — `89` for an Ada card,
`86` for Ampere, and so on:

```bash
PX_CUDA_ARCH=89 cargo run --release --example gpu_knn --features cuda
```

On Windows that's `set PX_CUDA_ARCH=89` (cmd) or `$env:PX_CUDA_ARCH=89`
(PowerShell) before the `cargo` line.

## Step 2: build and run

```bash
cargo run --release --example gpu_knn --features cuda
# custom sizes: N base, dim, num-queries, k, device, reps
cargo run --release --example gpu_knn --features cuda -- 1000000 128 2000 10
```

The output reports three timings (CPU single-thread, CPU all-cores, GPU), the
throughput and speedup for each, and a top-k agreement figure.

The **agreement check is the important part**. The GPU and the CPU flat index
are both exact, so their top-k must match to ~1.0. That's what proves the kernel
is correct, not just fast — a fast kernel that quietly returns the wrong
neighbors is worse than no kernel.

The example times each phase `reps` times and reports the median, so a single
invocation already absorbs whatever else the machine was doing. It defaults to 3;
pass a larger count as the sixth argument when you are recording a number:

```bash
cargo run --release --example gpu_knn --features cuda -- 1000000 128 2000 10 0 5
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvcc: command not found` | CUDA Toolkit not installed or not on `PATH` (Linux: add `/usr/local/cuda/bin`; Windows: reopen the VS Native Tools prompt). |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | Toolkit older than 12.8. Upgrade it, or set `PX_CUDA_ARCH` to an arch your toolkit knows. |
| `cl.exe not found` / host compiler error (Windows) | Build from the **x64 Native Tools Command Prompt**, or run `vcvars64.bat` first. |
| Linker can't find `cudart` | Set `CUDA_PATH` to the toolkit root (the build script adds `$CUDA_PATH/lib/x64` on Windows, `/lib64` on Linux). |
| `no kernel image is available` (CUDA error 209) | The compiled arch doesn't match the card. Set `PX_CUDA_ARCH` per Step 1 and `cargo clean -p proxima-core` before rebuilding. |
| CUDA out of memory | Lower `N` (the first CLI arg); the base matrix is `N × dim × 4` bytes and has to fit alongside the query and result buffers. |

Most build errors here come down to a toolchain or `PATH` issue rather than
the kernel itself; the table above covers the common ones.
