from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from proxima import FlatIndex, Metric

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent
LATENCY_SAMPLES = 10_000
THROUGHPUT_TRIALS = 7
LATENCY_SEED = 20240917


@dataclass
class BenchResult:
    name: str
    n_base: int
    dim: int
    k: int
    recall_at_k: float | None
    p50_ms: float
    p99_ms: float
    mean_ms: float
    qps_1t: float
    qps_mt: float
    memory_mb: float
    latency_samples: int = 0
    build_seconds: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None,
                              capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _first_int(text: str | None) -> int | None:
    for token in (text or "").split():
        try:
            return int(token)
        except ValueError:
            continue
    return None


def _cpu_model() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif system == "Darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand
    elif system == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            ) as key:
                name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
            if name:
                return name
        except Exception:
            pass
        name = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                     "(Get-CimInstance Win32_Processor).Name"])
        if name:
            return name.splitlines()[0].strip()
    return platform.processor() or platform.uname().processor or "unknown"


def _physical_cores() -> int | None:
    try:
        import psutil

        count = psutil.cpu_count(logical=False)
        if count:
            return int(count)
    except Exception:
        pass
    system = platform.system()
    if system == "Linux":
        try:
            pairs, package = set(), None
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("physical id"):
                    package = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    pairs.add((package, line.split(":", 1)[1].strip()))
            if pairs:
                return len(pairs)
        except Exception:
            pass
    elif system == "Darwin":
        return _first_int(_run(["sysctl", "-n", "hw.physicalcpu"]))
    elif system == "Windows":
        return _first_int(_run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "(Get-CimInstance Win32_Processor | "
             "Measure-Object -Property NumberOfCores -Sum).Sum"]))
    return None


def _git_info() -> dict:
    sha = _run(["git", "rev-parse", "HEAD"], REPO_ROOT)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], REPO_ROOT)
    status = _run(["git", "status", "--porcelain"], REPO_ROOT)
    return {"sha": sha or None, "branch": branch or None,
            "dirty": None if status is None else bool(status)}


def _module_version(name: str) -> str | None:
    module = sys.modules.get(name)
    version = getattr(module, "__version__", None) if module is not None else None
    if version:
        return str(version)
    try:
        from importlib.metadata import version as metadata_version

        return metadata_version(name)
    except Exception:
        return None


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def collect_provenance() -> dict:
    try:
        return {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hostname": _safe(platform.node) or _safe(socket.gethostname) or "unknown",
            "argv": list(sys.argv),
            "cwd": _safe(os.getcwd),
            "os": {
                "platform": _safe(platform.platform),
                "system": _safe(platform.system),
                "release": _safe(platform.release),
                "machine": _safe(platform.machine),
            },
            "cpu": {
                "model": _safe(_cpu_model, "unknown"),
                "cores_physical": _safe(_physical_cores),
                "cores_logical": _safe(os.cpu_count),
            },
            "git": _safe(_git_info, {}),
            "versions": {
                "python": _safe(platform.python_version),
                "python_executable": sys.executable,
                "numpy": _safe(lambda: _module_version("numpy")),
                "faiss": _safe(lambda: _module_version("faiss")),
                "rustc": _safe(lambda: _run(["rustc", "--version"])),
            },
        }
    except Exception as exc:
        return {"error": f"provenance collection failed: {type(exc).__name__}: {exc}"}


def results_envelope(results: list[BenchResult]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": collect_provenance(),
        "results": [r.as_dict() for r in results],
    }


def write_results(path: str | Path, results: list[BenchResult]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results_envelope(results), indent=2))
    return out


def exact_ground_truth(base: np.ndarray, queries: np.ndarray, k: int,
                       metric: Metric = Metric.InnerProduct) -> np.ndarray:
    idx = FlatIndex(dim=base.shape[1], metric=metric)
    idx.add(np.ascontiguousarray(base, dtype=np.float32))
    ids, _ = idx.search_batch(np.ascontiguousarray(queries, dtype=np.float32),
                              k=k, num_threads=0)
    return ids


def recall_at_k(approx_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    nq = len(approx_ids)
    total = 0.0
    for i in range(nq):
        approx = {int(x) for x in approx_ids[i][:k] if x >= 0}
        truth = {int(x) for x in gt_ids[i][:k] if x >= 0}
        if truth:
            total += len(approx & truth) / len(truth)
    return total / nq if nq else 0.0


def pin_single_thread() -> None:
    faiss = sys.modules.get("faiss")
    if faiss is None:
        return
    setter = getattr(faiss, "omp_set_num_threads", None)
    if setter is None:
        return
    setter(1)
    getter = getattr(faiss, "omp_get_max_threads", None)
    if getter is not None and getter() != 1:
        raise RuntimeError(
            "FAISS still reports more than one OpenMP thread after "
            "omp_set_num_threads(1); single-query latency would not be a "
            "controlled comparison against Proxima's single-threaded path"
        )


def measure_latency(index, queries: np.ndarray, k: int,
                    samples: int = LATENCY_SAMPLES, seed: int = LATENCY_SEED
                    ) -> tuple[float, float, float, int]:
    pin_single_thread()
    if len(queries) == 0 or samples <= 0:
        return 0.0, 0.0, 0.0, 0
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(queries), size=samples)
    times = np.empty(samples, dtype=np.float64)
    for i, qi in enumerate(picks):
        q = queries[qi]
        t0 = time.perf_counter()
        index.search(q, k=k)
        times[i] = (time.perf_counter() - t0) * 1e3
    return (float(np.percentile(times, 50)), float(np.percentile(times, 99)),
            float(times.mean()), int(samples))


def measure_throughput(index, queries: np.ndarray, k: int, num_threads: int,
                       trials: int = THROUGHPUT_TRIALS) -> float:
    index.search_batch(queries[: min(64, len(queries))], k=k, num_threads=num_threads)
    rates = []
    for _ in range(max(1, trials)):
        t0 = time.perf_counter()
        index.search_batch(queries, k=k, num_threads=num_threads)
        dt = time.perf_counter() - t0
        rates.append(len(queries) / dt if dt > 0 else float("inf"))
    return statistics.median(rates)


def benchmark_index(name: str, index, queries: np.ndarray, k: int,
                    gt_ids: np.ndarray | None,
                    latency_samples: int = LATENCY_SAMPLES,
                    build_seconds: float | None = None) -> BenchResult:
    q = np.ascontiguousarray(queries, dtype=np.float32)
    approx_ids, _ = index.search_batch(q, k=k, num_threads=0)
    recall = recall_at_k(approx_ids, gt_ids, k) if gt_ids is not None else None
    p50, p99, mean, n_latency = measure_latency(index, q, k, samples=latency_samples)
    qps_1t = measure_throughput(index, q, k, num_threads=1)
    qps_mt = measure_throughput(index, q, k, num_threads=0)
    return BenchResult(
        name=name, n_base=index.size, dim=index.dim, k=k,
        recall_at_k=recall, p50_ms=p50, p99_ms=p99, mean_ms=mean,
        qps_1t=qps_1t, qps_mt=qps_mt, memory_mb=index.memory_bytes / 1e6,
        latency_samples=n_latency, build_seconds=build_seconds,
    )


def format_table(results: list[BenchResult]) -> str:
    cols = ["name", "n_base", "dim", "k", "recall_at_k", "p50_ms", "p99_ms",
            "qps_1t", "qps_mt", "memory_mb"]
    headers = ["index", "N", "dim", "k", "recall@k", "p50(ms)", "p99(ms)",
               "QPS(1t)", "QPS(mt)", "mem(MB)"]

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}" if v < 100 else f"{v:,.0f}"
        return f"{v:,}" if isinstance(v, int) else str(v)

    rows = [[fmt(getattr(r, c)) for c in cols] for r in results]
    widths = [max([len(headers[i])] + [len(row[i]) for row in rows]) for i in range(len(cols))]
    line = lambda parts: "  ".join(p.rjust(widths[i]) for i, p in enumerate(parts))
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])
