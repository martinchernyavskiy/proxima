#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.use("svg")

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "docs" / "assets"

BLUE = "#2a78d6"
AQUA = "#1baf7a"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8980"


def _header(fig, title: str, subtitle: str) -> None:
    fig.text(0.03, 0.97, title, color=INK_PRIMARY, fontsize=13,
             fontweight="bold", ha="left", va="top")
    fig.text(0.03, 0.905, subtitle, color=INK_SECONDARY, fontsize=9.5,
             ha="left", va="top")


def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, length=0, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)


def _series(ax, points, color, label):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round",
             solid_joinstyle="round", zorder=3, label=label)
    ax.scatter(xs, ys, s=70, color=color, edgecolors=SURFACE, linewidths=2,
               zorder=4)
    return xs, ys


def _label_point(ax, x, y, text, xytext, ha="left"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=xytext,
                fontsize=8, color=INK_MUTED, ha=ha)


def _set_log_ticks(ax, ticks_ms: list[float], data_min: float, data_max: float,
                   pad_frac: float = 0.22) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ticks_ms)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g} ms"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(data_min * (1 - pad_frac), data_max * (1 + pad_frac))


def _save(fig, stem: str) -> None:
    svg_path = ASSETS / f"{stem}.svg"
    png_path = ASSETS / f"{stem}.png"
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png", dpi=200)
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")


def plot_sift1m(results: list[dict]) -> None:
    by_name = {r["name"]: r for r in results}
    px = [(by_name[f"PX HNSW(ef={ef})"]["p50_ms"], by_name[f"PX HNSW(ef={ef})"]["recall_at_k"], ef)
          for ef in (32, 64, 128)]
    fs = [(by_name[f"FAISS HNSW(ef={ef})"]["p50_ms"], by_name[f"FAISS HNSW(ef={ef})"]["recall_at_k"], ef)
          for ef in (32, 64, 128)]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.11, right=0.97)

    fx, fy = _series(ax, fs, AQUA, "FAISS HNSW")
    sx, sy = _series(ax, px, BLUE, "Proxima HNSW")
    _label_point(ax, fx[0], fy[0], "ef=32", (7, -3))
    _label_point(ax, fx[-1], fy[-1], "ef=128", (7, -3))
    _label_point(ax, sx[0], sy[0], "ef=32", (7, -3))
    _label_point(ax, sx[-1], sy[-1], "ef=128", (7, -3))

    all_x = fx + sx
    _set_log_ticks(ax, [0.07, 0.1, 0.15, 0.2, 0.3], min(all_x), max(all_x))
    ax.set_ylim(0.88, 1.005)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))
    _style(ax, "p50 query latency (log scale)", "recall@10")

    _header(fig, "SIFT1M: recall vs. latency, Proxima vs. FAISS",
            "1M vectors, 128-dim, k=10, identical queries + ground truth")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
             handlelength=1.6)
    _save(fig, "recall_latency_sift1m")
    plt.close(fig)


def plot_wiki(results: list[dict]) -> None:
    by_name = {r["name"]: r for r in results}
    sweep = [(by_name[f"HNSW(ef={ef})"]["p50_ms"], by_name[f"HNSW(ef={ef})"]["recall_at_k"], ef)
             for ef in (16, 32, 64, 128)]
    flat = by_name["FlatIndex (exact)"]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.11, right=0.97)

    xs, ys = _series(ax, sweep, BLUE, "HNSW (ef_search sweep)")
    _label_point(ax, xs[0], ys[0], "ef=16", (7, -3))
    _label_point(ax, xs[-1], ys[-1], "ef=128", (-10, -14), ha="right")

    fx, fy = flat["p50_ms"], flat["recall_at_k"]
    ax.scatter([fx], [fy], s=90, marker="D", color=INK_SECONDARY,
               edgecolors=SURFACE, linewidths=2, zorder=4, label="Flat (exact)")
    slowdown = fx / xs[-1]
    _label_point(ax, fx, fy, f"exact, ~{slowdown:.1f}x slower", (-10, 8), ha="right")

    all_x = xs + [fx]
    _set_log_ticks(ax, [0.2, 0.5, 1, 2, 5], min(all_x), max(all_x))
    ax.set_ylim(0.87, 1.006)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}"))
    _style(ax, "p50 query latency (log scale)", "recall@10")

    _header(fig, "Wikipedia 100k: HNSW's recall/latency dial vs. exact search",
            "100k articles, 384-dim, k=10, ef_search is the tuning knob")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
             handlelength=1.6)
    _save(fig, "recall_latency_wiki")
    plt.close(fig)


def load_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        print(f"{path.name}: no provenance recorded (pre-envelope results file)")
        return data
    prov = data.get("provenance") or {}
    git = prov.get("git") or {}
    sha = git.get("sha") or "?"
    dirty = " +dirty" if git.get("dirty") else ""
    cpu = (prov.get("cpu") or {}).get("model") or "?"
    print(f"{path.name}: {prov.get('hostname', '?')} ({cpu}) "
          f"at {prov.get('timestamp_utc', '?')}, git {sha[:12]}{dirty}")
    return data.get("results", [])


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    sift = load_results(HERE / "results" / "sift1m.json")
    wiki = load_results(HERE / "results" / "wiki_simple_100k.json")
    plot_sift1m(sift)
    plot_wiki(wiki)


if __name__ == "__main__":
    main()
