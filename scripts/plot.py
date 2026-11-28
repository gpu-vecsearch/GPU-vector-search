"""Plot benchmark, roofline, scaling and baseline results.

Figures (PNG, written to results/plots/; a figure whose input is missing is skipped):

  throughput.png  queries/s vs query batch size for every version and FAISS
                  (results/benchmark.csv, rows with the chosen k)
  roofline.png    each profiled kernel against the roof from the same Nsight
                  Compute run (results/roofline.csv), plus measured ceilings from
                  results/micro.csv when present
  scaling.png     v2 OpenMP speedup vs threads, from benchmark.csv rows labelled
                  v2-t<threads>  (bench_v2 --version v2-t8 with OMP_NUM_THREADS=8)
  gemm.png        cuBLAS GEMM GFLOP/s vs batch, FP32 and FP16 (results/cublas_gemm.csv)

When the same (label, batch, k) appears more than once in a CSV, the last row wins,
so re-running a benchmark replaces its point.

Needs matplotlib: scripts/setup_faiss.sh installs it into the FAISS venv
($HOME/.venvs/vecsearch-faiss), or pip install matplotlib anywhere.

Usage: python3 scripts/plot.py [--results results] [--out results/plots] [--k 10]
                               [--scaling-batch B]
"""

import argparse
import csv
import math
import os
import re
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
except ImportError:
    sys.exit("matplotlib is not installed; run: bash scripts/setup_faiss.sh "
             "(then use $HOME/.venvs/vecsearch-faiss/bin/python)")

# ---------------------------------------------------------------------------
# Palette and chrome (validated categorical palette, light mode, fixed order).
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Color follows the entity, never its rank: each label owns one slot in every figure.
ENTITIES = [
    ("v1", "v1 serial CPU"),
    ("v2", "v2 OpenMP CPU"),
    ("v3", "v3 naive CUDA"),
    ("v4", "v4 tiled CUDA"),
    ("v5", "v5 fused top-k"),
    ("v6", "v6 WMMA FP16"),
    ("faiss-gpu-resident", "FAISS GPU (resident)"),
    ("faiss-gpu-e2e", "FAISS GPU (end-to-end)"),
]
SLOT = {label: i for i, (label, _) in enumerate(ENTITIES)}
NAME = dict(ENTITIES)

KERNEL_NAMES = {"l2_naive_kernel": "distance", "topk_rows_kernel": "top-k"}


def style(fig, ax, title, subtitle):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=10, width=1, length=4)
    ax.tick_params(which="minor", length=0)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=INK)
    fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=10, color=INK_2)


def label_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10.5)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10.5)


def compact(v, _pos=None):
    """1, 10, 100, 1K, 10K, 1M - one style per axis."""
    if v <= 0:
        return "0"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= div:
            x = v / div
            return ("%.0f%s" if x >= 10 or x == int(x) else "%.1f%s") % (x, suffix)
    if v >= 1:
        return "%.0f" % v
    return ("%.2f" % v).rstrip("0").rstrip(".")


def flops_text(gflops):
    return "%.1f TFLOP/s" % (gflops / 1e3) if gflops >= 1e3 else "%.0f GFLOP/s" % gflops


def log_axis(axis, lo=None, hi=None):
    """Log axis with compact labels; adds 2x and 5x ticks when it spans < ~2.5 decades."""
    subs = (1.0,)
    if lo and hi and hi / lo < 10 ** 2.5:
        subs = (1.0, 2.0, 5.0)
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_minor_formatter(NullFormatter())
    axis.set_major_formatter(FuncFormatter(compact))


def label_along(ax, p0, p1, text, at, color, fontsize=9.5, offset=7):
    """Place text parallel to the segment p0-p1 (data coords), just above point `at`."""
    (x0, y0), (x1, y1) = ax.transData.transform([p0, p1])
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    ax.annotate(text, at, xytext=(-offset * math.sin(math.radians(angle)),
                                  offset * math.cos(math.radians(angle))),
                textcoords="offset points", rotation=angle, rotation_mode="anchor",
                ha="left", va="bottom", fontsize=fontsize, color=color)


def legend(ax, handles, outside=False, loc="upper left"):
    kw = dict(handles=handles, frameon=False, fontsize=9.5, labelcolor=INK_2, handlelength=1.4,
              borderaxespad=0.2)
    if outside:
        return ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), **kw)
    return ax.legend(loc=loc, **kw)


def save(fig, path):
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", path)


def read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def line_with_markers(ax, xs, ys, color, label):
    (h,) = ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                   marker="o", markersize=8, markerfacecolor=color, markeredgecolor=SURFACE,
                   markeredgewidth=2, label=label, zorder=3)
    return h


def end_labels(fig, ax, series, min_gap_pt=13):
    """Direct labels at line ends (only for <= 4 series), in text ink. If any two ends
    are closer than a label's height, skip them all: the legend carries identity, and
    nudged labels would detach from their lines."""
    ends = sorted(ax.transData.transform((xs[-1], ys[-1]))[1] for xs, ys, _ in series)
    min_gap = min_gap_pt * fig.dpi / 72.0
    if any(b - a < min_gap for a, b in zip(ends, ends[1:])):
        return False
    for xs, ys, text in series:
        ax.annotate(text, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9.5, color=INK)
    return True


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_throughput(rows, k, out):
    points = {}
    for r in rows:
        if int(r["k"]) != k:
            continue
        label = r["version"]
        if label not in SLOT:
            continue  # scaling rows (v2-t8) and unknown labels are not plotted here
        points.setdefault(label, {})[int(r["batch"])] = float(r["throughput_qps"])
    unknown = sorted({r["version"] for r in rows} - set(SLOT) -
                     {r["version"] for r in rows if re.match(r"^v2-t\d+$", r["version"])})
    if unknown:
        print("throughput: ignoring unknown labels %s (add them to ENTITIES)" % ", ".join(unknown))
    if not points:
        print("throughput: no rows with k=%d in benchmark.csv, skipped" % k)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(top=0.84)
    style(fig, ax, "Throughput vs. query batch size",
          "SIFT1M (1M x 128), exact top-%d, median of timed runs. GPU versions copy the database to "
          "the device on every call; FAISS (resident) keeps it there." % k)
    handles, ends = [], []
    for label, _ in ENTITIES:
        if label not in points:
            continue
        xs = sorted(points[label])
        ys = [points[label][x] for x in xs]
        handles.append(line_with_markers(ax, xs, ys, SERIES[SLOT[label]], NAME[label]))
        ends.append((xs, ys, NAME[label]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_axis(ax.xaxis)
    log_axis(ax.yaxis)
    ally = [y for _, ys, _ in ends for y in ys]
    log_axis(ax.yaxis, min(ally), max(ally))
    label_axes(ax, "Queries per batch", "Queries per second")
    if len(handles) >= 2:
        legend(ax, handles, outside=len(handles) > 4)
    if len(handles) <= 4:
        right = ax.get_xlim()[1]
        ax.set_xlim(right=max(max(xs) for xs, _, _ in ends) * 4)
        if not end_labels(fig, ax, ends):
            ax.set_xlim(right=right)
    save(fig, os.path.join(out, "throughput.png"))


def place_point_labels(fig, ax, labels):
    """Direct-label every point (identity never rests on color alone), trying a few
    positions per label and keeping the first that overlaps no earlier text."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    fig.canvas.draw()  # labels and transforms must be laid out before measuring
    renderer = fig.canvas.get_renderer()
    taken = [Text.get_window_extent(t, renderer) for t in ax.texts]
    pad = 9 * fig.dpi / 72.0  # roughly a marker's radius plus its ring, in pixels
    for (x, y), _ in labels:
        cx, cy = ax.transData.transform((x, y))
        taken.append(Bbox([[cx - pad, cy - pad], [cx + pad, cy + pad]]))
    # Points sit under the roof, so try spots below a point before spots above it.
    candidates = [((10, -2), "left", "top"), ((-10, -2), "right", "top"),
                  ((10, -32), "left", "top"), ((-10, -32), "right", "top"),
                  ((10, -62), "left", "top"), ((-10, -62), "right", "top"),
                  ((10, 4), "left", "bottom"), ((-10, 4), "right", "bottom")]
    for (x, y), text in sorted(labels, key=lambda item: item[0][0]):
        placed = None
        for offset, ha, va in candidates:
            # A label set away from its point gets a thin leader line back to it.
            leader = dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=6) \
                if abs(offset[1]) > 20 else None
            t = ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va,
                            fontsize=9, color=INK, linespacing=1.3, arrowprops=leader, zorder=5,
                            bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))
            box = Text.get_window_extent(t, renderer)  # the text alone, not its leader line
            if not any(box.overlaps(b) for b in taken):
                placed = box
                break
            t.remove()
        if placed is None:  # nothing free: keep the default spot rather than drop the name
            t = ax.annotate(text, (x, y), xytext=candidates[0][0], textcoords="offset points",
                            ha="left", va="top", fontsize=9, color=INK, linespacing=1.3)
            placed = t.get_window_extent(renderer)
        taken.append(placed)


def kernel_label(version, kernel):
    base = re.sub(r"<.*>$", "", kernel)
    return "%s %s" % (version, KERNEL_NAMES.get(base, re.sub(r"_kernel$", "", base)))


def plot_roofline(rows, micro, out):
    pts = [r for r in rows if float(r["gflop"]) > 0 and float(r["achieved_gflops"]) > 0]
    skipped = [kernel_label(r["version"], r["kernel"]) for r in rows if r not in pts]
    if not pts:
        print("roofline: no kernels with FP32 work in roofline.csv, skipped")
        return
    if skipped:
        print("roofline: not plotted (no FP32 FLOPs, e.g. top-k compares only): %s" % ", ".join(skipped))

    peak = float(pts[0]["peak_gflops"])
    bw = float(pts[0]["peak_dram_gbs"])
    ais = [float(r["ai_flop_per_byte"]) for r in pts]
    xmin = 10 ** math.floor(math.log10(min(ais + [0.1]) / 2))
    xmax = 10 ** math.ceil(math.log10(max(ais + [peak / bw]) * 20))
    xs = [xmin * (xmax / xmin) ** (i / 200.0) for i in range(201)]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(top=0.85)
    style(fig, ax, "Roofline: where each kernel sits",
          "Achieved FP32 GFLOP/s vs. arithmetic intensity (DRAM). Roof and points from the same "
          "Nsight Compute run, at profiling clocks%s." % (", " + pts[0]["device"].replace("NVIDIA ", "")
                                                          if pts[0].get("device") else ""))
    ax.plot(xs, [min(peak, x * bw) for x in xs], color=INK_2, linewidth=1.5, zorder=2)
    ridge = peak / bw

    # Measured ceilings (boost clocks) from the microbenchmarks, only if they were
    # measured on the same GPU model as the profile (Euler mixes GPU models).
    device = pts[0].get("device", "")
    latest = {}
    for r in micro:
        if device and r.get("device", "") != device:
            continue
        latest[r["test"]] = float(r["value"])
    other = sorted({r.get("device", "") for r in micro} - {device}) if device else []
    if micro and not latest:
        print("roofline: measured ceilings not drawn: micro.csv is from %s, the profile from %s"
              % (", ".join(other) or "an unknown GPU", device or "an unknown GPU"))
    m_bw = max([latest[t] for t in ("dram_read", "dram_copy") if t in latest] or [0])
    m_fl = latest.get("fp32_fma", 0)
    if m_bw and m_fl:
        ax.plot(xs, [min(m_fl, x * m_bw) for x in xs], color=MUTED, linewidth=1, zorder=1)

    handles, seen, labels = [], set(), []
    for r in pts:
        v = r["version"]
        color = SERIES[SLOT[v]] if v in SLOT else INK_2
        x, y = float(r["ai_flop_per_byte"]), float(r["achieved_gflops"])
        h = ax.scatter([x], [y], s=90, color=color, edgecolors=SURFACE, linewidths=2, zorder=4,
                       label=NAME.get(v, v))
        if v not in seen:
            handles.append(h)
            seen.add(v)
        labels.append(((x, y), "%s\n%.0f GFLOP/s, %.0f%% of roof"
                       % (kernel_label(v, r["kernel"]), y, float(r["pct_of_roof"]))))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    lows = [float(r["achieved_gflops"]) for r in pts]
    ax.set_ylim(10 ** math.floor(math.log10(min(lows + [xmin * bw]) / 3)), peak * 4)
    log_axis(ax.xaxis)
    log_axis(ax.yaxis)
    label_axes(ax, "Arithmetic intensity (FLOP per DRAM byte)", "GFLOP/s")
    x_lab = xmin * 2
    label_along(ax, (xmin, xmin * bw), (ridge, peak), "DRAM %.0f GB/s" % bw, (x_lab, x_lab * bw), INK_2)
    # Ceiling labels live in the right margin, beside the line ends, where no data can be.
    # The profiled roof's label hangs below its line; the measured one sits above its own.
    ax.annotate("FP32 peak %s\nridge %.0f FLOP/byte" % (flops_text(peak), ridge), (xmax, peak),
                xytext=(8, 0), textcoords="offset points", fontsize=9.5, color=INK_2,
                ha="left", va="top", annotation_clip=False, linespacing=1.3)
    if m_bw and m_fl:
        ax.annotate("measured (boost clocks)\n%s, %.0f GB/s" % (flops_text(m_fl), m_bw),
                    (xmax, max(m_fl, peak)), xytext=(8, 2), textcoords="offset points",
                    fontsize=8.5, color=MUTED, ha="left", va="bottom", annotation_clip=False,
                    linespacing=1.3)
    place_point_labels(fig, ax, labels)
    if len(handles) >= 2:
        legend(ax, handles, loc="lower right")
    save(fig, os.path.join(out, "roofline.png"))


def plot_scaling(rows, k, batch, out):
    by_t = {}
    for r in rows:
        m = re.match(r"^v2-t(\d+)$", r["version"])
        if m and int(r["k"]) == k:
            by_t.setdefault(int(r["batch"]), {})[int(m.group(1))] = float(r["latency_ms"])
    if not by_t:
        print("scaling: no v2-t<threads> rows in benchmark.csv, skipped "
              "(run bench_v2 --version v2-t<N> with OMP_NUM_THREADS=<N>)")
        return
    if batch is None:
        batch = max(by_t, key=lambda b: (len(by_t[b]), b))
    if batch not in by_t:
        print("scaling: no v2-t rows at batch %d, skipped" % batch)
        return
    lat = by_t[batch]
    threads = sorted(lat)
    t0 = threads[0]
    speedup = [lat[t0] / lat[t] * t0 for t in threads]  # relative to 1 thread (scaled if t0 > 1)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    fig.subplots_adjust(top=0.84)
    style(fig, ax, "v2 OpenMP speedup vs. threads",
          "SIFT1M, batch %d, top-%d; speedup over %d thread%s." % (batch, k, t0, "" if t0 == 1 else "s")
          + ("" if t0 == 1 else " (scaled to 1-thread units)"))
    ax.plot(threads, threads, color=MUTED, linewidth=1, zorder=2)
    ax.annotate("ideal", (threads[-1], threads[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9.5, color=MUTED)
    line_with_markers(ax, threads, speedup, SERIES[SLOT["v2"]], NAME["v2"])
    ax.annotate("%.1fx at %d threads" % (speedup[-1], threads[-1]), (threads[-1], speedup[-1]),
                xytext=(6, -12), textcoords="offset points", va="center", fontsize=9.5, color=INK)
    ax.set_xlim(0, threads[-1] * 1.15)
    ax.set_ylim(0, max(threads[-1], max(speedup)) * 1.1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
    label_axes(ax, "OpenMP threads", "Speedup")
    save(fig, os.path.join(out, "scaling.png"))


def plot_gemm(rows, out):
    pts = {}
    for r in rows:
        pts.setdefault(r["precision"], {})[int(r["batch"])] = float(r["gflops"])
    if not pts:
        print("gemm: no results/cublas_gemm.csv rows, skipped")
        return
    names = {"fp32": "cuBLAS FP32 (SGEMM)", "fp16": "cuBLAS FP16, FP32 accumulate (Tensor Cores)"}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(top=0.84)
    style(fig, ax, "cuBLAS GEMM throughput vs. batch size",
          "C = Q x X^T with X = 1M x 128 (the distance GEMM). Kernel time only; no transfers or top-k.")
    handles, ends = [], []
    for i, prec in enumerate(p for p in ("fp32", "fp16") if p in pts):
        xs = sorted(pts[prec])
        ys = [pts[prec][x] for x in xs]
        handles.append(line_with_markers(ax, xs, ys, SERIES[i], names[prec]))
        ends.append((xs, ys, prec.upper()))
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_axis(ax.xaxis)
    ally = [y for _, ys, _ in ends for y in ys]
    log_axis(ax.yaxis, min(ally), max(ally))
    label_axes(ax, "Queries per batch (B)", "GFLOP/s")
    if len(handles) >= 2:
        legend(ax, handles)
    right = ax.get_xlim()[1]
    ax.set_xlim(right=max(max(xs) for xs, _, _ in ends) * 4)
    if not end_labels(fig, ax, ends):
        ax.set_xlim(right=right)
    save(fig, os.path.join(out, "gemm.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="default: <results>/plots")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--scaling-batch", type=int, default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "plots")
    os.makedirs(out, exist_ok=True)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    bench = read_csv(os.path.join(args.results, "benchmark.csv"))
    plot_throughput(bench, args.k, out)
    plot_roofline(read_csv(os.path.join(args.results, "roofline.csv")),
                  read_csv(os.path.join(args.results, "micro.csv")), out)
    plot_scaling(bench, args.k, args.scaling_batch, out)
    plot_gemm(read_csv(os.path.join(args.results, "cublas_gemm.csv")), out)


if __name__ == "__main__":
    main()
