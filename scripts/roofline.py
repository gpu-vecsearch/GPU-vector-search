"""Turn Nsight Compute raw CSVs into roofline points.

Reads every results/ncu/*_raw.csv written by slurm/ncu.sbatch (or the files
given on the command line), aggregates launches of the same kernel, and writes
results/roofline.csv with one row per (profile, kernel):

  arithmetic intensity  = FP32 FLOPs / DRAM bytes   (FLOPs = 2*FFMA + FADD + FMUL)
  achieved GFLOP/s      = FLOPs / kernel time
  roof at that AI       = min(peak GFLOP/s, AI * peak DRAM GB/s)

Peaks come from the same ncu run (FFMA/cycle * SM clock, DRAM bytes/cycle *
DRAM clock), so they match the clocks ncu profiled at.

Usage: python3 scripts/roofline.py [RAW_CSV ...] [--out results/roofline.csv]
Plain Python 3.6+, no third-party packages.
"""

import csv
import glob
import os
import re
import sys

TIME = "gpu__time_duration.sum"
DRAM_R = "dram__bytes_read.sum"
DRAM_W = "dram__bytes_write.sum"
L2 = "lts__t_bytes.sum"
FFMA = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
FADD = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
FMUL = "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
PEAK_FFMA = "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"
SM_CLK = "sm__cycles_elapsed.avg.per_second"
PEAK_DRAM = "dram__bytes.sum.peak_sustained"
DRAM_CLK = "dram__cycles_elapsed.avg.per_second"
SM_PCT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM_PCT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"

SUMMED = [TIME, DRAM_R, DRAM_W, L2, FFMA, FADD, FMUL]
PREFIX = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
          "m": 1e-3, "u": 1e-6, "n": 1e-9}
BASE_UNITS = ("byte", "second", "cycle", "inst", "Hz", "sector", "warp", "thread")

OUT_FIELDS = ["version", "profile", "kernel", "launches", "time_ms", "gflop", "dram_gb",
              "ai_flop_per_byte", "achieved_gflops", "achieved_dram_gbs",
              "peak_gflops", "peak_dram_gbs", "ridge_ai", "bound", "pct_of_roof",
              "l2_gb", "ai_l2", "sm_pct_of_peak", "dram_pct_of_peak"]


def unit_scale(unit):
    """Factor converting a value in `unit` to base SI units (s, byte, 1/s)."""
    unit = unit.strip()
    if not unit or unit == "%":
        return 1.0
    if "/" in unit:
        num, den = unit.split("/", 1)
        return unit_scale(num) / unit_scale(den)
    for base in BASE_UNITS:
        if unit.endswith(base):
            prefix = unit[: -len(base)]
            if prefix in PREFIX:
                return PREFIX[prefix]
    return 1.0


def parse_number(text):
    text = text.strip().replace(",", "")
    if not text or text.lower() in ("n/a", "nan"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def short_kernel_name(name):
    name = name.replace("(anonymous namespace)::", "")
    name = re.sub(r"\(.*\)\s*$", "", name)  # drop the parameter list
    name = re.sub(r"^void\s+", "", name)       # template kernels carry a return type
    return name.strip()


def read_ncu_raw(path):
    """Yield one {metric: value-in-base-units, 'kernel': name} dict per launch."""
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    # ncu may print ==PROF== lines before the CSV; the header starts with "ID".
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if start is None:
        raise ValueError("no ncu CSV header found in %s" % path)
    rows = list(csv.reader(lines[start:]))
    header, units, data = rows[0], rows[1], rows[2:]
    # With --print-units base the second row holds units; detect it by its ID column.
    if units and units[0].strip() not in ("",):
        data = rows[1:]
        units = [""] * len(header)
    for row in data:
        if len(row) != len(header):
            continue
        rec = {"kernel": short_kernel_name(row[header.index("Kernel Name")])}
        for col, unit, value in zip(header, units, row):
            num = parse_number(value)
            if num is not None and "__" in col:
                rec[col] = num * unit_scale(unit)
        yield rec


def summarize(path):
    profile = os.path.basename(path)
    if profile.endswith("_raw.csv"):
        profile = profile[: -len("_raw.csv")]
    version = profile.split("_", 1)[0]

    kernels = {}
    order = []
    for rec in read_ncu_raw(path):
        k = rec["kernel"]
        if k not in kernels:
            kernels[k] = {"launches": 0, "peaks": rec, "pct": []}
            for m in SUMMED:
                kernels[k][m] = 0.0
            order.append(k)
        agg = kernels[k]
        agg["launches"] += 1
        for m in SUMMED:
            agg[m] += rec.get(m, 0.0)
        agg["pct"].append((rec.get(SM_PCT), rec.get(DRAM_PCT)))

    out = []
    for k in order:
        agg = kernels[k]
        p = agg["peaks"]
        t = agg[TIME]
        flops = 2.0 * agg[FFMA] + agg[FADD] + agg[FMUL]
        dram = agg[DRAM_R] + agg[DRAM_W]
        peak_gflops = 2.0 * p.get(PEAK_FFMA, 0.0) * p.get(SM_CLK, 0.0) / 1e9
        peak_gbs = p.get(PEAK_DRAM, 0.0) * p.get(DRAM_CLK, 0.0) / 1e9
        ai = flops / dram if dram else float("inf")
        achieved = flops / t / 1e9 if t else 0.0
        ridge = peak_gflops / peak_gbs if peak_gbs else 0.0
        roof = min(peak_gflops, ai * peak_gbs) if peak_gbs else peak_gflops
        sm_pct = [s for s, _ in agg["pct"] if s is not None]
        dram_pct = [d for _, d in agg["pct"] if d is not None]
        out.append({
            "version": version,
            "profile": profile,
            "kernel": k,
            "launches": agg["launches"],
            "time_ms": t * 1e3,
            "gflop": flops / 1e9,
            "dram_gb": dram / 1e9,
            "ai_flop_per_byte": ai,
            "achieved_gflops": achieved,
            "achieved_dram_gbs": dram / t / 1e9 if t else 0.0,
            "peak_gflops": peak_gflops,
            "peak_dram_gbs": peak_gbs,
            "ridge_ai": ridge,
            "bound": "memory" if ai < ridge else "compute",
            "pct_of_roof": 100.0 * achieved / roof if roof else 0.0,
            "l2_gb": agg[L2] / 1e9,
            "ai_l2": flops / agg[L2] if agg[L2] else float("inf"),
            "sm_pct_of_peak": sum(sm_pct) / len(sm_pct) if sm_pct else float("nan"),
            "dram_pct_of_peak": sum(dram_pct) / len(dram_pct) if dram_pct else float("nan"),
        })
    return out


def fmt(v):
    if isinstance(v, float):
        return "%.4g" % v
    return str(v)


def main(argv):
    out_path = "results/roofline.csv"
    paths = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out_path = argv[i + 1]
            i += 2
        elif argv[i] in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            paths.append(argv[i])
            i += 1
    if not paths:
        paths = sorted(glob.glob("results/ncu/*_raw.csv"))
    if not paths:
        print("no ncu raw CSVs found (expected results/ncu/*_raw.csv)", file=sys.stderr)
        return 1

    rows = []
    for path in paths:
        rows.extend(summarize(path))

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})

    cols = ["profile", "kernel", "time_ms", "ai_flop_per_byte", "achieved_gflops",
            "achieved_dram_gbs", "bound", "pct_of_roof"]
    print("  ".join("%-18s" % c for c in cols))
    for r in rows:
        print("  ".join("%-18s" % fmt(r[c])[:18] for c in cols))
    if rows:
        print("peaks: %.0f GFLOP/s FP32, %.0f GB/s DRAM, ridge %.1f FLOP/byte"
              % (rows[0]["peak_gflops"], rows[0]["peak_dram_gbs"], rows[0]["ridge_ai"]))
    print("wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
