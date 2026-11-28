"""FAISS GPU exact-search baseline on SIFT1M (comparison only, not a version).

Uses faiss.GpuIndexFlatL2 (exact, FP32) and appends rows to results/benchmark.csv
in the same format as bench/bench.cpp, under two labels:

  faiss-gpu-resident  database added to the GPU index once; each timed call is
                      index.search(queries) only. How FAISS is used in practice.
  faiss-gpu-e2e       each timed call creates the index, copies the 1M-vector
                      database to the GPU and searches. Same work as our search()
                      interface, which copies the database on every call.

Latency is the median of --repeat timed calls after one untimed warm-up.
Recall is plain recall@k against the SIFT1M ground truth; the log also prints
tie-aware recall (as bench does).

Setup: bash scripts/setup_faiss.sh (once, on a login node).
Usage: python3 baselines/faiss_flat.py [--batches 1,10,100,1000,10000] [--k 10]
                                       [--data data/sift] [--repeat 5] [--csv PATH | --no-csv]
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("faiss is not installed in this Python; run: bash scripts/setup_faiss.sh")

HEADER = "version,batch,k,latency_ms,throughput_qps,recall_at_k"


def read_vecs(path, dtype, max_rows=None):
    """Read a TEXMEX .fvecs/.ivecs file: each record is int32 dim, then dim 4-byte values."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError("empty file: " + path)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    if not (rows[:, 0] == dim).all():
        raise ValueError("inconsistent dimensions in " + path)
    if max_rows is not None:
        rows = rows[:max_rows]
    return np.ascontiguousarray(rows[:, 1:]).view(dtype)


def recall_at_k(found, truth, k):
    hits = (found[:, :, None] == truth[:, None, :k]).any(axis=2).sum()
    return hits / float(found.shape[0] * k)


def tie_aware(found, truth, base, queries, k):
    """Count a returned id correct if it is no farther (exact, float64) than the k-th true neighbor."""
    q = queries.astype(np.float64)
    kth = ((q - base[truth[:, k - 1]].astype(np.float64)) ** 2).sum(axis=1)
    ids = np.clip(found, 0, base.shape[0] - 1)
    dist = ((q[:, None, :] - base[ids].astype(np.float64)) ** 2).sum(axis=2)
    ok = (dist <= kth[:, None]) & (found >= 0)
    in_truth = (found[:, :, None] == truth[:, None, :k]).any(axis=2)
    misses = int((~in_truth).sum())
    tied = int(((~in_truth) & (dist == kth[:, None])).sum())
    farther = int(((~in_truth) & (dist > kth[:, None])).sum())
    return ok.sum() / float(ok.size), misses, tied, farther


def append_row(path, row):
    empty, newline = True, True
    if os.path.exists(path) and os.path.getsize(path) > 0:
        empty = False
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            newline = f.read(1) == b"\n"
    with open(path, "a") as f:
        if empty:
            f.write(HEADER + "\n")
        elif not newline:
            f.write("\n")
        f.write(row + "\n")


def timed(fn, repeat):
    fn()  # warm-up
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", default="1,10,100,1000,10000")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--data", default="data/sift")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--csv", default="results/benchmark.csv")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--modes", default="resident,e2e", help="comma list of: resident, e2e")
    args = ap.parse_args()

    if faiss.get_num_gpus() < 1:
        sys.exit("FAISS sees no GPU (run inside a GPU job, and check the faiss-gpu install)")

    t0 = time.perf_counter()
    base = read_vecs(os.path.join(args.data, "sift_base.fvecs"), np.float32)
    queries_all = read_vecs(os.path.join(args.data, "sift_query.fvecs"), np.float32)
    truth_all = read_vecs(os.path.join(args.data, "sift_groundtruth.ivecs"), np.int32)
    d = base.shape[1]
    print("FAISS %s, N=%d d=%d, loaded in %.0f ms" % (faiss.__version__, base.shape[0], d,
                                                      (time.perf_counter() - t0) * 1e3))
    if args.k > truth_all.shape[1]:
        sys.exit("--k exceeds ground-truth depth %d" % truth_all.shape[1])

    res = faiss.StandardGpuResources()
    resident = faiss.GpuIndexFlatL2(res, d)
    t0 = time.perf_counter()
    resident.add(base)
    print("resident index: added database in %.0f ms" % ((time.perf_counter() - t0) * 1e3))

    def e2e_search(q):
        index = faiss.GpuIndexFlatL2(res, d)
        index.add(base)
        out = index.search(q, args.k)
        del index
        return out

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for batch in [int(b) for b in args.batches.split(",") if b.strip()]:
        if batch < 1 or batch > queries_all.shape[0]:
            print("  skip batch %d (have %d queries)" % (batch, queries_all.shape[0]))
            continue
        q = np.ascontiguousarray(queries_all[:batch])
        truth = truth_all[:batch]
        for mode in modes:
            if mode == "resident":
                fn = lambda: resident.search(q, args.k)
            elif mode == "e2e":
                fn = lambda: e2e_search(q)
            else:
                sys.exit("unknown mode " + mode)
            ms, (dist, ids) = timed(fn, args.repeat)
            qps = batch / (ms / 1e3)
            rec = recall_at_k(ids, truth, args.k)
            tie_rec, misses, tied, farther = tie_aware(ids, truth, base, q, args.k)
            label = "faiss-gpu-" + mode
            print("[%s] batch=%d k=%d  latency %.3f ms  %.1f queries/s  recall@%d %.4f  "
                  "tie-aware %.4f (misses %d: %d tied, %d farther)"
                  % (label, batch, args.k, ms, qps, args.k, rec, tie_rec, misses, tied, farther))
            if not args.no_csv:
                append_row(args.csv, "%s,%d,%d,%.3f,%.1f,%.4f" % (label, batch, args.k, ms, qps, rec))
    if not args.no_csv:
        print("appended to %s" % args.csv)


if __name__ == "__main__":
    main()
