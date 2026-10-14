"""Summarise smoke-test runs: how stable is each method across repeats, and did CPU speed explain the changes?

Usage:  python compare.py [results_folder]      (default: newest folder in results/)
"""
import glob, json, os, statistics, sys

def med(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return statistics.median(xs) if xs else None

def load(d):
    met = json.load(open(f"{d}/metrics.json"))["dataset"]["metrics"]
    fm = json.load(open(f"{d}/faiss_metrics.json"))
    hw = json.load(open(f"{d}/hardware_metrics.json"))
    return dict(
        name=os.path.basename(d),
        recall=met["recall@k"]["1"],
        med_ms=met["latency"]["median_ns"] / 1e6,
        qps=met["qps"],
        setup_s=fm["trainingTimeNs"] / 1e9 + fm["buildTimeMs"] / 1e3,
        threads=fm.get("numThreadsUsed"),
        q_mhz=med(fm.get("queryBatchMHz") or []),
        b_mhz=med(fm.get("buildBatchMHz") or []),
        cores=fm.get("cpusObservedDuringQueries") or {},
        temp=((hw.get("power") or {}).get("temperatureC") or {}).get("mean"),
    )

def spread(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2: return None
    m = statistics.mean(vals)
    return (max(vals) - min(vals)) / m * 100 if m else None

def pct(v): return f"{v:5.0f}%" if v is not None else "   n/a"

folder = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "2*")))[-1]
print(f"Results: {folder}\n")

all_runs = {}
for m in ("flat", "ivf", "hnsw"):
    runs = []
    for d in sorted(glob.glob(f"{folder}/fashion_mnist_{m}_run*"), key=lambda p: int(p.rsplit("run", 1)[1])):
        try: runs.append(load(d))
        except Exception as e: print(f"skipped {os.path.basename(d)}: {e}")
    all_runs[m] = runs

# ---- 1. per-run detail ----
print("Per run (in the order they ran):")
print(f"  {'run':26} {'ms/query':>9} {'query MHz':>9} {'setup s':>8} {'build MHz':>9} {'temp C':>6}  cores used")
for m, runs in all_runs.items():
    for r in runs:
        qm = f"{r['q_mhz']:.0f}" if r["q_mhz"] else "n/a"; bm = f"{r['b_mhz']:.0f}" if r["b_mhz"] else "n/a"
        t = f"{r['temp']:.1f}" if r["temp"] is not None else "n/a"
        print(f"  {r['name']:26} {r['med_ms']:9.4f} {qm:>9} {r['setup_s']:8.2f} {bm:>9} {t:>6}  {r['cores']}")

# ---- 2. summary ----
print(f"\n{'method':6} {'runs':>4} {'thr':>3} | {'recall@1':>8} | {'median ms/query':>15} {'spread':>6} | {'setup s':>7} {'spread':>6} | {'MHz-adjusted spread':>19} | verdict")
print("-" * 104)
for m, runs in all_runs.items():
    if not runs:
        print(f"{m:6} {0:>4} | no results"); continue
    md = [r["med_ms"] for r in runs]; su = [r["setup_s"] for r in runs]
    # time x clock speed = CPU cycles per query; if this is stable while time is not, the phone's speed changes explain the noise
    cyc = [r["med_ms"] * r["q_mhz"] for r in runs if r["q_mhz"]]
    th = sorted({r["threads"] for r in runs if r["threads"] is not None})
    s_md = spread(md)
    verdict = "too few runs" if len(runs) < 3 else ("STABLE" if s_md <= 15 else ("borderline" if s_md <= 25 else "NOISY"))
    print(f"{m:6} {len(runs):>4} {','.join(map(str, th)) or '?':>3} | {statistics.mean(r['recall'] for r in runs):8.4f} | "
          f"{statistics.median(md):15.4f} {pct(s_md)} | {statistics.median(su):7.2f} {pct(spread(su))} | {pct(spread(cyc) if len(cyc) == len(runs) else None):>19} | {verdict}")

print("""
ms/query       = median time per query in that run (ignores rare slow batches).
query/build MHz = clock speed of the core FAISS ran on, read right after each batch (median per run).
spread         = (max - min) / mean across runs.  STABLE <= 15%, borderline <= 25%, NOISY above.
MHz-adjusted   = spread of (ms/query x MHz). If this is much lower than the plain spread,
                 the changes came from the phone changing its CPU speed, not from the method or the code.""")
