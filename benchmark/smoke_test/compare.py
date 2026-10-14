"""Summarise smoke-test runs: how stable is each method across repeats?

Usage:  python compare.py <results_folder>
The folder holds fashion_mnist_<method>_run<N>/ sub-folders.
"""
import glob, json, os, statistics, sys

def load(d):
    met = json.load(open(f"{d}/metrics.json"))["dataset"]["metrics"]
    fm = json.load(open(f"{d}/faiss_metrics.json"))
    hw = json.load(open(f"{d}/hardware_metrics.json"))
    return dict(recall=met["recall@k"]["1"], qps=met["qps"],
                setup_s=fm["trainingTimeNs"] / 1e9 + fm["buildTimeMs"] / 1e3,
                threads=fm.get("numThreadsUsed"),
                cpu=(hw.get("cpu") or {}).get("processUsagePercent", {}).get("mean"),
                temp=((hw.get("power") or {}).get("temperatureC") or {}).get("mean"),
                ram_gb=hw["memory"]["beforeMB"] / 1024)

def spread(vals):
    m = statistics.mean(vals)
    return (max(vals) - min(vals)) / m * 100 if m else 0.0

folder = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "2*")))[-1]
print(f"Results: {folder}\n")
print(f"{'method':6} {'runs':>4} {'threads':>7} | {'recall@1':>8} | {'QPS mean':>9} {'[min - max]':>21} {'spread':>6} | {'CPU %':>5} | {'setup s':>7} {'spread':>6} | verdict")
print("-" * 112)
notes = []
for m in ("flat", "ivf", "hnsw"):
    runs = []
    for d in sorted(glob.glob(f"{folder}/fashion_mnist_{m}_run*")):
        try: runs.append(load(d))
        except Exception as e: notes.append(f"skipped {os.path.basename(d)}: {e}")
    if not runs:
        print(f"{m:6} {0:>4} | no results"); continue
    q = [r["qps"] for r in runs]; s = [r["setup_s"] for r in runs]
    cpu = [r["cpu"] for r in runs if r["cpu"] is not None]
    th = sorted({r["threads"] for r in runs if r["threads"] is not None})
    qs = spread(q)
    verdict = "too few runs" if len(runs) < 3 else ("STABLE" if qs <= 15 else ("borderline" if qs <= 25 else "NOISY"))
    print(f"{m:6} {len(runs):>4} {','.join(map(str, th)) or '?':>7} | {statistics.mean(r['recall'] for r in runs):8.4f} | "
          f"{statistics.mean(q):9.1f} [{min(q):9.1f} - {max(q):9.1f}] {qs:5.0f}% | {statistics.mean(cpu) if cpu else float('nan'):5.0f} | "
          f"{statistics.mean(s):7.2f} {spread(s):5.0f}% | {verdict}")
    temps = [r["temp"] for r in runs if r["temp"] is not None]
    notes.append(f"{m}: CPU % per run {[round(c) for c in cpu]}, RAM in use before {min(r['ram_gb'] for r in runs):.1f}-{max(r['ram_gb'] for r in runs):.1f} GB"
                 + (f", battery {min(temps):.1f}-{max(temps):.1f} C" if temps else ""))
print("\n" + "\n".join(notes))
print("\nspread = (max - min) / mean across runs.  STABLE <= 15%, borderline <= 25%, NOISY above.")
print("CPU % = average CPU use of the app (100% = one full core). With 1 thread it should sit near 100%.")
print("setup = training + building.")
