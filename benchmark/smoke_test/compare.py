"""Summarise smoke-test runs, grouped by the type of CPU core each query batch ran on.

Usage:  python compare.py [results_folder]      (default: newest folder in results/)

Phones mix fast and slow cores. Android decides where the benchmark runs, and a slow
"little" core is 3-4x slower. So the timings are split by core type: the main result is
the core type most batches ran on, and anything else is reported next to it.
"""
import glob, json, os, statistics, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "client"))
from utils.ann_recall import tie_aware_recall
from collections import defaultdict

def cluster_names(freqs_khz):
    """Map core id -> readable core type, from each core's max frequency."""
    if not freqs_khz or any(f <= 0 for f in freqs_khz):
        return {}
    levels = sorted(set(freqs_khz), reverse=True)
    names = {}
    for rank, f in enumerate(levels):
        label = "fastest" if rank == 0 else ("slowest" if rank == len(levels) - 1 else f"tier{rank + 1}")
        for c, fc in enumerate(freqs_khz):
            if fc == f:
                names[c] = f"{label} ({f // 1000} MHz)"
    return names

def load(d):
    fm = json.load(open(f"{d}/faiss_metrics.json"))
    met = json.load(open(f"{d}/metrics.json"))["dataset"]["metrics"]
    cfg = json.load(open(f"{d}/config.json"))
    hw = json.load(open(f"{d}/hardware_metrics.json"))
    items = json.load(open(f"{d}/results.json"))["items"]
    bs = cfg["faiss"]["batch_size"]
    batch_ms = [items[i]["retrievalLatencyNs"] / 1e6 for i in range(0, len(items), bs)]
    cores = fm.get("queryBatchCores") or []
    names = cluster_names(fm.get("coreMaxFreqsKHz") or [])
    by = defaultdict(list)
    for ms, c in zip(batch_ms, cores):
        by[names.get(c, f"core {c}")].append(ms)
    # tie-aware recall: from metrics.json if present, otherwise computed from the saved files
    tar = met.get("tie_aware_recall@k")
    if not tar:
        ds_dir = os.path.join(ROOT, "client", "ann_files", "ann_dataset", cfg["ann_dataset"]["name"])
        try:
            meta = json.load(open(f"{ds_dir}/meta.json"))
            K = len(items[0]["neighbors"])
            nb = np.array([it["neighbors"] if it.get("neighbors") else [-1] * K for it in items])
            tar = tie_aware_recall(nb, ds_dir, meta, angular=(cfg["faiss"]["metric"] == "IP"), ks=(1, K))
        except Exception as e:
            tar = {}
    std = met.get("standard_recall@k") or {}
    kmax = max(std, key=lambda x: int(x)) if std else None
    kk = max((int(x) for x in tar if x.isdigit()), default=None)
    return dict(name=os.path.basename(d), recall=met["recall@k"]["1"],
                tar1=tar.get("1"), tark=tar.get(str(kk)) if kk else None,
                std_recall=std.get(kmax) if kmax else None, k=kmax,
                total_setup_s=(fm.get("totalSetupTimeMs") or 0) / 1000,
                index_mb=fm.get("indexMemoryMB"), peak_extra_mb=fm.get("buildPeakExtraMB"),
                file_mb=fm.get("indexFileMB"), load_s=(fm.get("loadTimeNs") or 0) / 1e9,
                ef=(cfg["faiss"]["config"] or {}).get("ef_search"),
                source=json.load(open(f"{d}/overall.json")).get("indexSource"),
                med_ms=statistics.median(batch_ms), by=dict(by), n=len(batch_ms),
                setup_s=fm["trainingTimeNs"] / 1e9 + fm["buildTimeMs"] / 1e3,
                cpuset=fm.get("cpusetAtQueryStart"), cpuset_end=fm.get("cpusetAtQueryEnd"),
                temp=((hw.get("power") or {}).get("temperatureC") or {}).get("mean"))

def spread(vals):
    if len(vals) < 2: return None
    m = statistics.mean(vals)
    return (max(vals) - min(vals)) / m * 100 if m else None

def pct(v): return f"{v:4.0f}%" if v is not None else "  n/a"

folder = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "2*")))[-1]
print(f"Results: {folder}\n")

data = {}
ORDER = ["flat", "ivf", "hnsw", "ivfpq", "binary", "binaryivf"]
# dataset name: from the run folders' config files (names like ny_times contain underscores)
DS = None
for cfgp in glob.glob(f"{folder}/*_run*/config.json"):
    DS = json.load(open(cfgp))["ann_dataset"]["name"]; break
DS = DS or "fashion_mnist"
print(f"Dataset: {DS}\n")
found = {os.path.basename(p)[len(DS) + 1:].rsplit("_run", 1)[0] for p in glob.glob(f"{folder}/{DS}_*_run*")}
for m in [x for x in ORDER if x in found] + sorted(found - set(ORDER)):
    runs = []
    for d in sorted(glob.glob(f"{folder}/{DS}_{m}_run*"), key=lambda p: int(p.rsplit("run", 1)[1])):
        try: runs.append(load(d))
        except Exception as e: print(f"skipped {os.path.basename(d)}: {e}")
    data[m] = runs

# ---- per run ----
print("Per run: share of query batches on each core type (median ms/query there)")
for m, runs in data.items():
    for r in runs:
        parts = ", ".join(f"{k}: {len(v) * 100 // r['n']}% ({statistics.median(v):.4f} ms)" for k, v in sorted(r["by"].items()))
        cs = r["cpuset"] or "?"
        if r["cpuset_end"] and r["cpuset_end"] != r["cpuset"]: cs += f" -> {r['cpuset_end']}"
        print(f"  {r['name']:24} setup {r['setup_s']:6.2f}s  group {cs:14} | {parts}")

# ---- summary ----
print(f"\n{'method':6} {'runs':>4} | {'recall@1':>8} | {'main core type':24} {'batches there':>13} | {'ms/query':>8} {'spread':>6} | {'all batches spread':>18} | {'setup s':>7} {'spread':>6} | verdict")
print("-" * 132)
for m, runs in data.items():
    if not runs:
        print(f"{m:6} {0:>4} | no results"); continue
    pooled = defaultdict(int)
    for r in runs:
        for k, v in r["by"].items(): pooled[k] += len(v)
    total = sum(pooled.values())
    main = max(pooled, key=pooled.get) if pooled else "?"
    per_run = [statistics.median(r["by"][main]) for r in runs if len(r["by"].get(main, [])) >= 10]
    s_main = spread(per_run)
    s_all = spread([r["med_ms"] for r in runs])
    su = [r["setup_s"] for r in runs]
    share = pooled.get(main, 0) * 100 // total if total else 0
    if len(per_run) < 3: verdict = "too few runs on main core type"
    elif s_main <= 15 and share >= 90: verdict = "STABLE"
    elif s_main <= 15: verdict = f"stable on main type, but {100 - share}% elsewhere"
    else: verdict = "NOISY"
    print(f"{m:6} {len(runs):>4} | {statistics.mean(r['recall'] for r in runs):8.4f} | {main:24} {share:12}% | "
          f"{statistics.median(per_run) if per_run else float('nan'):8.4f} {pct(s_main)} | {pct(s_all):>18} | "
          f"{statistics.median(su):7.2f} {pct(spread(su))} | {verdict}")

# ---- the numbers you would report ----
# Two-phase runs: <ds>_<m>_build = one fresh build (setup cost, build memory, file size);
# <ds>_<m>_runN = queries on the saved index (speed, recall, memory when loaded).
# Older one-phase runs have no _build folder; then everything comes from the runs.
builds = {}
for m in data:
    bd = f"{folder}/{DS}_{m}_build"
    if os.path.exists(f"{bd}/metrics.json"):
        try: builds[m] = load(bd)
        except Exception as e: print(f"skipped {os.path.basename(bd)}: {e}")
two_phase = bool(builds)

print(f"\nResults to report ({'setup from one build; queries = medians over runs on the saved index' if two_phase else 'medians over runs'}):")
print(f"  {'method':9} | {'tie-aware r@1':>13} | {'tie-aware r@k':>13} | {'ms/query':>8} | {'total setup s':>13} | {'MB built':>8} | {'MB loaded':>9} | {'file MB':>7} | {'load s':>6}")
checks = []
for m, runs in data.items():
    if not runs: continue
    def mdn(key, rs=runs):
        v = [r[key] for r in rs if r.get(key) is not None]
        return statistics.median(v) if v else float("nan")
    k = runs[0]["k"] or "-"
    b = builds.get(m)
    setup = b["total_setup_s"] if b else mdn("total_setup_s")
    mb_built = b["index_mb"] if b else mdn("index_mb")
    mb_loaded = mdn("index_mb") if b else float("nan")
    file_mb = b["file_mb"] if b else mdn("file_mb")
    print(f"  {m:9} | {mdn('tar1'):13.4f} | {mdn('tark'):8.4f} @{str(k):<3} | {mdn('med_ms'):8.4f} | {setup:13.2f} | "
          f"{mb_built:8.1f} | {mb_loaded:9.1f} | {file_mb:7.1f} | {mdn('load_s'):6.2f}")
    if b:
        srcs = {str(r["source"]) for r in runs}
        same = all(abs((r["tar1"] or 0) - (b["tar1"] or 0)) < 0.002 for r in runs)
        checks.append(f"  {m:9} build source={b['source']}, query runs source={','.join(sorted(srcs))}, "
                      f"same recall as the fresh build: {'yes' if same else 'NO'}  "
                      f"| build-run query speed {b['med_ms']:.4f} ms vs cool runs {mdn('med_ms'):.4f} ms")
if checks:
    print("\nChecks (build must be SCRATCH, query runs CACHE with identical recall):")
    print("\n".join(checks))

# ---- cache check ----
cc = [load(d) for d in sorted(glob.glob(f"{folder}/cache_check_ef*")) if os.path.exists(f"{d}/metrics.json")]
if cc:
    print("\nCache check:")
    for r in cc:
        print(f"  ef_search={r['ef']:<4} index source={r['source']!s:8} total setup {r['total_setup_s']:6.2f}s  load {r['load_s']:5.2f}s  "
              f"std recall@{r['k']} {r['std_recall']:.4f}  ms/query {r['med_ms']:.4f}")
    if len(cc) == 2:
        a, b = sorted(cc, key=lambda r: r["ef"])
        ok_src = a["source"] == "SCRATCH" and b["source"] == "CACHE"
        ok_ef = b["std_recall"] >= a["std_recall"] and b["med_ms"] > a["med_ms"]
        print(f"  -> {'PASS' if ok_src else 'FAIL'}: second run {'loaded the cached index' if ok_src else 'did not load from cache'}; "
              f"{'PASS' if ok_ef else 'CHECK'}: higher ef_search {'gave higher/equal recall and slower queries' if ok_ef else 'did not behave as expected (ef_search may not be applied after loading)'}")

print("""
tie-aware r@k  = share of the top-k results that are at least as close as the true k-th neighbour
                 (ann-benchmarks convention; equally distant duplicates count as correct).
total setup    = time until the index is ready (create + train + add); IVF training included.
MB built       = native memory of the index right after building it on the phone (includes spare
                 room that growing arrays reserve); MB loaded = same index loaded from its file.
group          = Android priority group of the app during queries. "/top-app" means Android treated it as
                 the app in use (fast cores). "/foreground" or "/background" means lower priority.
main core type = the core type most query batches ran on, pooled over all runs of that method.
ms/query       = median per run on the main core type; spread = (max - min) / mean across runs.
all batches    = spread when core type is ignored (what the noise looks like without this split).
STABLE         = spread <= 15% on the main core type AND at least 90% of batches ran there.""")
