#!/usr/bin/env python3
"""Compare traditional and agentic pipelines on the same questions through one litert/gpu engine.

    python3 agentic/compare_pipelines.py -d 9146d9a3 --samples 5

Ensures the on-device task data and FAISS index match --samples N (regenerating
and rebuilding on the phone when missing), then sweeps:

    pipeline A  traditional       retrieve top_k, answer once
    pipeline B  agentic           retrieve -> Evidence Investigator loop
                                 (QUERY / ANSWERABLE, <=3 rounds) -> answer
("traditional" is sent to the device as aug_method; Agentic.kt runs the simple
concatenation path for any value that is not none/agentic.)

Any agent failure fails that question (recorded in report.json, sweep continues).
"""
import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

import agent as A  # noqa: E402
from helpers import (  # noqa: E402
    HTTP_BASE,
    SDCARD_BASE,
    TASK_ROOT,
    ensure_server,
    prepare_dirs,
    pull,
    push_dir_files,
    restart_server,
)
from classes.common import TestType  # noqa: E402
from score import answer_of, em, f1, retrieval_hit  # noqa: E402
from utils.log import display_status  # noqa: E402

TRIVIA_DIR = f"{TASK_ROOT}/trivia_qa"
CLIENT_TASK_DIR = Path(__file__).resolve().parent.parent / "task_files" / "downstream_task"
PIPELINES = ["traditional", "agentic"]
REBUILD_HINT = "No cached index"

console = Console()

# ponytail: substring list is a crude proxy. Upgrade with a judge model if the
# abstention rate ever needs to be a headline number rather than a smoke check.
HEDGES = ("don't know", "do not know", "cannot", "can't", "unable", "not enough",
          "no information", "insufficient", "not specified", "not provided", "unclear",
          "not in evidence")


def ask(device, query, verbose=False):
    stream = A.stream_device(query)
    # default: consume silently (one result line from the sweep); the full
    # agent trace is always on disk as jsonl. --verbose renders it live.
    events = A.run(stream) if verbose else list(stream)
    errors = [e for e in events if e["step"] == "error"]
    # Save-or-fail, never raise: failed questions become rows, not dead sweeps.
    error = errors[-1].get("detail", "") if errors else None
    final = [e for e in events if e["detail"] == "final answer"]
    if not error and not final:
        error = "completed without a final answer"
    pred = answer_of(final[0]["output"]) if final and not error else ""
    # startswith: since the config-proof logging change the detail is
    # "generating · systemPrompt=..." — exact == would count 0 calls
    llm_calls = len([e for e in events if e["detail"].startswith("generating")])
    # one Retriever/end per retrieval attempt; the none pipeline still emits one but
    # flags it "skipped", so exclude that to report true retrieval calls.
    retrievals = len([e for e in events if e["agent"] == "Retriever"
                      and e["step"] == "end" and "skipped" not in e["detail"]])
    # device-measured first-token latency of the answer-generating call (0 if none)
    ttft = final[0].get("ttftMs", 0) if final and not error else 0
    span = events[-1]["tMs"] if events else 0
    return pred, llm_calls, retrievals, span, ttft, events, error


# ── device task data + index ───────────────────────────────────────────────────

def status_stream(snapshots=None):
    """Poll /status until the benchmark run finishes (copied from main.py:
    importing main pulls the whole HF/datasets stack even on fast-path runs).
    Every status seen is appended to snapshots for metadata."""
    errors = 0
    while True:
        try:
            r = requests.get(f"{HTTP_BASE}/status", timeout=5)
            r.raise_for_status()
            st = r.json()
            errors = 0
        except Exception:
            errors += 1
            if errors == 10:
                raise RuntimeError("server stopped answering /status")
            st = None
        if st is not None:
            if snapshots is not None:
                snapshots.append(st)
            yield st
            if st.get("overallState") in ("COMPLETED", "FAILED"):
                return
        time.sleep(1)


def index_cache_path(device):
    """Replicate CacheManager.buildTaskSignature + hashSignature client-side so
    we can stat the right faiss_<hash>.index without a device round-trip.

    Verified byte-identical against a known on-device hash (limit=7 run).
    """
    c = device_config(device)
    faiss, task, emb = c["rag_pipeline"]["faiss"], c["downstream_task"], c["rag_pipeline"]["embedding"]
    chunker = emb["chunker"]
    # FaissInnerConfig: Config.kt SerialNames, nulls dropped (explicitNulls=false)
    inner = {k: faiss["config"][k] for k in
             ("nprobe", "nlist", "num_training_vectors", "m", "ef_construction", "ef_search")
             if faiss["config"].get(k) is not None}
    # data-class field order from CacheManager.kt
    sig = {
        "faiss": {"method": faiss["method"], "metric": faiss["metric"],
                  "backend": faiss["backend"], "batchSize": faiss["batch_size"],
                  "config": inner},
        "task": {"name": task["name"], "samplingMethod": task["sampling_method"],
                 "corpusLimit": task["corpus_limit"], "seed": task["seed"],
                 "limit": task["limit"]},
        "embedding": {"modelName": emb["model_name"], "dtype": emb["dtype"],
                      "backend": emb["backend"]},
        "chunker": {"method": chunker["method"], "size": chunker["size"],
                    "overlapEnabled": chunker["overlap_enabled"],
                    "overlapSize": chunker["overlap_size"]},
    }
    h = hashlib.sha256(
        json.dumps(sig, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return f"{SDCARD_BASE}/task_files/cache/faiss_{h}.index"


def adb_stat(device, path):
    # one quoted string: a bare "|" in argv form is eaten by the device shell as a pipe
    out = subprocess.run(["adb", "-s", device, "shell", f"stat -c '%s|%Y' '{path}'"],
                         capture_output=True, text=True)
    if out.returncode != 0 or "|" not in out.stdout:
        return None
    size, mtime = out.stdout.strip().split("|", 1)
    return {"bytes": int(size), "mtime_epoch": int(mtime),
            "mtime": datetime.fromtimestamp(int(mtime), tz=timezone.utc).isoformat()}


def device_config(device):
    return pull("config.json", device, f"{SDCARD_BASE}/task_files")


def ensure_task_data(device, task, samples):
    """Make the phone's questions/references/documents match N samples.

    Returns {"count", "regenerated", "wall_ms"}; parse_task may sample fewer
    than asked so count is authoritative.
    """
    if samples is None:
        questions = pull("questions.json", device, f"{TASK_ROOT}/{task}")
        return {"count": len(questions), "regenerated": False, "wall_ms": 0}

    cfg = device_config(device)["downstream_task"]
    questions = pull("questions.json", device, f"{TASK_ROOT}/{task}", required=False)
    # name too: same N across tasks must regen, or the config/index stay on the
    # previous task while questions come from this task's folder
    if (questions is not None and len(questions) == samples
            and cfg.get("name") == task
            and cfg.get("limit") == samples and cfg.get("corpus_limit") == samples):
        console.print(f"  [green]✓[/] task data: {samples} samples already on device")
        return {"count": samples, "regenerated": False, "wall_ms": 0}

    t0 = time.time()
    console.print(f"  [yellow]…[/] regenerating [bold]{task}[/] with {samples} samples "
                  f"(HuggingFace download)")
    # imported lazily: datasets is heavy and the fast path never needs it
    from classes.downstream_task import DownstreamTask, DownstreamTaskName
    from classes.common import SamplingMethod
    from parse_downstream_task import parse_task

    down = DownstreamTask(
        name=DownstreamTaskName(task),
        sampling_method=SamplingMethod(cfg.get("sampling_method", "random")),
        # corpus = exactly the sampled items' paragraphs, so samples controls
        # both the questions asked and the index content
        corpus_limit=samples,
        limit=samples,
        seed=cfg.get("seed", 42),
    )
    parse_task(down, None, str(CLIENT_TASK_DIR))
    actual = down.limit

    prepare_dirs([
        "task_files",
        "task_files/downstream_task",
        f"task_files/downstream_task/{task}",
        "task_files/results",
        f"task_files/results/{task}",
    ])
    # helpers.adb_push_files prints a line per file and adb writes its own
    # progress to stderr; silence both to keep the sweep UI clean
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        push_dir_files(device, str(CLIENT_TASK_DIR / task),
                       f"{SDCARD_BASE}/task_files/downstream_task/{task}")

    def sync(c):
        c["downstream_task"]["name"] = task
        c["downstream_task"]["limit"] = actual
        # keep the device-side index signature equal to the corpus we just pushed
        c["downstream_task"]["corpus_limit"] = actual
    A.push_config(device, sync)

    wall_ms = int((time.time() - t0) * 1000)
    console.print(f"  [green]✓[/] pushed {actual} questions, {actual} references, "
                  f"documents + config (limit={actual})")
    return {"count": actual, "regenerated": True, "wall_ms": wall_ms}


def _index_cached(device):
    """True if the device serves the cached index (probe query succeeds)."""
    events = list(A.stream_device("What is the capital of France?"))
    return not any(REBUILD_HINT in e.get("detail", "") for e in events if e["step"] == "error")


def ensure_index(device, task, engine, backend):
    """Probe the cached FAISS index; rebuild it via one benchmark run if missing.

    Returns a metadata dict: source (cached/rebuilt), rebuild wall time, live
    index progress (documents/chunks) from the benchmark's /status snapshots,
    cache-file stat, and the device's faiss_metrics.json when the benchmark
    completed without crashing.

    shortcut: rebuild = full task benchmark, which also answers every question
    once with the LLM (waste ~N answers, and its eval pass can OOM the app —
    the index is written before eval starts, so we re-probe after a crash).
    Upgrade: an index-only CLI mode in TaskBenchmark.kt when that matters.
    """
    A.set_engine(device, engine, backend, None)
    # traditional hits retrievalFor() before any LLM call, so a missing index
    # fails instantly and for free; a present index costs one small probe answer.
    A.set_aug_method(device, "traditional")
    cache_file = index_cache_path(device)
    info = {"source": "cached", "rebuilt": False, "rebuild_wall_ms": None,
            "recovered_after_crash": False, "error": None, "progress": None,
            "device_faiss_metrics": None, "cache_file": cache_file}
    info.update(adb_stat(device, cache_file) or {})
    if _index_cached(device):
        console.print(f"  [green]✓[/] FAISS index for current config already cached"
                      + (f" ({info.get('bytes', 0) / 1024:.0f} KB)" if info.get("bytes") else ""))
        return info

    info["source"], info["rebuilt"] = "rebuilt", True
    console.print("  [yellow]…[/] FAISS index missing — rebuilding on device "
                  "(benchmark run, index + one eval pass)")
    # fresh process like main.py: the probe loaded the LLM into this server, and
    # benchmark eval OOMs when it stacks its own ONNX copy on top
    restart_server(device)
    time.sleep(2)
    t0 = time.time()
    requests.post(f"{HTTP_BASE}/start_benchmark",
                  json={"test_type": TestType.TASK.value, "resume": False},
                  timeout=30).raise_for_status()
    snapshots = []
    try:
        display_status(status_stream(snapshots), TestType.TASK)
        r = requests.get(f"{HTTP_BASE}/status", timeout=5).json()
        if r.get("overallState") == "FAILED":
            info["error"] = f"index rebuild failed: {r.get('lastError')}"
            info["rebuild_wall_ms"] = int((time.time() - t0) * 1000)
            return info
        # full completion: the device also wrote its own metrics file
        metrics = pull("faiss_metrics.json", device,
                       f"{SDCARD_BASE}/task_files/results/{task}", required=False)
        if metrics is not None:
            info["device_faiss_metrics"] = metrics
    except RuntimeError:
        # app died mid-benchmark (usually eval-pass OOM); the index file is
        # written before eval, so restart and check rather than assume failure
        info["recovered_after_crash"] = True
        console.print("  [yellow]…[/] device server died during rebuild — restarting "
                      "and re-checking the index cache")
        restart_server(device)
        time.sleep(2)
    info["rebuild_wall_ms"] = int((time.time() - t0) * 1000)
    # keep the index-phase progress from the last live snapshot
    for st in reversed(snapshots):
        prog = (st.get("task") or {}).get("indexProgress")
        if prog:
            info["progress"] = prog
            break
    if _index_cached(device):
        info.update(adb_stat(device, cache_file) or {})
        chunks = (info.get("progress") or {}).get("chunksProcessed")
        extra = f" · {chunks} chunks" if chunks else ""
        console.print(f"  [green]✓[/] index built in {info['rebuild_wall_ms'] / 1000:.1f}s"
                      f"{extra} and cached")
    else:
        info["error"] = info["error"] or "index rebuild did not produce a usable index"
        console.print(f"  [red]✗[/] {info['error']}; check `adb logcat`")
    return info


def negatives(args):
    """Abstention smoke check: these questions are unanswerable from the hotpot
    corpus, so a correct pipeline should decline rather than invent an answer."""
    questions = pull("questions.json", args.device, TRIVIA_DIR)
    A.set_engine(args.device, args.engine, args.backend, None)
    A.set_aug_method(args.device, "agentic")
    console.print("[bold]negative control[/]: trivia_qa questions against the hotpot index\n")
    hedged = 0
    for i, q in enumerate(questions, 1):
        pred, calls, _ret, _ms, _ttft, events, error = ask(args.device, q, verbose=args.verbose)
        Path(args.save_dir, f"negative_q{i}.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n")
        low = pred.lower()
        hit = any(h in low for h in HEDGES)
        hedged += hit
        status = f"[red]FAIL {error}[/]" if error else f"abstain={str(hit):<5}"
        console.print(f"  Q{i} {status} calls={calls} -> {pred!r}")
    console.print(f"\nabstained {hedged}/{len(questions)} (higher is better; the corpus "
                  f"cannot answer any of these)")
    return 0


def sweep(args, questions, references):
    report, sweep_ms = {}, {}
    for pipeline in args.pipelines:
        aug = "agentic" if pipeline.startswith("agentic") else pipeline
        A.set_aug_method(args.device, aug)
        # Any agent failure fails the question instead of silently degrading.
        rows = []
        t0 = time.time()
        bar = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        )
        with bar:
            task_id = bar.add_task(pipeline, total=len(questions))
            for i, q in enumerate(questions, 1):
                golds = references[i - 1]
                pred, calls, retrievals, span, ttft, events, error = ask(
                    args.device, q, verbose=args.verbose)
                Path(args.save_dir, f"{pipeline}_q{i}.jsonl").write_text(
                    "\n".join(json.dumps(e) for e in events) + "\n")
                rows.append(dict(q=q, gold=golds, pred=pred, em=em(pred, golds),
                                 f1=f1(pred, golds),
                                 retrieval_hit=retrieval_hit(golds, events),
                                 failed=bool(error), error=error,
                                 calls=calls, retrievals=retrievals, ms=span, ttft=ttft))
                if error:
                    line = f"  [{pipeline}] Q{i} [red]FAIL {error}[/]"
                else:
                    # full pred is always in report.json + the qN.jsonl trace
                    disp = pred if len(pred) <= 90 else pred[:89] + "…"
                    line = (f"  [{pipeline}] Q{i} em={rows[-1]['em']} "
                            f"f1={rows[-1]['f1']:.2f} hit={rows[-1]['retrieval_hit']} "
                            f"calls={calls} retr={retrievals} "
                            f"{span}ms ttft={ttft}ms  pred={disp!r}")
                bar.console.print(line)
                bar.advance(task_id)
        sweep_ms[pipeline] = int((time.time() - t0) * 1000)
        report[pipeline] = rows
        # Incremental save: a crash mid-sweep keeps the finished arms.
        Path(args.save_dir, "report.json").write_text(json.dumps(report, indent=2))
    return report, sweep_ms


def summary(report):
    # acc% = EM rate; ttft = final-answer-call first-token mean (ms, answered only);
    # latency = end-to-end span (ms). JSON keys (em/failed/calls/...) stay stable.
    # hit% = retrieval_hit rate (gold answer present in retrieved chunks)
    table = Table(title="pipeline comparison")
    for col in ("pipeline", "EM", "acc%", "F1", "hit%", "failures", "LLM Calls",
                "retrievals", "latency", "ttft"):
        table.add_column(col, justify="right" if col != "pipeline" else "left")
    for pipeline, rows in report.items():
        n = len(rows)
        em_sum = sum(r["em"] for r in rows)
        ttfts = [r.get("ttft", 0) for r in rows if r.get("ttft", 0) > 0]
        ttft_cell = f"{sum(ttfts) / len(ttfts):.0f}" if ttfts else "-"
        table.add_row(
            pipeline, f"{em_sum}/{n}", f"{100 * em_sum / n:.1f}",
            f"{sum(r['f1'] for r in rows) / n:.2f}",
            f"{100 * sum(r.get('retrieval_hit', 0) for r in rows) / n:.1f}",
            str(sum(r["failed"] for r in rows)),
            f"{sum(r['calls'] for r in rows) / n:.1f}",
            f"{sum(r['retrievals'] for r in rows) / n:.1f}",
            f"{sum(r['ms'] for r in rows) / n:.0f}", ttft_cell,
        )
    console.print(table)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", "-d", required=True)
    p.add_argument("--engine", default="litert")
    p.add_argument("--backend", default="gpu")
    p.add_argument("--save-dir", default=None,
                   help="default: runs/compare_<task>_<samples>_<timestamp>")
    p.add_argument("--task", default="hotpot_qa",
                   help="downstream task dir on device, e.g. hotpot_qa or musique")
    p.add_argument("--pipelines", nargs="+", choices=PIPELINES, default=PIPELINES,
                   help="pipelines to compare (default: both arms)")
    p.add_argument("--samples", type=int, default=None,
                   help="ensure task data + FAISS index for N samples on the device "
                        "(regenerating/rebuilding if missing), then run all N")
    p.add_argument("--negatives", action="store_true",
                   help="ask trivia_qa questions against the hotpot corpus: unanswerable by construction")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="render the full agent trace live per question "
                        "(traces are always saved to jsonl regardless)")
    args = p.parse_args()

    # this phone stalls for tens of seconds at a time — give ensure_server
    # several attempts before declaring the device unusable
    for _ in range(6):
        if ensure_server(args.device) is not None:
            break
        time.sleep(4)
    else:
        sys.exit("device server unavailable")

    # agentic_pipeline_config.json drives both arms; arm-level flags (engine,
    # backend, aug) are applied later per pipeline and still win
    A.sync_llm_config(args.device)

    if args.save_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = args.task.replace("_", "")          # hotpot_qa -> hotpotqa
        samples_tag = str(args.samples) if args.samples else "all"
        args.save_dir = str(Path(__file__).parent / "runs" /
                            f"compare_{tag}_{samples_tag}_{stamp}")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if args.negatives:
        return negatives(args)

    console.rule(f"[bold]{args.task}[/] · device {args.device}")
    top_k = device_config(args.device)["rag_pipeline"]["faiss"]["top_k"]
    console.print(f"[dim]engine {args.engine}/{args.backend} · top_k={top_k} retrieved "
                  f"docs/query · samples={args.samples or 'all on device'}[/]")
    t_start = time.time()
    console.print("[bold cyan]1/3[/] task data")
    data_info = ensure_task_data(args.device, args.task, args.samples)

    console.print("[bold cyan]2/3[/] FAISS index")
    t_idx = time.time()
    index_info = ensure_index(args.device, args.task, args.engine, args.backend)
    index_ensure_ms = int((time.time() - t_idx) * 1000)

    def write_metadata(extra=None):
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": args.device,
            "task": args.task,
            "samples": args.samples,
            "engine": args.engine,
            "backend": args.backend,
            "top_k": top_k,
            "pipelines": args.pipelines,
            "questions": data_info["count"],
            "data": data_info,
            "index": index_info,
            "save_dir": str(args.save_dir),
        }
        if extra:
            meta.update(extra)
        Path(args.save_dir, "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta

    if index_info.get("error"):
        write_metadata()
        sys.exit(index_info["error"])

    console.print(f"[bold cyan]3/3[/] pipelines on {data_info['count']} questions")
    questions = pull("questions.json", args.device, f"{TASK_ROOT}/{args.task}")
    references = pull("references.json", args.device, f"{TASK_ROOT}/{args.task}")
    if args.samples is not None:
        n = data_info["count"]
        questions, references = questions[:n], references[:n]
    if not questions or len(questions) != len(references):
        sys.exit(f"invalid task data: {len(questions)} questions, {len(references)} references")

    report, sweep_ms = sweep(args, questions, references)
    summary(report)
    write_metadata({
        "phases": {
            "data_prep_ms": data_info["wall_ms"],
            "index_ensure_ms": index_ensure_ms,
            "index_rebuild_wall_ms": index_info.get("rebuild_wall_ms"),
            "sweep_ms": sweep_ms,
            "total_ms": int((time.time() - t_start) * 1000),
        },
    })
    console.print(f"\nrows -> [bold]{args.save_dir}/report.json[/]")
    console.print(f"meta -> [bold]{args.save_dir}/metadata.json[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
