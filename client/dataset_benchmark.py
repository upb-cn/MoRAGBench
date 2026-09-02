#!/usr/bin/env python3
"""Minimal MoRAGBench - run questions through both backends, one at a time."""
import argparse
import csv
import sys
import time
from datetime import datetime

from datasets import load_dataset
from rich.console import Console
from rich.progress import track

from helpers import (
    LITERT_MODEL,
    ONNX_MODEL_DIR,
    SDCARD_BASE,
    _connect_refused,
    check_model_on_device,
    ensure_server,
    generate_post,
    set_engine,
)

console = Console()

DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "validation"),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc", "validation"),
}


def load_questions(dataset_name, n):
    path, subset, split = DATASETS[dataset_name]
    console.print(f"Loading {dataset_name} ({split} split)...")
    ds = load_dataset(path, subset, split=split, streaming=True)
    questions = []
    for item in ds:
        questions.append(item["question"])
        if len(questions) >= n:
            break
    console.print(f"Loaded {len(questions)} questions.")
    return questions


def run_one(device, engine, question, max_tokens):
    """Run single question; model stays resident across retries. None on failure."""
    try:
        r = generate_post("generate", {"prompt": question, "max_tokens": max_tokens})
    except Exception as e:
        console.print(f"  [yellow]{e}[/]")
        return None
    m = r.get("metrics", r)
    gen = m.get("generated_tokens", 0) or 0
    ttft = m.get("ttft_ms", 0) or 0
    dur = m.get("overall_duration_ms", 0) or 0
    speed = m.get("decoding_speed_tokens_per_sec", 0.0) or 0.0
    tbt = (dur - ttft) / (gen - 1) if gen > 1 else 0.0
    return {
        "question": question,
        "backend": engine,
        "ttft_ms": round(ttft, 2),
        "decode_speed_tps": round(speed, 2),
        "tbt_ms": round(tbt, 2),
        "overall_ms": round(dur, 2),
        "tokens": gen,
    }


def run_engine(device, questions, engine, max_tokens, csv_writer=False):
    """Run all questions for one engine, writing each result immediately."""
    console.print(f"\n[bold]Benchmarking {engine.upper()}...[/]")
    set_engine(device, engine)
    # Busy (timeout) is fine; only abort if the process is truly gone.
    if ensure_server(device) is None and _connect_refused():
        console.print("[red]Server is not running and could not be started.[/]")
        return 0

    # warmup
    try:
        generate_post("generate", {"prompt": "Hello", "max_tokens": 8})
    except Exception as e:
        console.print(f"[red]Warmup failed: {e}[/]")
        return 0
    console.print("  [green]Warmup OK.[/]")

    success = 0
    for idx, q in enumerate(track(questions, description=f"[cyan]{engine}[/]"), 1):
        # Periodic health check BETWEEN questions: recover only if the process is
        # truly gone; a busy (timeout) server is left to finish loading.
        if idx % 20 == 0 and _connect_refused():
            console.print("  [yellow]Server is down — recovering...[/]")
            if ensure_server(device) is None:
                console.print("  [red]Server recovery failed; skipping rest of run.[/]")
                break
        res = run_one(device, engine, q, max_tokens)
        if res:
            success += 1
            if csv_writer:
                csv_writer.writerow(res)
            console.print(f"  [green]OK[/] ({res['decode_speed_tps']:.1f} tok/s)")
        else:
            console.print(f"  [red]FAILED after retries[/]")
    return success


def print_summary(rows):
    if not rows:
        return
    from collections import defaultdict
    import statistics

    by_engine = defaultdict(list)
    for r in rows:
        by_engine[r["backend"]].append(r)

    console.print(f"\n{'='*60}")
    console.print("RESULTS SUMMARY")
    console.print(f"{'='*60}")
    for engine, runs in by_engine.items():
        n = len(runs)
        console.print(f"\n[bold]{engine.upper()}[/] ({n} questions)")
        for key in ["ttft_ms", "decode_speed_tps", "tbt_ms", "overall_ms"]:
            vals = [float(r[key]) for r in runs if r[key] not in (None, "")]
            if not vals:
                continue
            avg = statistics.mean(vals)
            med = statistics.median(vals)
            label = key.replace("_", " ").title()
            console.print(f"  {label:.<30} avg={avg:.2f}  med={med:.2f}")
    console.print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--device", required=True, help="ADB device serial")
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    console.print(f"Device: [bold]{args.device}[/]  Dataset: [bold]{args.dataset}[/]  Samples: {args.samples}")

    for engine_name, model_dir in [("onnx", ONNX_MODEL_DIR), ("litert", LITERT_MODEL)]:
        path = f"{SDCARD_BASE}/{'task_files/llm/' + model_dir + '/model.onnx' if engine_name == 'onnx' else model_dir}"
        if not check_model_on_device(args.device, path):
            console.print(f"[red]{engine_name.upper()} model not found: {path}[/]")
            sys.exit(1)

    questions = load_questions(args.dataset, args.samples)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results_{args.dataset}_{ts}.csv"
    fields = ["question", "backend", "ttft_ms", "decode_speed_tps", "tbt_ms", "overall_ms", "tokens"]
    all_rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for engine in ["litert", "onnx"]:
            success = run_engine(args.device, questions, engine, args.max_tokens, csv_writer=writer)
            console.print(f"  [green]{engine}: {success}/{len(questions)} succeeded[/]")
            if engine == "litert":
                time.sleep(3)

    with open(csv_path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    console.print(f"\n[green]CSV saved: {csv_path}[/]")
    print_summary(all_rows)


if __name__ == "__main__":
    main()