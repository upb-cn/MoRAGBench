#!/usr/bin/env python3
"""Run MoRAGBench questions through one or both backends, streaming to JSONL."""
import argparse
import json
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

ENGINES = ["litert", "onnx"]


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


def run_one(engine, backend, question, max_tokens):
    """Run single question; model stays resident across retries. None on failure."""
    try:
        r = generate_post("generate", {"prompt": question, "max_tokens": max_tokens})
    except Exception as e:
        console.print(f"  [yellow]{e}[/]")
        return None
    m = r.get("metrics", r)
    status = str(m.get("status", "")).upper()
    gen = int(m.get("generated_tokens", 0) or 0)
    dur = float(m.get("overall_duration_ms", 0) or 0)
    # A non-OK status or no generated tokens means this was not a real inference.
    # Reject it so failed/timed-out requests don't pollute the stats as "success".
    if status not in ("", "OK", "SUCCESS") or gen <= 0 or dur <= 0:
        console.print(f"  [yellow]bad result: status={status!r} tokens={gen} overall={dur}[/]")
        return None
    ttft = float(m.get("ttft_ms", 0) or 0)
    speed = float(m.get("decoding_speed_tokens_per_sec", 0.0) or 0.0)
    tbt = (dur - ttft) / (gen - 1) if gen > 1 else 0.0
    return {
        "question": question,
        "response": r.get("response", ""),
        "engine": engine,
        "backend": backend,
        "ttft_ms": round(ttft, 2),
        "decode_speed_tps": round(speed, 2),
        "tbt_ms": round(tbt, 2),
        "overall_ms": round(dur, 2),
        "tokens": gen,
    }


def run_engine(device, questions, engine, backend, max_tokens, out_path):
    """Run all questions for one engine, appending each result to JSONL immediately."""
    console.print(f"\n[bold]Benchmarking {engine.upper()}...[/]")
    set_engine(device, engine, backend)
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
    with open(out_path, "a") as f:
        for idx, q in enumerate(track(questions, description=f"[cyan]{engine}[/]"), 1):
            # Periodic health check BETWEEN questions: recover only if the process is
            # truly gone; a busy (timeout) server is left to finish loading.
            if idx % 20 == 0 and _connect_refused():
                console.print("  [yellow]Server is down — recovering...[/]")
                if ensure_server(device) is None:
                    console.print("  [red]Server recovery failed; skipping rest of run.[/]")
                    break
            res = run_one(engine, backend, q, max_tokens)
            if res:
                success += 1
                f.write(json.dumps(res) + "\n")
                f.flush()
                console.print(f"  [green]OK[/] ({res['decode_speed_tps']:.1f} tok/s)")
            else:
                console.print(f"  [red]FAILED after retries[/]")
    return success


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--device", required=True, help="ADB device serial")
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--engine", choices=ENGINES + ["both"], default="both",
                        help="which engine(s) to run; 'both' runs litert then onnx")
    parser.add_argument("--backend", choices=["gpu", "nnapi", "cpu"], default=None,
                        help="override backend; default: litert->gpu, onnx->nnapi")
    args = parser.parse_args()

    console.print(f"Device: [bold]{args.device}[/]  Dataset: [bold]{args.dataset}[/]  "
                  f"Samples: {args.samples}  Backend: {args.backend or 'auto'}")

    # preflight only the engines actually requested
    requested = ENGINES if args.engine == "both" else [args.engine]
    for engine_name in requested:
        model_dir = LITERT_MODEL if engine_name == "litert" else ONNX_MODEL_DIR
        path = (f"{SDCARD_BASE}/{model_dir}" if engine_name == "litert"
                else f"{SDCARD_BASE}/task_files/llm/{model_dir}/model.onnx")
        if not check_model_on_device(args.device, path):
            console.print(f"[red]{engine_name.upper()} model not found: {path}[/]")
            sys.exit(1)

    questions = load_questions(args.dataset, args.samples)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for engine in requested:
        backend = args.backend or ("gpu" if engine == "litert" else "nnapi")
        out_path = f"results_{args.dataset}_{engine}_{backend}_{ts}.jsonl"
        console.print(f"\n[bold]Target: engine={engine} backend={backend} -> {out_path}[/]")
        success = run_engine(args.device, questions, engine, backend, args.max_tokens, out_path)
        console.print(f"  [green]{engine} ({backend}): {success}/{len(questions)} succeeded[/]")
        if engine != requested[-1]:
            time.sleep(3)


if __name__ == "__main__":
    main()
