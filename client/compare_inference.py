#!/usr/bin/env python3
"""MoRAGBench - Compare ONNX Runtime vs LiteRT LM inference on Android.

With --iters 1 (default): single-run comparison with a simple side-by-side table.
With --iters N:           multi-iteration comparison with averaged metrics,
                          per-iteration breakdown, error recovery, and cooldown.
"""
import argparse
import statistics
import subprocess
import sys
import time
from datetime import datetime

import requests
from rich.console import Console

SERVER_PACKAGE = "com.example.cli"
PORT = 18080
HTTP_BASE = f"http://127.0.0.1:{PORT}"
SDCARD_BASE = f"/sdcard/Android/data/{SERVER_PACKAGE}/files"

DEFAULT_PROMPT = (
    "Write a short paragraph about the impact of artificial intelligence "
    "on modern society."
)
MAX_TOKENS = 128

ONNX_MODEL_DIR = "qwen2.5-1.5B-instruct_int8"
LITERT_MODEL = "Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm"

console = Console()


# ── helpers ───────────────────────────────────────────────────────────────────

def adb_cmd(device, *args):
    return ["adb", "-s", device] + list(args)


def adb_run(device, *args, check=True):
    cmd = adb_cmd(device, *args)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}\n{r.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()


def start_service(device):
    subprocess.run(
        adb_cmd(device, "shell", "am", "start-foreground-service",
                f"{SERVER_PACKAGE}/.ServerService"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def stop_service(device):
    subprocess.run(
        adb_cmd(device, "shell", "am", "stopservice", "-n",
                f"{SERVER_PACKAGE}/.ServerService"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def adb_forward(device):
    subprocess.run(
        adb_cmd(device, "forward", f"tcp:{PORT}", f"tcp:{PORT}"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def ensure_server(device):
    try:
        r = requests.get(f"{HTTP_BASE}/ping", timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    stop_service(device)
    time.sleep(1)
    start_service(device)
    adb_forward(device)
    time.sleep(2)
    return {}


def restart_server(device):
    stop_service(device)
    time.sleep(1)
    start_service(device)
    adb_forward(device)
    time.sleep(2)


def post(endpoint, payload):
    r = requests.post(f"{HTTP_BASE}/{endpoint}", json=payload, timeout=600)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()


def check_model_on_device(device, model_path):
    return adb_run(device, "shell", "test", "-f", model_path, check=False) == ""


def prepare_dirs(dirs):
    r = requests.post(f"{HTTP_BASE}/prepare_dirs", json={"dirs": dirs}, timeout=10)
    r.raise_for_status()


# ── device info ───────────────────────────────────────────────────────────────

def get_device_info(device):
    model = adb_run(device, "shell", "getprop", "ro.product.model")
    brand = adb_run(device, "shell", "getprop", "ro.product.brand")
    return f"{brand} {model}"


# ── model verification ───────────────────────────────────────────────────────

def verify_models(device):
    onnx_path = f"{SDCARD_BASE}/task_files/llm/{ONNX_MODEL_DIR}/model.onnx"
    litert_path = f"{SDCARD_BASE}/{LITERT_MODEL}"

    onnx_ok = check_model_on_device(device, onnx_path)
    litert_ok = check_model_on_device(device, litert_path)

    if not onnx_ok:
        console.print(f"[yellow]WARNING: ONNX model not found at {onnx_path}[/]")
    if not litert_ok:
        console.print(f"[yellow]WARNING: LiteRT model not found at {litert_path}[/]")
    return onnx_ok, litert_ok


# ── metric extraction ─────────────────────────────────────────────────────────

def extract_metrics(result):
    gen = result.get("generated_tokens", 0)
    ttft = result.get("ttft_ms", 0)
    dur = result.get("overall_duration_ms", 0)
    load = result.get("load_time_ms", 0)
    speed = result.get("decoding_speed_tokens_per_sec", 0.0)
    tbt = (dur - ttft) / (gen - 1) if gen > 1 else 0.0
    return {
        "load_time": float(load),
        "ttft": float(ttft),
        "decode_speed": float(speed),
        "tbt": float(tbt),
        "tokens": int(gen),
        "overall": float(dur),
    }


def calculate_average(metrics_list):
    if not metrics_list:
        return None
    keys = metrics_list[0].keys()
    return {k: statistics.mean(m[k] for m in metrics_list) for k in keys}


# ── backend runners ───────────────────────────────────────────────────────────

def run_onnx(device, prompt, max_tokens):
    model_path = f"{SDCARD_BASE}/task_files/llm/{ONNX_MODEL_DIR}/model.onnx"
    if not check_model_on_device(device, model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")

    prepare_dirs(["task_files", "task_files/llm", f"task_files/llm/{ONNX_MODEL_DIR}"])
    return post("generate", {"prompt": prompt, "max_tokens": max_tokens})


def run_litert(device, prompt, max_tokens):
    model_path = f"{SDCARD_BASE}/{LITERT_MODEL}"
    if not check_model_on_device(device, model_path):
        console.print(f"[yellow]WARNING: LiteRT model might be missing at {model_path}[/]")

    return post("generate", {"prompt": prompt, "max_tokens": max_tokens})


BACKENDS = [
    ("ONNX Runtime", run_onnx),
    ("LiteRT LM", run_litert),
]


def benchmark_backend(name, runner, device, prompt, max_tokens, iterations, delay):
    runs = []
    total_start = time.time()

    try:
        ensure_server(device)
    except Exception as e:
        console.print(f"[red][{name}] Server start failed: {e}[/]")
        return {"runs": [], "average": None, "success": 0, "total_ms": 0}

    for i in range(1, iterations + 1):
        iter_start = time.time()
        try:
            result = runner(device, prompt, max_tokens)
            metrics = extract_metrics(result)
            metrics["iteration_total_time"] = (time.time() - iter_start) * 1000
            runs.append(metrics)
            console.print(
                f"  [{name}] Iter {i}/{iterations} OK "
                f"(TTFT: {metrics['ttft']:.0f}ms, Speed: {metrics['decode_speed']:.2f} tok/s)"
            )
        except Exception as e:
            console.print(f"  [{name}] Iter {i}/{iterations} FAILED: {e}")
            if any(kw in str(e) for kw in ["Connection aborted", "RemoteDisconnected", "Read timed out"]):
                console.print(f"  [{name}] Attempting server recovery...")
                try:
                    restart_server(device)
                    time.sleep(2)
                except Exception as re:
                    console.print(f"  [{name}] Recovery failed: {re}")

        if i < iterations and delay > 0:
            time.sleep(delay)

    avg = calculate_average(runs)
    return {
        "runs": runs,
        "average": avg,
        "success": len(runs),
        "total_ms": (time.time() - total_start) * 1000,
    }


# ── output ────────────────────────────────────────────────────────────────────

def print_simple_table(onnx_metrics, litert_metrics):
    rows = [
        ("Load Time", "load_time", "ms"),
        ("TTFT", "ttft", "ms"),
        ("Decode Speed", "decode_speed", "tok/s"),
        ("TBT", "tbt", "ms"),
    ]

    col1 = max(len(r[0]) for r in rows) + 2
    col2, col3 = 22, 22
    sep = f"+{'-'*col1}+{'-'*col2}+{'-'*col3}+"
    hdr = f"| {'Metric':<{col1-2}} | {'ONNX Runtime':<{col2-2}} | {'LiteRT LM':<{col3-2}} |"

    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for label, key, unit in rows:
        v1 = f"{onnx_metrics[key]:.2f} {unit}" if key == "decode_speed" else f"{onnx_metrics[key]:.1f} {unit}"
        v2 = f"{litert_metrics[key]:.2f} {unit}" if key == "decode_speed" else f"{litert_metrics[key]:.1f} {unit}"
        print(f"| {label:<{col1-2}} | {v1:<{col2-2}} | {v2:<{col3-2}} |")
    print(sep)
    print(f"\nTokens : {onnx_metrics['tokens']} (ONNX)  {litert_metrics['tokens']} (LiteRT)")


def print_advanced_table(onnx_data, litert_data, iters, prompt, device_info):
    rows = [
        ("Load Time", "load_time", "ms"),
        ("TTFT", "ttft", "ms"),
        ("Decode Speed", "decode_speed", "tok/s"),
        ("TBT", "tbt", "ms"),
        ("Overall Inference", "overall", "ms"),
        ("Iteration Wall-Clock", "iteration_total_time", "ms"),
    ]

    print("\n" + "=" * 80)
    print(f"BENCHMARK REPORT: {device_info}")
    print(f"Prompt: {prompt[:70]}...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    col1, col2, col3 = 25, 25, 25
    sep = f"+{'-'*col1}+{'-'*col2}+{'-'*col3}+"
    hdr = f"| {'Metric (Average)':<{col1-2}} | {'ONNX Runtime':<{col2-2}} | {'LiteRT LM':<{col3-2}} |"

    print(sep)
    print(hdr)
    onnx_s = f"{onnx_data['success']}/{iters}"
    litert_s = f"{litert_data['success']}/{iters}"
    print(f"| {'Success Count':<{col1-2}} | {onnx_s:<{col2-2}} | {litert_s:<{col3-2}} |")
    print(sep)

    for label, key, unit in rows:
        v1 = onnx_data["average"].get(key, 0) if onnx_data["average"] else 0
        v2 = litert_data["average"].get(key, 0) if litert_data["average"] else 0
        fmt = lambda v, u: f"{v:.2f} {u}" if key == "decode_speed" else f"{v:.1f} {u}"
        print(f"| {label:<{col1-2}} | {fmt(v1, unit):<{col2-2}} | {fmt(v2, unit):<{col3-2}} |")
    print(sep)

    def print_iters(name, data):
        if not data["runs"]:
            return
        print(f"\nINDIVIDUAL ITERATIONS: {name}")
        sub_sep = f"+{'-'*8}+{'-'*12}+{'-'*12}+{'-'*14}+{'-'*12}+{'-'*12}+{'-'*14}+"
        sub_hdr = (
            f"| {'Iter':<6} | {'TTFT':<10} | {'TBT':<10} | {'Speed':<12} "
            f"| {'Load':<10} | {'Inference':<10} | {'Wall-Clock':<12} |"
        )
        print(sub_sep)
        print(sub_hdr)
        print(sub_sep)
        for i, r in enumerate(data["runs"], 1):
            print(
                f"| {i:<6} "
                f"| {r['ttft']:>7.0f} ms "
                f"| {r['tbt']:>7.1f} ms "
                f"| {r['decode_speed']:>7.2f} t/s "
                f"| {r['load_time']:>7.0f} ms "
                f"| {r['overall']:>7.0f} ms "
                f"| {r['iteration_total_time']/1000:>10.2f} s |"
            )
        print(sub_sep)

    print_iters("ONNX Runtime", onnx_data)
    print_iters("LiteRT LM", litert_data)

    onnx_tok = onnx_data["average"].get("tokens", 0) if onnx_data["average"] else 0
    litert_tok = litert_data["average"].get("tokens", 0) if litert_data["average"] else 0
    print(f"\nAvg Tokens : {onnx_tok:.0f} (ONNX) | {litert_tok:.0f} (LiteRT)")
    print(
        f"Total Time : {onnx_data['total_ms']/1000:.1f}s (ONNX) "
        f"+ {litert_data['total_ms']/1000:.1f}s (LiteRT)"
    )
    print("=" * 80 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare ONNX Runtime vs LiteRT LM on Android")
    parser.add_argument("--device", "-d", required=True, help="ADB device serial")
    parser.add_argument("--prompt", "-p", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--iters", type=int, default=1, help="Iterations per backend (1 = simple table, >1 = advanced)")
    parser.add_argument("--delay", type=int, default=0, help="Cooldown seconds between iterations")
    args = parser.parse_args()

    device_info = get_device_info(args.device)
    console.print(f"Target device: [bold]{device_info}[/] ({args.device})")

    verify_models(args.device)

    results = {}
    for name, runner in BACKENDS:
        console.print(f"\n[bold]{'='*40}[/]")
        console.print(f"[bold]{name} benchmark[/]")
        console.print(f"[bold]{'='*40}[/]")
        results[name] = benchmark_backend(
            name, runner, args.device, args.prompt, args.max_tokens,
            args.iters, args.delay,
        )
        console.print(f"[green]Done.[/]")

    onnx = results["ONNX Runtime"]
    litert = results["LiteRT LM"]

    if args.iters == 1:
        if onnx["average"] and litert["average"]:
            print_simple_table(onnx["average"], litert["average"])
        else:
            console.print("[red]One or both backends failed. Cannot produce comparison table.[/]")
    else:
        print_advanced_table(onnx, litert, args.iters, args.prompt, device_info)


if __name__ == "__main__":
    main()
