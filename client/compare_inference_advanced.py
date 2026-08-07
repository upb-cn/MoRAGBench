#!/usr/bin/env python3
"""
Advanced Benchmark: ONNX Runtime vs LiteRT LM on Android.
Includes multiple iterations, error handling, averaging of metrics, 
and detailed metadata (iteration times, total duration, device info).
"""
import argparse
import subprocess
import time
import logging
import statistics
from datetime import datetime

import requests

# Constants
SERVER_PACKAGE = "com.example.cli"
PORT = 18080
ONNX_MODEL_DIR  = "qwen2.5-1.5B-instruct_int8"
LITERT_MODEL    = "Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm"
SDCARD_BASE     = f"/sdcard/Android/data/{SERVER_PACKAGE}/files"
DEFAULT_PROMPT  = "Write a short paragraph about the impact of artificial intelligence on modern society."
MAX_TOKENS      = 128
ITERATIONS      = 3

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AdvancedBenchmark")

# ── helpers ──────────────────────────────────────────────────────────────────

def adb(device, *args):
    cmd = ["adb", "-s", device] + list(args)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        logger.debug(f"ADB command failed: {' '.join(cmd)}\nError: {r.stderr}")
    return r

def get_device_info(device):
    model = adb(device, "shell", "getprop", "ro.product.model").stdout.strip()
    brand = adb(device, "shell", "getprop", "ro.product.brand").stdout.strip()
    return f"{brand} {model}"

def restart_server(device):
    """Stop → wait → start → forward."""
    logger.info("Restarting Android server service...")
    adb(device, "shell", "am", "stopservice", "-n", f"{SERVER_PACKAGE}/.ServerService")
    time.sleep(1.5)
    adb(device, "shell", "am", "start-foreground-service", f"{SERVER_PACKAGE}/.ServerService")
    adb(device, "forward", f"tcp:{PORT}", f"tcp:{PORT}")
    time.sleep(2.5)   # give the HTTP server time to bind

def ping_server():
    """Returns (alive, onnx_loaded, litert_loaded) tuple."""
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/ping", timeout=5)
        if r.ok:
            data = r.json()
            return True, data.get("onnx_loaded", False), data.get("litert_loaded", False)
    except Exception:
        pass
    return False, False, False

def ensure_server_alive(device):
    """Only restarts the server if it's not responding. Preserves in-memory models."""
    alive, onnx_loaded, litert_loaded = ping_server()
    if alive:
        logger.info(f"Server already running. ONNX loaded: {onnx_loaded}, LiteRT loaded: {litert_loaded}")
        return
    logger.info("Server not responding — restarting...")
    restart_server(device)

def post(endpoint, payload):
    r = requests.post(f"http://127.0.0.1:{PORT}/{endpoint}", json=payload, timeout=600)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()

# ── benchmark runners ─────────────────────────────────────────────────────────

def run_onnx(device, prompt, max_tokens):
    # Verify model is on device
    model_path = f"{SDCARD_BASE}/task_files/llm/{ONNX_MODEL_DIR}/model.onnx"
    if adb(device, "shell", "test", "-f", model_path).returncode != 0:
        raise FileNotFoundError(f"ONNX model not found at {model_path}")

    # Ensure dirs exist
    requests.post(f"http://127.0.0.1:{PORT}/prepare_dirs",
                  json={"dirs": ["task_files", "task_files/llm", f"task_files/llm/{ONNX_MODEL_DIR}"]},
                  timeout=15)

    logger.debug("Requesting /generate (ONNX)...")
    return post("generate", {"prompt": prompt, "max_tokens": max_tokens})

def run_litert(device, prompt, max_tokens):
    # Verify model is on device
    model_path = f"{SDCARD_BASE}/{LITERT_MODEL}"
    if adb(device, "shell", "test", "-f", model_path).returncode != 0:
        logger.warning(f"LiteRT model might be missing at {model_path}")

    logger.debug("Requesting /generate_litert (LiteRT)...")
    return post("generate_litert", {"prompt": prompt, "max_tokens": max_tokens})

# ── metrics ───────────────────────────────────────────────────────────────────

def extract_metrics(result):
    """Returns raw numeric dictionary of metrics."""
    m = result.get("metrics", {})
    gen   = m.get("generated_tokens", 0)
    ttft  = m.get("ttft_ms", 0)
    dur   = m.get("overall_duration_ms", 0)
    load  = m.get("load_time_ms", 0)
    speed = m.get("decoding_speed_tokens_per_sec", 0.0)
    # Calculate TBT: (Overall - TTFT) / (Generated - 1)
    tbt   = (dur - ttft) / (gen - 1) if gen > 1 else 0.0
    
    return {
        "load_time":    float(load),
        "ttft":         float(ttft),
        "decode_speed": float(speed),
        "tbt":          float(tbt),
        "tokens":       int(gen),
        "overall":      float(dur)
    }

def calculate_average(metrics_list):
    if not metrics_list:
        return None
    keys = metrics_list[0].keys()
    averages = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        averages[key] = statistics.mean(values) if values else 0.0
    return averages

# ── core loop ─────────────────────────────────────────────────────────────────

def benchmark_backend(name, runner_func, device, prompt, max_tokens, iterations, delay_secs=0):
    logger.info(f"Starting {name} benchmark suite ({iterations} iterations)...")
    successful_runs = []
    total_start_time = time.time()
    
    # Only restart the server if it's not already alive (preserves loaded models)
    try:
        ensure_server_alive(device)
    except Exception as e:
        logger.error(f"[{name}] initial server restart failed: {e}")
        return {
            "runs": [],
            "average": None,
            "success_count": 0,
            "total_test_time_ms": 0
        }

    for i in range(1, iterations + 1):
        iter_start = time.time()
        logger.info(f"[{name}] Iteration {i}/{iterations}...")
        try:
            # We no longer restart_server(device) here
            result = runner_func(device, prompt, max_tokens)
            metrics = extract_metrics(result)
            
            iter_end = time.time()
            iter_duration = (iter_end - iter_start) * 1000 # to ms
            metrics["iteration_total_time"] = iter_duration
            
            successful_runs.append(metrics)
            logger.info(f"[{name}] Iteration {i} success. (TTFT: {metrics['ttft']:.0f}ms, Speed: {metrics['decode_speed']:.2f} tok/s)")
        except Exception as e:
            logger.error(f"[{name}] Iteration {i} failed: {e}")
            # If we get a connection error, the server likely crashed. 
            # Force a restart for the next iteration to recover.
            if any(err in str(e) for err in ["Connection aborted", "RemoteDisconnected", "Read timed out"]):
                logger.warning(f"[{name}] Server seems dead. Attempting recovery restart for next iteration...")
                try:
                    restart_server(device)  # Force-restart on crash (model is gone anyway)
                    time.sleep(2)
                except Exception as re:
                    logger.error(f"[{name}] Recovery restart failed: {re}")
        
        # Optional cooldown delay between iterations (skip after last one)
        if i < iterations and delay_secs > 0:
            logger.info(f"Cooling down for {delay_secs}s before next iteration...")
            time.sleep(delay_secs)
            
    total_end_time = time.time()
    avg_metrics = calculate_average(successful_runs)
    
    return {
        "runs": successful_runs,
        "average": avg_metrics,
        "success_count": len(successful_runs),
        "total_test_time_ms": (total_end_time - total_start_time) * 1000
    }

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Advanced Comparison: ONNX vs LiteRT on Android")
    parser.add_argument("--device", "-d", required=True, help="ADB device serial")
    parser.add_argument("--prompt", "-p", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--iters", type=int, default=ITERATIONS, help="Number of iterations per backend")
    parser.add_argument("--delay", type=int, default=0, help="Cooldown delay in seconds between iterations (default: 0)")
    args = parser.parse_args()

    device_info = get_device_info(args.device)
    logger.info(f"Target Device: {device_info} ({args.device})")

    # 1. Run ONNX
    onnx_data = benchmark_backend("ONNX Runtime", run_onnx, args.device, args.prompt, args.max_tokens, args.iters, args.delay)
    
    # 2. Run LiteRT
    litert_data = benchmark_backend("LiteRT LM", run_litert, args.device, args.prompt, args.max_tokens, args.iters, args.delay)

    # 3. Print Results Table
    print("\n" + "="*80)
    print(f"BENCHMARK REPORT: {device_info}")
    print(f"Prompt: {args.prompt[:70]}...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    rows = [
        ("Load Time", "load_time", "ms"),
        ("TTFT", "ttft", "ms"),
        ("Decode Speed", "decode_speed", "tok/s"),
        ("TBT", "tbt", "ms"),
        ("Overall Inference", "overall", "ms"),
        ("Iteration Wall-Clock", "iteration_total_time", "ms"),
    ]
    
    col1, col2, col3 = 25, 25, 25
    sep = f"+{'-'*col1}+{'-'*col2}+{'-'*col3}+"
    hdr = f"| {'Metric (Average)':<{col1-2}} | {'ONNX Runtime':<{col2-2}} | {'LiteRT LM (GPU)':<{col3-2}} |"
    
    print(sep)
    print(hdr)
    print(f"| {'Success Count':<{col1-2}} | {f'{onnx_data['success_count']}/{args.iters}':<{col2-2}} | {f'{litert_data['success_count']}/{args.iters}':<{col3-2}} |")
    print(sep)

    for label, key, unit in rows:
        v1 = onnx_data["average"].get(key, 0) if onnx_data["average"] else 0
        v2 = litert_data["average"].get(key, 0) if litert_data["average"] else 0
        
        if key == "decode_speed":
            s1, s2 = f"{v1:.2f} {unit}", f"{v2:.2f} {unit}"
        else:
            s1, s2 = f"{v1:.1f} {unit}", f"{v2:.1f} {unit}"
            
        print(f"| {label:<{col1-2}} | {s1:<{col2-2}} | {s2:<{col3-2}} |")
    
    print(sep)
    
    # 3.1 Individual Iterations
    def print_iters(name, data):
        if not data["runs"]:
            return
        print(f"\nINDIVIDUAL ITERATIONS: {name}")
        sub_hdr = f"| {'Iter':<6} | {'TTFT':<10} | {'TBT':<10} | {'Speed':<12} | {'Load':<10} | {'Inference':<10} | {'Wall-Clock':<12} |"
        sub_sep = f"+{'-'*8}+{'-'*12}+{'-'*12}+{'-'*14}+{'-'*12}+{'-'*12}+{'-'*14}+"
        print(sub_sep)
        print(sub_hdr)
        print(sub_sep)
        for i, run in enumerate(data["runs"], 1):
            print(f"| {i:<6} | {run['ttft']:>7.0f} ms | {run['tbt']:>7.1f} ms | {run['decode_speed']:>7.2f} t/s | {run['load_time']:>7.0f} ms | {run['overall']:>7.0f} ms | {run['iteration_total_time']/1000:>10.2f} s |")
        print(sub_sep)

    print_iters("ONNX Runtime", onnx_data)
    print_iters("LiteRT LM", litert_data)

    # Extra Metadata
    total_tokens_onnx = onnx_data["average"].get("tokens", 0) if onnx_data["average"] else 0
    total_tokens_litert = litert_data["average"].get("tokens", 0) if litert_data["average"] else 0
    
    print(f"\nAvg Tokens : {total_tokens_onnx:.0f} (ONNX) | {total_tokens_litert:.0f} (LiteRT)")
    print(f"Total Test Time: {onnx_data['total_test_time_ms']/1000:.1f}s (ONNX Suite) + {litert_data['total_test_time_ms']/1000:.1f}s (LiteRT Suite)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
