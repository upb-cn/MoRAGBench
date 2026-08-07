#!/usr/bin/env python3
"""
Compare ONNX Runtime vs LiteRT LM inference on Android.
Uses the exact same server start/stop logic as onnx_inference.py and litert_inference.py.
"""
import argparse
import subprocess
import time

import requests

SERVER_PACKAGE = "com.example.cli"
PORT = 18080

ONNX_MODEL_DIR  = "qwen2.5-1.5B-instruct_int8"   # must match onnx_inference.py
LITERT_MODEL    = "Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm"
SDCARD_BASE     = f"/sdcard/Android/data/{SERVER_PACKAGE}/files"

DEFAULT_PROMPT  = "Write a short paragraph about the impact of artificial intelligence on modern society."
MAX_TOKENS      = 128


# ── helpers ──────────────────────────────────────────────────────────────────

def adb(device, *args):
    cmd = ["adb", "-s", device] + list(args)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r


def restart_server(device):
    """Stop → wait → start → forward.  Exact copy of onnx_inference.py logic."""
    print("\n=== Restarting server ===")
    adb(device, "shell", "am", "stopservice", "-n", f"{SERVER_PACKAGE}/.ServerService")
    time.sleep(1)
    adb(device, "shell", "am", "start-foreground-service", f"{SERVER_PACKAGE}/.ServerService")
    adb(device, "forward", f"tcp:{PORT}", f"tcp:{PORT}")
    time.sleep(2)   # give the HTTP server time to bind


def post(endpoint, payload):
    r = requests.post(f"http://127.0.0.1:{PORT}/{endpoint}", json=payload, timeout=600)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()


# ── benchmark runners ─────────────────────────────────────────────────────────

def run_onnx(device, prompt, max_tokens):
    """Replica of onnx_inference.py's inference call."""
    # Verify model is on device
    model_path = f"{SDCARD_BASE}/task_files/llm/{ONNX_MODEL_DIR}/model.onnx"
    if adb(device, "shell", "test", "-f", model_path).returncode != 0:
        raise FileNotFoundError(
            f"ONNX model not found at {model_path}\n"
            f"Run onnx_inference.py once to push it first."
        )

    # Ensure dirs exist (same call as onnx_inference.py)
    requests.post(f"http://127.0.0.1:{PORT}/prepare_dirs",
                  json={"dirs": ["task_files", "task_files/llm", f"task_files/llm/{ONNX_MODEL_DIR}"]},
                  timeout=10)

    print("Sending /generate request...")
    return post("generate", {"prompt": prompt, "max_tokens": max_tokens})


def run_litert(device, prompt, max_tokens):
    """Replica of litert_inference.py's inference call."""
    # Verify model is on device
    model_path = f"{SDCARD_BASE}/{LITERT_MODEL}"
    if adb(device, "shell", "test", "-f", model_path).returncode != 0:
        print(f"WARNING: LiteRT model not found at {model_path}")

    print("Sending /generate_litert request...")
    return post("generate_litert", {"prompt": prompt, "max_tokens": max_tokens})


# ── metrics ───────────────────────────────────────────────────────────────────

def extract(result):
    m = result.get("metrics", {})
    gen   = m.get("generated_tokens", 0)
    ttft  = m.get("ttft_ms", 0)
    dur   = m.get("overall_duration_ms", 0)
    load  = m.get("load_time_ms", 0)
    speed = m.get("decoding_speed_tokens_per_sec", 0.0)
    tbt   = (dur - ttft) / (gen - 1) if gen > 1 else 0.0
    return {
        "Load Time":    f"{load} ms",
        "TTFT":         f"{ttft} ms",
        "Decode Speed": f"{speed:.2f} tok/s",
        "TBT":          f"{tbt:.1f} ms",
        "Tokens":        gen,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare ONNX Runtime vs LiteRT LM on Android")
    parser.add_argument("--device", "-d", required=True, help="ADB device serial (adb get-serialno)")
    parser.add_argument("--prompt", "-p", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    # ── 1. ONNX ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ONNX Runtime benchmark")
    print("="*60)
    restart_server(args.device)
    onnx_result  = run_onnx(args.device, args.prompt, args.max_tokens)
    onnx_metrics = extract(onnx_result)
    print("Done.")

    # ── 2. LiteRT ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("LiteRT LM benchmark")
    print("="*60)
    restart_server(args.device)
    litert_result  = run_litert(args.device, args.prompt, args.max_tokens)
    litert_metrics = extract(litert_result)
    print("Done.")

    # ── 3. Table ──────────────────────────────────────────────────────────────
    rows = ["Load Time", "TTFT", "Decode Speed", "TBT"]
    col1 = max(len(r) for r in rows) + 2
    col2 = 22
    col3 = 22

    sep  = f"+{'-'*col1}+{'-'*col2}+{'-'*col3}+"
    hdr  = f"| {'Metric':<{col1-2}} | {'ONNX Runtime':<{col2-2}} | {'LiteRT LM (GPU)':<{col3-2}} |"

    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for row in rows:
        v1 = onnx_metrics[row]
        v2 = litert_metrics[row]
        print(f"| {row:<{col1-2}} | {v1:<{col2-2}} | {v2:<{col3-2}} |")
    print(sep)
    print(f"\nModel  : Qwen 2.5 1.5B Instruct")
    print(f"Prompt : {args.prompt[:80]}")
    print(f"Tokens : {onnx_metrics['Tokens']} (ONNX)  {litert_metrics['Tokens']} (LiteRT)")


if __name__ == "__main__":
    main()
