#!/usr/bin/env python3
"""Shared ADB / HTTP helpers for MoRAGBench client scripts."""

import subprocess
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

_env = dotenv_values(Path(__file__).resolve().parent / ".env")

SERVER_PACKAGE = _env["SERVER_PACKAGE"]
PORT = int(_env["PORT"])
HTTP_BASE = _env["HTTP_BASE"]
SDCARD_BASE = _env["SDCARD_BASE"]
CONFIG_PATH = _env["CONFIG_PATH"]
DEFAULT_PROMPT = _env["DEFAULT_PROMPT"]
MAX_TOKENS = int(_env["MAX_TOKENS"])
LITERT_MODEL = _env["LITERT_MODEL"]
ONNX_MODEL_DIR = _env["ONNX_MODEL_DIR"]

import requests


# ── ADB primitives ──────────────────────────────────────────────────────────────


def adb_cmd(device, *args):
    return ["adb", "-s", device] + list(args)


def adb_run(device, *args, check=True):
    cmd = adb_cmd(device, *args)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}\n{r.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()


def run(cmd):
    print(">", " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("ERROR:", r.stderr)
        raise RuntimeError("command failed")
    return r.stdout.strip()


# ── server lifecycle ────────────────────────────────────────────────────────────


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


# ── device helpers ──────────────────────────────────────────────────────────────


def check_model_on_device(device, model_path):
    return adb_run(device, "shell", "test", "-f", model_path, check=False) == ""


def create_server_dir(device, path):
    subprocess.run(
        adb_cmd(device, "shell", "mkdir", "-p", path),
        check=True,
    )


def adb_push_files(device, local_path, destination_path):
    import os
    if not os.path.exists(local_path):
        raise ValueError(f"Local path does not exist: {local_path}")
    try:
        subprocess.run(
            adb_cmd(device, "push", local_path, destination_path),
            check=True,
        )
        print("Files copied successfully.")
    except subprocess.CalledProcessError as e:
        print("ADB copy failed.")
        raise e


def adb_pull(device, phone_path, local_path):
    run(adb_cmd(device, "pull", phone_path, local_path))


def push_dir_files(device, local_dir, remote_dir):
    """Push every file (non-recursive) from local_dir into remote_dir."""
    import os
    remote_dir = remote_dir.rstrip("/") + "/"
    for name in os.listdir(local_dir):
        src = os.path.join(local_dir, name)
        if os.path.isfile(src):
            adb_push_files(device, src, remote_dir)


# ── HTTP helpers ────────────────────────────────────────────────────────────────


def post(endpoint, payload):
    r = requests.post(f"{HTTP_BASE}/{endpoint}", json=payload, timeout=600)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()


def prepare_dirs(dirs):
    r = requests.post(f"{HTTP_BASE}/prepare_dirs", json={"dirs": dirs}, timeout=10)
    r.raise_for_status()


def set_engine(device, engine):
    cmd = adb_cmd(device, "shell",
        f"sed -i 's/\"engine\": \"[^\"]*\"/\"engine\": \"{engine}\"/' {CONFIG_PATH}; "
        f"if ! grep -q '\"engine\"' {CONFIG_PATH}; then "
        f"sed -i '/\"llm\": {{/a\\    \"engine\": \"{engine}\",' {CONFIG_PATH}; fi; "
        f"grep '\"engine\"' {CONFIG_PATH}"
    )
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if engine not in r.stdout.strip():
        print(f"Failed to set engine to {engine} in config.")
        sys.exit(1)
