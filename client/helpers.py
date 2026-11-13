#!/usr/bin/env python3
"""Shared ADB / HTTP helpers for MoRAGBench client scripts."""

import json
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
TASK_ROOT = f"{SDCARD_BASE}/task_files/downstream_task"

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
    try:
        subprocess.run(
            adb_cmd(device, "shell", "am", "start-foreground-service",
                    f"{SERVER_PACKAGE}/.ServerService"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
    except subprocess.TimeoutExpired:
        # ADB shell can hang; the service may still come up. The caller decides
        # liveness via ping rather than crashing the whole run here.
        pass


def stop_service(device):
    try:
        subprocess.run(
            adb_cmd(device, "shell", "am", "force-stop", SERVER_PACKAGE),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
        )
    except subprocess.TimeoutExpired:
        pass


def adb_forward(device):
    subprocess.run(
        adb_cmd(device, "forward", f"tcp:{PORT}", f"tcp:{PORT}"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
    )


def ping_server():
    try:
        r = requests.get(f"{HTTP_BASE}/ping", timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def _connect_refused():
    """True if the server port is closed (process gone), vs merely busy (loading).

    The adb forward to a force-stopped app surfaces as 'Connection aborted /
    RemoteDisconnected / refused', NOT a clean 'Connection refused'. A read- or
    connect- *timeout* means the server is BUSY (mid load/generate) — restarting
    it would kill an in-flight load and force a slow reload (the OOM/crash
    source). So: any timeout = busy (False); any hard connection error = dead (True).
    """
    try:
        requests.get(f"{HTTP_BASE}/ping", timeout=3)
        return False
    except requests.exceptions.Timeout:
        # Server accepted the connection but didn't answer in time: it's busy.
        return False
    except Exception:
        # Connection refused, aborted, or remote-closed: the process is gone.
        return True


def ensure_server(device):
    """Return /ping result if server is up; otherwise (re)start it.

    Never restarts a merely-busy server: a read-timeout means it's mid-load or
    mid-generate (restarting would force a slow model reload and cause the
    "blows up after a point" OOM churn). Only a hard connection-refused (process
    gone) triggers a restart. Returns ping dict, or None if it came up.
    """
    result = ping_with_retries(3, 1.0)
    if result is not None:
        return result
    if not _connect_refused():
        # Server is alive but busy (loading/generating). Leave it alone.
        return None
    stop_service(device)
    time.sleep(1)
    start_service(device)
    adb_forward(device)
    time.sleep(2)
    return ping_with_retries(5, 1.0)


def restart_server(device):
    stop_service(device)
    time.sleep(1)
    start_service(device)
    adb_forward(device)
    time.sleep(2)
    if ping_server() is None:
        adb_forward(device)
        time.sleep(2)


# ── device helpers ──────────────────────────────────────────────────────────────


def check_model_on_device(device, model_path):
    return adb_run(device, "shell", "test", "-f", model_path, check=False) == ""


def pull(path, device, task_dir, required=True):
    """Read a JSON file from the device via adb cat."""
    out = subprocess.run(adb_cmd(device, "shell", "cat", f"{task_dir}/{path}"),
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        if required:
            sys.exit(f"could not read {path} from device")
        return None
    return json.loads(out.stdout)


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
    """Pull a file or directory from the device.

    Falls back to the app's /file endpoint when `adb pull` fails. On Android 11+
    the FUSE layer forces files the app writes into its external files dir to
    mode 0660 app-owned; adb shell (a different uid) can then neither read nor
    chmod them, so the only way to get results off the device is to have the
    app stream them over the forwarded control port.
    """
    try:
        run(adb_cmd(device, "pull", phone_path, local_path))
        return
    except RuntimeError:
        pass
    print("adb pull failed, falling back to HTTP /file endpoint.")
    _http_fetch_recursive(device, phone_path, local_path)


def _http_fetch_recursive(device, phone_path, local_path):
    import os
    import posixpath
    import urllib.parse

    rel = posixpath.relpath(phone_path.rstrip("/"), SDCARD_BASE)
    if rel == ".":
        rel = "."
    os.makedirs(local_path, exist_ok=True)
    entries = requests.get(
        f"{HTTP_BASE}/file",
        params={"path": rel},
        timeout=60,
    ).json()["entries"]
    for name in entries:
        child_phone = posixpath.join(phone_path.rstrip("/"), name)
        child_local = os.path.join(local_path, name)
        if os.path.isdir(child_local):
            _http_fetch_recursive(device, child_phone, child_local)
            continue
        with open(child_local, "wb") as f:
            f.write(
                requests.get(
                    f"{HTTP_BASE}/file",
                    params={"path": posixpath.relpath(child_phone, SDCARD_BASE)},
                    timeout=300,
                ).content
            )


def push_dir_files(device, local_dir, remote_dir):
    """Push every file (non-recursive) from local_dir into remote_dir."""
    import os
    remote_dir = remote_dir.rstrip("/")
    for name in os.listdir(local_dir):
        src = os.path.join(local_dir, name)
        if os.path.isfile(src):
            # Full destination file path, not "dir/": some ROMs (ColorOS) fail
            # with EISDIR when adb push targets a path ending in "/".
            adb_push_files(device, src, f"{remote_dir}/{name}")


def remote_dir_has_files(device, remote_dir):
    # is this folder already on the phone with something in it
    check = f"[ -d '{remote_dir}' ] && [ \"$(ls -A '{remote_dir}' 2>/dev/null)\" ] && echo NONEMPTY || echo EMPTY"
    return run(adb_cmd(device, "shell", check)) == "NONEMPTY"


def push_dir_files_if_missing(device, local_dir, remote_dir, label):
    # skip the push if the folder is already there from an earlier run
    if remote_dir_has_files(device, remote_dir):
        print(f"Skipping push for {label}, already exists on device")
    else:
        push_dir_files(device, local_dir, remote_dir)


def push_dir_files_if_settings_match(device, local_dir, remote_dir, label, note_name):
    # the app looks the folder up by dataset name so the name can't change,
    # instead the settings are saved in a small note inside the folder.
    # same note on the phone means same data, so no need to push again
    import os
    with open(os.path.join(local_dir, note_name), "r") as f:
        local_note = f.read().strip()
    remote_note = run(adb_cmd(device, "shell", f"cat '{remote_dir}/{note_name}' 2>/dev/null; true"))
    if remote_note == local_note:
        print(f"Skipping push for {label}, same settings already on device")
    else:
        push_dir_files(device, local_dir, remote_dir)


# ── HTTP helpers ────────────────────────────────────────────────────────────────

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Reuse one TCP connection per host instead of a fresh handshake per request.
# Fresh connections through ADB forwarding are the main source of the
# "blows up after a point" connection-abort failures on long runs.
_session = requests.Session()
_session.mount(
    HTTP_BASE,
    HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
    ),
)


def post(endpoint, payload):
    r = _session.post(f"{HTTP_BASE}/{endpoint}", json=payload, timeout=600)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()


def ping_with_retries(n=3, delay=1.0):
    """Poll /ping up to n times. Returns response dict, or None if never up."""
    for _ in range(n):
        result = ping_server()
        if result is not None:
            return result
        time.sleep(delay)
    return None


def generate_post(endpoint, payload, attempts=4, base_delay=2.0):
    """POST, retrying transient connection drops with backoff.

    The server keeps the model resident in RAM, so we retry against the SAME
    server rather than restarting it (which would force a slow, memory-heavy
    model reload). If all attempts fail, the last error is re-raised so callers
    see the real cause instead of a swallowed "failed after retries".
    """
    last_err = None
    for attempt in range(attempts):
        try:
            return post(endpoint, payload)
        except Exception as e:
            msg = str(e)
            transient = any(
                kw in msg
                for kw in ["Connection aborted", "RemoteDisconnected", "Read timed out",
                           "ConnectionResetError", "ConnectionError", "Max retries",
                           "ProtocolError", "Timeout"]
            )
            if not transient:
                raise
            last_err = e
            time.sleep(base_delay * (2 ** attempt))
    raise last_err


def prepare_dirs(dirs):
    r = requests.post(f"{HTTP_BASE}/prepare_dirs", json={"dirs": dirs}, timeout=10)
    r.raise_for_status()


def set_engine(device, engine, backend=None):
    import json
    import os
    import tempfile

    local = tempfile.mktemp(suffix=".json")
    try:
        subprocess.run(
            adb_cmd(device, "pull", CONFIG_PATH, local),
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        with open(local) as f:
            config = json.load(f)
        config["rag_pipeline"]["llm"]["engine"] = engine
        config["rag_pipeline"]["llm"]["backend"] = backend or ("gpu" if engine == "litert" else "nnapi")
        with open(local, "w") as f:
            json.dump(config, f, indent=4)
        subprocess.run(
            adb_cmd(device, "push", local, CONFIG_PATH),
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    finally:
        if os.path.exists(local):
            os.unlink(local)

    # verify
    r = subprocess.run(
        adb_cmd(device, "shell", f'grep "\\"engine\\"" {CONFIG_PATH}'),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if engine not in r.stdout:
        print(f"Failed to set engine to {engine} in config.", file=sys.stderr)
        sys.exit(1)
