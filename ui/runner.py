"""Launch client/main.py in the background and watch it.

The job lives in a process-wide registry (st.cache_resource), so a browser
refresh does not lose a running benchmark. Live progress comes from the
phone's /status endpoint, the same one client/utils/log.py polls.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests


def _http_base(client_dir: Path) -> str:
    env = client_dir / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("HTTP_BASE="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "http://127.0.0.1:18080"


def adb_devices() -> tuple[list[str], str | None]:
    """Return (serials, error message)."""
    if not shutil.which("adb"):
        return [], "adb not found on PATH"
    try:
        out = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return [], str(e)
    serials = [l.split()[0] for l in out.splitlines()[1:] if l.strip().endswith("device")]
    return serials, None if serials else "no device connected (check USB debugging)"


@dataclass
class Job:
    config_path: Path
    output_path: Path
    device: str
    cmd: list[str]
    started: float = field(default_factory=time.time)
    proc: subprocess.Popen | None = None
    log: list[str] = field(default_factory=list)
    finished: float | None = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self.proc is None else self.proc.poll()


class Runner:
    def __init__(self, client_dir: Path):
        self.client_dir = client_dir
        self.http_base = _http_base(client_dir)
        self.jobs: list[Job] = []
        self._lock = threading.Lock()

    @property
    def current(self) -> Job | None:
        return self.jobs[-1] if self.jobs else None

    def start(self, config_path: Path, output_path: Path, device: str, resume: bool = False,
              overrides: list[str] | None = None) -> Job:
        if self.current and self.current.running:
            raise RuntimeError("A benchmark is already running.")
        cmd = [sys.executable, "-u", "main.py", "--config", str(config_path),
               "--device", device, "--output_path", str(output_path)]
        if resume:
            cmd.append("--resume")
        for o in overrides or []:
            cmd += ["--set", o]
        # never show secrets in the UI
        shown = ["hf_token=***" if c.startswith("hf_token=") else c for c in cmd]
        job = Job(config_path, output_path, device, shown)
        env = os.environ | {"PYTHONIOENCODING": "utf-8", "TERM": "dumb", "COLUMNS": "120"}
        job.proc = subprocess.Popen(
            cmd, cwd=self.client_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()
        with self._lock:
            self.jobs.append(job)
        return job

    def _pump(self, job: Job):
        assert job.proc and job.proc.stdout
        for line in job.proc.stdout:
            job.log.append(line.rstrip("\n"))
            if len(job.log) > 5000:
                del job.log[:1000]
        job.proc.wait()
        job.finished = time.time()

    def stop(self):
        job = self.current
        if job and job.running:
            job.proc.terminate()

    def status(self) -> dict | None:
        try:
            r = requests.get(f"{self.http_base}/status", timeout=2)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            return None


def progress_from_status(st: dict) -> dict:
    """Reduce a /status payload to what the UI shows (same fields as log.py)."""
    out = {"state": st.get("overallState"), "phase": st.get("currentPhase"), "error": st.get("lastError")}
    if "task" in st and st["task"]:
        idx, ev = st["task"].get("indexProgress", {}), st["task"].get("evaluationProgress", {})
        out |= {
            "index": (idx.get("documentsProcessed"), idx.get("documentsTotal"), "documents"),
            "eval": (ev.get("questionsProcessed"), ev.get("questionsTotal"), "questions"),
            "index_state": idx.get("state"), "index_source": idx.get("source"),
            "chunks": idx.get("chunksProcessed"),
        }
    elif "dataset" in st and st["dataset"]:
        idx, ev = st["dataset"].get("indexProgress", {}), st["dataset"].get("evaluationProgress", {})
        out |= {
            "index": (idx.get("vectorsProcessed"), idx.get("trainTotal"), "vectors"),
            "eval": (ev.get("vectorsProcessed"), ev.get("testTotal"), "queries"),
            "index_state": idx.get("state"), "index_source": idx.get("source"),
        }
    return out
