#!/usr/bin/env python3
"""Print benchmark stats from one or more results_*.jsonl files as a Rich table."""
import argparse
import glob
import json
import statistics

from rich.console import Console
from rich.table import Table

console = Console()

METRICS = [
    ("ttft_ms", "TTFT (ms)"),
    ("decode_speed_tps", "Decode (tok/s)"),
    ("tbt_ms", "TBT (ms)"),
    ("overall_ms", "Overall (ms)"),
    ("tokens", "Tokens"),
]


def load_rows(paths):
    rows = []
    for p in paths:
        inferred_backend = infer_backend(p)
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    # Old files predate the per-row backend field; backfill from the
                    # filename (results_<ds>_<engine>_<backend>_<ts>.jsonl).
                    r.setdefault("backend", inferred_backend)
                    # Guard against corrupt/failed rows recorded by earlier buggy
                    # writers: drop anything that isn't a physically possible inference.
                    if is_impossible(r):
                        continue
                    rows.append(r)
        except FileNotFoundError:
            console.print(f"[red]not found: {p}[/]")
    return rows


def infer_backend(path):
    base = path.rsplit("/", 1)[-1]
    parts = base.split("_")
    for i, p in enumerate(parts):
        if p in ("gpu", "nnapi", "cpu") and i > 0 and parts[i - 1] in ("litert", "onnx"):
            return p
    return "unknown"


def is_impossible(r):
    for key in ("ttft_ms", "tbt_ms", "overall_ms"):
        v = r.get(key)
        if v is None:
            return True
        try:
            if float(v) <= 0:
                return True
        except (TypeError, ValueError):
            return True
    try:
        if int(r.get("tokens", 0)) <= 0:
            return True
        if float(r.get("decode_speed_tps", 0)) <= 0:
            return True
    except (TypeError, ValueError):
        return True
    return False


def default_paths():
    return sorted(glob.glob("results_*.jsonl"))


def main():
    parser = argparse.ArgumentParser(description="Summarize MoRAGBench JSONL results.")
    parser.add_argument("files", nargs="*", help="results_*.jsonl files (default: all)")
    parser.add_argument("--raw", action="store_true", help="also print each sample's rows")
    args = parser.parse_args()

    paths = args.files or default_paths()
    if not paths:
        console.print("[red]No results_*.jsonl files found.[/]")
        return

    rows = load_rows(paths)
    if not rows:
        console.print("[red]No rows in the given files.[/]")
        return

    # group strictly by (engine, backend); default missing backend to "unknown"
    # so different backends are never merged into one bucket.
    groups = {}
    for r in rows:
        key = (r.get("engine", "?"), r.get("backend") or "unknown")
        groups.setdefault(key, []).append(r)

    for (engine, backend), runs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(runs)
        label = f"{engine} / {backend}"
        table = Table(title=f"{label} ({n} samples)")
        table.add_column("Metric", style="cyan")
        table.add_column("Avg", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        for key, name in METRICS:
            vals = [float(r[key]) for r in runs if r.get(key) is not None and r[key] != ""]
            if not vals:
                continue
            table.add_row(
                name,
                f"{statistics.mean(vals):.2f}",
                f"{statistics.median(vals):.2f}",
                f"{min(vals):.2f}",
                f"{max(vals):.2f}",
            )
        table.add_row("Succeeded", f"{n}", "", "", "")
        console.print(table)

    if args.raw:
        console.print("\n[bold]Raw rows:[/]")
        for r in rows:
            console.print(json.dumps(r))


if __name__ == "__main__":
    main()
