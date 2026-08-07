#!/usr/bin/env python3
"""MoRAGBench - LiteRT LM inference against the Android CLI server.

The server exposes a single /generate endpoint; the engine (onnx/litert)
is selected through the `engine` field in the on-device config.json.
This script pins the engine to LiteRT before running.
"""
import argparse
import subprocess
import sys
import time

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

SERVER_PACKAGE = "com.example.cli"
PORT = 18080
CONFIG_PATH = (
    f"/storage/emulated/0/Android/data/{SERVER_PACKAGE}/files/task_files/config.json"
)
MODEL_FILENAME = "Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm"
ENGINE = "litert"

console = Console()


def run(cmd, check=True):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and r.returncode != 0:
        console.print(f"[red]ERROR: {' '.join(cmd)}\n{r.stderr.strip()}[/]")
        sys.exit(1)
    return r.stdout.strip()


def ensure_server(adb):
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/ping", timeout=5)
        if r.ok:
            return r.json()
        raise RuntimeError(f"server returned {r.status_code}")
    except Exception:
        console.print("[yellow]Server not responding - starting it...[/]")
        run(adb + ["shell", "am", "stopservice", "-n", f"{SERVER_PACKAGE}/.ServerService"])
        time.sleep(1)
        run(adb + ["shell", "am", "start-foreground-service", f"{SERVER_PACKAGE}/.ServerService"])
        run(adb + ["forward", f"tcp:{PORT}", f"tcp:{PORT}"])
        time.sleep(2)
        return {}


def set_engine(adb):
    with console.status(f"[bold]Setting engine to [cyan]{ENGINE}[/] in device config...[/]"):
        script = (
            f"sed -i 's/\"engine\": \"[^\"]*\"/\"engine\": \"{ENGINE}\"/' {CONFIG_PATH}; "
            f"if ! grep -q '\"engine\"' {CONFIG_PATH}; then "
            f"sed -i '/\"llm\": {{/a\\    \"engine\": \"{ENGINE}\",' {CONFIG_PATH}; fi; "
            f"grep '\"engine\"' {CONFIG_PATH}"
        )
        applied = run(adb + ["shell", script])
    if ENGINE not in applied:
        console.print("[red]Failed to set engine in config.[/]")
        sys.exit(1)


def check_model(adb):
    model_path = f"/storage/emulated/0/Android/data/{SERVER_PACKAGE}/files/{MODEL_FILENAME}"
    with console.status("[bold]Checking model on device...[/]"):
        found = (
            subprocess.run(
                adb + ["shell", "test", "-f", model_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    if not found:
        console.print(
            Panel(
                f"LiteRT model not found on device:\n{model_path}\n\n"
                "Push it to the device before running LiteRT inference.",
                title="[red]Warning[/]",
                border_style="red",
            )
        )
    else:
        console.print(f"[green]LiteRT model found on device.[/]")


def show_input(args):
    table = Table(show_header=False, expand=False)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Device", args.device)
    table.add_row("Engine", ENGINE)
    table.add_row("Prompt", args.prompt)
    table.add_row("System prompt", args.system_prompt or "(none)")
    table.add_row("Max tokens", str(args.max_tokens))
    console.print(Panel(table, title="[bold]Inference request[/]", border_style="blue"))


def show_metrics(m):
    gen = m.get("generated_tokens", 0)
    speed = m.get("decoding_speed_tokens_per_sec", 0.0) or 0.0
    avg_tbt = (1000.0 / speed) if speed > 0 else None

    table = Table(title="[bold]Metrics[/]", border_style="green")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")
    table.add_row("Status", m.get("status", "?"))
    table.add_row("Input tokens", str(m.get("input_tokens", "?")))
    table.add_row("Generated tokens", str(gen))
    table.add_row("Load time", f"{m.get('load_time_ms', 0)} ms")
    table.add_row("TTFT", f"{m.get('ttft_ms', 0)} ms")
    table.add_row("TBT (avg)", f"{avg_tbt:.2f} ms" if avg_tbt else "-")
    table.add_row("Decoding speed", f"{speed:.2f} tok/s")
    table.add_row("Overall duration", f"{m.get('overall_duration_ms', 0)} ms")
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Run LiteRT LM inference on the Android device",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", "-d", required=True, help="ADB device serial")
    parser.add_argument(
        "--prompt", "-p",
        default="Write a short paragraph about the impact of artificial intelligence "
                "on modern society.",
        help="Input prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generated tokens")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt")
    args = parser.parse_args()

    adb = ["adb", "-s", args.device]

    console.print(
        Panel(
            f"MoRAGBench - [cyan]{ENGINE.upper()}[/] inference on [bold]{args.device}[/]",
            border_style="blue",
        )
    )

    show_input(args)
    check_model(adb)

    with console.status("[bold]Checking server...[/]"):
        ping = ensure_server(adb)
    loaded = ping.get("litert_loaded", False)
    console.print(
        f"[green]Server up.[/] Model loaded: [{'green' if loaded else 'yellow'}]{loaded}[/]"
        + ("" if loaded else " (first request may take a while to load the model)")
    )

    set_engine(adb)

    payload = {"prompt": args.prompt, "max_tokens": args.max_tokens}
    if args.system_prompt:
        payload["system_prompt"] = args.system_prompt

    try:
        with console.status("[bold yellow]Generating...[/]"):
            r = requests.post(f"http://127.0.0.1:{PORT}/generate", json=payload, timeout=600)
            r.raise_for_status()
        result = r.json()
    except Exception as e:
        console.print(f"[red]Request failed: {e}[/]")
        sys.exit(1)

    console.print(
        Panel(
            Text(result.get("response", "")),
            title="[bold]Response[/]",
            border_style="green",
        )
    )

    if "metrics" in result:
        show_metrics(result["metrics"])


if __name__ == "__main__":
    main()
