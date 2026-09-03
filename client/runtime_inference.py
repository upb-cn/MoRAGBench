#!/usr/bin/env python3
"""MoRAGBench - LLM inference against the Android CLI server.

Runs inference using either ONNX or LiteRT engine. The engine is selected
via the --engine flag and written to the on-device config.json before the
request is made.
"""
import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from helpers import (
    DEFAULT_PROMPT,
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


def check_model(device, engine):
    if engine == "litert":
        model_path = f"{SDCARD_BASE}/{LITERT_MODEL}"
    else:
        model_path = f"{SDCARD_BASE}/task_files/llm/{ONNX_MODEL_DIR}/model.onnx"

    if not check_model_on_device(device, model_path):
        console.print(
            Panel(
                f"{engine.upper()} model not found on device:\n{model_path}\n\n"
                "Push it to the device before running inference.",
                title="[red]Warning[/]",
                border_style="red",
            )
        )
    else:
        console.print(f"[green]{engine.upper()} model found on device.[/]")


def show_input(args, engine):
    table = Table(show_header=False, expand=False)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Device", args.device)
    table.add_row("Engine", engine)
    table.add_row("Prompt", args.prompt)
    table.add_row("System prompt", args.system_prompt or "(none)")
    table.add_row("Max tokens", str(args.max_tokens))
    console.print(Panel(table, title="[bold]Inference request[/]", border_style="blue"))


def show_response(result):
    # Server may nest metrics under a "metrics" key
    m = result.get("metrics", result)

    console.print(
        Panel(
            Text(result.get("response", "")),
            title="[bold]Response[/]",
            border_style="green",
        )
    )

    status = m.get("status", result.get("status", "?"))
    if status and status != "OK":
        console.print(f"[red]Status: {status}[/]")
        if m.get("error") or result.get("error"):
            console.print(f"[red]Error: {m.get('error') or result.get('error')}[/]")
        return

    speed = m.get("decoding_speed_tokens_per_sec", 0.0) or 0.0
    gen = m.get("generated_tokens", 0) or 0
    avg_tbt = (1000.0 / speed) if speed > 0 else None

    table = Table(title="[bold]Metrics[/]", border_style="green")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")
    table.add_row("Status", status or "?")
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
        description="Run LLM inference on the Android device",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", "-d", required=True, help="ADB device serial")
    parser.add_argument(
        "--engine", "-e",
        required=True,
        choices=["onnx", "litert"],
        help="Inference engine to use",
    )
    parser.add_argument("--prompt", "-p", default=DEFAULT_PROMPT, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generated tokens")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt")
    args = parser.parse_args()

    engine = args.engine

    console.print(
        Panel(
            f"MoRAGBench - [cyan]{engine.upper()}[/] inference on [bold]{args.device}[/]",
            border_style="blue",
        )
    )

    show_input(args, engine)
    check_model(args.device, engine)

    with console.status("[bold]Checking server...[/]"):
        ping = ensure_server(args.device)
    if ping is None and _connect_refused():
        console.print("[red]Server is not running and could not be started. Check the device and ADB port forwarding.[/]")
        sys.exit(1)
    loaded = ping.get(f"{engine}_loaded", False) if ping else False
    console.print(
        f"[green]Server up.[/] Model loaded: "
        f"[{'green' if loaded else 'yellow'}]{loaded}[/]"
        + ("" if loaded else " (first request may take a while to load the model)")
    )

    set_engine(args.device, engine)

    payload = {"prompt": args.prompt, "max_tokens": args.max_tokens}
    if args.system_prompt:
        payload["system_prompt"] = args.system_prompt

    try:
        with console.status("[bold yellow]Generating...[/]"):
            result = generate_post("generate", payload)
    except Exception as e:
        console.print(f"[red]Request failed: {e}[/]")
        sys.exit(1)

    show_response(result)


if __name__ == "__main__":
    main()
