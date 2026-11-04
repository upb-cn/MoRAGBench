#!/usr/bin/env python3
"""MoRAGBench - interactive agentic RAG client.

Posts a query to the on-device agentic pipeline and renders the agent trace live:
every step, which agent is working, what it returned, and per-agent timings.

    python agent.py -d 9146d9a3 -q "Who founded Foo Corp and when?"
    python agent.py --trace agent_trace.jsonl        # replay a saved run, no device
    python agent.py -d 9146d9a3 -q "..." --save t.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers import (  # noqa: E402
    HTTP_BASE,
    CONFIG_PATH,
    SDCARD_BASE,
    TASK_ROOT,
    adb_cmd,
    ensure_server,
    generate_post,
    pull,
    _connect_refused,
)
import score  # noqa: E402

console = Console()

AGENT_STYLE = {
    "Evidence Investigator": "bold magenta",
    "Answerer": "bold green",
    "Retriever": "bold blue",
    "system": "dim",
    "error": "bold red",
}

# Steps that end a phase rather than finish an agent.
TERMINAL = "end"
POLL_S = 0.1
LEVEL = "full"


def push_config(device, mutate):
    """Pull the device task config, apply mutate(config), push it back."""
    import subprocess
    import tempfile

    local = tempfile.mktemp(suffix=".json")
    try:
        subprocess.run(
            adb_cmd(device, "pull", CONFIG_PATH, local),
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        with open(local) as f:
            config = json.load(f)
        mutate(config)
        with open(local, "w") as f:
            json.dump(config, f, indent=4)
        subprocess.run(
            adb_cmd(device, "push", local, CONFIG_PATH),
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    finally:
        Path(local).unlink(missing_ok=True)


def set_aug_method(device, aug):
    push_config(device, lambda c: c["rag_pipeline"]["llm"].__setitem__("aug_method", aug))


def set_engine(device, engine, backend, model):
    """Switch the device LLM engine/backend/model. None means leave unchanged."""

    def mutate(c):
        llm = c["rag_pipeline"]["llm"]
        if engine:
            llm["engine"] = engine
        if backend:
            llm["backend"] = backend
        if model:
            llm["model_name"] = model

    push_config(device, mutate)


# The agentic pipeline's config file: prompts and generation params edited here
# are pushed to the device before every run (agent.py, compare_pipelines.py).
AGENTIC_CONFIG = Path(__file__).resolve().parent / "agentic_pipeline_config.json"


def sync_llm_config(device, cfg_path=None):
    """Push rag_pipeline.llm from the local agentic config file onto the device.

    The file wins over whatever the device holds; callers run set_aug_method /
    set_engine AFTER this, so CLI flags still override the file per run.
    Null-valued keys are skipped (older files shouldn't wipe device settings)."""
    cfg_path = Path(cfg_path) if cfg_path else AGENTIC_CONFIG
    if not cfg_path.exists():
        return False
    llm_cfg = (json.loads(cfg_path.read_text()).get("rag_pipeline") or {}).get("llm") or {}
    if not llm_cfg:
        return False
    fields = {k: v for k, v in llm_cfg.items() if v is not None}
    push_config(device, lambda c: c["rag_pipeline"]["llm"].update(fields))
    console.print(f"[dim]config: {len(fields)} llm fields synced from {cfg_path.name}[/]")
    return True


def stream_device(query, force_complex=False):
    """POST the query, then yield events as the device produces them."""
    for _ in range(100):
        try:
            generate_post("agent_query", {"query": query, "force_complex": force_complex})
            break
        except RuntimeError as e:
            # 409 = a previous (orphaned) task is still running on the device;
            # wait it out — the event log resets when the next task starts.
            if "409" not in str(e):
                raise
            time.sleep(3)
    yield from _poll()


def _poll():
    """Yield new events until the device reports the run left RUNNING."""
    since = 0
    started = False
    stalls = 0
    while True:
        try:
            r = requests.get(f"{HTTP_BASE}/agent_events?since={since}", timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            # The phone freezes for minutes under load (swap stalls) — keep
            # waiting those out (~5 min). A refused connection means the app
            # is gone: give up fast instead of stalling every question.
            if _connect_refused():
                console.print(f"[red]event poll failed: {e}[/]")
                if started:
                    console.print("[red]Lost the device. The app probably died; check "
                                  "`adb logcat` and that this engine/backend combination is "
                                  "supported on this phone.[/]")
                return
            stalls += 1
            if stalls > 25:
                console.print(f"[red]event poll stalled ~5 min, giving up: {e}[/]")
                return
            time.sleep(2)
            continue
        stalls = 0

        for raw in data.get("events", []):
            ev = json.loads(raw) if isinstance(raw, str) else raw
            since = ev["seq"] + 1
            started = True
            yield ev

        state = data.get("state", "")
        if state and state != "RUNNING":
            return
        time.sleep(POLL_S)


def stream_file(path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def ts(ev):
    return f"[dim]+{ev['tMs'] / 1000:6.2f}s[/]"


def dur(ev):
    ms = ev.get("durationMs", 0)
    if not ms:
        return ""
    bits = [f"{ms / 1000:.2f}s"]
    if ev.get("genTokens"):
        bits.append(f"{ev['genTokens']} tok")
    if ev.get("decodeTps"):
        bits.append(f"{ev['decodeTps']:.1f} tok/s")
    if ev.get("ttftMs"):
        bits.append(f"ttft {ev['ttftMs']}ms")
    return "[dim]" + " · ".join(bits) + "[/]"


def style(ev):
    return AGENT_STYLE.get(ev["agent"], "white")


def render(ev):
    agent, step, detail, out = ev["agent"], ev["step"], ev["detail"], ev["output"]
    s = style(ev)

    if step == "start":
        return  # shown by the live spinner instead

    if step == "item":
        console.print(f"  {ts(ev)} [dim]│[/] {out}")
        return

    if step == "fallback":
        console.print(f"  {ts(ev)} [{s}]● {agent}[/] [{s}]{detail}[/]  {dur(ev)}")
        if out:
            console.print(f"  {ts(ev)} [dim]│ raw:[/] [dim]{one_line(out)}[/]")
        return

    if step == "error":
        console.print(f"  {ts(ev)} [bold red]✗ {agent}[/] [red]{detail}[/]")
        return

    if step == TERMINAL:
        if detail == "final answer":
            console.print(f"  {ts(ev)} [{s}]● {agent}[/]  {dur(ev)}\n")
            body = Text(out.strip() or "(empty)", style="bold")
            console.print(Panel(body, title="[bold]Final Answer[/]",
                                border_style="green", padding=(1, 2)))
        else:
            # When detail *is* the response print it only
            # once, on the grey response line, so every LLM step reads the
            # same: header, then the model's raw answer beneath it.
            head = "" if one_line(out) == detail else f" [{s}]{detail}[/]"
            console.print(f"  {ts(ev)} [{s}]● {agent}[/]{head}  {dur(ev)}")
            if out and agent != "Retriever":
                console.print(Panel(one_line(out), title=f"[dim]{agent} response[/]",
                                    border_style="grey37", padding=(0, 1)))


def one_line(s, n=110):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def summary(evs):
    total_ms = max((e["tMs"] for e in evs), default=0)
    calls = [e for e in evs if e["step"] == TERMINAL and e.get("genTokens")]
    chips = [
        f"[bold]{total_ms / 1000:.2f}s[/] total",
        f"[bold]{len(calls)}[/] LLM calls",
        f"{sum(e['inputTokens'] for e in calls)} in / "
        f"{sum(e['genTokens'] for e in calls)} out tok",
    ]
    return "  ·  ".join(chips)


def run(stream):
    evs = []
    with console.status("[dim]connecting…[/]") as status:
        for ev in stream:
            evs.append(ev)
            if ev["step"] == "start":
                status.update(f"[{style(ev)}]{ev['agent']}[/]  [dim]{ev['detail']}[/]")
            elif ev["step"] not in ("context",) and (
                    LEVEL == "full" or ev["detail"] == "final answer"):
                render(ev)
    console.print()
    console.rule("[dim]summary[/]", style="dim")
    console.print("  " + summary(evs))
    return evs


def print_metrics(device, query, evs):
    """EM/F1/retrieval_hit via score.py (the same functions compare_pipelines
    sweeps with); gold from the device task's questions/references files."""
    qs, refs, golds = None, None, None
    cfg = pull("config.json", device, f"{SDCARD_BASE}/task_files", required=False)
    if cfg:
        task = cfg["downstream_task"]["name"]
        qs = pull("questions.json", device, f"{TASK_ROOT}/{task}", required=False)
        refs = pull("references.json", device, f"{TASK_ROOT}/{task}", required=False)
    if qs and refs and query.strip() in [q.strip() for q in qs]:
        golds = refs[[q.strip() for q in qs].index(query.strip())]
    if golds is None:
        console.print(f"[dim]query not in {cfg['downstream_task']['name'] if cfg else '?'} task "
                      f"questions — no gold, skipping em/f1/retrieval_hit[/]")
        return

    errors = [e for e in evs if e["step"] == "error"]
    final = [e for e in evs if e["detail"] == "final answer"]
    pred = score.answer_of(final[0]["output"]) if final and not errors else ""
    retrievals = len([e for e in evs if e["agent"] == "Retriever"
                      and e["step"] == "end" and "skipped" not in e["detail"]])
    ttft = final[0].get("ttftMs", 0) if final and not errors else 0
    console.print(
        f"  em={int(score.em(pred, golds))}  ·  f1={score.f1(pred, golds):.2f}  ·  "
        f"retrieval_hit={score.retrieval_hit(golds, evs)}  ·  retrievals={retrievals}  ·  "
        f"ttft={ttft}ms  ·  gold={golds!r}  ·  pred={pred!r}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", "-d", help="ADB serial. Omit with --trace.")
    p.add_argument("--query", "-q", help="Query to send.")
    p.add_argument("--trace", help="Replay a saved trace instead of using a device.")
    p.add_argument("--save", help="Write the event stream to this .jsonl path.")
    p.add_argument("--level", default="full", choices=["full", "quiet"],
                   help="'full' shows every agent's raw output; 'quiet' shows only "
                        "the final answer.")
    p.add_argument("--aug", default="agentic",
                   choices=["none", "concatenation", "agentic"],
                   help="Device pipeline to run. 'none' = closed book, no retrieval; "
                        "'concatenation' = traditional single-shot RAG.")
    p.add_argument("--engine", choices=["onnx", "litert"], help="Override device engine.")
    p.add_argument("--backend", help="Override device backend (onnx: cpu|xnnpack|nnapi; litert: cpu|gpu|npu).")
    p.add_argument("--model", help="Override device model_name.")
    p.add_argument("--force-complex", action="store_true",
                   help="Mandate a QUERY in investigator round 1 (skip the "
                        "ANSWERABLE shortcut). Small models answer too eagerly; "
                        "use this to exercise multi-round retrieval.")
    args = p.parse_args()
    global LEVEL
    LEVEL = args.level

    if args.trace:
        evs = run(stream_file(args.trace))
    else:
        if not args.device or not args.query:
            p.error("--device and --query are required without --trace")
        if ensure_server(args.device) is None and _connect_refused():
            console.print("[red]Device server is not running and could not be started.[/]")
            return 1
        # agentic_pipeline_config.json drives the run; flags below override it
        sync_llm_config(args.device)
        set_aug_method(args.device, args.aug)
        if args.engine or args.backend or args.model:
            set_engine(args.device, args.engine, args.backend, args.model)
        console.print(f"[dim]{args.aug} pipeline · {args.device}[/]\n")
        evs = run(stream_device(args.query, args.force_complex))

    if args.save is None and evs:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent / "runs" / f"agent_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.save = str(out_dir / "trace.jsonl")
    if args.save:
        with open(args.save, "w") as f:
            for ev in evs:
                f.write(json.dumps(ev) + "\n")
        console.print(f"[dim]trace saved to {args.save}[/]")
    if not args.trace and args.query:
        print_metrics(args.device, args.query, evs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
