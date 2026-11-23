"""MoRAGBench web UI: configure, run and inspect benchmarks.

    streamlit run ui/app.py

Runs on the laptop next to the Python client. The phone stays the benchmark
server; this UI only writes configs, calls client/main.py and reads results.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

UI_DIR = Path(__file__).resolve().parent
REPO = UI_DIR.parent
CLIENT = REPO / "client"
BENCH = REPO / "benchmark"
UI_CONFIG_DIR = BENCH / "ui_configs"
UI_RESULTS_DIR = BENCH / "ui_results"

sys.path.insert(0, str(CLIENT))  # import the client's Pydantic config classes
sys.path.insert(0, str(UI_DIR))

from classes.benchmark import ANNBenchmark, TaskBenchmark  # noqa: E402
import results_data as rd  # noqa: E402
import schema_form as sf  # noqa: E402
from runner import Runner, adb_devices, progress_from_status  # noqa: E402

KINDS = {"task": TaskBenchmark, "ann": ANNBenchmark}
KIND_LABEL = {"task": "Task benchmark (end-to-end RAG)", "ann": "ANN benchmark (retrieval only)"}

st.set_page_config(page_title="MoRAGBench", page_icon="📱", layout="wide")


@st.cache_resource
def get_runner() -> Runner:
    return Runner(CLIENT)


def kind_of(cfg: dict) -> str:
    return "task" if "downstream_task" in cfg else "ann"


def existing_configs(kind: str) -> list[Path]:
    paths = []
    for pattern in ("ui_configs/*.json", "configs/*.json", "*/completed_configs/*.json"):
        paths += BENCH.glob(pattern)
    out = []
    for p in sorted(set(paths), key=lambda p: (p.parent.name != "ui_configs", str(p))):
        try:
            if kind_of(json.loads(p.read_text(encoding="utf-8"))) == kind:
                out.append(p)
        except (OSError, json.JSONDecodeError):
            pass
    return out


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


# ════════════════════════════════════════════════════════════════════════════
# Configure
# ════════════════════════════════════════════════════════════════════════════


def sections_of(model_cls):
    """Split a benchmark model into tabs: nested models of nested models
    (rag_pipeline.embedding, ...) become their own tab."""
    tabs, leftovers = [], []
    for name, f in model_cls.model_fields.items():
        tp = f.annotation
        if sf._is_model(tp):
            subs = tp.model_fields
            if subs and all(sf._is_model(s.annotation) for s in subs.values()):
                tabs += [((name, sub), s.annotation) for sub, s in subs.items()]
            else:
                tabs.append(((name,), tp))
        else:
            leftovers.append(name)
    return tabs, leftovers


def configure_page():
    st.title("Configure a benchmark")
    st.caption("Every option below is read from the Pydantic classes in `client/classes/`, "
               "and validated exactly like `main.py` does.")

    kind = st.segmented_control("Benchmark type", list(KINDS), format_func=KIND_LABEL.get,
                                default=st.session_state.get("cfg_kind", "task"), key="cfg_kind_widget")
    kind = kind or "task"
    if st.session_state.get("cfg_kind") != kind:
        st.session_state.cfg_kind = kind
        st.session_state.cfg_initial = {}
        sf.reset_form_state()
    model_cls = KINDS[kind]
    initial = st.session_state.get("cfg_initial", {})

    with st.expander("Start from an existing config", expanded=False):
        c1, c2 = st.columns([3, 2])
        options = existing_configs(kind)
        pick = c1.selectbox("Saved configs", options, format_func=rel, index=None,
                            placeholder=f"{len(options)} configs found in benchmark/")
        upload = c2.file_uploader("…or upload a JSON", type="json")
        b1, b2 = st.columns([1, 6])
        if b1.button("Load", disabled=not (pick or upload)):
            data = json.load(upload) if upload else json.loads(pick.read_text(encoding="utf-8"))
            if kind_of(data) != kind:
                st.error("That file is a different benchmark type.")
            else:
                data.pop("hf_token", None)
                st.session_state.cfg_initial = data
                st.session_state.cfg_name = (Path(upload.name).stem if upload else pick.stem)
                sf.reset_form_state()
                st.rerun()
        if b2.button("Reset to defaults"):
            st.session_state.cfg_initial = {}
            sf.reset_form_state()
            st.rerun()

    form_col, side_col = st.columns([3, 2], gap="large")

    with form_col:
        tabs_spec, leftovers = sections_of(model_cls)
        tabs = st.tabs([path[-1] for path, _ in tabs_spec] + (["other"] if leftovers else []))
        cfg: dict = {}
        for tab, (path, cls) in zip(tabs, tabs_spec):
            with tab:
                sub_init = initial
                for p in path:
                    sub_init = sub_init.get(p, {}) if isinstance(sub_init, dict) else {}
                values = sf.render_model(cls, sub_init, prefix=".".join(path), section=path[-1])
                target = cfg
                for p in path[:-1]:
                    target = target.setdefault(p, {})
                target[path[-1]] = values
        if leftovers:
            with tabs[-1]:
                lo_fields = {n: model_cls.model_fields[n] for n in leftovers}
                for n, f in lo_fields.items():
                    v, emit = sf._render_field(n, f, initial, "")
                    if emit:
                        cfg[n] = v

    with side_col:
        model, errors = sf.validate(model_cls, cfg)
        if errors:
            st.error("**Invalid configuration**\n\n" + "\n".join(f"- {e}" for e in errors))
        else:
            st.success("Valid configuration")

        public = {k: v for k, v in cfg.items() if k != "hf_token"}
        st.code(json.dumps(public, indent=2), language="json", height=420)

        name = st.text_input("File name", value=st.session_state.get("cfg_name", f"{kind}_config"))
        c1, c2, c3 = st.columns(3)
        save_path = UI_CONFIG_DIR / f"{name}.json"
        if c1.button("Save", disabled=bool(errors), width="stretch",
                     help=f"Writes {rel(save_path)}"):
            UI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(public, indent=2), encoding="utf-8")
            st.toast(f"Saved {rel(save_path)}")
        c2.download_button("Download", json.dumps(public, indent=2), file_name=f"{name}.json",
                           mime="application/json", disabled=bool(errors), width="stretch")
        if c3.button("Run it", type="primary", disabled=bool(errors), width="stretch"):
            UI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(public, indent=2), encoding="utf-8")
            st.session_state.run_config = str(save_path)
            st.session_state.run_token = cfg.get("hf_token")
            st.switch_page(RUN_PAGE)
        st.caption("Fields left at their default are omitted, as in the hand-written configs. "
                   "`main.py` fills them in and lists them as `[default]`.")


# ════════════════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════════════════


def run_page():
    st.title("Run on the phone")
    runner = get_runner()

    if not (CLIENT / ".env").exists():
        st.error("`client/.env` is missing. Copy `client/.env.example` to `client/.env` first; "
                 "`main.py` reads the port and phone paths from it.")

    job = runner.current
    busy = bool(job and job.running)

    with st.container(border=True):
        c1, c2 = st.columns(2)
        configs = existing_configs("task") + existing_configs("ann")
        preset = st.session_state.get("run_config")
        idx = next((i for i, p in enumerate(configs) if str(p) == preset), 0) if configs else None
        config = c1.selectbox("Config", configs, index=idx, format_func=rel, disabled=busy)

        devices, dev_err = adb_devices()
        if devices:
            device = c2.selectbox("Device", devices, disabled=busy)
        else:
            device = c2.text_input("Device serial", disabled=busy, help=dev_err)
            if dev_err:
                c2.caption(f"⚠ {dev_err}")

        default_out = UI_RESULTS_DIR / (f"{config.stem}_{datetime.now():%Y%m%d-%H%M}" if config else "run")
        out = st.text_input("Output folder", value=rel(default_out), disabled=busy)
        resume = st.checkbox("Resume an interrupted run (--resume)", disabled=busy)

        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("Start benchmark", type="primary", disabled=busy or not (config and device)):
            out_path = (REPO / out) if not Path(out).is_absolute() else Path(out)
            token = st.session_state.get("run_token")
            overrides = [f"hf_token={token}"] if token and str(config) == preset else []
            runner.start(config.resolve(), out_path.resolve(), device, resume=resume, overrides=overrides)
            st.rerun()
        if b2.button("Stop", disabled=not busy):
            runner.stop()
            st.rerun()

    if job:
        live_panel()
    else:
        st.info("No benchmark started in this session yet.")


@st.fragment(run_every=2)
def live_panel():
    runner = get_runner()
    job = runner.current
    if job is None:
        return

    elapsed = (job.finished or time.time()) - job.started
    if job.running:
        state = "Running"
    elif job.returncode == 0:
        state = "Finished"
    else:
        state = f"Failed (exit {job.returncode})"

    c1, c2, c3 = st.columns(3)
    c1.metric("Client", state)
    c2.metric("Elapsed", f"{int(elapsed // 60)}m {int(elapsed % 60)}s")
    c3.metric("Config", job.config_path.stem)

    if job.running:
        st_ = runner.status()
        if st_:
            p = progress_from_status(st_)
            st.caption(f"Phone: **{p['state']}** · phase {p['phase']} · index {p.get('index_state')} "
                       f"({p.get('index_source')})")
            for label, keyname in (("Indexing", "index"), ("Evaluating", "eval")):
                done, total, unit = p.get(keyname, (None, None, ""))
                if total:
                    st.progress(min(done / total, 1.0), text=f"{label}: {done}/{total} {unit}")
                else:
                    st.progress(0.0, text=f"{label}: waiting")
            if p.get("error"):
                st.error(p["error"])
        else:
            st.caption("Waiting for the phone server (parsing, transferring files or starting the service)…")

    if not job.running and job.returncode == 0:
        st.success(f"Results saved to `{rel(job.output_path)}`")
        if st.button("Open in Results"):
            st.session_state.results_focus = str(job.output_path)
            st.cache_data.clear()
            st.switch_page(RESULTS_PAGE)

    with st.expander("Client log", expanded=job.running or job.returncode not in (0, None)):
        st.code("\n".join(job.log[-300:]) or "(no output yet)", language=None, height=360)


# ════════════════════════════════════════════════════════════════════════════
# Results
# ════════════════════════════════════════════════════════════════════════════

TASK_METRICS = ["f1", "exact_match", "contains", "rouge1", "rougeL", "bleu", "ttft_avg_ms", "ttft_p95_ms",
                "decode_tok_s", "tpot_avg_ms", "query_embed_ms", "retrieve_ms", "index_build_s", "total_min",
                "mem_peak_mb", "mem_mean_mb", "mem_before_mb", "power_mean_mw", "temp_peak_c"]
ANN_METRICS = ["recall", "precision", "qps", "latency_mean_ms", "latency_p95_ms", "build_ms", "train_ms",
               "mem_peak_mb", "mem_mean_mb", "power_mean_mw"]
LABELS = {
    "f1": "F1", "exact_match": "Exact match", "contains": "Contains", "rouge1": "ROUGE-1", "rougeL": "ROUGE-L",
    "bleu": "BLEU", "ttft_avg_ms": "TTFT avg (ms)", "ttft_p95_ms": "TTFT p95 (ms)",
    "decode_tok_s": "Decoding (tok/s)", "tpot_avg_ms": "Time per output token (ms)",
    "query_embed_ms": "Query embedding (ms)", "retrieve_ms": "Retrieval (ms)", "index_build_s": "Index build (s)",
    "total_min": "Total time (min)", "mem_peak_mb": "Peak memory (MB)", "mem_mean_mb": "Mean memory (MB)",
    "mem_before_mb": "Memory before (MB)", "power_mean_mw": "Mean power (mW)", "temp_peak_c": "Peak temp (°C)",
    "recall": "Recall@k", "precision": "Precision@k", "qps": "Queries / s", "latency_mean_ms": "Latency mean (ms)",
    "latency_p95_ms": "Latency p95 (ms)", "build_ms": "Index build (ms)", "train_ms": "Training (ms)",
}
LOWER_IS_BETTER = {"ttft_avg_ms", "ttft_p95_ms", "tpot_avg_ms", "query_embed_ms", "retrieve_ms", "index_build_s",
                   "total_min", "mem_peak_mb", "mem_mean_mb", "power_mean_mw", "temp_peak_c",
                   "latency_mean_ms", "latency_p95_ms", "build_ms", "train_ms"}


@st.cache_data(show_spinner="Reading result folders…")
def cached_runs(roots: tuple[str, ...]):
    return rd.load_runs([Path(r) for r in roots], REPO)


@st.cache_data
def cached_generations(path: str):
    return rd.load_generations(Path(path))


def results_page():
    st.title("Results")

    with st.sidebar:
        st.subheader("Result folders")
        extra = st.text_area("Extra folders (one per line)", key="extra_roots",
                             help="Any folder containing run folders with metrics.json + config.json.")
        if st.button("Reload results"):
            st.cache_data.clear()
    roots = [str(BENCH), str(CLIENT / "benchmark_results")] + [l.strip() for l in extra.splitlines() if l.strip()]
    task_df, ann_df = cached_runs(tuple(roots))

    if task_df.empty and ann_df.empty:
        st.info("No results found. Run a benchmark or add a folder in the sidebar.")
        return

    focus = st.session_state.get("results_focus")
    default_kind = "task"
    if focus and not ann_df.empty and focus in set(ann_df.path):
        default_kind = "ann"
    kind = st.segmented_control(
        "Benchmark type", ["task", "ann"], default=default_kind, key="res_kind",
        format_func=lambda k: f"Task runs ({len(task_df)})" if k == "task" else f"ANN runs ({len(ann_df)})",
    ) or "task"
    df = task_df if kind == "task" else ann_df
    if df.empty:
        st.info("No runs of this type.")
        return
    metrics = [m for m in (TASK_METRICS if kind == "task" else ANN_METRICS) if m in df.columns]
    cfg_cols = rd.config_columns(df, kind)

    # ── filters ──
    with st.expander("Filters", expanded=False):
        varying = [c for c in cfg_cols if df[c].nunique(dropna=True) > 1 and c != "index_params"]
        fcols = st.columns(4)
        mask = pd.Series(True, index=df.index)
        for i, c in enumerate(varying):
            vals = sorted(df[c].dropna().unique().tolist(), key=str)
            chosen = fcols[i % 4].multiselect(c, vals, key=f"f_{kind}_{c}")
            if chosen:
                mask &= df[c].isin(chosen)
    view = df[mask].copy()
    st.caption(f"{len(view)} of {len(df)} runs")

    t_table, t_compare, t_detail = st.tabs(["Runs", "Compare", "Run detail"])

    with t_table:
        show_cols = ["suite", "run"] + [c for c in cfg_cols if c != "suite" and view[c].nunique(dropna=True) > 1] \
            + metrics
        show_cols = list(dict.fromkeys(show_cols))
        colcfg = {m: st.column_config.NumberColumn(LABELS.get(m, m), format="%.3f" if view[m].abs().max() < 10
                                                   else "%.1f") for m in metrics if m in view}
        event = st.dataframe(view[show_cols], hide_index=True, width="stretch", height=460,
                             column_config=colcfg, on_select="rerun", selection_mode="multi-row",
                             key=f"tbl_{kind}")
        sel_rows = event.selection.rows if event and event.selection else []
        st.session_state[f"selected_{kind}"] = view.iloc[sel_rows]["path"].tolist() if sel_rows else []
        st.caption("Select rows to compare them side by side or open them in *Run detail*.")
        st.download_button("Download table as CSV", view[show_cols].to_csv(index=False),
                           file_name=f"moragbench_{kind}_runs.csv", mime="text/csv")

    with t_compare:
        compare_tab(view, kind, metrics, cfg_cols)

    with t_detail:
        detail_tab(view, kind)


def compare_tab(view: pd.DataFrame, kind: str, metrics: list[str], cfg_cols: list[str]):
    selected = st.session_state.get(f"selected_{kind}", [])
    if len(selected) >= 2:
        st.subheader("Selected runs side by side")
        sub = view[view.path.isin(selected)].set_index("run")
        differing = [c for c in cfg_cols if sub[c].astype(str).nunique() > 1]
        table = sub[differing + metrics].T
        table.index = [LABELS.get(i, i) for i in table.index]
        st.dataframe(table.astype(str), width="stretch")

    st.subheader("Preset")
    if kind == "task":
        color = st.selectbox("Colour by", [c for c in cfg_cols if view[c].nunique() > 1] or cfg_cols,
                             index=0, key="preset_color")
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(scatter(view, "ttft_avg_ms", "f1", color, "Accuracy vs. time to first token"),
                            width="stretch")
        with c2:
            st.altair_chart(scatter(view, "mem_peak_mb", "decode_tok_s", color, "Decoding speed vs. peak memory"),
                            width="stretch")
    else:
        ds_opts = sorted(view.dataset.dropna().unique())
        ds = st.selectbox("Dataset", ds_opts, key="ann_ds")
        sub = view[view.dataset == ds]
        st.altair_chart(ann_tradeoff(sub, ds), width="stretch")
        st.caption("The classic ANN-Benchmarks plot: upper-right is better. "
                   "The line connects each index type's Pareto-optimal configurations.")

    st.subheader("Custom chart")
    c1, c2, c3, c4 = st.columns(4)
    x = c1.selectbox("X axis", cfg_cols + metrics, index=cfg_cols.index("index") if "index" in cfg_cols else 0,
                     key=f"cx_{kind}")
    y = c2.selectbox("Y axis", metrics, key=f"cy_{kind}", format_func=lambda m: LABELS.get(m, m))
    color = c3.selectbox("Colour", ["(none)"] + cfg_cols, key=f"cc_{kind}")
    color = None if color == "(none)" else color
    logy = c4.toggle("Log Y", key=f"cl_{kind}")
    st.altair_chart(custom_chart(view, x, y, color, logy, x in cfg_cols), width="stretch")


def scatter(df, x, y, color, title):
    d = df.dropna(subset=[x, y])
    return (alt.Chart(d, title=title).mark_circle(size=80, opacity=0.8)
            .encode(x=alt.X(x, title=LABELS.get(x, x), scale=alt.Scale(zero=False)),
                    y=alt.Y(y, title=LABELS.get(y, y), scale=alt.Scale(zero=False)),
                    color=alt.Color(f"{color}:N", title=color),
                    tooltip=["run", "suite", color, alt.Tooltip(x, format=".2f"), alt.Tooltip(y, format=".3f")])
            .interactive().properties(height=340))


def pareto(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Points not dominated on (higher x, higher y)."""
    d = df.dropna(subset=[x, y]).sort_values(x, ascending=False)
    best, keep = -float("inf"), []
    for i, r in d.iterrows():
        if r[y] > best:
            keep.append(i)
            best = r[y]
    return d.loc[keep].sort_values(x)


def ann_tradeoff(sub: pd.DataFrame, ds: str):
    tip = ["run", "index_params", alt.Tooltip("recall", format=".4f"), alt.Tooltip("qps", format=",.0f")]
    base = alt.Chart(sub, title=f"{ds}: recall vs. throughput").encode(
        x=alt.X("recall", title="Recall@k", scale=alt.Scale(zero=False)),
        y=alt.Y("qps", title="Queries / s (log)", scale=alt.Scale(type="log")),
        color=alt.Color("index:N", title="index"))
    pts = base.mark_circle(size=60, opacity=0.5).encode(tooltip=tip)
    front = pd.concat([pareto(g, "recall", "qps") for _, g in sub.groupby("index")]) if not sub.empty else sub
    line = alt.Chart(front).mark_line(point=True, strokeWidth=2).encode(
        x="recall", y=alt.Y("qps", scale=alt.Scale(type="log")), color="index:N", tooltip=tip)
    return (pts + line).interactive().properties(height=420)


def custom_chart(df, x, y, color, logy, x_is_cfg):
    d = df.dropna(subset=[y]).copy()
    yscale = alt.Scale(type="log") if logy else alt.Scale(zero=not logy)
    if x_is_cfg:
        d[x] = d[x].astype(str)
        enc = dict(x=alt.X(f"{x}:N", title=x, sort=None),
                   y=alt.Y(f"mean({y}):Q", title=f"mean {LABELS.get(y, y)}", scale=yscale),
                   tooltip=[f"{x}:N", alt.Tooltip(f"mean({y}):Q", format=".3f"), "count():Q"])
        if color:
            enc |= dict(color=alt.Color(f"{color}:N"), xOffset=alt.XOffset(f"{color}:N"))
        return alt.Chart(d).mark_bar().encode(**enc).properties(height=360)
    return scatter(d, x, y, color or "index", f"{LABELS.get(y, y)} vs. {LABELS.get(x, x)}")


def detail_tab(view: pd.DataFrame, kind: str):
    selected = st.session_state.get(f"selected_{kind}", [])
    focus = st.session_state.get("results_focus")
    paths = view.path.tolist()
    if not paths:
        st.info("No runs match the filters.")
        return
    labels = dict(zip(view.path, view.suite + " / " + view.run))
    default = focus if focus in paths else (selected[0] if selected else paths[0])
    path = st.selectbox("Run", paths, index=paths.index(default), format_func=labels.get, key=f"det_{kind}")
    row = view[view.path == path].iloc[0]

    if kind == "task":
        k = st.columns(6)
        for col, m in zip(k, ["f1", "exact_match", "contains", "ttft_avg_ms", "decode_tok_s", "mem_peak_mb"]):
            col.metric(LABELS[m], fmt(row[m]))
        gens = cached_generations(path)
        if not gens.empty:
            c1, c2 = st.columns(2)
            with c1:
                parts = pd.DataFrame({
                    "stage": ["Query embedding", "Retrieval", "Prefill", "Decoding"],
                    "ms": [gens.query_embed_ms.mean(), gens.retrieve_ms.mean(),
                           gens.prefill_ms.mean(), gens.decoding_ms.mean()],
                })
                st.altair_chart(alt.Chart(parts, title="Average latency per question").mark_bar().encode(
                    x=alt.X("ms:Q", title="ms"), y=alt.Y("stage:N", sort=None, title=None),
                    tooltip=["stage", alt.Tooltip("ms", format=".2f")]).properties(height=220),
                    width="stretch")
            with c2:
                st.altair_chart(alt.Chart(gens, title="TTFT distribution").mark_bar().encode(
                    x=alt.X("ttft_ms:Q", bin=alt.Bin(maxbins=40), title="TTFT (ms)"),
                    y=alt.Y("count():Q", title="questions")).properties(height=220),
                    width="stretch")
            st.markdown("**Per-question output**")
            q = st.text_input("Search questions or responses", key="q_search")
            g = gens
            if q:
                g = g[g.question.str.contains(q, case=False, na=False) | g.response.str.contains(q, case=False, na=False)]
            st.dataframe(g[["index", "question", "response", "ttft_ms", "tok_s", "input_tokens",
                            "generated_tokens", "error"]], hide_index=True, width="stretch", height=320)
            with st.expander("Show retrieved context for one question"):
                qi = st.number_input("Question index", 0, int(gens["index"].max()), 0)
                r = gens[gens["index"] == qi]
                if not r.empty:
                    st.text(r.iloc[0].context or "(none)")
    else:
        k = st.columns(5)
        for col, m in zip(k, ["recall", "precision", "qps", "latency_mean_ms", "mem_peak_mb"]):
            col.metric(LABELS[m], fmt(row[m]))

    run_dir = Path(path)
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    fm = (m.get("task") or m.get("dataset") or {}).get("faiss_metrics") or {}
    stages = {n: (fm.get(f"{n}TimeNs") or {}).get("totalNs") for n in ("chunking", "embedding", "add", "read")}
    stages["training"] = fm.get("trainingTimeNs")
    st_df = pd.DataFrame([{"stage": k_, "s": v * 1e-9} for k_, v in stages.items() if v])
    c1, c2 = st.columns(2)
    with c1:
        if not st_df.empty:
            st.altair_chart(alt.Chart(st_df, title=f"Index construction ({fm.get('buildTimeMs', 0) / 1000:.1f} s build)")
                            .mark_bar().encode(x=alt.X("s:Q", title="seconds"), y=alt.Y("stage:N", sort="-x", title=None),
                                               tooltip=["stage", alt.Tooltip("s", format=".2f")]).properties(height=200),
                            width="stretch")
    with c2:
        hw = m.get("hardware") or {}
        mem, pw = hw.get("memory") or {}, hw.get("power") or {}
        st.markdown("**Hardware**")
        st.markdown(
            f"Memory: {mem.get('beforeMB', '–')} MB before · {mem.get('meanMB', '–')} MB mean · "
            f"**{mem.get('peakMB', '–')} MB peak**  \n"
            f"Power: {fmt((pw.get('powerMw') or {}).get('mean'))} mW mean · "
            f"{fmt((pw.get('powerMw') or {}).get('peak'))} mW peak  \n"
            f"Temperature: {fmt((pw.get('temperatureC') or {}).get('mean'))} °C mean · "
            f"{fmt((pw.get('temperatureC') or {}).get('peak'))} °C peak"
        )
    c1, c2 = st.columns(2)
    with c1.expander("config.json"):
        st.json(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    with c2.expander("metrics.json"):
        st.json(m)


def fmt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if isinstance(v, (int, float)):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3f}"
    return str(v)


CONFIG_PAGE = st.Page(configure_page, title="Configure", icon="⚙️", default=True)
RUN_PAGE = st.Page(run_page, title="Run", icon="▶️", url_path="run")
RESULTS_PAGE = st.Page(results_page, title="Results", icon="📊", url_path="results")

st.navigation([CONFIG_PAGE, RUN_PAGE, RESULTS_PAGE]).run()
