"""Find benchmark result folders and flatten them into one table.

A result folder is any directory holding both ``metrics.json`` and
``config.json`` (what client/main.py writes to --output_path).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

NS_TO_MS = 1e-6


def _get(d: Any, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def find_result_dirs(roots: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for m in root.rglob("metrics.json"):
            if (m.parent / "config.json").exists():
                found.add(m.parent.resolve())
    return sorted(found)


def _faiss_params(faiss: dict) -> str:
    cfg = faiss.get("config") or {}
    return ", ".join(f"{k}={v}" for k, v in cfg.items()) or "-"


def flatten_task(run_dir: Path, cfg: dict, m: dict) -> dict:
    dt, rp = cfg.get("downstream_task", {}), cfg.get("rag_pipeline", {})
    emb, llm, faiss = rp.get("embedding", {}), rp.get("llm", {}), rp.get("faiss", {})
    gen = _get(m, "task", "generation_metrics", default={})
    acc = gen.get("accuracy", {}) or {}
    fm = _get(m, "task", "faiss_metrics", default={}) or {}
    hw = m.get("hardware", {}) or {}
    retrieve_ns = _get(gen, "retrieve_top_k_docs_ns", "avg")
    return {
        # config
        "dataset": dt.get("name"),
        "questions": dt.get("limit"),
        "corpus": dt.get("corpus_limit"),
        "llm": llm.get("model_name"),
        "llm_dtype": llm.get("dtype"),
        "llm_backend": llm.get("backend"),
        "aug_method": llm.get("aug_method"),
        "max_tokens": llm.get("max_tokens"),
        "embedding": emb.get("model_name"),
        "emb_dtype": emb.get("dtype"),
        "emb_backend": emb.get("backend"),
        "chunker": _get(emb, "chunker", "method"),
        "chunk_size": _get(emb, "chunker", "size"),
        "index": faiss.get("method"),
        "metric": faiss.get("metric"),
        "top_k": faiss.get("top_k"),
        "index_params": _faiss_params(faiss),
        # accuracy
        "f1": acc.get("f1"),
        "exact_match": acc.get("exact_match"),
        "contains": acc.get("contains"),
        "rouge1": acc.get("rouge1"),
        "rougeL": acc.get("rougeL"),
        "bleu": acc.get("bleu"),
        # latency / throughput
        "ttft_avg_ms": _get(gen, "ttft_ms", "avg"),
        "ttft_p95_ms": _get(gen, "ttft_ms", "p95"),
        "decode_tok_s": _get(gen, "decoding_speed_toks_per_sec", "avg"),
        "tpot_avg_ms": _get(gen, "tpot_ms", "avg"),
        "query_embed_ms": _get(gen, "query_embeddings_ms", "avg"),
        "retrieve_ms": retrieve_ns * NS_TO_MS if retrieve_ns is not None else None,
        "index_build_s": (fm.get("buildTimeMs") or 0) / 1000 if fm.get("buildTimeMs") is not None else None,
        "total_min": (_get(m, "overall", "benchmarkTimeMs") or 0) / 60000 or None,
        # hardware
        "mem_peak_mb": _get(hw, "memory", "peakMB"),
        "mem_mean_mb": _get(hw, "memory", "meanMB"),
        "mem_before_mb": _get(hw, "memory", "beforeMB"),
        "power_mean_mw": _get(hw, "power", "powerMw", "mean"),
        "temp_peak_c": _get(hw, "power", "temperatureC", "peak"),
    }


def flatten_ann(run_dir: Path, cfg: dict, m: dict) -> dict:
    ds, faiss = cfg.get("ann_dataset", {}), cfg.get("faiss", {})
    dm = _get(m, "dataset", "metrics", default={}) or {}
    fm = _get(m, "dataset", "faiss_metrics", default={}) or {}
    hw = m.get("hardware", {}) or {}
    k = str(faiss.get("top_k"))
    lat = dm.get("latency", {}) or {}
    row = {
        "dataset": ds.get("name"),
        "vectors": ds.get("limit"),
        "index": faiss.get("method"),
        "metric": faiss.get("metric"),
        "top_k": faiss.get("top_k"),
        "batch_size": faiss.get("batch_size"),
        "index_params": _faiss_params(faiss),
        "recall": _get(dm, "recall@k", k),
        "precision": _get(dm, "precision@k", k),
        "qps": dm.get("qps"),
        "latency_mean_ms": lat["mean_ns"] * NS_TO_MS if lat.get("mean_ns") is not None else None,
        "latency_p95_ms": lat["p95_ns"] * NS_TO_MS if lat.get("p95_ns") is not None else None,
        "build_ms": fm.get("buildTimeMs"),
        "train_ms": fm["trainingTimeNs"] * NS_TO_MS if fm.get("trainingTimeNs") is not None else None,
        "mem_peak_mb": _get(hw, "memory", "peakMB"),
        "mem_mean_mb": _get(hw, "memory", "meanMB"),
        "power_mean_mw": _get(hw, "power", "powerMw", "mean"),
    }
    # individual index hyperparameters as their own columns (nlist, nprobe, m, ...)
    for p, v in (faiss.get("config") or {}).items():
        row[p] = v
    return row


def load_runs(roots: list[Path], repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (task_runs, ann_runs). Index = run path relative to the repo."""
    task_rows, ann_rows = [], []
    for d in find_result_dirs(roots):
        try:
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            rel = str(d.relative_to(repo_root))
        except ValueError:
            rel = str(d)
        suite = str(Path(rel).parent).replace("\\", "/")
        suite = suite.removeprefix("benchmark/").removesuffix("/results") or suite
        base = {"run": d.name, "suite": suite, "path": str(d)}
        if "downstream_task" in cfg:
            task_rows.append(base | flatten_task(d, cfg, m))
        elif "ann_dataset" in cfg:
            ann_rows.append(base | flatten_ann(d, cfg, m))
    task = pd.DataFrame(task_rows)
    ann = pd.DataFrame(ann_rows)
    return task, ann


def load_generations(run_dir: Path) -> pd.DataFrame:
    """Per-question records from generation_metrics.jsonl (task runs only)."""
    rows = []
    for f in sorted(run_dir.glob("generation_metrics*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                mt = r.get("metrics") or {}
                rows.append({
                    "index": r.get("index"),
                    "question": r.get("question"),
                    "response": r.get("response"),
                    "context": r.get("context_text"),
                    "error": r.get("error"),
                    "input_tokens": mt.get("input_tokens"),
                    "generated_tokens": mt.get("generated_tokens"),
                    "ttft_ms": mt.get("ttft_ms"),
                    "prefill_ms": mt.get("prefill_ms"),
                    "decoding_ms": mt.get("decoding_duration_ms"),
                    "tok_s": mt.get("decoding_speed_toks_per_sec"),
                    "query_embed_ms": mt.get("query_embeddings_ms"),
                    "retrieve_ms": (mt.get("retrieve_top_k_docs_ns") or 0) * NS_TO_MS,
                    "total_ms": mt.get("overall_duration_ms"),
                })
    return pd.DataFrame(rows).drop_duplicates("index", keep="last") if rows else pd.DataFrame()


def config_columns(df: pd.DataFrame, kind: str) -> list[str]:
    """Columns describing the configuration (used for filters / grouping)."""
    if kind == "task":
        cols = ["suite", "dataset", "questions", "corpus", "llm", "llm_dtype", "llm_backend", "aug_method",
                "max_tokens", "embedding", "emb_dtype", "emb_backend", "chunker", "chunk_size", "index",
                "metric", "top_k", "index_params"]
    else:
        cols = ["suite", "dataset", "vectors", "index", "metric", "top_k", "batch_size", "index_params",
                "nlist", "nprobe", "num_training_vectors", "m", "ef_construction", "ef_search"]
    return [c for c in cols if c in df.columns]
