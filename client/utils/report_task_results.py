import json
import numpy as np
from utils.metrics import get_task_metrics
from classes.downstream_task import DownstreamTask
import os

def read_base_and_resumes(results_dir, file_name):
    """Return (base, resumes) for a metrics file: the base run's JSON (or None)
    and the list of <name>_resume_N.json files in order."""
    base = None
    base_path = os.path.join(results_dir, file_name)
    if os.path.exists(base_path):
        with open(base_path, "r") as f:
            base = json.load(f)

    resumes = []
    resume_index = 1
    while True:
        resume_path = os.path.join(results_dir, f"{file_name.removesuffix('.json')}_resume_{resume_index}.json")
        if not os.path.exists(resume_path):
            break
        with open(resume_path, "r") as f:
            resumes.append(json.load(f))
        resume_index += 1
    return base, resumes

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compute_stats(values):
    if not values:
        return {
            "avg": None,
            "min": None,
            "max": None,
            "p95": None,
            "p99": None,
        }
    values = np.array(values)
    return {
        "avg": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def report_task_results(
    results_dir: str,
    task_dir: str,
    generation_metrics_file: str,
    faiss_metrics_file: str,
    overall_metrics_file: str,
    hardware_metrics_file: str,
    task: DownstreamTask
):
    results = {
        "overall": {},
        "task": {}
    }

    # ---- Read overall and hardware metrics ----
    # Like faiss_metrics: "overall" / "hardware" stay the base run's dict ({} if the first
    # run never wrote one), resumed runs go to "overall_resumes" / "hardware_resumes".
    base, resumes = read_base_and_resumes(results_dir, overall_metrics_file)
    results["overall"] = base if base is not None else {}
    results["overall_resumes"] = resumes

    base, resumes = read_base_and_resumes(results_dir, hardware_metrics_file)
    results["hardware"] = base if base is not None else {}
    results["hardware_resumes"] = resumes

    result_obj = {
        "name": task.name.value,
        "generation_metrics": {},
        "faiss_metrics": {}
    }
    
    # ---- Read generation metrics ----
    total_input_tokens = 0
    total_output_tokens = 0
    total_duration_ms = 0

    ttft_values = []
    decoding_speed_values = []
    tpot_values = []
    query_embeddings = []
    retrieve_docs = []

    predictions = []
    references = []

    # Read JSONL
    for obj in load_jsonl(f"{results_dir}/{generation_metrics_file}"):
        metrics = obj.get("metrics") or {}

        total_input_tokens += metrics.get("input_tokens", 0)
        total_output_tokens += metrics.get("generated_tokens", 0)
        total_duration_ms += metrics.get("overall_duration_ms", 0)

        if "ttft_ms" in metrics:
            ttft_values.append(metrics["ttft_ms"])

        if "decoding_speed_toks_per_sec" in metrics:
            decoding_speed_values.append(
                metrics["decoding_speed_toks_per_sec"]
            )

        if "tbt" in metrics:
            tpot_values.extend(metrics["tbt"])
            
        if "query_embeddings_ms" in metrics:
            query_embeddings.append(
                metrics["query_embeddings_ms"]
            )
            
        if "retrieve_top_k_docs_ns" in metrics:
            retrieve_docs.append(
                metrics["retrieve_top_k_docs_ns"]
            )

        predictions.append(obj.get("response"))

    # Load references
    with open(f"{task_dir}/downstream_task/{task.name.value}/references.json", "r") as f:
        references = json.load(f)

    task_metrics = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "overall_duration_ms": total_duration_ms,
        "ttft_ms": compute_stats(ttft_values),
        "decoding_speed_toks_per_sec": compute_stats(decoding_speed_values),
        "query_embeddings_ms": compute_stats(query_embeddings),
        "retrieve_top_k_docs_ns": compute_stats(retrieve_docs),
        "tpot_ms": compute_stats(tpot_values),
        "accuracy": get_task_metrics(predictions, references)
    }
    
    result_obj["generation_metrics"] = task_metrics
    
    
    # ---- Read faiss metrics ----
    # faiss_metrics stays the base run's dict (empty if the first run never wrote one),
    # resumed runs go to faiss_metrics_resumes, so code reading the old key does not break.
    base, resumes = read_base_and_resumes(results_dir, faiss_metrics_file)
    result_obj["faiss_metrics"] = base if base is not None else {}
    result_obj["faiss_metrics_resumes"] = resumes
    
    
    results["task"] = result_obj

    return results