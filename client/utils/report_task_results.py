import json
import numpy as np
from utils.metrics import get_task_metrics
from classes.downstream_task import DownstreamTask
import os

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

    # ---- Read overall metrics (all runs) ----
    overall_runs = []
    
    # Read base run
    base_overall_path = f"{results_dir}/{overall_metrics_file}"
    if os.path.exists(base_overall_path):
        with open(base_overall_path, "r") as f:
            overall_runs.append(json.load(f))
    
    # Read resume runs
    resume_index = 1
    while True:
        resume_path = f"{results_dir}/{overall_metrics_file.removesuffix('.json')}_resume_{resume_index}.json"  
        if not os.path.exists(resume_path):
            break
        with open(resume_path, "r") as f:
            overall_runs.append(json.load(f))
        resume_index += 1


    
    results["overall"] = overall_runs if len(overall_runs) > 1 else (overall_runs[0] if overall_runs else {})

    # ---- Read hardware metrics (all runs) ----
    hardware_runs = []
    
    # Read base run
    base_hardware_path = f"{results_dir}/{hardware_metrics_file}"
    if os.path.exists(base_hardware_path):
        with open(base_hardware_path, "r") as f:
            hardware_runs.append(json.load(f))
    
    # Read resume runs
    resume_index = 1
    while True:
        resume_path = f"{results_dir}/{hardware_metrics_file.removesuffix('.json')}_resume_{resume_index}.json"
        if not os.path.exists(resume_path):
            break
        with open(resume_path, "r") as f:
            hardware_runs.append(json.load(f))
        resume_index += 1
    
    results["hardware"] = hardware_runs if len(hardware_runs) > 1 else (hardware_runs[0] if hardware_runs else {})   
     
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
    
    
    # ---- Read faiss metrics (all runs) ----
    faiss_runs = []

    # Read base run
    base_faiss_path = f"{results_dir}/{faiss_metrics_file}"
    if os.path.exists(base_faiss_path):
        with open(base_faiss_path, "r") as f:
            faiss_runs.append(json.load(f))

    # Read resume runs
    resume_index = 1
    while True:
        resume_path = f"{results_dir}/{faiss_metrics_file.removesuffix('.json')}_resume_{resume_index}.json"
        if not os.path.exists(resume_path):
            break
        with open(resume_path, "r") as f:
            faiss_runs.append(json.load(f))
        resume_index += 1

    result_obj["faiss_metrics"] = faiss_runs if len(faiss_runs) > 1 else (faiss_runs[0] if faiss_runs else {})
    
    
    results["task"] = result_obj

    return results