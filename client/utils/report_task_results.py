import json
import numpy as np
from utils.metrics import get_task_metrics
from classes.downstream_task import DownstreamTask

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compute_stats(values):
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
    
    with open(f"{results_dir}/{overall_metrics_file}", "r") as f:
        overall_metrics = json.load(f)
    
    results["overall"] = overall_metrics
    
    with open(f"{results_dir}/{hardware_metrics_file}", "r") as f:
        hardware_metrics = json.load(f)
    
    results["hardware"] = hardware_metrics
    
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
        metrics = obj.get("metrics", {})

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
    with open(f"{results_dir}/{faiss_metrics_file}") as f:
        faiss_metrics = json.load(f)
        
    result_obj["faiss_metrics"] = faiss_metrics
    
    
    results["task"] = result_obj

    return results