import json
import numpy as np
from classes.ann_dataset import ANNDataset
import numpy as np
from utils.metrics import get_ann_metrics
from utils.ann_recall import tie_aware_recall

def load_array(path, shape, dtype):
    arr = np.fromfile(path, dtype=dtype)
    return arr.reshape(shape)

def report_ann_results(
    results_dir: str,
    ann_dataset_dir: str,
    results_file: str,
    faiss_metrics_file: str,
    overall_metrics_file: str,
    hardware_metrics_file: str,
    dataset: ANNDataset
):
    results = {
        "overall": {},
        "dataset": {},
        "hardware": {}
    }

    with open(f"{results_dir}/{overall_metrics_file}", "r") as f:
        overall_metrics = json.load(f)
    
    results["overall"] = overall_metrics
    
    with open(f"{results_dir}/{hardware_metrics_file}", "r") as f:
        hardware_metrics = json.load(f)
    
    results["hardware"] = hardware_metrics
    
    result_obj = {
        "name": dataset.name.value,
        "metrics": {},
        "faiss_metrics": {}
    }

    # ---- Read faiss metrics ----
    with open(f"{results_dir}/{faiss_metrics_file}") as f:
        faiss_metrics = json.load(f)
        
    result_obj["faiss_metrics"] = faiss_metrics
    
    # ---- Read results ----
    
    # Read results
    with open(f"{results_dir}/{results_file}") as f:
        ann_results = json.load(f)["items"]
        
    # Read true neighbors from the dataset
    dataset_dir = f"{ann_dataset_dir}/{dataset.name.value}"
    
    # Read meta file
    with open(f"{dataset_dir}/meta.json", "r") as f:
        meta = json.load(f)
        
    # Load neighbors and distances arrays
    true_neighbors = load_array(
        path=f"{dataset_dir}/neighbors.i32",
        shape=meta["neighbors"]["shape"],
        dtype=meta["neighbors"]["dtype"]
    )
    
    # Compute metrics
    metrics = get_ann_metrics(
        results=ann_results,
        true_neighbors=true_neighbors
    )
    
    # Tie-aware recall (ann-benchmarks convention): counts equally distant items as hits
    try:
        from parse_ann_dataset import DATASET_CONFIGS
        angular = DATASET_CONFIGS[dataset.name][1]
        K = len(ann_results[0]["neighbors"])
        nb = np.array([r["neighbors"] if r.get("neighbors") else [-1] * K for r in ann_results])
        metrics["tie_aware_recall@k"] = tie_aware_recall(nb, dataset_dir, meta, angular, ks=(1, K))
    except Exception as e:
        print(f"WARN: tie-aware recall not computed: {e}")

    result_obj["metrics"] = metrics
        
    results["dataset"] = result_obj

    return results