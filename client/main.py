import argparse
import os
import json
import time

import requests

from parse_config import parse_config
from parse_downstream_task import parse_task
from parse_ann_dataset import parse_ann
from parse_llm import parse_llm
from parse_embedding_model import parse_embedding
from utils.report_task_results import report_task_results
from utils.report_ann_results import report_ann_results
from utils.log import display_status

from classes.common import TestType
from helpers import (
    HTTP_BASE,
    PORT,
    SERVER_BASE,
    adb_forward,
    adb_pull,
    adb_push_files,
    prepare_dirs,
    push_dir_files,
    start_service,
    stop_service,
)

# Common dirs
TASK_DIR = "task_files"
TASK_RESULT_DIR = f"task_results"
TASK_CONFIG_PATH = f"{TASK_DIR}/config.json"
GENERATION_METRICS_FILE = "generation_metrics.jsonl"
FAISS_METRICS_FILE = "faiss_metrics.json"
OVERALL_METRICS_FILE = f"overall.json"
HARDWARE_METRICS_FILE = f"hardware_metrics.json"

ANN_DIR = "ann_files"
ANN_RESULT_DIR = f"ann_results"
ANN_RESULTS_FILE = "results.json"
ANN_CONFIG_PATH = f"{ANN_DIR}/config.json"

# Server dirs
SERVER_RESULTS_DIR = "results"
SERVER_ANN_DATASET_DIR = f"{SERVER_BASE}/{ANN_DIR}/ann_dataset"

# Client dir
CLIENT_BASE = "."
DOWNSTREAM_TASK_DIR = f"{CLIENT_BASE}/{TASK_DIR}/downstream_task"
LLM_DIR = f"{CLIENT_BASE}/{TASK_DIR}/llm"
EMBEDDING_DIR = f"{CLIENT_BASE}/{TASK_DIR}/embedding"
ANN_DATASET_DIR = f"{CLIENT_BASE}/{ANN_DIR}/ann_dataset"
CACHE_DIR = f"{CLIENT_BASE}/cache"

def start_benchmark(device, test_type: TestType, resume: bool = False):
    payload = {"test_type": test_type.value, "resume": resume}
    r = requests.post(HTTP_BASE + "/start_benchmark", json=payload)
    r.raise_for_status()
    print("Started benchmark")

def get_status():
    try:
        r = requests.get(HTTP_BASE + "/status")
        r.raise_for_status()
        return r.json()
    except:
        print("Failed to get status from server")
        return None

def status_stream():
    error_count = 0
    while True:
        st = get_status()
        
        if st is None:
            error_count += 1
            if error_count == 10:
                print("Too many errors getting status from server. Exiting.")
                raise RuntimeError("Failed to get status from server. Server is probably down.")
        else:
            error_count = 0
        
        yield st

        if st is not None and st.get("overallState") in ("COMPLETED", "FAILED"):
            if st["overallState"] == "FAILED":
                print(f"Benchmark failed with error: {st.get('lastError')}")
            break

        time.sleep(1)

def main(args):
    device = args.device

    # Infer test type from config
    # Simple idea: If "downstream_task" in config, it's a TASK test, else ANN test
    with open(args.config, "r") as f:
        raw_cfg = json.load(f)
    
    if "downstream_task" in raw_cfg:
        test_type = TestType.TASK
    else:
        test_type = TestType.ANN
    
    # Parse config
    print("\n==== Parsing Config ====\n")
    print(f"Config Path: {args.config}")
    config_output_path = TASK_CONFIG_PATH if test_type == TestType.TASK else ANN_CONFIG_PATH
    bench_model = parse_config(args.set, raw_cfg, test_type, config_output_path)
    
    # Save config file again
    with open(config_output_path, "w") as f:
        json.dump(bench_model.model_dump(mode="json"), f, indent=2)
    
    print("\n==== Start Server ====\n")
    # Stop any existing service first
    stop_service(device)
    start_service(device)

    print(f"\n==== Forwarding Port {PORT} ====\n")
    adb_forward(device)
    
    print("\n==== Preparing App-Owned Directories ====\n")
    time.sleep(2)  # let the server come up before HTTP calls

    if test_type == TestType.TASK:
        task_name = bench_model.downstream_task.name.value
        llm_name = f"{bench_model.rag_pipeline.llm.model_name.value}_{bench_model.rag_pipeline.llm.dtype.value}"
        embed_name = f"{bench_model.rag_pipeline.embedding.model_name.value}_{bench_model.rag_pipeline.embedding.dtype.value}"

        dirs_to_create = [
            TASK_DIR,
            f"{TASK_DIR}/downstream_task",
            f"{TASK_DIR}/downstream_task/{task_name}",
            f"{TASK_DIR}/llm",
            f"{TASK_DIR}/llm/{llm_name}",
            f"{TASK_DIR}/embedding",
            f"{TASK_DIR}/embedding/{embed_name}",
            f"{TASK_DIR}/{SERVER_RESULTS_DIR}",
        ]
        prepare_dirs(dirs_to_create)

        print("\n==== Parsing Downstream Tasks ====\n")
        parse_task(bench_model.downstream_task, bench_model.hf_token, DOWNSTREAM_TASK_DIR)

        print("\n==== Parsing LLM ====\n")
        parse_llm(bench_model.rag_pipeline.llm, bench_model.hf_token, LLM_DIR)

        print("\n==== Parsing Embedding Model ====\n")
        parse_embedding(bench_model.rag_pipeline.embedding, bench_model.hf_token, EMBEDDING_DIR)

        print("\n==== Moving Files to Server ====\n")
        push_dir_files(
            device,
            f"{DOWNSTREAM_TASK_DIR}/{task_name}",
            f"{SERVER_BASE}/{TASK_DIR}/downstream_task/{task_name}",
        )
        push_dir_files(
            device,
            f"{LLM_DIR}/{llm_name}",
            f"{SERVER_BASE}/{TASK_DIR}/llm/{llm_name}",
        )
        push_dir_files(
            device,
            f"{EMBEDDING_DIR}/{embed_name}",
            f"{SERVER_BASE}/{TASK_DIR}/embedding/{embed_name}",
        )
        adb_push_files(device, f"{CLIENT_BASE}/{TASK_CONFIG_PATH}", f"{SERVER_BASE}/{TASK_CONFIG_PATH}")
    else:
        dataset_name = bench_model.ann_dataset.name.value

        dirs_to_create = [
            ANN_DIR,
            f"{ANN_DIR}/ann_dataset",
            f"{ANN_DIR}/ann_dataset/{dataset_name}",
            f"{ANN_DIR}/{SERVER_RESULTS_DIR}",
        ]
        prepare_dirs(dirs_to_create)

        print("\n==== Parsing ANN Dataset ====\n")
        parse_ann(bench_model.ann_dataset, ANN_DATASET_DIR, CACHE_DIR)

        print("\n==== Moving Files to Server ====\n")
        push_dir_files(
            device,
            f"{CLIENT_BASE}/{ANN_DATASET_DIR}/{dataset_name}",
            f"{SERVER_ANN_DATASET_DIR}/{dataset_name}",
        )
        adb_push_files(device, f"{CLIENT_BASE}/{ANN_CONFIG_PATH}", f"{SERVER_BASE}/{ANN_CONFIG_PATH}")
        

    print("\n==== Starting Benchmark ====\n")
    time.sleep(2)  # wait a bit for server to be ready
    start_benchmark(device, test_type, resume=args.resume)

    # Poll for status
    display_status(status_stream(), test_type)
    
    # Pull results files
    print("\n==== Pulling Results from Server ====\n")
    if test_type == TestType.TASK:
        results_phone_path = f"{SERVER_BASE}/{TASK_DIR}/{SERVER_RESULTS_DIR}/{bench_model.downstream_task.name.value}/."
    else:
        results_phone_path = f"{SERVER_BASE}/{ANN_DIR}/{SERVER_RESULTS_DIR}/{bench_model.ann_dataset.name.value}/."

    # Clean up old local results if not resuming
    if not args.resume and os.path.exists(args.output_path):
        for f in os.listdir(args.output_path):
            os.remove(os.path.join(args.output_path, f))
            
    os.makedirs(args.output_path, exist_ok=True)
    adb_pull(device, results_phone_path, args.output_path)
        
    # Report results
    print("\n==== Reporting Results ====\n")
    if test_type == TestType.TASK:
        results = report_task_results(
            results_dir = args.output_path,
            task_dir = f"{CLIENT_BASE}/{TASK_DIR}",
            generation_metrics_file = GENERATION_METRICS_FILE,
            faiss_metrics_file = FAISS_METRICS_FILE,
            overall_metrics_file = OVERALL_METRICS_FILE,
            hardware_metrics_file=HARDWARE_METRICS_FILE,
            task = bench_model.downstream_task
        )
    else:
        results = report_ann_results(
            results_dir=args.output_path,
            ann_dataset_dir=ANN_DATASET_DIR,
            results_file=ANN_RESULTS_FILE,
            faiss_metrics_file=FAISS_METRICS_FILE,
            overall_metrics_file=OVERALL_METRICS_FILE,
            hardware_metrics_file=HARDWARE_METRICS_FILE,
            dataset=bench_model.ann_dataset
        )
    
    # Write metric results
    with open(f"{args.output_path}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Write the config file used
    with open(f"{args.output_path}/config.json", "w") as f:
        json.dump(bench_model.model_dump(mode="json"), f, indent=2)
        
    print(f"\nResults are Saved in: {args.output_path}\n")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load BenchmarkTest JSON config (Pydantic) with validations")
    parser.add_argument("--config", "-c", required=True, help="Path to config.json")
    parser.add_argument("--device", "-d", required=True, help="ADB device serial number (adb devices)")
    parser.add_argument("--output_path", "-o", help="Path to output results json file.", default=f"{CLIENT_BASE}/benchmark_results")
    parser.add_argument("--set", action="append", default=[], help="Override dotted-key PATH=VALUE")
    parser.add_argument("--resume", "-r",action="store_true",default=False,help="Resume a previously interrupted benchmark")
    args = parser.parse_args()
    
    main(args)
