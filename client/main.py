import argparse
import subprocess, requests
import os
import json
import time

from parse_config import parse_config
from parse_downstream_task import parse_task
from parse_ann_dataset import parse_ann
from parse_llm import parse_llm
from parse_embedding_model import parse_embedding
from utils.report_task_results import report_task_results
from utils.report_ann_results import report_ann_results
from utils.log import display_status

from classes.common import TestType

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
SERVER_PACKAGE = "com.example.cli"
SERVER_BASE = f"/sdcard/Android/data/{SERVER_PACKAGE}/files"
SERVER_RESULTS_DIR = "results"
SERVER_ANN_DATASET_DIR = f"{SERVER_BASE}/{ANN_DIR}/ann_dataset"

# Client dir
CLIENT_BASE = "."
DOWNSTREAM_TASK_DIR = f"{CLIENT_BASE}/{TASK_DIR}/downstream_task"
LLM_DIR = f"{CLIENT_BASE}/{TASK_DIR}/llm"
EMBEDDING_DIR = f"{CLIENT_BASE}/{TASK_DIR}/embedding"
ANN_DATASET_DIR = f"{CLIENT_BASE}/{ANN_DIR}/ann_dataset"
CACHE_DIR = f"{CLIENT_BASE}/cache"

PORT = 18080
HTTP_BASE = f"http://127.0.0.1:{PORT}"

def run(cmd):
    print(">", " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("ERROR:", r.stderr)
        raise RuntimeError("adb command failed")
    return r.stdout.strip()

def create_server_dir(path: str):
    subprocess.run(
        ["adb", "shell", "mkdir", "-p", path],
        check=True
    )

def adb_push_files(local_path: str, destination_path: str):
    """
    Copies a local file/dir (with all subdirectories and files)
    to an Android device using adb.

    :param local_path: Path to local file/dir
    :param destination_path: Destination path on Android (e.g. /sdcard/Download)
    """

    if not os.path.exists(local_path):
        raise ValueError(f"Local path does not exist: {local_path}")

    try:
        subprocess.run(
            ["adb", "push", local_path, destination_path],
            check=True
        )
        
        print("Files copied successfully.")
    except subprocess.CalledProcessError as e:
        print("ADB copy failed.")
        raise e


def adb_pull(phone_path, local_path):
    run(["adb", "pull", phone_path, local_path])

def start_service():
    run(["adb", "shell", "am", "start-foreground-service", f"{SERVER_PACKAGE}/.ServerService"])
    
def stop_service():
    try:
        run(["adb", "shell", "am", "stopservice", "-n", f"{SERVER_PACKAGE}/.ServerService"])
    except:
        pass

def adb_forward():
    run(["adb", "forward", f"tcp:{PORT}", f"tcp:{PORT}"])

def start_benchmark(test_type: TestType):
    payload = {"test_type": test_type.value}
    r = requests.post(
        HTTP_BASE + "/start_benchmark",
        json=payload
    )
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
    
    if test_type == TestType.TASK:
        print("\n==== Parsing Downstream Tasks ====\n")
        parse_task(bench_model.downstream_task, bench_model.hf_token, DOWNSTREAM_TASK_DIR)
        
        print("\n==== Parsing LLM ====\n")
        parse_llm(bench_model.rag_pipeline.llm, bench_model.hf_token, LLM_DIR)
        
        print("\n==== Parsing Embedding Model ====\n")
        parse_embedding(bench_model.rag_pipeline.embedding, bench_model.hf_token, EMBEDDING_DIR)
        
        print("\n==== Moving Files to Server ====\n")
        create_server_dir(f"{SERVER_BASE}/{TASK_DIR}")
        adb_push_files(DOWNSTREAM_TASK_DIR, f"{SERVER_BASE}/{TASK_DIR}")
        adb_push_files(LLM_DIR, f"{SERVER_BASE}/{TASK_DIR}")
        adb_push_files(EMBEDDING_DIR, f"{SERVER_BASE}/{TASK_DIR}")
        adb_push_files(f"{CLIENT_BASE}/{TASK_CONFIG_PATH}", f"{SERVER_BASE}/{TASK_CONFIG_PATH}")
    else:
        print("\n==== Parsing ANN Dataset ====\n")
        parse_ann(bench_model.ann_dataset, ANN_DATASET_DIR, CACHE_DIR)
        
        print("\n==== Moving Files to Server ====\n")
        dataset_name = bench_model.ann_dataset.name.value
        create_server_dir(SERVER_ANN_DATASET_DIR)
        adb_push_files(f"{CLIENT_BASE}/{ANN_DATASET_DIR}/{dataset_name}", SERVER_ANN_DATASET_DIR)
        adb_push_files(f"{CLIENT_BASE}/{ANN_CONFIG_PATH}", f"{SERVER_BASE}/{ANN_CONFIG_PATH}")
        
    # Save config file again
    with open(config_output_path, "w") as f:
        json.dump(bench_model.model_dump(mode="json"), f, indent=2)
    
    print("\n==== Start Server ====\n")
    # Stop any existing service first
    stop_service()
    start_service()

    print(f"\n==== Forwarding Port {PORT} ====\n")
    adb_forward()

    print("\n==== Starting Benchmark ====\n")
    time.sleep(2)  # wait a bit for server to be ready
    start_benchmark(test_type)

    # Poll for status
    display_status(status_stream(), test_type)
    
    # Pull results files
    print("\n==== Pulling Results from Server ====\n")
    if test_type == TestType.TASK:
        results_phone_path = f"{SERVER_BASE}/{TASK_DIR}/{SERVER_RESULTS_DIR}/{bench_model.downstream_task.name.value}/."
    else:
        results_phone_path = f"{SERVER_BASE}/{ANN_DIR}/{SERVER_RESULTS_DIR}/{bench_model.ann_dataset.name.value}/."
    os.makedirs(args.output_path, exist_ok=True)
    adb_pull(results_phone_path, args.output_path)
        
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
    parser.add_argument("--output_path", "-o", help="Path to output results json file.", default=f"{CLIENT_BASE}/benchmark_results")
    parser.add_argument("--set", action="append", default=[], help="Override dotted-key PATH=VALUE")
    args = parser.parse_args()
    
    main(args)
