package com.example.cli


object Constants {
    // Common variables
    const val PORT = 18080
    const val LOG_FILE = "server.log"
    const val CONFIG_FILE = "config.json"
    const val HARDWARE_METRICS_FILE = "hardware_metrics.json"

    // Task variables
    const val TASK_BASE_DIR = "task_files"
    const val DOWNSTREAM_TASK_DIR = "$TASK_BASE_DIR/downstream_task"
    const val EMBEDDING_DIR = "$TASK_BASE_DIR/embedding"
    const val LLM_DIR = "$TASK_BASE_DIR/llm"
    const val TASK_RESULTS_DIR = "$TASK_BASE_DIR/results"
    const val MAIN_RESULTS_FILE = "generation_metrics.jsonl"
    const val FAISS_SETUP_FILE = "faiss_metrics.json"
    const val OVERALL_METRICS_FILE = "overall.json"
    const val TASK_CACHE_DIR = "$TASK_BASE_DIR/cache"

    // ANN variables
    const val ANN_BASE_DIR = "ann_files"
    const val ANN_DATASET_DIR = "$ANN_BASE_DIR/ann_dataset"
    const val ANN_CACHE_DIR = "$ANN_BASE_DIR/cache"
    const val ANN_RESULTS_DIR = "$ANN_BASE_DIR/results"
    const val ANN_RESULTS_FILE = "results.json"
}