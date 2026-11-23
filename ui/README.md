# MoRAGBench Web UI

A browser interface for configuring benchmarks, running them on the phone and inspecting the results. It runs on the laptop next to the Python client. The phone stays the benchmark server; the UI only writes config files, calls `client/main.py` and reads result folders.

## Start

From the repository root, with the client's virtual environment active:

```
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Then open http://localhost:8501.

Running benchmarks needs the same setup as the CLI: `client/.env` (copy `client/.env.example`), a phone connected over ADB, and the `cli` app installed on it. Configuring and browsing results work without a phone.

## Pages

**Configure**: builds a config with a form instead of hand-editing JSON.
- Every option, default and numeric bound is read from the Pydantic classes in `client/classes/`. A new model, dataset, dtype or index type added there appears in the form automatically; the UI has no list of its own.
- Settings that have no effect are hidden, using the same rules as `parse_config.py` (sampling parameters when `use_sampling` is off, `overlap_size` when overlap is off, `seed` unless sampling is random).
- The index parameters follow the chosen `faiss.method` (none for flat, nlist/nprobe/num_training_vectors for IVF, m/ef_construction/ef_search for HNSW).
- The config is validated live with the exact Pydantic validation `main.py` runs, including cross-field checks such as nprobe ≤ nlist.
- Existing configs in `benchmark/` can be loaded as a starting point. Saved configs go to `benchmark/ui_configs/`.

**Run**: starts `client/main.py` for a saved config on a chosen ADB device.
- Shows live indexing and evaluation progress from the phone's `/status` endpoint, plus the client log.
- Supports `--resume`. A Hugging Face token entered on the Configure page is passed with `--set` and is never written to the config file.
- Results go to `benchmark/ui_results/<config>_<timestamp>/` by default.
- The job keeps running if the browser tab is refreshed.

**Results**: reads every folder under `benchmark/` (and `client/benchmark_results/`) that contains `metrics.json` and `config.json`. More folders can be added in the sidebar.
- *Runs*: one row per run with its configuration and metrics, filterable by any setting, downloadable as CSV.
- *Compare*: accuracy vs. TTFT and decoding speed vs. memory for task runs; recall vs. QPS with each index type's Pareto front for ANN runs; a custom chart builder; selected runs side by side.
- *Run detail*: key metrics, average latency per stage, TTFT distribution, index construction breakdown, memory/power/temperature, and every question with its response and retrieved context.

## Files

| File | Purpose |
|---|---|
| `app.py` | The three pages |
| `schema_form.py` | Turns the Pydantic config classes into form widgets |
| `runner.py` | Starts `main.py` in the background and polls the phone's status |
| `results_data.py` | Finds result folders and flattens them into tables |
