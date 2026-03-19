import os
import json
import math
from itertools import product

OUTPUT_DIR = "./configs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_TRAINING_VECTORS_SPACE_MB = 400

# ----------------------------
# Dataset definitions (excluding Kosarak & MovieLens-10M)
# ----------------------------
DATASETS = {
    "deep1b": {"N": 9_990_000, "test_size": 10_000, "dim": 96, "distance": "Angular"},
    "fashion_mnist": {"N": 60_000, "test_size": 10_000, "dim": 784, "distance": "Euclidean"},
    "gist": {"N": 1_000_000, "test_size": 1_000, "dim": 960, "distance": "Euclidean"},
    "glove_25": {"N": 1_183_514, "test_size": 10_000, "dim": 25, "distance": "Angular"},
    "glove_50": {"N": 1_183_514, "test_size": 10_000, "dim": 50, "distance": "Angular"},
    "glove_100": {"N": 1_183_514, "test_size": 10_000, "dim": 100, "distance": "Angular"},
    "glove_200": {"N": 1_183_514, "test_size": 10_000, "dim": 200, "distance": "Angular"},
    "mnist": {"N": 60_000, "test_size": 10_000, "dim": 784, "distance": "Euclidean"},
    "ny_times": {"N": 290_000, "test_size": 10_000, "dim": 256, "distance": "Angular"},
    "sift": {"N": 1_000_000, "test_size": 10_000, "dim": 128, "distance": "Euclidean"},
    "last_fm": {"N": 292_385, "test_size": 50_000, "dim": 65, "distance": "Angular"},
    "coco_i2i": {"N": 113_287, "test_size": 10_000, "dim": 512, "distance": "Angular"},
    "coco_t2i": {"N": 113_287, "test_size": 10_000, "dim": 512, "distance": "Angular"},
}

DISTANCE_TO_METRIC = {
    "Angular": "IP",
    "Euclidean": "L2"
}

TOP_K_VALUES = [1, 3, 5]


def round_int(x):
    return max(1, int(round(x)))


def generate_ivf_configs(N, dim):
    sqrtN = math.sqrt(N)
    nlist_values = [
        round_int(sqrtN),
        round_int(4 * sqrtN),
        round_int(8 * sqrtN),
        round_int(16 * sqrtN),
    ]

    configs = []

    for nlist in nlist_values:
        nprobe_values = [
            max(1, round_int(nlist / 10)),
            max(1, round_int(nlist / 25)),
            max(1, round_int(nlist / 50)),
            max(1, round_int(nlist / 100)),
        ]

        
        training_values = [N]

        for nprobe, num_train in product(nprobe_values, training_values):
            memory_per_vector_bytes = dim * 4  # float32
            current_memory_mb = (num_train * memory_per_vector_bytes) / (1024 * 1024)
            
            if current_memory_mb > MAX_TRAINING_VECTORS_SPACE_MB:
                num_train = int(MAX_TRAINING_VECTORS_SPACE_MB * 1024 * 1024 / (dim * 4))
                print(f"WARNING: Current number of taining vectors will lead to {current_memory_mb:.2f} MB memory usage. The max is {MAX_TRAINING_VECTORS_SPACE_MB} MB. Capping num_training_vectors to {num_train}.")
            

            configs.append({
                "nlist": nlist,
                "nprobe": nprobe,
                "num_training_vectors": num_train,
            })

    return configs


def generate_hnsw_configs(N):
    if N < 100_000:
        m = 8
    elif N < 1_000_000:
        m = 16
    else:
        m = 32

    ef_construction_values = [5 * m, 10 * m, 20 * m]
    ef_search_values = [2 * m, 5 * m, 10 * m]

    configs = []
    for ef_c, ef_s in product(ef_construction_values, ef_search_values):
        configs.append({
            "m": m,
            "ef_construction": ef_c,
            "ef_search": ef_s,
        })
    return configs


def save_config(dataset_name, method, metric, top_k, ann_limit, faiss_cfg):
    config = {
        "ann_dataset": {
            "name": dataset_name,
            "sampling_method": "first_n",
            "limit": ann_limit,
            "seed": 42
        },
        "faiss": {
            "method": method,
            "backend": "cpu",
            "metric": metric,
            "top_k": top_k,
            "batch_size": 2000,
            "config": faiss_cfg,
            "use_cache": True
        }
    }

    # filename encoding configuration
    cfg_part = "_".join(f"{k}-{v}" for k, v in faiss_cfg.items()) if faiss_cfg else "default"
    filename = f"{dataset_name}__{method}__metric-{metric}__k-{top_k}__{cfg_part}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)


def main():
    total = 0

    for dataset_name, info in DATASETS.items():
        N = info["N"]
        limit = info["test_size"]
        dim = info["dim"]
        metric = DISTANCE_TO_METRIC[info["distance"]]

        # FLAT
        for top_k in TOP_K_VALUES:
            save_config(dataset_name, "flat", metric, top_k, limit, {})
            total += 1

        # IVF
        ivf_configs = generate_ivf_configs(N, dim)
        for top_k in TOP_K_VALUES:
            for cfg in ivf_configs:
                save_config(dataset_name, "ivf", metric, top_k, limit, cfg)
                total += 1

        # HNSW
        hnsw_configs = generate_hnsw_configs(N)
        for top_k in TOP_K_VALUES:
            for cfg in hnsw_configs:
                save_config(dataset_name, "hnsw", metric, top_k, limit, cfg)
                total += 1

    print(f"Generated {total} configuration files in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
