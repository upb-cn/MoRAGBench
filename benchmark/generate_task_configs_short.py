import math
import os
import json
import copy

# In General, these are the things that I want to tune:
# Embedding:
#   - Model: all-minilm-l6-v2 or all-minilm-l12-v2
# FAISS:
#   - top-k: 1, 3, or 5
#   - Method: flat, ivf, or hnsw
#   - For ivf config: 
#       - num_training_vectors will be always everything
#       - Tune nprobe and nlist
#   - For hnsw config:
#       - I will use different combinations, rather than tuning
# LLM:
#   - Model: qwen2.5-0.5B or qwen2.5-1.5B
#   - dtype: int8, q4

OUTPUT_DIR = "./configs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

baseline = {
    "downstream_task": {
        "name": "trivia_qa",
        "sampling_method": "first_n",
        "limit": 1000,
        "corpus_limit": 2500
    },
    "rag_pipeline": {
        "embedding": {
            "backend": "cpu",
            "model_name": "all-minilm-l6-v2",
            "dtype": "int8",
            "chunker": {
                "method": "token",
                "size": 100,
                "overlap_enabled": True,
                "overlap_size": 10
            }
        },
        "faiss": {
            "method": "flat",
            "backend": "cpu",
            "metric": "L2",
            "top_k": 3,
            "config": {},
            "batch_size": 2000,
            "use_cache": False
        },
        "llm": {
            "aug_method": "concatenation",
            "backend": "cpu",
            "model_name": "qwen2.5-0.5B",
            "use_sampling": False,
            "dtype": "int8",
            "kv_window": 4096,
            "prefill_chunk_size": 1024,
            "max_tokens": 50,
            "ignore_eos": False,
            "generate_until": ["\n", "\n\n"],
            "system_prompt": "You are a factual QA assistant.\nAnswer using only the provided documents.\nIf the answer is not in the documents, say 'I don't know.'"
        }
    }
}


# For trivia_qa first 10000 items, there are ~X docs
# With current chunking config (which I'll keep constant),
# this will lead to 247709 chunks

N_values = {
    "trivia_qa": 247709,
}

for dataset in ["trivia_qa"]:
    
    N = N_values[dataset]

    generated_configs = []

    for embedding_model in ["all-minilm-l6-v2", "all-minilm-l12-v2"]:
        new_copy = copy.deepcopy(baseline)
        new_copy["rag_pipeline"]["embedding"]["model_name"] = embedding_model
        generated_configs.append(new_copy)
        
    for embedding_backend in ["xnnpack", "nnapi"]:
        new_copy = copy.deepcopy(baseline)
        new_copy["rag_pipeline"]["embedding"]["backend"] = embedding_backend
        generated_configs.append(new_copy)

    # Configs for IVF index:

    # For nlist, a common heuristic is nlist ≈ sqrt(N)
    # So I will choose sqrt(N), sqrt(N) / 2, and sqrt(N) * 2
    # For nprobe, a common heuristic is nprobe ≈ 1% – 10% of nlist
    # So I will choose 1%, 5%, and 10%

    num_training_vectors = 50_000
    sqrt_n = math.sqrt(N)
    nlist_baseline = int(sqrt_n)
    nprobe_baseline = int(0.05 * nlist_baseline)

    # Tune nlist
    for nlist in [int(sqrt_n / 2), int(sqrt_n), int(sqrt_n * 2)]:
        new_copy = copy.deepcopy(baseline)
        new_copy["rag_pipeline"]["faiss"]["method"] = "ivf"
        new_copy["rag_pipeline"]["faiss"]["config"] = {
            "num_training_vectors": num_training_vectors,
            "nlist": nlist,
            "nprobe": nprobe_baseline
        }
        generated_configs.append(new_copy)
        
        # Also loop over top_k for everyone
        # k = 3 is already covered in the baseline
        for top_k in [1, 5]:
            new_copy = copy.deepcopy(new_copy)
            new_copy["rag_pipeline"]["faiss"]["top_k"] = top_k
            generated_configs.append(new_copy)
            
    # Tune nprobe
    for nprobe in [int(0.01 * nlist_baseline), int(0.1 * nlist_baseline)]:
        new_copy = copy.deepcopy(baseline)
        new_copy["rag_pipeline"]["faiss"]["method"] = "ivf"
        new_copy["rag_pipeline"]["faiss"]["config"] = {
            "num_training_vectors": num_training_vectors,
            "nlist": nlist_baseline,
            "nprobe": nprobe
        }
        generated_configs.append(new_copy)
        
        # Also loop over top_k for everyone
        # k = 3 is already covered in the baseline
        for top_k in [1, 5]:
            new_copy = copy.deepcopy(new_copy)
            new_copy["rag_pipeline"]["faiss"]["top_k"] = top_k
            generated_configs.append(new_copy)
            
            
            
    # Configs for HNSW index:

    # M controls how many edges each node has
    # Usually it's set to M ≈ 12–48, depending on the dataset size
    # ef_construction controls how thoroughly neighbors
    # are selected when building the graph
    # A rule of thumb is to be set it to ef_construction ≈ 4–10 × M
    # ef_search controls mow many neighbor nodes are explored during query time
    # A rule of thumb is to set it to ef_search ≈ 2–5 × M

    # Tune HNSW with 3 different combinations:
    # 1) Speed-optimized
    comb1 = {
        "m": 16,
        "ef_construction": 96,   # 6 × M
        "ef_search": 32,         # 2 × M
    }
    # 2) Balanced
    comb2 = {
        "m": 24,
        "ef_construction": 192,  # 8 × M
        "ef_search": 72,         # 3 × M
    }
    # 3) High-recall
    comb3 = {
        "m": 32,
        "ef_construction": 320,  # 10 × M
        "ef_search": 128,        # 4 × M
    }

    for comb in [comb1, comb2, comb3]:
        new_copy = copy.deepcopy(baseline)
        new_copy["rag_pipeline"]["faiss"]["method"] = "hnsw"
        new_copy["rag_pipeline"]["faiss"]["config"] = {
            "m": comb["m"],
            "ef_construction": comb["ef_construction"],
            "ef_search": comb["ef_search"]
        }
        generated_configs.append(new_copy)
        
        # Also loop over top_k for everyone
        # k = 3 is already covered in the baseline
        for top_k in [1, 5]:
            new_copy = copy.deepcopy(new_copy)
            new_copy["rag_pipeline"]["faiss"]["top_k"] = top_k
            generated_configs.append(new_copy)
            
            
    # Different LLMs
    for llm_model in ["qwen2.5-1.5B"]:
        new_copy = copy.deepcopy(baseline)
        new_copy["rag_pipeline"]["llm"]["model_name"] = llm_model
        generated_configs.append(new_copy)
        
    # Different dtypes
    # int8 is covered in the baseline
    for llm_dtype in ["q4"]:
        new_copy = copy.deepcopy(baseline)
        new_copy["rag_pipeline"]["llm"]["dtype"] = llm_dtype
        generated_configs.append(new_copy)
            
    for llm_backend in ["xnnpack", "nnapi"]:
        new_copy = copy.deepcopy(baseline)
        new_copy["rag_pipeline"]["llm"]["backend"] = llm_backend
        generated_configs.append(new_copy)

    # Write output configs
    for i,config in enumerate(generated_configs):
        filepath = os.path.join(OUTPUT_DIR, f"{dataset}_{i}.json")

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)