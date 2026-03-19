import urllib
from classes.ann_dataset import ANNDataset, ANNDatasetName
import os
import shutil
from utils.shared import sample_items
import h5py
import numpy as np
import json
import faiss


# Tuple of (dataset_path, should_normalize)
DATASET_CONFIGS = {
    ANNDatasetName.DEEP1B: ("https://ann-benchmarks.com/deep-image-96-angular.hdf5", True),
    ANNDatasetName.FASHION_MNIST: ("https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5", False),
    ANNDatasetName.GIST: ("https://ann-benchmarks.com/gist-960-euclidean.hdf5", False),
    ANNDatasetName.GLOVE_25: ("https://ann-benchmarks.com/glove-25-angular.hdf5", True),
    ANNDatasetName.GLOVE_50: ("https://ann-benchmarks.com/glove-50-angular.hdf5", True),
    ANNDatasetName.GLOVE_100: ("https://ann-benchmarks.com/glove-100-angular.hdf5", True),
    ANNDatasetName.GLOVE_200: ("https://ann-benchmarks.com/glove-200-angular.hdf5", True),
    ANNDatasetName.MNIST: ("https://ann-benchmarks.com/mnist-784-euclidean.hdf5", False),
    ANNDatasetName.NYTIMES: ("https://ann-benchmarks.com/nytimes-256-angular.hdf5", True),
    ANNDatasetName.SIFT: ("https://ann-benchmarks.com/sift-128-euclidean.hdf5", False),
    ANNDatasetName.LAST_FM: ("https://ann-benchmarks.com/lastfm-64-dot.hdf5", True),
    ANNDatasetName.COCO_I2I: ("https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-i2i-512-angular.hdf5", True),
    ANNDatasetName.COCO_T2I: ("https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-t2i-512-angular.hdf5", True),
}

def download_file(url, path):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response, open(path, "wb") as out_file:
        out_file.write(response.read())


def save_array(arr, path, dtype):
    arr = np.asarray(arr, dtype=dtype)
    arr.tofile(path)
    return {
        "path": path.split("/")[-1],
        "shape": list(arr.shape),
        "dtype": str(arr.dtype)
    }

def parse_ann(dataset: ANNDataset, ann_dataset_dir: str, cache_dir: str):
    name = dataset.name
    limit = dataset.limit
    sampling_method = dataset.sampling_method
    seed = dataset.seed
    
    if limit < 0:
        limit = -1
        print(f"INFO: Limit for dataset {name} is set to a negative value. All items will be processed")
    
    # Prepare config
    dataset_url = DATASET_CONFIGS[name][0]
    should_normalize = DATASET_CONFIGS[name][1]
    full_dataset_path = f"{cache_dir}/{name.value}.hd5"
    os.makedirs(cache_dir, exist_ok=True)

    # Downloading full dataset    
    if not os.path.exists(full_dataset_path):
        print(f"INFO: Downloading dataset...")
        download_file(dataset_url, full_dataset_path)
        print("INFO: Download complete.")
    else:
        print(f"INFO: Dataset file {full_dataset_path} already exists.")
    
    # Read full dataset
    with h5py.File(f"./cache/{name.value}.hd5", "r") as f:
        train = f["train"][:]
        test = f["test"][:]
        neighbors = f["neighbors"][:]
        distances = f["distances"][:]
        
    train = train.astype(np.float32, copy=False)
    test = test.astype(np.float32, copy=False)
        
    if should_normalize:
        faiss.normalize_L2(train)
        faiss.normalize_L2(test)
    
    # Sample from the items
    test_sub, idx = sample_items(
        task_name=name.value,
        items=test,
        limit=limit,
        sampling_method=sampling_method,
        seed=seed
    )
    # Apply same sampling for neighbors and distances
    neighbors_sub = neighbors[idx]
    distances_sub = distances[idx]
    
    print("INFO: Normalizing vectors (L2)...")

    # Change limit to actual number of items
    dataset.limit = len(test_sub)
    
    # Delete if exists
    if os.path.exists(f"{ann_dataset_dir}/{name.value}"):
        shutil.rmtree(f"{ann_dataset_dir}/{name.value}")
    
    # Create directory
    os.makedirs(ann_dataset_dir, exist_ok=True)
    
    # Save new sampled dataset
    # Now convert it to .f32 format
    print("INFO: Converting + writing binary files...")
    
    output_file = f"{ann_dataset_dir}/{name.value}"
    os.makedirs(output_file, exist_ok=True)
    meta = {}

    meta["train"] = save_array(
        train,
        f"{output_file}/train.f32",
        np.float32
    )

    meta["test"] = save_array(
        test_sub,
        f"{output_file}/test.f32",
        np.float32
    )

    meta["neighbors"] = save_array(
        neighbors_sub,
        f"{output_file}/neighbors.i32",
        np.int32
    )

    meta["distances"] = save_array(
        distances_sub,
        f"{output_file}/distances.f32",
        np.float32
    )

    with open(f"{output_file}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(
        f"\nINFO: Finished parsing and convreting ann dataset: {name.value}", 
        f"\nINFO: Documents saved at: {output_file}"
    )