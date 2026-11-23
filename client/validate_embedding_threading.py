#!/usr/bin/env python3
"""Compare saved Android embeddings against a Python ONNX Runtime CPU reference."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


SENTENCE = "The quick brown fox jumps over the lazy dog."

MODELS = {
    "bge-small-en-v1.5": {
        "directory": "bge-small-en-v1.5_float32",
        "dtype": "float32",
        "prefix": "Represent this sentence for searching relevant passages: ",
        "pooling": "cls",
        "dim": 384,
    },
    "bge-base-en-v1.5": {
        "directory": "bge-base-en-v1.5_FLOAT32",
        "dtype": "float32",
        "prefix": "Represent this sentence for searching relevant passages: ",
        "pooling": "cls",
        "dim": 768,
    },
    "all-minilm-l6-v2": {
        "directory": "all-minilm-l6-v2",
        "dtype": "int8",
        "prefix": "",
        "pooling": "mean",
        "dim": 384,
    },
}


def reference_embedding(model_dir, config):
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    encoded = tokenizer.encode(config["prefix"] + SENTENCE)

    inputs = {
        "input_ids": np.asarray([encoded.ids], dtype=np.int64),
        "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
        "token_type_ids": np.asarray([encoded.type_ids], dtype=np.int64),
    }

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_dir / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    required_inputs = {item.name for item in session.get_inputs()}
    output = session.run(None, {k: v for k, v in inputs.items() if k in required_inputs})[0]

    if output.ndim == 3:
        tokens = output[0]
        if config["pooling"] == "cls":
            pooled = tokens[0]
        else:
            mask = inputs["attention_mask"][0].astype(np.float32)
            pooled = (tokens * mask[:, None]).sum(axis=0) / max(float(mask.sum()), 1.0)
    elif output.ndim == 2:
        pooled = output[0]
    else:
        raise ValueError(f"Unexpected ONNX output shape: {output.shape}")

    vector = np.asarray(pooled, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    assert vector.size == config["dim"]
    return vector


def compare(reference, android):
    android = np.asarray(android, dtype=np.float64)
    if android.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {android.shape} versus {reference.shape}")
    cosine = float(np.dot(reference, android) / (
        np.linalg.norm(reference) * np.linalg.norm(android)
    ))
    max_difference = float(np.max(np.abs(reference - android)))
    return cosine, max_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("client/task_files/embedding"),
    )
    parser.add_argument(
        "--android-results",
        type=Path,
        default=Path("client/embedding_threading_validation"),
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs=2,
        metavar=("RUN_1", "RUN_2"),
        help="Two Android result directories; defaults to the original results and repeat_2",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        action="append",
        help="Validate only the selected model; may be specified more than once",
    )
    args = parser.parse_args()
    result_dirs = (
        tuple(args.run_dirs)
        if args.run_dirs
        else (args.android_results, args.android_results / "repeat_2")
    )
    selected_models = args.model or [
        "bge-base-en-v1.5",
        "all-minilm-l6-v2",
    ]

    print(f"Python ONNX Runtime: {ort.__version__}")
    print(f"NumPy: {np.__version__}")
    print("Requested execution provider: CPUExecutionProvider")
    print(f"Available execution providers: {ort.get_available_providers()}")

    for name in selected_models:
        config = MODELS[name]
        model_dir = args.models_dir / config["directory"]
        reference = reference_embedding(model_dir, config)
        print(f"\n{name} ({config['dtype']}, {config['pooling']} pooling)")

        stable_vectors = []
        for result_dir in result_dirs:
            if not result_dir.is_dir():
                raise FileNotFoundError(f"Missing Android results directory: {result_dir}")
            for threading in ("default", "single"):
                for repetition in range(3):
                    path = result_dir / (
                        f"{name}_{config['dtype']}_{threading}_{repetition}.json"
                    )
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing Android result: {path}")
                    data = json.loads(path.read_text())
                    expected = {
                        "model": name,
                        "dtype": config["dtype"],
                        "threading": threading,
                        "intra_op_threads": 1 if threading == "single" else None,
                        "repetition": repetition,
                        "input_text": config["prefix"] + SENTENCE,
                        "pooling": config["pooling"].upper(),
                        "normalized": True,
                    }
                    for field, value in expected.items():
                        if data.get(field) != value:
                            raise ValueError(
                                f"{path}: {field} expected {value!r}, "
                                f"got {data.get(field)!r}"
                            )
                    if len(data["embedding"]) != config["dim"]:
                        raise ValueError(f"Unexpected embedding dimensions in {path}")

                    if name.startswith("bge-") and threading == "single":
                        stable_vectors.append(np.asarray(data["embedding"], dtype=np.float64))
                    cosine, max_diff = compare(reference, data["embedding"])
                    print(
                        f"  {result_dir.name:>30} {threading:>7} run={repetition}"
                        f" cosine={cosine:.12f} max_abs_diff={max_diff:.9g}"
                    )
        if name.startswith("bge-"):
            if len(stable_vectors) != 6 or any(not np.array_equal(stable_vectors[0], vector) for vector in stable_vectors[1:]):
                raise AssertionError(f"{name} single-thread Android embeddings are not identical across all six runs")
            print(f"  PASS: all six {name} single-thread Android embeddings are identical")


if __name__ == "__main__":
    main()
