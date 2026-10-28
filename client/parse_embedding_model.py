from classes.embedding import SupportedEmbeddingDType, SupportedEmbeddingModel, Embedding
import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import shutil
import onnx
import json


# LLM Tuple of (hf_model_path, not_supported_dtype)
EMBEDDING_CONFIGS = {
    SupportedEmbeddingModel.ALL_MINILM_L6_V2: (
        "sentence-transformers/all-MiniLM-L6-v2",
        [SupportedEmbeddingDType.FP16,
         SupportedEmbeddingDType.Q4,
         SupportedEmbeddingDType.Q4_NO_GATHER,
         SupportedEmbeddingDType.QUANTIZED]
    ),
    SupportedEmbeddingModel.ALL_MINILM_L12_V2: (
        "sentence-transformers/all-MiniLM-L12-v2",
        [SupportedEmbeddingDType.FP16,
         SupportedEmbeddingDType.Q4,
         SupportedEmbeddingDType.Q4_NO_GATHER,
         SupportedEmbeddingDType.QUANTIZED]
    ),
    SupportedEmbeddingModel.EMBEDDINGGEMMA: (
        "onnx-community/embeddinggemma-300m-ONNX",
        [SupportedEmbeddingDType.INT8,
         SupportedEmbeddingDType.FLOAT32_O1,
         SupportedEmbeddingDType.FLOAT32_O2,
         SupportedEmbeddingDType.FLOAT32_O3,
         SupportedEmbeddingDType.FLOAT32_O4]
         # FP16 is available but not recommended - see EMBEDDINGGEMMA_MODEL_BY_DTYPE
    ),
    SupportedEmbeddingModel.BGE_SMALL_EN_V1_5: (
        "BAAI/bge-small-en-v1.5",
        [SupportedEmbeddingDType.FLOAT32_O1,
         SupportedEmbeddingDType.FLOAT32_O2,
         SupportedEmbeddingDType.FLOAT32_O3,
         SupportedEmbeddingDType.FLOAT32_O4,
         SupportedEmbeddingDType.INT8,
         SupportedEmbeddingDType.Q4,
         SupportedEmbeddingDType.Q4_NO_GATHER,
         SupportedEmbeddingDType.QUANTIZED,
         SupportedEmbeddingDType.FP16]
    ),
    SupportedEmbeddingModel.BGE_BASE_EN_V1_5: (
        "BAAI/bge-base-en-v1.5",
        [SupportedEmbeddingDType.FLOAT32_O1,
         SupportedEmbeddingDType.FLOAT32_O2,
         SupportedEmbeddingDType.FLOAT32_O3,
         SupportedEmbeddingDType.FLOAT32_O4,
         SupportedEmbeddingDType.INT8,
         SupportedEmbeddingDType.Q4,
         SupportedEmbeddingDType.Q4_NO_GATHER,
         SupportedEmbeddingDType.QUANTIZED,
         SupportedEmbeddingDType.FP16]
    ),
}

EMBEDDINGGEMMA_MODEL_BY_DTYPE = {
    SupportedEmbeddingDType.FLOAT32: ("model.onnx", "model.onnx_data"),
    SupportedEmbeddingDType.QUANTIZED: ("model_quantized.onnx", "model_quantized.onnx_data"),
    SupportedEmbeddingDType.Q4: ("model_q4.onnx", "model_q4.onnx_data"),
    SupportedEmbeddingDType.Q4_NO_GATHER: ("model_no_gather_q4.onnx", "model_no_gather_q4.onnx_data"),
    # NOTE: FP16 is listed here for completeness but is NOT supported as of June 2026.
    # EmbeddingGemma activations do not support fp16 or its derivatives.
    # See: https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX
    SupportedEmbeddingDType.FP16: ("model_fp16.onnx", "model_fp16.onnx_data"),
}


def parse_embedding(embedding: Embedding, token: str | None, embedding_dir: str):
    name = embedding.model_name
    dtype = embedding.dtype
    
    # Prepare config
    embedding_hf_path = EMBEDDING_CONFIGS[name][0]
    non_supported_dtype = EMBEDDING_CONFIGS[name][1]
    
    # Check dtype support
    if non_supported_dtype is not None and dtype in non_supported_dtype:
        raise ValueError(f"Dtype {dtype.value} is not supported for Embedding model {name.value}")

    # Folder name encodes both model name and dtype to keep variants separate
    dir_path = f"{embedding_dir}/{name.value}_{dtype.value}"
    os.makedirs(dir_path, exist_ok=True)

    model_path = os.path.join(dir_path, "model.onnx")
    tokenizer_path = os.path.join(dir_path, "tokenizer.json")

    # Skip download if this specific dtype variant already exists
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        print(f"\nINFO: Embedding model already exists at: {dir_path}, skipping download")
        return

    if name == SupportedEmbeddingModel.ALL_MINILM_L6_V2 or name == SupportedEmbeddingModel.ALL_MINILM_L12_V2:
        MODEL_NAME_BY_DTYPE = {
            SupportedEmbeddingDType.FLOAT32: "model.onnx",
            SupportedEmbeddingDType.FLOAT32_O1: "model_O1.onnx",
            SupportedEmbeddingDType.FLOAT32_O2: "model_O2.onnx",
            SupportedEmbeddingDType.FLOAT32_O3: "model_O3.onnx",
            SupportedEmbeddingDType.FLOAT32_O4: "model_O4.onnx",
            SupportedEmbeddingDType.INT8: "model_qint8_arm64.onnx",
        }
        model_name = MODEL_NAME_BY_DTYPE.get(dtype)
        
        try:
            snapshot_download(
                repo_id=embedding_hf_path,
                local_dir=dir_path,
                token=token,
                allow_patterns=[
                    "tokenizer.json",
                    f"onnx/{model_name}",
                ],
            )
            
            shutil.move(
                os.path.join(dir_path, "onnx", model_name),
                model_path,
            )
            os.removedirs(os.path.join(dir_path, "onnx"))
            
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                raise RuntimeError("Invalid or missing Hugging Face token") from e
            else:
                raise RuntimeError("Error downloading Embedding model") from e

    elif name in (
        SupportedEmbeddingModel.BGE_SMALL_EN_V1_5,
        SupportedEmbeddingModel.BGE_BASE_EN_V1_5,
    ):
        try:
            snapshot_download(
                repo_id=embedding_hf_path,
                local_dir=dir_path,
                token=token,
                allow_patterns=[
                    "tokenizer.json",
                    "onnx/model.onnx",
                ],
            )

            shutil.move(
                os.path.join(dir_path, "onnx", "model.onnx"),
                model_path,
            )
            os.removedirs(os.path.join(dir_path, "onnx"))

        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                raise RuntimeError("Invalid or missing Hugging Face token") from e
            else:
                raise RuntimeError("Error downloading BGE embedding model") from e

    elif name == SupportedEmbeddingModel.EMBEDDINGGEMMA:
        model_file, data_file = EMBEDDINGGEMMA_MODEL_BY_DTYPE.get(dtype)
        
        try:
            snapshot_download(
                repo_id=embedding_hf_path,
                local_dir=dir_path,
                token=token,
                allow_patterns=[
                    "tokenizer.json",
                    f"onnx/{model_file}",
                    f"onnx/{data_file}",
                ],
            )
            
            model = onnx.load(
                os.path.join(dir_path, "onnx", model_file),
                load_external_data=True
            )
            onnx.save(
                model,
                model_path,
                save_as_external_data=False
            )
            
            shutil.rmtree(os.path.join(dir_path, "onnx"))
            
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                raise RuntimeError("Invalid or missing Hugging Face token") from e
            else:
                raise RuntimeError("Error downloading EmbeddingGemma model") from e

    print(f"\nINFO: Embedding model has been downloaded and saved at: {dir_path}")