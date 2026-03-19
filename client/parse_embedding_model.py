from classes.embedding import SupportedEmbeddingDType, SupportedEmbeddingModel, Embedding
import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import shutil


# LLM Tuple of (hf_model_path, not_supported_dtype)
EMBEDDING_CONFIGS = {
    SupportedEmbeddingModel.ALL_MINILM_L6_V2: ("sentence-transformers/all-MiniLM-L6-v2", None),
    SupportedEmbeddingModel.ALL_MINILM_L12_V2: ("sentence-transformers/all-MiniLM-L12-v2", None),
}


def parse_embedding(embedding: Embedding, token: str | None, embedding_dir: str):
    name = embedding.model_name
    dtype = embedding.dtype
    
    # Prepare config
    embedding_hf_path = EMBEDDING_CONFIGS[name][0]
    non_supported_dtype = EMBEDDING_CONFIGS[name][1]
    
    # Check dtype support
    if non_supported_dtype is not None and dtype == non_supported_dtype:
        raise ValueError(f"Dtype {dtype.value} is not supported for Embedding model {name.value}")

    dir_path = f"{embedding_dir}/{name.value}"
    os.makedirs(dir_path, exist_ok=True)

    if name == SupportedEmbeddingModel.ALL_MINILM_L6_V2 or name == SupportedEmbeddingModel.ALL_MINILM_L12_V2:
        # Figure out model file name based on dtype
        MODEL_NAME_BY_DTYPE = {
            SupportedEmbeddingDType.FLOAT32: "model.onnx",
            SupportedEmbeddingDType.FLOAT32_O1: "model_O1.onnx",
            SupportedEmbeddingDType.FLOAT32_O2: "model_O2.onnx",
            SupportedEmbeddingDType.FLOAT32_O3: "model_O3.onnx",
            SupportedEmbeddingDType.FLOAT32_O4: "model_O4.onnx",
            SupportedEmbeddingDType.INT8: "model_qint8_arm64.onnx",
        }
        model_name = MODEL_NAME_BY_DTYPE.get(dtype)
        
        # Load LLM
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
            
            # Now move onnx/model_name to to dir_path/model.onnx
            shutil.move(
                os.path.join(dir_path, "onnx", model_name),
                os.path.join(dir_path, "model.onnx"),
            )
            
            # Remove empty onnx directory
            os.removedirs(os.path.join(dir_path, "onnx"))
            
        except HfHubHTTPError as e:
            # Authentication / authorization errors
            if e.response is not None and e.response.status_code in (401, 403):
                raise RuntimeError("Invalid or missing Hugging Face token") from e
            else:
                raise RuntimeError("Error downloading Embedding model") from e
    
    print(f"\nINFO: LLM has been downloaded and saved at: {dir_path}")