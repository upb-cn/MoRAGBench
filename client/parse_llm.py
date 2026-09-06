from classes.llm import SupportedLLMDType, SupportedLLM, LLM
import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import shutil

# LLM Tuple of (hf_model_path, not_supported_dtype)
LLM_CONFIGS = {
    SupportedLLM.QWEN25_0_5B: ("onnx-community/Qwen2.5-0.5B", None),
    SupportedLLM.QWEN25_1_5B: ("onnx-community/Qwen2.5-1.5B", None),
    SupportedLLM.LLAMA32_1B: ("onnx-community/Llama-3.2-1B-Instruct-ONNX", None),
    SupportedLLM.SMOLLM2_1_7B: ("HuggingFaceTB/SmolLM2-1.7B-Instruct", None),
}

def parse_llm(llm: LLM, token: str | None, llm_dir: str):
    name = llm.model_name
    dtype = llm.dtype

    # Prepare config
    llm_hf_path = LLM_CONFIGS[name][0]
    non_supported_dtype = LLM_CONFIGS[name][1]

    # Check dtype support
    if non_supported_dtype is not None and dtype == non_supported_dtype:
        raise ValueError(f"Dtype {dtype.value} is not supported for LLM {name.value}")

    dir_path = f"{llm_dir}/{name.value}_{dtype.value}"
    os.makedirs(dir_path, exist_ok=True)

    if name == SupportedLLM.QWEN25_0_5B or name == SupportedLLM.QWEN25_1_5B:
        # Figure out model file name based on dtype
        MODEL_NAME_BY_DTYPE = {
            SupportedLLMDType.FLOAT32: "model.onnx",
            SupportedLLMDType.FLOAT16: "model_fp16.onnx",
            SupportedLLMDType.INT8: "model_int8.onnx",
            SupportedLLMDType.UINT8: "model_uint8.onnx",
            SupportedLLMDType.BNB4: "model_bnb4.onnx",
            SupportedLLMDType.Q4: "model_q4.onnx",
            SupportedLLMDType.Q4F16: "model_q4f16.onnx",
        }
        model_name = MODEL_NAME_BY_DTYPE.get(dtype)

        # Load LLM
        try:
            snapshot_download(
                repo_id=llm_hf_path,
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
                raise RuntimeError("Error downloading LLM model") from e

    # Llama 3.2
    elif name == SupportedLLM.LLAMA32_1B:
        MODEL_NAME_BY_DTYPE = {
            SupportedLLMDType.FLOAT32: ("model.onnx", ["model.onnx_data", "model.onnx_data_1", "model.onnx_data_2"]),
            SupportedLLMDType.FLOAT16: ("model_fp16.onnx", ["model_fp16.onnx_data", "model_fp16.onnx_data_1"]),
            SupportedLLMDType.INT8: ("model_int8.onnx", []),
            SupportedLLMDType.UINT8: ("model_uint8.onnx", []),
            SupportedLLMDType.BNB4: ("model_bnb4.onnx", []),
            SupportedLLMDType.Q4: ("model_q4.onnx", ["model_q4.onnx_data"]),
            SupportedLLMDType.Q4F16: ("model_q4f16.onnx", ["model_q4f16.onnx_data"]),
        }

        if dtype not in MODEL_NAME_BY_DTYPE:
            raise ValueError(f"Dtype {dtype.value} is not supported for {name.value}")

        model_name, sidecar_names = MODEL_NAME_BY_DTYPE[dtype]

        # Build list of model files to download from new repo (better maintained, all dtypes)
        model_patterns = [f"onnx/{model_name}"]
        for sidecar in sidecar_names:
            model_patterns.append(f"onnx/{sidecar}")

        try:
            # Download model weights from new repo
            snapshot_download(
                repo_id=llm_hf_path,
                local_dir=dir_path,
                token=token,
                allow_patterns=model_patterns,
            )

            # Download tokenizer from GENAI-ONNX repo — required because the
            # native qwen_tokenizer library fails to initialize with the new
            # repo's tokenizer.json format (confirmed: "Failed to init tokenizer")
            genai_folder = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
            snapshot_download(
                repo_id="onnx-community/Llama-3.2-1B-Instruct-GENAI-ONNX",
                local_dir=dir_path,
                token=token,
                allow_patterns=[f"{genai_folder}/tokenizer.json"],
            )

            # Move model file to dir_path root as model.onnx
            shutil.move(
                os.path.join(dir_path, "onnx", model_name),
                os.path.join(dir_path, "model.onnx"),
            )

            # Move sidecar files keeping their original names
            for sidecar in sidecar_names:
                shutil.move(
                    os.path.join(dir_path, "onnx", sidecar),
                    os.path.join(dir_path, sidecar),
                )

            # Move tokenizer from GENAI-ONNX nested folder to root
            shutil.move(
                os.path.join(dir_path, genai_folder, "tokenizer.json"),
                os.path.join(dir_path, "tokenizer.json"),
            )

            # Clean up empty folders
            shutil.rmtree(os.path.join(dir_path, "cpu_and_mobile"), ignore_errors=True)
            os.removedirs(os.path.join(dir_path, "onnx"))

        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                raise RuntimeError("Invalid or missing Hugging Face token") from e
            else:
                raise RuntimeError("Error downloading LLM model") from e
    # SmolLM2 1.7B
    elif name == SupportedLLM.SMOLLM2_1_7B:
        MODEL_NAME_BY_DTYPE = {
            SupportedLLMDType.FLOAT32: ("model.onnx", ["model.onnx_data"]),
            SupportedLLMDType.FLOAT16: ("model_fp16.onnx", ["model_fp16.onnx_data"]),
            SupportedLLMDType.INT8: ("model_int8.onnx", []),
            SupportedLLMDType.UINT8: ("model_uint8.onnx", []),
            SupportedLLMDType.BNB4: ("model_bnb4.onnx", []),
            SupportedLLMDType.Q4: ("model_q4.onnx", []),
            SupportedLLMDType.Q4F16: ("model_q4f16.onnx", []),
        }

        if dtype not in MODEL_NAME_BY_DTYPE:
            raise ValueError(f"Dtype {dtype.value} is not supported for {name.value}")

        model_name, sidecar_names = MODEL_NAME_BY_DTYPE[dtype]

        patterns = ["tokenizer.json", f"onnx/{model_name}"]
        for sidecar in sidecar_names:
            patterns.append(f"onnx/{sidecar}")

        try:
            snapshot_download(
                repo_id=llm_hf_path,
                local_dir=dir_path,
                token=token,
                allow_patterns=patterns,
            )

            shutil.move(
                os.path.join(dir_path, "onnx", model_name),
                os.path.join(dir_path, "model.onnx"),
            )

            for sidecar in sidecar_names:
                shutil.move(
                    os.path.join(dir_path, "onnx", sidecar),
                    os.path.join(dir_path, sidecar),
                )

            os.removedirs(os.path.join(dir_path, "onnx"))

        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                raise RuntimeError("Invalid or missing Hugging Face token") from e
            else:
                raise RuntimeError("Error downloading LLM model") from e
    print(f"\nINFO: LLM has been downloaded and saved at: {dir_path}")