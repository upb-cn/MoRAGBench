from typing import List
from classes.common import Backend
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, PositiveFloat, PositiveInt, Field


class AugmentationMethod(Enum):
    # Only this is supported for now
    CONCATENATION = "concatenation"
    
class SupportedLLM(Enum):
    QWEN25_0_5B = "qwen2.5-0.5B"
    QWEN25_1_5B = "qwen2.5-1.5B"
    
class SupportedLLMDType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"
    UINT8 = "uint8"
    BNB4 = "bnb4"
    Q4 = "q4"
    Q4F16 = "q4f16"

@dataclass
class LLM(BaseModel):
    model_name: SupportedLLM
    aug_method: AugmentationMethod = AugmentationMethod.CONCATENATION
    backend: Backend = Backend.CPU
    use_sampling: bool = False
    repetition_penalty: float = Field(1, ge=1)
    dtype: SupportedLLMDType = SupportedLLMDType.INT8
    temp: PositiveFloat = 0.8
    top_p: float = Field(0.95, gt=0, le=1)
    top_k: PositiveInt = 0
    system_prompt: str = "You are a helpful assistant. Use the following retrieved documents to answer the user's query:"
    kv_window: PositiveInt = 2048
    prefill_chunk_size: PositiveInt = 512
    max_tokens: PositiveInt = 512
    ignore_eos: bool = True
    generate_until: List[str] | None = None