from typing import List, Optional
from classes.common import Backend
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, PositiveFloat, PositiveInt, Field, model_validator


class AugmentationMethod(Enum):
    CONCATENATION = "concatenation"
    # Skip retrieval entirely and prompt the LLM with the question only.
    # Used to measure the no-RAG baseline for a downstream task.
    NONE = "none"
    
class SupportedLLM(Enum):
    QWEN25_0_5B = "qwen2.5-0.5B"
    QWEN25_1_5B = "qwen2.5-1.5B"
    LLAMA32_1B = "llama-3.2-1B"
    SMOLLM2_1_7B = "smollm2-1.7B"
    
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
    # Budget for the prompt, kept separate from the generation budget above.
    # Defaults to whatever the KV window leaves once the generated tokens are
    # reserved. Sharing a single budget used to silently truncate prompts.
    max_prompt_tokens: Optional[PositiveInt] = None
    ignore_eos: bool = True
    generate_until: List[str] | None = None

    @model_validator(mode='after')
    def set_max_prompt_tokens_default(self) -> 'LLM':
        if self.max_prompt_tokens is None:
            self.max_prompt_tokens = max(1, self.kv_window - self.max_tokens)
        elif self.max_prompt_tokens + self.max_tokens > self.kv_window:
            raise ValueError(
                f"max_prompt_tokens ({self.max_prompt_tokens}) + max_tokens ({self.max_tokens}) "
                f"exceeds kv_window ({self.kv_window})"
            )
        return self
