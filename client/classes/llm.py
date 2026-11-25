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
    engine: str = "onnx"
    backend: Backend = Backend.CPU
    use_sampling: bool = False
    repetition_penalty: float = Field(1, ge=1)
    dtype: SupportedLLMDType = SupportedLLMDType.INT8
    temp: PositiveFloat = 0.8
    top_p: float = Field(0.95, gt=0, le=1)
    top_k: PositiveInt = 0
    system_prompt: str = "You are a helpful assistant. Use the following retrieved documents to answer the user's query:"
    investigator_system_prompt: str = "Follow the instructions exactly."
    # every LLM instruction for the agentic pipeline (client-editable, pushed
    # to the device before each run); %s placeholders in order:
    #   answerer_prompt: question, bridge-line block, evidence
    #   investigator_prompt: question, searches already tried, evidence so far
    answerer_prompt: str = (
        "Answer the question using only the evidence below.\n"
        "\n"
        "Question: %s\n"
        "%sEvidence: %s\n"
        "\n"
        "Write one line starting with REASON: first say which fact the question asks\n"
        "for, then the passage facts in order that lead to that fact, making sure the\n"
        "last fact is the requested one and not the other side of a relation.\n"
        "Then one line starting with ANSWER: with only the shortest exact answer,\n"
        "copied from the evidence, at most 5 words, with no extra or changed words.\n"
        "Do not answer with an intermediate entity.\n"
        "For yes/no questions answer exactly yes or no.\n"
        "Only when no evidence supports the answer, answer: not in evidence.\n"
        "\n"
        "Example:\n"
        "REASON: the question asks who founded Kestrel Corp; passage 2 says <founder> did.\n"
        "ANSWER: <short answer>"
    )
    investigator_prompt: str = (
        "Check every link needed to answer, in order. If a link is missing, query\n"
        "for it using the known entity. Never repeat the question or a tried query.\n"
        "\n"
        "Question: \"%s\"\n"
        "Searches already tried:\n"
        "%s\n"
        "Evidence so far:\n"
        "%s\n"
        "\n"
        "If all links are supported: ANSWERABLE: <answer>\n"
        "Otherwise: QUERY: <search>\n"
        "One line only."
    )
    answerer_retry_prompt: str = "\nKeep the REASON/ANSWER lines; the ANSWER line is one short phrase, no list"
    investigator_retry_prompt: str = "\nExactly one line: QUERY: <search> or ANSWERABLE: <answer>."
    bridge_line: str = "Intermediate fact (established): %s"
    force_complex_mandate: str = "\nAlways reply with QUERY: one more lookup for this question."
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
