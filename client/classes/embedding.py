from classes.common import Backend
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field, PositiveInt, conint, model_validator


class ChunkMethod(Enum):
    TOKEN = "token"
    WORD = "word"
    CHARACTER = "character"
    SENTENCE = "sentence"
    
class SupportedEmbeddingModel(Enum):
    ALL_MINILM_L6_V2 = "all-minilm-l6-v2"
    ALL_MINILM_L12_V2 = "all-minilm-l12-v2"
    EMBEDDINGGEMMA = "embeddinggemma"
    BGE_SMALL_EN_V1_5 = "bge-small-en-v1.5"
    BGE_BASE_EN_V1_5 = "bge-base-en-v1.5"
    
class SupportedEmbeddingDType(Enum):
    FLOAT32 = "float32"
    FLOAT32_O1 = "float32-O1"
    FLOAT32_O2 = "float32-O2"
    FLOAT32_O3 = "float32-O3"
    FLOAT32_O4 = "float32-O4"
    INT8 = "int8"
    Q4 = "q4"
    Q4_NO_GATHER = "q4-no-gather"
    QUANTIZED = "quantized"         
    FP16 = "fp16"                    


@dataclass 
class Chunker(BaseModel):
    method: ChunkMethod = ChunkMethod.TOKEN
    size: PositiveInt = 256
    overlap_enabled: bool = True
    overlap_size: PositiveInt = 50
    
    @model_validator(mode="after")
    def check_overlap_vs_size(self):
        if self.overlap_enabled and self.overlap_size > self.size:
            raise ValueError(
                "overlap_size cannot be larger than size"
            )
        return self


@dataclass
class Embedding(BaseModel):
    model_name: SupportedEmbeddingModel
    backend: Backend = Backend.CPU
    chunker: Chunker = Field(default_factory=Chunker)
    dtype: SupportedEmbeddingDType = SupportedEmbeddingDType.INT8