from typing import Union
from classes.common import Backend
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, PositiveInt, model_validator

class DistanceMetric(Enum):
    L2 = "L2"
    INNER_PRODUCT = "IP"
    
class IndexMethod(Enum):
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"

@dataclass
class FlatConfig(BaseModel):
    """No hyperparameters (for now)."""
    pass

@dataclass
class IVFConfig(BaseModel):
    nprobe: PositiveInt
    num_training_vectors: PositiveInt
    nlist: PositiveInt


@dataclass
class HNSWConfig(BaseModel):
    m: PositiveInt
    ef_construction: PositiveInt
    ef_search: PositiveInt

@dataclass
class Faiss(BaseModel):
    config: Union[FlatConfig, IVFConfig, HNSWConfig]
    backend: Backend = Backend.CPU
    metric: DistanceMetric = DistanceMetric.L2
    top_k: PositiveInt = 3
    batch_size: PositiveInt = 1000
    method: IndexMethod = IndexMethod.FLAT
    use_cache: bool = True
    
    @model_validator(mode="after")
    def validate_method_config(self):

        config = self.config
        method = self.method

        is_flat = isinstance(config, FlatConfig)
        is_ivf = isinstance(config, IVFConfig)
        is_hnsw = isinstance(config, HNSWConfig)
        
        method_val = method.value

        if method_val == "flat":
            if not is_flat:
                print("WARN: faiss.method is 'flat' but config of type "
                    f"{type(config).__name__} was provided. "
                    "No config is required for flat; it will be ignored."
                )

        elif method_val == "ivf":
            if not is_ivf:
                raise ValueError(
                    f"faiss.method == 'ivf' requires IVFConfig "
                    f"{tuple(IVFConfig.model_fields.keys())}"
                )

            # cross-field IVF constraint
            if config.nprobe > config.nlist:
                raise ValueError(
                    "faiss.config.nprobe cannot be larger than faiss.config.nlist"
                )
                
            if config.nlist > config.num_training_vectors:
                raise ValueError(
                    "faiss.config.nlist should be <= faiss.config.num_training_vectors"
                )

        elif method_val == "hnsw":
            if not is_hnsw:
                raise ValueError(
                    f"faiss.method == 'hnsw' requires HNSWConfig "
                    f"{tuple(HNSWConfig.model_fields.keys())}"
                )

        return self
