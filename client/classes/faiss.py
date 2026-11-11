from typing import Literal, Optional, Union
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
    IVFPQ = "ivfpq"
    BINARY = "binary"
    BINARY_IVF = "binaryivf"

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
class IVFPQConfig(BaseModel):
    nprobe: PositiveInt
    num_training_vectors: PositiveInt
    nlist: PositiveInt
    m: PositiveInt
    nbits: PositiveInt = 8

@dataclass
class BinaryConfig(BaseModel):
    """No hyperparameters: binary flat is exhaustive Hamming search."""
    pass

@dataclass
class BinaryIVFConfig(BaseModel):
    nprobe: PositiveInt
    num_training_vectors: PositiveInt
    nlist: PositiveInt

_METHOD_CONFIG_CLASS = {
    "flat": FlatConfig,
    "ivf": IVFConfig,
    "hnsw": HNSWConfig,
    "ivfpq": IVFPQConfig,
    "binary": BinaryConfig,
    "binaryivf": BinaryIVFConfig,
}

@dataclass
class Faiss(BaseModel):
    config: Union[FlatConfig, IVFConfig, HNSWConfig, IVFPQConfig, BinaryConfig, BinaryIVFConfig]
    backend: Backend = Backend.CPU
    metric: DistanceMetric = DistanceMetric.L2
    top_k: PositiveInt = 3
    batch_size: PositiveInt = 1000
    method: IndexMethod = IndexMethod.FLAT
    use_cache: bool = True
    num_threads: Optional[PositiveInt] = None  # CPU threads for FAISS; None = all cores
    cpu_pin: Optional[Literal["fastest", "mid", "slowest"]] = None  # pin FAISS to one core type

    @model_validator(mode="before")
    @classmethod
    def _coerce_config_by_method(cls, data):
        """
        Several of the config classes above (e.g. IVFConfig and BinaryIVFConfig) share the
        exact same field names, so pydantic's automatic Union matching can't tell them apart
        by shape alone and will silently pick whichever one comes first in the Union — not
        necessarily the one for the actual method. To avoid that ambiguity, explicitly build
        the correct config class based on `method` before the Union validation runs.
        """
        if isinstance(data, dict):
            method_raw = data.get("method", "flat")
            method_val = method_raw.value if isinstance(method_raw, IndexMethod) else method_raw
            config_cls = _METHOD_CONFIG_CLASS.get(method_val)
            raw_config = data.get("config")
            if config_cls is not None and isinstance(raw_config, dict):
                data = dict(data)
                data["config"] = config_cls.model_validate(raw_config)
        return data

    @model_validator(mode="after")
    def validate_method_config(self):

        config = self.config
        method = self.method

        is_flat = isinstance(config, FlatConfig)
        is_ivf = isinstance(config, IVFConfig)
        is_hnsw = isinstance(config, HNSWConfig)
        is_ivfpq = isinstance(config, IVFPQConfig)
        is_binary = isinstance(config, BinaryConfig)
        is_binary_ivf = isinstance(config, BinaryIVFConfig)

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

        elif method_val == "ivfpq":
            if not is_ivfpq:
                raise ValueError(
                    f"faiss.method == 'ivfpq' requires IVFPQConfig "
                    f"{tuple(IVFPQConfig.model_fields.keys())}"
                )

            # cross-field IVF constraints (same as plain IVF)
            if config.nprobe > config.nlist:
                raise ValueError(
                    "faiss.config.nprobe cannot be larger than faiss.config.nlist"
                )

            if config.nlist > config.num_training_vectors:
                raise ValueError(
                    "faiss.config.nlist should be <= faiss.config.num_training_vectors"
                )

            # PQ training needs at least one point per centroid (2 ** nbits per subquantizer);
            # otherwise FAISS throws and the index stays untrained.
            if not 1 <= config.nbits <= 16:
                raise ValueError("faiss.config.nbits must be between 1 and 16")
            if config.num_training_vectors < 2 ** config.nbits:
                raise ValueError(
                    f"faiss.config.num_training_vectors ({config.num_training_vectors}) must be "
                    f">= 2 ** nbits ({2 ** config.nbits})"
                )

        elif method_val == "binary":
            if not is_binary:
                print("WARN: faiss.method is 'binary' but config of type "
                    f"{type(config).__name__} was provided. "
                    "No config is required for binary; it will be ignored."
                )

        elif method_val == "binaryivf":
            if not is_binary_ivf:
                raise ValueError(
                    f"faiss.method == 'binaryivf' requires BinaryIVFConfig "
                    f"{tuple(BinaryIVFConfig.model_fields.keys())}"
                )

            if config.nprobe > config.nlist:
                raise ValueError(
                    "faiss.config.nprobe cannot be larger than faiss.config.nlist"
                )

            if config.nlist > config.num_training_vectors:
                raise ValueError(
                    "faiss.config.nlist should be <= faiss.config.num_training_vectors"
                )

        return self
