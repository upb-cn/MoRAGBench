from classes.embedding import Embedding
from classes.faiss import Faiss
from classes.llm import LLM
from classes.downstream_task import DownstreamTask
from classes.ann_dataset import ANNDataset

from pydantic import BaseModel, model_validator

# TaskBenchmark (on the phone) only builds these index types and does not apply
# num_threads / cpu_pin yet. Reject the rest here instead of failing on the device.
TASK_SUPPORTED_METHODS = ("flat", "ivf", "hnsw")

class RAGPipeline(BaseModel):
    embedding: Embedding
    faiss: Faiss
    llm: LLM

    @model_validator(mode="after")
    def _only_task_supported_faiss(self):
        method = self.faiss.method.value
        if method not in TASK_SUPPORTED_METHODS:
            raise ValueError(
                f"faiss.method '{method}' is only supported for ANN benchmarks; "
                f"task benchmarks support {TASK_SUPPORTED_METHODS}"
            )
        if self.faiss.num_threads is not None or self.faiss.cpu_pin is not None:
            raise ValueError("faiss.num_threads / faiss.cpu_pin are only supported for ANN benchmarks")
        return self
    
class TaskBenchmark(BaseModel):
    downstream_task: DownstreamTask
    rag_pipeline: RAGPipeline
    hf_token: str | None = None
    
class ANNBenchmark(BaseModel):
    ann_dataset: ANNDataset
    faiss: Faiss