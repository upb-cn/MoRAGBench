from classes.embedding import Embedding
from classes.faiss import Faiss
from classes.llm import LLM
from classes.downstream_task import DownstreamTask
from classes.ann_dataset import ANNDataset

from pydantic import BaseModel
    
class RAGPipeline(BaseModel):
    embedding: Embedding
    faiss: Faiss
    llm: LLM
    
class TaskBenchmark(BaseModel):
    downstream_task: DownstreamTask
    rag_pipeline: RAGPipeline
    hf_token: str | None = None
    
class ANNBenchmark(BaseModel):
    ann_dataset: ANNDataset
    faiss: Faiss