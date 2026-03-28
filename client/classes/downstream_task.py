from enum import Enum
from typing import Optional
from pydantic import BaseModel, model_validator
from classes.common import SamplingMethod

class DownstreamTaskName(Enum):
    TRIVIA_QA = "trivia_qa"
    SQUAD = "squad"
    HOTPOT_QA = "hotpot_qa"
    
class CorpusScope(Enum):
    LIMITED = "limited" # Use documents from the "limit" questions only
    ALL = "all" # Use all documents in the dataset
    
class DownstreamTask(BaseModel):
    name: DownstreamTaskName
    sampling_method: SamplingMethod = SamplingMethod.RANDOM
    corpus_limit: Optional[int] = None
    limit: int = -1
    seed: int = 42

    @model_validator(mode='after')
    def set_corpus_limit_default(self) -> 'DownstreamTask':
        if self.corpus_limit is None:
            self.corpus_limit = self.limit
        elif self.corpus_limit < self.limit:
            raise ValueError(f"corpus_limit ({self.corpus_limit}) must be at least equal to limit ({self.limit})")
        return self