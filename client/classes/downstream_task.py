from enum import Enum
from pydantic import BaseModel
from classes.common import SamplingMethod

class DownstreamTaskName(Enum):
    TRIVIA_QA = "trivia_qa"
    SQUAD = "squad"
    HOTPOT_QA = "hotpot_qa"
    
class DownstreamTask(BaseModel):
    name: DownstreamTaskName
    sampling_method: SamplingMethod = SamplingMethod.RANDOM
    limit: int = -1
    seed: int = 42