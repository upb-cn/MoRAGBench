from enum import Enum
from classes.common import SamplingMethod
from pydantic import BaseModel

class ANNDatasetName(Enum):
    DEEP1B = "deep1b"
    FASHION_MNIST = "fashion_mnist"
    GIST = "gist"
    GLOVE_25 = "glove_25"
    GLOVE_50 = "glove_50"
    GLOVE_100 = "glove_100"
    GLOVE_200 = "glove_200"
    KOSARAK = "kosarak"
    MNIST = "mnist"
    MOVIELENS_10M = "movielens_10m"
    NYTIMES = "ny_times"
    SIFT = "sift"
    LAST_FM = "last_fm"
    COCO_I2I = "coco_i2i"
    COCO_T2I = "coco_t2i"
    
class ANNDataset(BaseModel):
    name: ANNDatasetName
    sampling_method: SamplingMethod = SamplingMethod.RANDOM
    limit: int = -1
    seed: int = 42