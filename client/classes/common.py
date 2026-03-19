from enum import Enum

class Backend(Enum):
    CPU = "cpu"
    XNNPACK = "xnnpack"
    NNAPI = "nnapi"
    
class SamplingMethod(Enum):
    RANDOM = "random"
    FIRST_N = "first_n"
    LAST_N = "last_n"    
    
class TestType(Enum):
    TASK = "task"
    ANN = "ann"