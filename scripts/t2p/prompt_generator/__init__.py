from typing import List, Optional
from enum import Enum

class SamplingMethod(Enum):
    NONE = 0,
    TOP_K = 1,
    TOP_P = 2

class ProbabilityConversion(Enum):
    CUTOFF_AND_POWER = 0,
    SOFTMAX = 1

class GenerationSettings:
    def __init__(
        self,
        tag_range: int = 0,
        conversion: ProbabilityConversion = ProbabilityConversion.CUTOFF_AND_POWER,
        prob_power: float = 2,
        sampling: SamplingMethod = SamplingMethod.TOP_K, 
        n:int = 20, 
        k: Optional[int] = 50, 
        p: Optional[float] = 0.3, 
        weighted: bool = True):

        self.tag_range = tag_range
        self.sampling = sampling
        self.conversion = conversion
        self.n = n
        self.p = p
        self.k = k
        self.weighted = weighted
        self.prob_power = prob_power


class PromptGenerator:
    def clear(self): pass
    def load_data(self, model_id: str, data_id: str):
        raise NotImplementedError()
    def load_model(self, model_id: str):
        raise NotImplementedError()
    def ready(self) -> bool:
        raise NotImplementedError()
    def __call__(self, text: str, text_neg: str, neg_weight: float, settings: GenerationSettings) -> List[str]:
        raise NotImplementedError()