from typing import List, Optional
from enum import Enum

class SamplingMethod(Enum):
    NONE = 0,
    TOP_K = 1,
    TOP_P = 2

class ProbablityConversion(Enum):
    CUTOFF_AND_POWER = 0,
    SOFTMAX = 1

class GenerationSettings:
    def __init__(
        self, 
        conversion: ProbablityConversion = ProbablityConversion.CUTOFF_AND_POWER, 
        prob_power: float = 2,
        sampling: SamplingMethod = SamplingMethod.TOP_K, 
        n:int = 20, 
        k: Optional[int] = 50, 
        p: Optional[float] = 0.3, 
        weighted: bool = True):

        self.sampling = sampling
        self.conversion = conversion
        self.n = n
        self.p = p
        self.k = k
        self.weighted = weighted
        self.prob_power = prob_power


class PromptGenerator:
    def __call__(self, text: str, settings: GenerationSettings) -> List[str]:
        raise NotImplementedError()


from .wd_like import WDLike