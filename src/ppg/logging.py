import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class DistributionInfo:
    avg: float
    std: float
    med: float
    max: float
    min: float

    @staticmethod
    def from_array(array: np.array) -> "DistributionInfo":
        array = np.array(array)
        avg = np.mean(array)
        std = np.std(array)
        med = np.median(array)
        max_val = np.max(array)
        min_val = np.min(array)

        return DistributionInfo(avg, std, med, max_val, min_val)


@dataclass
class EpochInfo:
    epoch: int
    time: float
    returns: DistributionInfo
    episode_len: DistributionInfo
    distance: DistributionInfo
    reward_partials: Dict[str, DistributionInfo]