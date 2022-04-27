from dataclasses import dataclass
from typing import Dict

@dataclass
class DoneInfo:
    reward: float
    episode_len: float
    distance: float
    reward_partials: Dict[str, float]
@dataclass
class EpochInfo:
    epoch: int
    time: float
    info: DoneInfo