from pathlib import Path

import gym

from opensim_env.action.concrete import DumbExampleController
from opensim_env.context import OpensimEnvConfig
from opensim_env.data import TrainingData
from opensim_env.env import OpensimEnv
from opensim_env.interface.core import OpensimGymEnv

def make_env(env_type, visualize):
    data_path = Path("data") / "motion_AB06_13,297.csv"
    data = TrainingData(data_path, start_time=14.2)

    if env_type == "healthy":
        from opensim_env.models import HEALTHY_PATH
        from opensim_env.observation.concrete import RobinHealthyObserver
        from opensim_env.reward.concrete import RobinHealthyEvaluator

        return OpensimEnv(
            OpensimEnvConfig(HEALTHY_PATH, init_pose=data.get_row(0), visualize=visualize),
            lambda c: RobinHealthyObserver(c, data),
            DumbExampleController,
            lambda c: RobinHealthyEvaluator(c, data, 0.01, 1.0),
        )
    elif env_type == "healthy_terrain":
        from opensim_env.models import HEALTHY_ROUGH_TERRAIN_PATH
        from opensim_env.observation.concrete import RobinHealthyObserver
        from opensim_env.reward.concrete import RobinHealthyEvaluator

        return OpensimEnv(
            OpensimEnvConfig(HEALTHY_ROUGH_TERRAIN_PATH, init_pose=data.get_row(0), visualize=visualize),
            lambda c: RobinHealthyObserver(c, data),
            DumbExampleController,
            lambda c: RobinHealthyEvaluator(c, data, 0.01, 1.0),
        )
    
    elif env_type == "prosthesis":
        raise NotImplementedError()
    
    elif env_type == "prosthesis_terrain":
        raise NotImplementedError()
    
    else:
        raise ValueError("The environment type specified does not exist.")


def make_gym_env(env_type, visualize):
    gym.register(
        "OpenSimEnv-v1", entry_point=OpensimGymEnv, kwargs=dict(env=lambda: make_env(env_type, visualize))
    )
    return "OpenSimEnv-v1"