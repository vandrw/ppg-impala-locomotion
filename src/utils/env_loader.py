from pathlib import Path
import numpy as np

import gym

from opensim_env.action.concrete import DumbExampleController
from opensim_env.context import OpensimEnvConfig
from opensim_env.data import TrainingData
from opensim_env.env import OpensimEnv
from opensim_env.interface.core import OpensimGymEnv


def make_env(env_type, data_subject, visualize):

    if data_subject == "AB06":
        data_path = Path("data") / "AB06_transformed_inDegrees_14,2.csv"
        data = TrainingData(data_path, start_time=14.2)
    elif data_subject == "AB23":
        data_path = Path("data") / "AB23_transformed_inDegrees_6,52.csv"
        data = TrainingData(data_path, start_time=6.52)
    else:
        raise ValueError("The provided subject name does not exist.")

    if env_type == "healthy":
        from opensim_env.models import HEALTHY_PATH
        from opensim_env.observation.concrete import RobinHealthyObserver
        from opensim_env.reward.concrete import RobinHealthyEvaluator

        from opensim_env.observation.concrete.robin_healthy import (
            RobinHealthyObserverConfig,
        )
        from opensim_env.reward.concrete.robin_healthy import (
            RobinHealthyEvaluatorConfig,
        )

        obs_config = RobinHealthyObserverConfig(
            include_body_parts=True, include_imitation=False
        )

        rew_config = RobinHealthyEvaluatorConfig(
            data=data,
            target_pelvis_vel=1.17,
            include_imi_velocity=True,
            include_goal=True,
        )

        return OpensimEnv(
            OpensimEnvConfig(
                HEALTHY_PATH, init_pose=data.get_row(0), visualize=visualize
            ),
            lambda c: RobinHealthyObserver(c, data, obs_config),
            DumbExampleController,
            lambda c: RobinHealthyEvaluator(c, rew_config),
        )
    elif env_type == "healthy_terrain":
        from opensim_env.models import HEALTHY_ROUGH_TERRAIN_PATH
        from opensim_env.observation.concrete import RobinHealthyObserver
        from opensim_env.reward.concrete import RobinHealthyEvaluator

        from opensim_env.observation.concrete.robin_healthy import (
            RobinHealthyObserverConfig,
        )
        from opensim_env.reward.concrete.robin_healthy import (
            RobinHealthyEvaluatorConfig,
        )

        obs_config = RobinHealthyObserverConfig(
            include_body_parts=True, include_imitation=False
        )

        rew_config = RobinHealthyEvaluatorConfig(data=data, target_pelvis_vel=1.2)

        return OpensimEnv(
            OpensimEnvConfig(
                HEALTHY_ROUGH_TERRAIN_PATH,
                init_pose=data.get_row(0),
                visualize=visualize,
            ),
            lambda c: RobinHealthyObserver(c, data, obs_config),
            DumbExampleController,
            lambda c: RobinHealthyEvaluator(c, rew_config),
        )

    elif env_type == "healthy_leanne":
        from opensim_env.models import HEALTHY_PATH
        from opensim_env.observation.concrete import RobinHealthyObserver
        from opensim_env.reward.concrete.reward_22musc import RutgerHealthyEvaluator

        from opensim_env.observation.concrete.robin_healthy import (
            RobinHealthyObserverConfig,
        )

        obs_config = RobinHealthyObserverConfig(
            include_body_parts=True, include_imitation=False
        )

        return OpensimEnv(
            OpensimEnvConfig(
                HEALTHY_PATH, init_pose=data.get_row(0), visualize=visualize
            ),
            lambda c: RobinHealthyObserver(c, data, obs_config),
            DumbExampleController,
            lambda c: RutgerHealthyEvaluator(c, data, 1.0, 1.2),
        )

    elif env_type == "prosthesis":
        raise NotImplementedError()

    elif env_type == "prosthesis_terrain":
        raise NotImplementedError()

    else:
        raise ValueError("The environment type specified does not exist.")


def make_gym_env(env_type, data_subject, visualize):
    gym.register(
        "OpenSimEnv-v1",
        entry_point=OpensimGymEnv,
        kwargs=dict(env=lambda: make_env(env_type, data_subject, visualize)),
    )

    env = gym.make("OpenSimEnv-v1", disable_env_checker=True)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # env = gym.wrappers.NormalizeReward(env)

    return env
