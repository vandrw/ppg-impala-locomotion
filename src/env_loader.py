from pathlib import Path
import importlib

import gym

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "osim-models"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def load_ppg_env(env, visualize=False):
    if env == "healthy":
        from src.envs.healthy_env import HealthyOpenSimEnv

        model_path = DEFAULT_MODELS_DIR / "healthy-andrei.osim"
        data_path = DEFAULT_DATA_DIR / "AB06.csv"

        gym.envs.register(
            id="HealthyOpenSimEnv-v1",
            entry_point=HealthyOpenSimEnv,
            max_episode_steps=100000,
            kwargs={
                "visualize": visualize,
                "model_path": model_path,
                "data_path": data_path,
                "data_start_time": 7.0,
                "data_tempo": 0.9,
            },
        )

        return "HealthyOpenSimEnv-v1"
    elif env == "prosthesis":
        from src.envs.prosthesis_env import ProsthesisOpenSimEnv

        model_path = DEFAULT_MODELS_DIR / "OS4_gait14dof15musc_2act_LTFP_VR.osim"
        data_path = DEFAULT_DATA_DIR / "new.csv"

        gym.envs.register(
            id="ProsthesisOpenSimEnv-v1",
            entry_point=ProsthesisOpenSimEnv,
            max_episode_steps=100000,
            kwargs={
                "visualize": visualize,
                "model_path": model_path,
                "data_path": data_path,
                "data_start_time": 7.0,
                "data_tempo": 0.9,
            },
        )

        return "ProsthesisOpenSimEnv-v1"
    else:
        raise Exception("Invalid environment given: {}".format(env))
