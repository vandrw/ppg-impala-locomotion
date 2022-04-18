from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "osim-models"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def get_venv(name, visualize=False, **env_kwargs):
    # venv = get_procgen_venv(env_id=env_name, **env_kwargs)
    if name == "healthy":
        from src.envs.healthy_env import HealthyOpenSimEnv

        model_path = DEFAULT_MODELS_DIR / "healthy-andrei.osim"
        data_path = DEFAULT_DATA_DIR / "AB06.csv"

        return HealthyOpenSimEnv(
            visualize=visualize,
            model_path=model_path,
            data_path=data_path,
            data_start_time=7.0,
            data_tempo=0.9,
        )
    elif name == "prosthesis":
        from src.envs.prosthesis_env import ProsthesisOpenSimEnv

        model_path = DEFAULT_MODELS_DIR / "OS4_gait14dof15musc_2act_LTFP_VR.osim"
        data_path = DEFAULT_DATA_DIR / "new.csv"

        return ProsthesisOpenSimEnv(
            visualize=visualize,
            model_path=model_path,
            data_path=data_path,
            data_start_time=7.0,
            data_tempo=0.9,
        )
